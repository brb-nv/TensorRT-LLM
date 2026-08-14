#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Samples host memory composition on the node it runs on, for diagnosing host
# OOM kills and SIGBUS deaths during KV cache host tier allocation and use.
#
# Writes five files per node into the output directory:
#   hostmem_<node>.csv   node-wide /proc/meminfo, one row per sample
#   hostproc_<node>.csv  per-process memory for processes above a size floor
#   hostfrag_<node>.csv  free block counts per buddy allocator order
#   hostvm_<node>.csv    cumulative /proc/vmstat counters, including THP faults
#   hostoom_<node>.log   kernel ring buffer, which carries the OOM killer's
#                        own per-task memory report (only if readable)
#
# Node-wide numbers alone cannot attribute memory, and per-process numbers alone
# miss page cache, which is why both are collected.
#
# The fragmentation and vmstat files exist to tell apart the two ways a host tier
# can die. An OOM kill means the node ran out of pages outright, and the meminfo
# trace shows it. A SIGBUS with BUS_ADRERR instead means a fault could not be
# backed even though pages remained, which points at huge pages: a 2 MiB fault
# needs a physically contiguous block, so it can fail on a node that is merely
# fragmented. hostfrag shows whether such blocks existed, and the thp_fault_*
# counters in hostvm show whether faults were degrading to small pages instead.

set -uo pipefail

outdir="${1:?usage: host_mem_sampler.sh <outdir> [interval_s] [proc_floor_mib]}"
interval="${2:-2}"
# Per-process rows are only useful for processes large enough to matter against
# a several-hundred-GiB budget; the floor keeps the file small over a long run.
floor_mib="${3:-512}"

node="$(hostname -s)"
mem_csv="${outdir}/hostmem_${node}.csv"
proc_csv="${outdir}/hostproc_${node}.csv"
frag_csv="${outdir}/hostfrag_${node}.csv"
vm_csv="${outdir}/hostvm_${node}.csv"
oom_log="${outdir}/hostoom_${node}.log"

mkdir -p "${outdir}"

# MemFree excludes reclaimable page cache while MemAvailable includes it, so
# recording both distinguishes a genuinely full node from one whose memory is
# merely held as cache. Unevictable/Mlocked track pages pinned beyond reclaim.
mem_fields=(MemTotal MemFree MemAvailable Buffers Cached SwapCached Active Inactive
            "Active(file)" "Inactive(file)" "Active(anon)" Unevictable Mlocked Dirty
            Writeback AnonPages Mapped Shmem KReclaimable SReclaimable SUnreclaim
            AnonHugePages ShmemHugePages FileHugePages Hugepagesize
            CommitLimit Committed_AS)
# HugePages_* are page counts in /proc/meminfo, not kilobytes, so they are read
# separately and emitted raw; dividing them by 1024 as if they were sizes would
# silently report a full explicit hugetlb pool as empty.
mem_count_fields=(HugePages_Total HugePages_Free HugePages_Rsvd HugePages_Surp)
proc_fields=(VmRSS RssAnon RssFile RssShmem VmLck VmPin VmHWM VmSize)
# thp_fault_fallback rising means huge page faults degraded to small pages, which
# is survivable; a SIGBUS with these counters flat points elsewhere. compact_fail
# records the allocator giving up on making a contiguous block.
vm_fields=(nr_free_pages nr_anon_transparent_hugepages thp_fault_alloc
           thp_fault_fallback thp_fault_fallback_charge thp_collapse_alloc
           thp_collapse_alloc_failed compact_stall compact_fail compact_success
           pgmajfault unevictable_pgs_mlocked)
# Buddy allocator orders reported by /proc/buddyinfo. Order N holds blocks of
# 2^N base pages, so the order that matters for a huge page depends on the base
# page size; all orders are kept and Hugepagesize in hostmem resolves which.
max_order=10

{
    printf 'ts_epoch,ts_iso'
    printf ',%s_MiB' "${mem_fields[@]}"
    printf ',%s_count' "${mem_count_fields[@]}"
    printf '\n'
} > "${mem_csv}"

{
    printf 'ts_epoch,pid,comm'
    printf ',%s_MiB' "${proc_fields[@]}"
    printf '\n'
} > "${proc_csv}"

{
    printf 'ts_epoch,ts_iso'
    for order in $(seq 0 "${max_order}"); do printf ',order%s_blocks' "${order}"; done
    printf '\n'
} > "${frag_csv}"

{
    printf 'ts_epoch,ts_iso'
    printf ',%s' "${vm_fields[@]}"
    printf '\n'
} > "${vm_csv}"

# The OOM killer prints a full per-task memory table to the ring buffer. That
# report is the only record of the node's state at the instant of the kill, so
# stream it if the kernel allows unprivileged reads.
if dmesg -T &> /dev/null; then
    dmesg -TW &> "${oom_log}" &
    dmesg_pid=$!
else
    echo "dmesg not readable from this context; OOM killer report unavailable" > "${oom_log}"
    dmesg_pid=""
fi

cleanup() {
    [ -n "${dmesg_pid}" ] && kill "${dmesg_pid}" 2> /dev/null
    exit 0
}
trap cleanup TERM INT

echo "host_mem_sampler on ${node}: interval=${interval}s floor=${floor_mib}MiB -> ${mem_csv}"
echo "  base page size $(getconf PAGESIZE) B, $(awk '/Hugepagesize/ {print $2" kB huge pages"}' /proc/meminfo)"

while true; do
    now="$(date +%s)"
    iso="$(date -d "@${now}" +%Y-%m-%dT%H:%M:%S)"

    awk -v ts="${now}" -v iso="${iso}" -v fields="${mem_fields[*]}" \
        -v counts="${mem_count_fields[*]}" '
        BEGIN { n = split(fields, want, " "); m = split(counts, raw, " ") }
        {
            name = $1; sub(/:$/, "", name)
            val[name] = $2
        }
        END {
            printf "%s,%s", ts, iso
            for (i = 1; i <= n; i++) {
                # Absent fields are reported as empty rather than zero so that a
                # kernel without the counter is not mistaken for a zero reading.
                if (want[i] in val) printf ",%.2f", val[want[i]] / 1024
                else printf ","
            }
            for (i = 1; i <= m; i++) {
                if (raw[i] in val) printf ",%d", val[raw[i]]
                else printf ","
            }
            printf "\n"
        }
    ' /proc/meminfo >> "${mem_csv}"

    # Summed across zones: a fault needs a contiguous block from some zone, but
    # per-zone detail is more than is needed to see the node run out of them.
    awk -v ts="${now}" -v iso="${iso}" -v maxorder="${max_order}" '
        # Rows read "Node 0, zone <name> <order0> <order1> ...", so the counts
        # start at field 5 and order K is field K+5.
        $1 == "Node" && $3 == "zone" {
            for (i = 5; i <= NF; i++) blocks[i - 5] += $i
        }
        END {
            printf "%s,%s", ts, iso
            for (o = 0; o <= maxorder; o++) {
                if (o in blocks) printf ",%d", blocks[o]
                else printf ","
            }
            printf "\n"
        }
    ' /proc/buddyinfo >> "${frag_csv}" 2> /dev/null

    awk -v ts="${now}" -v iso="${iso}" -v fields="${vm_fields[*]}" '
        BEGIN { n = split(fields, want, " ") }
        { val[$1] = $2 }
        END {
            printf "%s,%s", ts, iso
            for (i = 1; i <= n; i++) {
                if (want[i] in val) printf ",%d", val[want[i]]
                else printf ","
            }
            printf "\n"
        }
    ' /proc/vmstat >> "${vm_csv}" 2> /dev/null

    # All status files go through a single awk pass; forking per process makes a
    # sample take longer than the sample interval on a busy node. FNR==1 marks
    # the start of the next file, which is where the previous one is flushed.
    awk -v ts="${now}" -v floor="${floor_mib}" -v fields="${proc_fields[*]}" '
        function emit() {
            if (pid != "" && ("VmRSS" in val) && val["VmRSS"] / 1024 >= floor) {
                printf "%s,%s,%s", ts, pid, comm
                for (i = 1; i <= n; i++) {
                    if (want[i] in val) printf ",%.2f", val[want[i]] / 1024
                    else printf ","
                }
                printf "\n"
            }
            delete val
            pid = ""; comm = ""
        }
        BEGIN { n = split(fields, want, " ") }
        FNR == 1 { emit() }
        {
            name = $1; sub(/:$/, "", name)
            if (name == "Name") comm = $2
            if (name == "Pid") pid = $2
            val[name] = $2
        }
        END { emit() }
    ' /proc/[0-9]*/status >> "${proc_csv}" 2> /dev/null

    sleep "${interval}"
done
