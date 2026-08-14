#!/usr/bin/env python3
"""Offline check of the KV dynamics trace bookkeeping.

Loads _dyn_trace.py in isolation (it depends only on the standard library) and
drives it through the sequences that matter, so the accounting can be trusted
before a wheel is built around it.
"""

import hashlib
import importlib.util
import os
import pathlib
import sys

os.environ["TLLM_KV_DYN_TRACE"] = "1"
os.environ["TLLM_KV_DYN_TRACE_INTERVAL_S"] = "3600"  # emit only when we ask
os.environ["TLLM_KV_DYN_GHOST_CAP"] = "4"
os.environ["TLLM_KV_DYN_PAGE_FRAC"] = "1"

SRC = (pathlib.Path(__file__).resolve().parents[5]
       / "tensorrt_llm/runtime/kv_cache_manager_v2/_dyn_trace.py")
spec = importlib.util.spec_from_file_location("_dyn_trace", SRC)
assert spec is not None and spec.loader is not None
dyn = importlib.util.module_from_spec(spec)
sys.modules["_dyn_trace"] = dyn
spec.loader.exec_module(dyn)

lines: list = []
dyn._log = lines.append  # the real one pulls in tensorrt_llm.logger

KEYS = [hashlib.sha256(str(i).encode()).digest() for i in range(8)]
st = dyn._state
failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
    print(f"  {'ok  ' if got == want else 'FAIL'} {label} = {got!r}")


print("1. cold lookup: nothing has ever been cached")
dyn.record_match(KEYS[:4], 0, 0)
check("requests", st.requests, 1)
check("blocks wanted", st.requested, 4)
check("cold misses", st.cold, 4)
check("eviction-induced misses", st.evicted, 0)

print("\n2. partial hit, two blocks usable")
dyn.record_match(KEYS[:4], 2, 2)
check("blocks matched", st.matched, 2)
check("cold misses accumulate", st.cold, 6)
check("hits tracked for matched blocks", len(st.hits), 2)

print("\n3. allocator drops the two matched blocks")
dyn.record_drop(KEYS[:2], level=1, free_frac=0.02)
check("drops at level 1", st.drops[1], 2)
check("ghost holds both", len(st.ghost), 2)
check("hit counts moved out of the live map", len(st.hits), 0)
check("per-page lines emitted", len([x for x in lines if "KVDYNPAGE" in x]), 2)
check("served count reached the sample ring", st.hits_at_death.count, 2)

print("\n4. the same prefix is wanted again")
dyn.record_match(KEYS[:4], 0, 0)
check("dropped blocks attributed to eviction", st.evicted, 2)
check("never-cached blocks still cold", st.cold, 8)

print("\n5. structural loss also enters the ghost")
dyn.record_unlink(KEYS[2])
check("unlink counted separately from drops", st.unlinked, 1)
check("unlink did not inflate drop counts", st.drops[1], 2)
dyn.record_match(KEYS[:4], 0, 0)
check("its later miss is eviction-induced", st.evicted, 5)

print("\n5b. a pruned subtree enters the ghost too")
# Blocks invalidated as collateral of a drop never reach the page destructor, so
# a later miss on one would look cold unless the prune itself is recorded.
dyn.record_match(KEYS[4:7], 3, 3)  # cache them so they have hit counts
dyn.record_prune(KEYS[4:7])
check("collateral blocks counted apart from drops", st.pruned_tree, 3)
check("drop counts untouched", st.drops[1], 2)
before = st.evicted
dyn.record_match(KEYS[4:7], 0, 0)
check("their later misses are eviction-induced", st.evicted - before, 3)

print("\n6. ghost capacity is enforced and reported")
dyn.record_drop(KEYS[3:8], level=1, free_frac=0.5)
check("ghost stays at the cap", len(st.ghost), 4)
check("forgotten keys are counted", st.ghost_full > 0, True)

print("\n6b. host fullness is bucketed at both eviction points")
# Bin edges are upper bounds on percent free, so a tier with 0.5% free lands in
# the fullest bucket and one with 30% free lands above the 25% edge.
check("0.5% free is the fullest bucket", dyn._free_bin(0.5), 0)
check("30% free sits above the 25% edge", dyn._free_bin(30.0), 5)
dev_before = list(st.dev_evict_hist)
dyn.record_device_evict(3, 0.005)
check("GPU spill binned by the room the host had",
      [a - b for a, b in zip(st.dev_evict_hist, dev_before)][0], 3)
check("spilled pages counted", st.dev_evicted_pages, 3)
host_before = list(st.host_drop_hist)
dyn.record_drop(KEYS[:2], level=1, free_frac=0.30)
check("host drop binned by the free space it was taken with",
      [a - b for a, b in zip(st.host_drop_hist, host_before)][5], 2)

print("\n7. summary line")
dyn._emit(dyn.time.monotonic())
summary = [x for x in lines if x.startswith("KVDYN ")][-1]
print("  " + summary)
check("window counters reset after emit", (st.requests, st.evicted, st.cold), (0, 0, 0))
check("fullness histograms reset after emit",
      (sum(st.dev_evict_hist), sum(st.host_drop_hist), st.dev_evicted_pages), (0, 0, 0))
for field in ("miss_evicted=", "miss_cold=", "evicted_share=", "free_at_drop_l1_p50=",
              "age_p50=", "served_p50=", "ghost_full=", "unlinked=", "dev_evicted=",
              "hostfree_at_dev_evict=", "hostfree_at_host_drop="):
    check(f"reports {field}", field in summary, True)

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all checks passed")
