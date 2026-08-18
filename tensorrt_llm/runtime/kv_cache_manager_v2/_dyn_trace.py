# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attribution for KV cache prefix-reuse misses.

The iteration counters report how many blocks were dropped from the host tier,
but not whether any of them were ever wanted again. A block that is evicted and
never re-requested costs nothing, so a large drop count is not on its own
evidence that the tier is undersized. This module closes that gap: it remembers
the key of every block that leaves the cache and classifies each later lookup
miss as one of

  cold     never cached, so no tier size and no policy would have helped;
  evicted  cached earlier, thrown out, and now wanted again.

Only the second kind is a capacity or policy failure. Splitting the miss rate
that way is the difference between "the tier is too small" and "the workload has
no reuse left to capture", which the aggregate hit rate alone cannot distinguish.

Two secondary questions are answered alongside it, because both came up while
reading drop counters that had no context:

  * how full each tier was at the moment of a drop -- drops observed while both
    tiers had free space are not capacity evictions and need another explanation;
  * how long an evicted block survives, and how many hits it served before dying.

Cost. Keys are truncated to 64 bits, which over a few million live entries makes
a collision (a miss misreported as eviction-induced) a ~1e-8 event, far below the
resolution of anything measured here. Memory is bounded by TLLM_KV_DYN_GHOST_CAP
at roughly 100 B/entry, so the 2M default costs ~200 MiB per rank. Everything is
inert unless TLLM_KV_DYN_TRACE=1; the hooks then reduce to one global bool test.
"""

import os
import time
from collections import deque
from typing import Final, Iterable, Sequence

# TLLM_KV_DYN_TRACE=1 enables per-block miss attribution. Off by default: it
# keeps a key per cached and per recently evicted block, which is bounded but not
# free, and it has no purpose outside a cache-behavior investigation.
ENABLED: Final[bool] = os.environ.get("TLLM_KV_DYN_TRACE", "0") == "1"

# TLLM_KV_DYN_ONBOARD_TIMING=0 disables the onboarding stall measurement while
# leaving the rest of the trace on.
#
# WHY THIS IS MEASURED WITH CUDA EVENTS AND NOT A HOST TIMER
#
# Onboarding a block from the host tier is issued on its own stream (see
# _storage_manager._batched_migrate, which runs each migration under a
# TemporaryCudaStream) and the consumer stream is then made to wait on the
# resulting event. That wait is stream ordering, enqueued and returned from
# immediately, so no host-side timer anywhere in the executor can observe it --
# including host_step_time, which is why a run can be stalling on onboarding and
# show nothing but good hit rates.
#
# The stall is the GPU-side gap between entering and leaving that wait on the
# consumer stream, which is what the event pair below brackets. Zero elapsed
# means the copy had already landed and the tier cost nothing; a large elapsed
# means compute sat idle waiting for it, which is the failure mode a bigger host
# tier makes worse rather than better.
#
# WHY THE STALL ALONE CANNOT ANSWER "DID ONBOARDING OVERLAP COMPUTE"
#
# A near-zero stall has two completely different causes and the fix differs: the
# copies were hidden behind compute, or there were barely any copies. onboard_mib
# separates those two, but neither number says what fraction of the transfer was
# hidden, which is the actual question.
#
# A second event pair on the migration stream answers it. `onboard_copy_begin`
# records after TemporaryCudaStream has enqueued its waits on the slots' prior
# events, so it completes exactly when the copy is free to start moving bytes, and
# `onboard_copy_end` records once the copies are enqueued. With that pair (A, B)
# and the existing consumer pair:
#
#   span   = B - A                 transfer itself, prior waits excluded
#   stall  = wait_end - wait_start exposed: compute idle
#   hidden = span - stall          transferred while compute was still busy
#
# B rather than wait_end is what makes the span honest. The consumer's wait cannot
# complete before the copy, so wait_end would do as an end marker -- but only when
# the copy is the last thing the wait is gated on. When overlap is perfect the
# consumer arrives after the copy has already finished, and wait_end then sits
# arbitrarily far past it: the span would grow with however long compute happened
# to run, still reporting 100% overlap but understating onboard_gbps without bound.
#
# and hidden/span is the overlap fraction. Note what an overlap of 0 does and does
# not mean: it says the copy began moving bytes only once compute had already
# arrived at the wait, which is what happens when the copy stream had to wait on
# the very slots the consumer was still using. That is a dependency-ordering
# problem, not a bandwidth one, and onboard_gbps is what separates the two.
#
# A stall larger than the span is not scored at all. The consumer waits on every
# task's ready event, not only the onboarded ones, so a resident page whose event
# lands after all the copies makes the wait outlast the transfer and the ratio
# meaningless. Those groups are counted as `onboard_blocked` rather than clamped to
# zero overlap, which would read as a transfer problem when the transfer was fine.
ONBOARD_TIMING: Final[bool] = (
    ENABLED and os.environ.get("TLLM_KV_DYN_ONBOARD_TIMING", "1") == "1"
)

# Seconds between summary lines. The window has to be short relative to the
# phenomenon: reuse collapses over minutes as the tier fills, so a whole-run
# average would hide the transition entirely.
INTERVAL_S: Final[float] = float(os.environ.get("TLLM_KV_DYN_TRACE_INTERVAL_S", "10"))

# Maximum remembered evicted keys. Once full, the oldest are forgotten, and a
# miss on a forgotten key is counted cold -- which understates eviction-induced
# misses. `ghost_full` in the summary reports when that is happening, so an
# undersized cap is visible rather than silent.
GHOST_CAP: Final[int] = int(os.environ.get("TLLM_KV_DYN_GHOST_CAP", "2000000"))

# TLLM_KV_DYN_PAGE_FRAC=N logs one in N individual drops. Aggregates say how much
# and when; these say what, and are the only way to inspect a specific block's
# history. Off by default because a busy rank drops thousands of pages a second.
PAGE_FRAC: Final[int] = int(os.environ.get("TLLM_KV_DYN_PAGE_FRAC", "0"))

# Samples kept per window for percentiles. 4096 is far more than needed for a p90
# and keeps the sort at emit time irrelevant next to a 10 s window.
_RESERVOIR = 4096


def _key64(key: bytes) -> int:
    """Truncate a block key to its leading 64 bits.

    The keys are SHA-256 digests, so any 64-bit slice is already uniform; no
    further mixing is needed to index a dict with it.
    """
    return int.from_bytes(key[:8], "little")


class _Ring:
    """Fixed-size sample buffer. Overwrites oldest; never allocates after init."""

    __slots__ = ("_buf", "_n", "_pos")

    def __init__(self) -> None:
        self._buf: list[float] = []
        self._pos = 0
        self._n = 0

    def add(self, value: float) -> None:
        if len(self._buf) < _RESERVOIR:
            self._buf.append(value)
        else:
            self._buf[self._pos] = value
            self._pos = (self._pos + 1) % _RESERVOIR
        self._n += 1

    def clear(self) -> None:
        self._buf.clear()
        self._pos = 0
        self._n = 0

    def percentile(self, frac: float) -> float:
        if not self._buf:
            return 0.0
        ordered = sorted(self._buf)
        idx = min(len(ordered) - 1, int(frac * len(ordered)))
        return ordered[idx]

    @property
    def count(self) -> int:
        return self._n


# Upper edges, in percent, of the host-fullness histogram attached to every
# eviction. A per-window percentile hides the shape: it cannot say whether a
# handful of drops came from a full tier and the rest from an empty one. The
# bins are narrow near zero because that is where a capacity eviction lives.
_FREE_BINS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.01)


def _free_bin(free_pct: float) -> int:
    for index, upper in enumerate(_FREE_BINS):
        if free_pct < upper:
            return index
    return len(_FREE_BINS) - 1


# Upper edges, in percent, of the per-group overlap histogram. A mean overlap
# fraction is the wrong summary if the truth is that most transfers hide
# completely and a few are fully exposed, which is what a prefetch that sometimes
# loses a race looks like -- so the extremes get their own narrow bins.
_OVERLAP_BINS = (5.0, 25.0, 50.0, 75.0, 95.0, 100.01)


def _overlap_bin(overlap_pct: float) -> int:
    for index, upper in enumerate(_OVERLAP_BINS):
        if overlap_pct < upper:
            return index
    return len(_OVERLAP_BINS) - 1


class _EventPool:
    """Free list of timing-enabled CUDA events.

    CachedCudaEvent cannot be reused here: it is created with
    CU_EVENT_DISABLE_TIMING, so cuEventElapsedTime rejects it. These are separate
    and deliberately few -- one pair per onboarding batch, not per block.
    """

    __slots__ = ("_free", "_drv")

    def __init__(self) -> None:
        import cuda.bindings.driver as drv

        self._drv = drv
        self._free: list = []

    def acquire(self):
        if self._free:
            return self._free.pop()
        err, ev = self._drv.cuEventCreate(self._drv.CUevent_flags.CU_EVENT_DEFAULT)
        if int(err) != int(self._drv.CUresult.CUDA_SUCCESS):
            raise RuntimeError(f"cuEventCreate failed: {err}")
        return ev

    def release(self, ev) -> None:
        # Capped so that a pathological burst cannot retain events forever.
        if len(self._free) < 64:
            self._free.append(ev)
        else:
            self._drv.cuEventDestroy(ev)

    def record(self, ev, stream) -> None:
        (err,) = self._drv.cuEventRecord(ev, stream)
        if int(err) != int(self._drv.CUresult.CUDA_SUCCESS):
            raise RuntimeError(f"cuEventRecord failed: {err}")

    def complete(self, ev) -> bool:
        (err,) = self._drv.cuEventQuery(ev)
        return int(err) == int(self._drv.CUresult.CUDA_SUCCESS)

    def elapsed_ms(self, start, end) -> float:
        err, ms = self._drv.cuEventElapsedTime(start, end)
        if int(err) != int(self._drv.CUresult.CUDA_SUCCESS):
            return 0.0
        return float(ms)


_event_pool: "_EventPool | None" = None
# Set on the first timing failure. A broken measurement reports zero stall, which
# is indistinguishable from perfect overlap and is the one wrong answer this
# instrumentation must never give quietly -- so the failure is both warned about
# and carried into the summary line as `onboard_timing=broken`.
_timing_broken: str = ""

# Copy-start event for the group being assembled. Three states, because "armed but
# nothing recorded yet" has to be distinguishable from "not measuring": _NOT_YET
# means a group is open and the next migration should be timed, an event means one
# already was, and None means no group is open.
_NOT_YET: Final = ...
_group_copy_start = None
_group_copy_end = None

if ONBOARD_TIMING:
    try:
        _event_pool = _EventPool()
    except Exception as exc:  # no driver or no context in this process
        _event_pool = None
        _timing_broken = f"init: {type(exc).__name__}"


class _State:
    """Per-process trace state. One instance per rank, created on import."""

    __slots__ = (
        "_ctx_free_frac",
        "_ctx_samples",
        "age",
        "cold",
        "dev_evict_hist",
        "dev_evicted_pages",
        "drops",
        "evicted",
        "host_drop_hist",
        "ghost",
        "ghost_full",
        "hits",
        "hits_at_death",
        "last_emit",
        "matched",
        "offload_bytes",
        "onboard_blocked",
        "onboard_blocks",
        "onboard_bytes",
        "onboard_copy_ms",
        "onboard_groups",
        "onboard_hidden_ms",
        "onboard_overlap_hist",
        "onboard_stalls",
        "onboard_wait_ms",
        "pending_waits",
        "pruned",
        "pruned_tree",
        "requested",
        "requests",
        "start",
        "tree_matched",
        "unlinked",
    )

    def __init__(self) -> None:
        # key64 -> milliseconds since `start` at which the block was last lost.
        # Always overwritten, so a block that is recomputed and lost again is
        # timed from its latest death rather than its first.
        self.ghost: dict[int, int] = {}
        # key64 -> hits served while cached. Only holds blocks that matched at
        # least once, so it stays much smaller than the set of cached blocks.
        self.hits: dict[int, int] = {}
        self.start = time.monotonic()
        self.last_emit = self.start
        self.requests = 0
        self.requested = 0  # blocks a request wanted, cached or not
        self.tree_matched = 0  # blocks found in the radix tree
        self.matched = 0  # blocks actually usable after pruning
        self.pruned = 0
        self.evicted = 0  # misses on blocks we had and dropped
        self.cold = 0  # misses on blocks never cached
        self.ghost_full = 0  # keys forgotten because the ghost was at capacity
        self.unlinked = 0  # blocks lost outside the allocator's drop path
        self.pruned_tree = 0  # blocks invalidated as collateral of a drop
        self.drops: list[int] = []  # indexed by cache level
        self.age = _Ring()  # seconds between a block's death and its re-request
        self.hits_at_death = _Ring()
        self._ctx_free_frac: list[_Ring] = []  # free slot fraction per level at drop
        self._ctx_samples = 0
        # Pages weighted into host-fullness bins, split by what forced the move:
        # a GPU eviction arriving at the host tier, versus a host drop leaving
        # the hierarchy. Both are measured against the host tier's own free
        # slots, so the two rows are directly comparable.
        self.dev_evict_hist = [0] * len(_FREE_BINS)
        self.host_drop_hist = [0] * len(_FREE_BINS)
        self.dev_evicted_pages = 0
        # Host tier traffic, split by direction. Onboarding is on the critical
        # path of the request that wanted the block; offloading is not, so they
        # are never summed.
        self.onboard_blocks = 0
        self.onboard_bytes = 0
        self.offload_bytes = 0
        self.onboard_wait_ms = 0.0
        self.onboard_stalls = 0
        # Overlap accounting, all in GPU time on the copy's own span. copy_ms is
        # the transfer itself with the copy stream's prior-event waits already
        # excluded, so onboard_bytes/copy_ms is the achieved DMA rate -- unlike
        # stall_gbps, which is only meaningful when overlap is poor.
        self.onboard_copy_ms = 0.0
        self.onboard_hidden_ms = 0.0
        self.onboard_groups = 0
        # Groups whose stall outlasted their transfer span, so the wait was gated by
        # something other than onboarding -- a resident page's ready event, since the
        # consumer waits on every task's. Unscorable rather than zero-overlap.
        self.onboard_blocked = 0
        self.onboard_overlap_hist = [0] * len(_OVERLAP_BINS)
        # (start_event, end_event, blocks, copy_start_event) awaiting GPU
        # completion. Drained at emit time and only when already complete, because
        # the whole point is to measure a stall without introducing one: a
        # cuEventSynchronize here would block the very stream being measured.
        self.pending_waits: deque = deque()

    def level_ring(self, level: int) -> _Ring:
        while len(self._ctx_free_frac) <= level:
            self._ctx_free_frac.append(_Ring())
        return self._ctx_free_frac[level]

    def bump_drop(self, level: int, count: int) -> None:
        while len(self.drops) <= level:
            self.drops.append(0)
        self.drops[level] += count

    def reset_window(self) -> None:
        self.requests = 0
        self.requested = 0
        self.tree_matched = 0
        self.matched = 0
        self.pruned = 0
        self.evicted = 0
        self.cold = 0
        self.ghost_full = 0
        self.unlinked = 0
        self.pruned_tree = 0
        self.drops = [0] * len(self.drops)
        self.age.clear()
        self.hits_at_death.clear()
        for ring in self._ctx_free_frac:
            ring.clear()
        self._ctx_samples = 0
        self.dev_evict_hist = [0] * len(_FREE_BINS)
        self.host_drop_hist = [0] * len(_FREE_BINS)
        self.dev_evicted_pages = 0
        self.onboard_blocks = 0
        self.onboard_bytes = 0
        self.offload_bytes = 0
        self.onboard_wait_ms = 0.0
        self.onboard_stalls = 0
        self.onboard_copy_ms = 0.0
        self.onboard_hidden_ms = 0.0
        self.onboard_groups = 0
        self.onboard_blocked = 0
        self.onboard_overlap_hist = [0] * len(_OVERLAP_BINS)
        # pending_waits is deliberately NOT cleared: an onboarding issued near a
        # window boundary completes in the next one and its stall is attributed
        # there, which is late by up to one window but never lost.

    def drain_waits(self) -> None:
        """Account every bracket whose end event has completed on the GPU.

        Walks from the front and stops at the first incomplete pair, since the
        pairs are recorded on one stream and therefore complete in order.

        Failures here are contained rather than raised: this runs from _emit,
        which runs from the cache manager's hot path, so a driver call behaving
        unexpectedly must cost the measurement and not the server.
        """
        global _timing_broken
        if _event_pool is None:
            return
        pending = self.pending_waits
        try:
            while pending:
                start, end, _blocks, (copy_start, copy_end) = pending[0]
                if not _event_pool.complete(end):
                    return
                pending.popleft()
                elapsed = _event_pool.elapsed_ms(start, end)
                self.onboard_wait_ms += elapsed
                if elapsed > 0.0:
                    self.onboard_stalls += 1
                if copy_start is not None:
                    # copy_end is on the migration stream and end on the consumer's,
                    # but the consumer waits on the copy, so copy_end has completed
                    # too and both elapsed calls are safe.
                    span = _event_pool.elapsed_ms(copy_start, copy_end)
                    self.onboard_groups += 1
                    if span <= 0.0 or elapsed > span:
                        self.onboard_blocked += 1
                    else:
                        self.onboard_copy_ms += span
                        self.onboard_hidden_ms += span - elapsed
                        self.onboard_overlap_hist[
                            _overlap_bin((span - elapsed) / span * 100.0)
                        ] += 1
                    _event_pool.release(copy_start)
                    _event_pool.release(copy_end)
                _event_pool.release(start)
                _event_pool.release(end)
        except Exception as exc:
            if not _timing_broken:
                _timing_broken = f"drain: {type(exc).__name__}: {exc}"
                _log(f"KVDYN onboarding stall timing disabled -- {_timing_broken}")
            pending.clear()


_state = _State() if ENABLED else None


def record_match(keys: Sequence[bytes], tree_matched: int, usable: int) -> None:
    """Classify one prefix lookup.

    `keys` is the request's full block-key chain, which is derived from the token
    sequence alone and so is known regardless of what is cached. The first
    `tree_matched` were found in the tree and `usable` of those survived pruning;
    everything past `tree_matched` is a miss to be attributed.
    """
    if _state is None:
        return
    st = _state
    st.requests += 1
    st.requested += len(keys)
    st.tree_matched += tree_matched
    st.matched += usable
    st.pruned += tree_matched - usable

    now = time.monotonic()
    for key in keys[:usable]:
        k = _key64(key)
        st.hits[k] = st.hits.get(k, 0) + 1
    elapsed_ms = (now - st.start) * 1000.0
    for key in keys[tree_matched:]:
        death_ms = st.ghost.get(_key64(key))
        if death_ms is None:
            st.cold += 1
        else:
            st.evicted += 1
            st.age.add(max(0.0, (elapsed_ms - death_ms) / 1000.0))

    if now - st.last_emit >= INTERVAL_S:
        _emit(now)


def _remember(st: "_State", k: int, death_ms: int) -> None:
    if k not in st.ghost and len(st.ghost) >= GHOST_CAP:
        st.ghost.pop(next(iter(st.ghost)))
        st.ghost_full += 1
    st.ghost[k] = death_ms


def record_drop(keys: Iterable[bytes], level: int, free_frac: float = -1.0) -> None:
    """Record blocks the allocator just reclaimed.

    `free_frac` is the fraction of slots free in `level` at that moment. A drop
    taken while the tier is mostly free is not a capacity eviction, and telling
    those apart from genuine pressure is the point of carrying it here.
    """
    if _state is None:
        return
    st = _state
    death_ms = int((time.monotonic() - st.start) * 1000.0)
    count = 0
    for key in keys:
        k = _key64(key)
        _remember(st, k, death_ms)
        served = st.hits.pop(k, 0)
        st.hits_at_death.add(float(served))
        count += 1
        if PAGE_FRAC and (k % PAGE_FRAC) == 0:
            _log(
                f"KVDYNPAGE key={k:016x} lvl={level} hits={served} "
                f"free_frac={free_frac:.4f} t={death_ms / 1000.0:.3f}"
            )
    if count:
        st.bump_drop(level, count)
        if free_frac >= 0.0:
            st.level_ring(level).add(free_frac)
            st.host_drop_hist[_free_bin(free_frac * 100.0)] += count


def record_device_evict(num_pages: int, host_free_frac: float) -> None:
    """Record pages evicted from GPU that are arriving at the host tier.

    `host_free_frac` is the host tier's free-slot fraction before it takes them,
    so this answers how much room the host had at the instant the GPU had to
    spill. Paired with the host-drop histogram it separates two different
    inefficiencies: spilling into a tier that is already full, and discarding
    from a tier that still had room.
    """
    if _state is None or num_pages <= 0:
        return
    st = _state
    st.dev_evicted_pages += num_pages
    if host_free_frac >= 0.0:
        st.dev_evict_hist[_free_bin(host_free_frac * 100.0)] += num_pages


def record_migration(dst_level: int, src_level: int, num_pages: int, num_bytes: int) -> None:
    """Record bytes moved between tiers, split by direction.

    Onboarding (towards level 0) is synchronous with respect to the request that
    wanted the block: it cannot run until the copy lands. Offloading is
    background work. Reporting one number for both would average the two.
    """
    if _state is None or num_pages <= 0:
        return
    if dst_level < src_level:
        _state.onboard_blocks += num_pages
        _state.onboard_bytes += num_bytes
    elif dst_level > src_level:
        _state.offload_bytes += num_bytes


def onboard_group_begin() -> None:
    """Arm copy-start capture for one batched lock's worth of onboarding.

    Only migrations issued between this and the matching `onboard_wait_end` are
    timed. Without the arming the span would also pick up onboarding from paths
    that have no consumer wait at all -- `_storage_manager` promotes pages from
    several places -- and a copy start belonging to no wait would be attributed to
    whichever wait came next, inflating its span and so its apparent overlap.
    """
    global _group_copy_start
    if _state is None or _event_pool is None:
        return
    _release_group_copy_start()
    _group_copy_start = _NOT_YET


def onboard_copy_begin(stream) -> None:
    """Mark, on the migration stream, the point the copy may start moving bytes.

    Called after the migration stream has enqueued its waits on the slots' prior
    events, so the recorded event completes when those clear rather than when the
    copy was enqueued. Only the first call in a group is kept: later batches are
    issued behind it and are inside the same span.
    """
    global _timing_broken, _group_copy_start
    if _state is None or _event_pool is None or _group_copy_start is not _NOT_YET:
        return
    try:
        ev = _event_pool.acquire()
        _event_pool.record(ev, stream)
        _group_copy_start = ev
    except Exception as exc:
        _group_copy_start = None
        if not _timing_broken:
            _timing_broken = f"copy_begin: {type(exc).__name__}: {exc}"
            _log(f"KVDYN onboarding overlap timing disabled -- {_timing_broken}")


def onboard_copy_end(stream) -> None:
    """Mark the end of this group's transfers on the migration stream.

    Kept as the latest such mark rather than the first, since a group can issue one
    migration per source level and pool group and the span has to cover all of them.
    """
    global _timing_broken, _group_copy_end
    if _state is None or _event_pool is None or _group_copy_start is None:
        return
    if _group_copy_start is _NOT_YET:
        return  # begin failed or was never reached; nothing to close
    try:
        if _group_copy_end is not None:
            _event_pool.release(_group_copy_end)
        ev = _event_pool.acquire()
        _event_pool.record(ev, stream)
        _group_copy_end = ev
    except Exception as exc:
        _group_copy_end = None
        if not _timing_broken:
            _timing_broken = f"copy_end: {type(exc).__name__}: {exc}"
            _log(f"KVDYN onboarding overlap timing disabled -- {_timing_broken}")


def onboard_group_abort() -> None:
    """Discard the open group, for the case where migration raised before the wait.

    Without this the event would be attributed to the next group's wait, and since
    it was recorded earlier the span would come out long and the overlap
    flatteringly high.
    """
    _release_group_copy_start()


def _release_group_copy_start() -> None:
    """Drop an armed-but-unconsumed copy bracket."""
    global _group_copy_start, _group_copy_end
    for ev in (_group_copy_start, _group_copy_end):
        if ev is not None and ev is not _NOT_YET and _event_pool is not None:
            _event_pool.release(ev)
    _group_copy_start = None
    _group_copy_end = None


def onboard_wait_begin(stream):
    """Open a stall bracket on `stream`, or return None if not measuring.

    Callers must pass the returned handle to `onboard_wait_end` on the same
    stream. Returns None whenever timing is off or unavailable, so the caller's
    fast path is one identity test.
    """
    global _timing_broken
    if _state is None or _event_pool is None:
        return None
    try:
        ev = _event_pool.acquire()
        _event_pool.record(ev, stream)
        return ev
    except Exception as exc:
        if not _timing_broken:
            _timing_broken = f"begin: {type(exc).__name__}: {exc}"
            _log(f"KVDYN onboarding stall timing disabled -- {_timing_broken}")
        return None


def onboard_wait_end(handle, stream, num_blocks: int) -> None:
    """Close the bracket opened by `onboard_wait_begin`."""
    global _timing_broken, _group_copy_start, _group_copy_end
    if handle is None or _state is None or _event_pool is None:
        _release_group_copy_start()
        return
    # Both or neither: a start without an end cannot be turned into a span.
    copy = (_group_copy_start, _group_copy_end)
    if _group_copy_start is _NOT_YET or _group_copy_end is None:
        _release_group_copy_start()
        copy = (None, None)
    _group_copy_start = None
    _group_copy_end = None
    try:
        end = _event_pool.acquire()
        _event_pool.record(end, stream)
        _state.pending_waits.append((handle, end, num_blocks, copy))
    except Exception as exc:
        _event_pool.release(handle)
        for ev in copy:
            if ev is not None:
                _event_pool.release(ev)
        if not _timing_broken:
            _timing_broken = f"end: {type(exc).__name__}: {exc}"
            _log(f"KVDYN onboarding stall timing disabled -- {_timing_broken}")


def record_prune(keys: Sequence[bytes]) -> None:
    """Record a whole subtree of blocks removed from the reusable set at once.

    Counted apart from the allocator's drops because these are consequences of
    one, not independent decisions: losing a block's page invalidates everything
    cached beneath it, since a longer prefix is unusable without its start.
    """
    if _state is None or not keys:
        return
    st = _state
    death_ms = int((time.monotonic() - st.start) * 1000.0)
    for key in keys:
        _remember(st, _key64(key), death_ms)
    st.pruned_tree += len(keys)


def record_unlink(key: bytes) -> None:
    """Record a block that lost its page without passing through the allocator.

    Structural losses -- a block replaced by one that covers it, a subtree pruned
    -- are not capacity evictions and are counted separately, but they still have
    to enter the ghost. A block missing from it would make a later miss on that
    block look cold, which would understate exactly the quantity being measured.
    """
    if _state is None:
        return
    st = _state
    _remember(st, _key64(key), int((time.monotonic() - st.start) * 1000.0))
    st.unlinked += 1


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _pct_f(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def _emit(now: float) -> None:
    st = _state
    assert st is not None
    st.drain_waits()
    window = now - st.last_emit
    misses = st.evicted + st.cold
    fields = [
        f"win={window:.1f}",
        f"t={now - st.start:.1f}",
        f"req={st.requests}",
        f"blk_want={st.requested}",
        f"blk_hit={st.matched}",
        f"hit_pct={_pct(st.matched, st.requested):.2f}",
        f"pruned={st.pruned}",
        f"miss={misses}",
        f"miss_evicted={st.evicted}",
        f"miss_cold={st.cold}",
        # The headline number: of the reuse we failed to get, how much we had
        # already paid for and threw away.
        f"evicted_share={_pct(st.evicted, misses):.2f}",
        f"age_p50={st.age.percentile(0.5):.1f}",
        f"age_p90={st.age.percentile(0.9):.1f}",
        f"served_p50={st.hits_at_death.percentile(0.5):.0f}",
        f"served_p90={st.hits_at_death.percentile(0.9):.0f}",
        f"ghost={len(st.ghost)}",
        f"ghost_full={st.ghost_full}",
        f"unlinked={st.unlinked}",
        f"pruned_tree={st.pruned_tree}",
        f"tracked_hits={len(st.hits)}",
    ]
    for level, count in enumerate(st.drops):
        fields.append(f"drop_l{level}={count}")
        ring = st._ctx_free_frac[level] if level < len(st._ctx_free_frac) else None
        if ring is not None and ring.count:
            fields.append(f"free_at_drop_l{level}_p50={ring.percentile(0.5):.4f}")
    fields.append(f"dev_evicted={st.dev_evicted_pages}")
    # Bin upper edges are fixed, so the reader can label these without being
    # told; emitting counts only keeps the line short.
    fields.append("hostfree_bins=" + ":".join(f"{edge:g}" for edge in _FREE_BINS))
    fields.append("hostfree_at_dev_evict=" + ":".join(str(c) for c in st.dev_evict_hist))
    fields.append("hostfree_at_host_drop=" + ":".join(str(c) for c in st.host_drop_hist))
    # Host tier traffic and what it cost. onboard_wait_ms is GPU time the consumer
    # stream spent blocked on onboarding copies, so it is directly comparable to
    # the window: 200 ms of stall in a 10 s window is 2% of the device, while a
    # figure approaching the window means the tier is serialising the engine.
    # onboard_wait_ms/onboard_stalls says whether a few large stalls or many small
    # ones, which distinguishes a bandwidth limit from an issue-timing problem.
    fields.append(f"onboard_blocks={st.onboard_blocks}")
    fields.append(f"onboard_mib={st.onboard_bytes / 2**20:.1f}")
    fields.append(f"offload_mib={st.offload_bytes / 2**20:.1f}")
    fields.append(f"onboard_wait_ms={st.onboard_wait_ms:.2f}")
    fields.append(f"onboard_stalls={st.onboard_stalls}")
    fields.append(f"onboard_wait_pct={_pct_f(st.onboard_wait_ms / 1000.0, window):.2f}")
    if st.onboard_bytes and st.onboard_wait_ms > 0.0:
        # Effective bandwidth *as observed from the stall*, not the link rate: if
        # copies overlap well this is meaninglessly high, and that is the answer.
        fields.append(
            f"stall_gbps={st.onboard_bytes / 2**30 / (st.onboard_wait_ms / 1000.0):.2f}"
        )
    # Overlap. onboard_overlap_pct is the share of measured transfer time that ran
    # while compute was still busy, so 100 is free onboarding and 0 is a tier that
    # serialises the engine. Reading it needs onboard_groups for weight and
    # onboard_blocked for the groups it had to exclude, where the wait outlasted the
    # transfer and so was gated by something other than onboarding.
    if st.onboard_groups:
        fields.append(f"onboard_groups={st.onboard_groups}")
        fields.append(f"onboard_blocked={st.onboard_blocked}")
    if st.onboard_copy_ms > 0.0:
        fields.append(f"onboard_copy_ms={st.onboard_copy_ms:.2f}")
        # Emitted rather than left to be derived as copy_ms - wait_ms: wait_ms also
        # carries stalls from groups that had no copy bracket or were unscorable,
        # so the subtraction is not this quantity.
        fields.append(f"onboard_hidden_ms={st.onboard_hidden_ms:.2f}")
        fields.append(
            f"onboard_overlap_pct={st.onboard_hidden_ms / st.onboard_copy_ms * 100.0:.1f}"
        )
        if st.onboard_bytes:
            # Achieved DMA rate over the transfer itself. Unlike stall_gbps this
            # stays meaningful when overlap is good, so it is what says whether a
            # stall is a bandwidth limit or an issue-timing one.
            fields.append(
                f"onboard_gbps={st.onboard_bytes / 2**30 / (st.onboard_copy_ms / 1000.0):.2f}"
            )
        fields.append("overlap_bins=" + ":".join(f"{e:g}" for e in _OVERLAP_BINS))
        fields.append(
            "overlap_hist=" + ":".join(str(c) for c in st.onboard_overlap_hist)
        )
    if st.pending_waits:
        fields.append(f"onboard_pending={len(st.pending_waits)}")
    if _timing_broken:
        # Without this a failed measurement and perfect overlap look identical.
        fields.append("onboard_timing=broken")
    elif not ONBOARD_TIMING:
        fields.append("onboard_timing=off")
    _log("KVDYN " + " ".join(fields))
    st.last_emit = now
    st.reset_window()


def _log(line: str) -> None:
    # Imported lazily: this module is pulled in by the cache manager's hot path,
    # and the logger drags in a good part of the package at import time.
    from tensorrt_llm.logger import logger

    logger.info(line)
