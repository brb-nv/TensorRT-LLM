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
from typing import Final, Iterable, Sequence

# TLLM_KV_DYN_TRACE=1 enables per-block miss attribution. Off by default: it
# keeps a key per cached and per recently evicted block, which is bounded but not
# free, and it has no purpose outside a cache-behavior investigation.
ENABLED: Final[bool] = os.environ.get("TLLM_KV_DYN_TRACE", "0") == "1"

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


class _State:
    """Per-process trace state. One instance per rank, created on import."""

    __slots__ = (
        "_ctx_free_frac",
        "_ctx_samples",
        "age",
        "cold",
        "drops",
        "evicted",
        "ghost",
        "ghost_full",
        "hits",
        "hits_at_death",
        "last_emit",
        "matched",
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


def _emit(now: float) -> None:
    st = _state
    assert st is not None
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
    _log("KVDYN " + " ".join(fields))
    st.last_emit = now
    st.reset_window()


def _log(line: str) -> None:
    # Imported lazily: this module is pulled in by the cache manager's hot path,
    # and the logger drags in a good part of the package at import time.
    from tensorrt_llm.logger import logger

    logger.info(line)
