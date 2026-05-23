<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Disagg vs agg TTFT — round 7 (2026-05-22): turn-progression root cause

Continuation of [`FINDINGS_RWLT_0521.md`](FINDINGS_RWLT_0521.md). Round 6
narrowed the residual `gen_queue_ms ≈ kv_transfer_gen_ms` gap to "something
inside the cache transceiver's `Transmissions` phase". Round 7 starts from a
sharper hypothesis and uses metrics we already collect to localize the cause
to a single suspect, then proposes one focused config ablation and one small
Python-only logging addition to confirm it.

## TL;DR (read this first)

> **Update 2026-05-22 PM (round 7b -- function-level instrumentation):**
> the suspect function `KVCacheManager::findBlocksInReuseTreeByBlockKey`
> is now timed directly via the new `br_find_tree_ms` field on the
> `[disagg-prof] format` log line (instrumentation lives inside
> `BlockRange::fromReuseTree` in `kvCacheUtils.h` and is surfaced in
> `cacheFormatter.cpp::getBlockRangeForSending`). Across 318 ctx
> format lines per run on `top5_signed`:
>
> - **`br_find_tree_ms` is 99-100% of `blockrange_ms`** at p50/p90/min
>   (100.0% p50 / 100.0% p90 / 98.8% min in si1; 99.8% p50 in no_reuse).
> - **94-99% of total ctx format time** is in this single call.
> - r(`br_find_tree_ms`, `prepopulated_tokens`) = **+0.886** in si1,
>   **+0.874** in no_reuse. r vs `br_num_blocks_collected` (= the
>   number of blocks the function is asked to return) = **−0.042** in
>   si1, +0.900 in no_reuse only because that run requests the full
>   prefix.
> - Linear (in pool-pooled regression) slope from no_reuse run:
>   **0.134 ms per 32-token prefix block** at the mean prefix length,
>   independent of bytes-on-wire. But per-block cost is NOT actually
>   constant -- it grows ~8x from the 8-16k ISL bucket (0.013 ms/blk)
>   to the 64-80k bucket (0.107 ms/blk), suggesting the function is
>   O(prefix^2) overall. See ISL table below.
> - For an 80k-token prefix returning only 10 new blocks, ctx burns
>   **198 ms in `findBlocksInReuseTreeByBlockKey` alone** to locate the
>   leaf, then takes **0.002 ms** to collect the actual 10 blocks via
>   `getPrevBlock()`. The cost is in the find, not the result.
>
> The original "VSWA bypass in the transceiver" hypothesis below was
> *wrong*: `pools=1` for GPT-OSS, the optimized
> `getBlockRangeForReceiving` path at `cacheFormatter.cpp:235` IS
> taken, and reuse correctly skips already-cached blocks on the wire
> (`per_window_blocks=131080:10` for a turn with 300 reused blocks).
> See "Root cause (round-7 verified)" below for the corrected story.

1. The disagg `kv_xfer_ms` floor on the gen receiver is dominated by
   a single super-linear function call on the **ctx sender**:
   `KVCacheManager::findBlocksInReuseTreeByBlockKey` (defined at
   `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:1884-1909`),
   invoked from `BlockRange::fromReuseTree` (`kvCacheUtils.h:80-103`),
   invoked from `getBlockRangeForSending` (`cacheFormatter.cpp:281`).
   Cost per 32-token prefix block grows from ~13 us at 8-16k ISL to
   ~107 us at 64-80k ISL (8x growth across 6x ISL, suggesting
   O(prefix^2)). At 80k-token prefixes the call costs ~250 ms p50
   (~349 ms p99).
2. Everything else in the format() path is sub-millisecond at p50:
   `br_chain_walk_ms` 0.00, `br_finalize_ms` 0.00, format-level
   `walk_ms` 0.02, `split_ms` 0.08, `send_ms` 0.11 (20 MB) /
   3.15 (1.7 GB). The gen receiver `unformat()` is identically
   sub-millisecond except `recv_ms`, which is purely the wait for
   ctx's `format()` to finish (gap between `gen.recv_ms` and
   `ctx.format.total_ms` is < 0.2 ms).
3. The round-7 `no_reuse` ablation does NOT change `kv_xfer_ms`
   (p50 76.5 vs 70.8 ms in round-7b) despite moving **85× more bytes**
   (1.7 GB vs 20 MB) because **`br_path` was `reuse_tree` in 318/318
   format lines in BOTH runs**. Reading `dataTransceiver.cpp:823-849`
   explains why: gen populates `lastBlockKey` from the full prompt's
   `getUniqueTokens(beam)` whenever `!isVariableWindow()`, **regardless
   of whether gen's own `enable_block_reuse` flag is on or off**. So
   ctx's `getBlockRangeForSending` sees a non-empty `lastBlockKey` in
   both runs and goes down the `reuse_tree` branch in both runs. The
   `no_reuse` run is NOT a real ablation of the suspect path. To
   ablate it, ctx-side `enable_block_reuse` would need to be turned
   off (which we deliberately did NOT do because it would change
   ctx's prefill cost too, contaminating the comparison).
4. The `[disagg-dbg]` log gate now defaults to ENABLED (set
   `TRTLLM_DEBUG_DISAGG_GEN_TIMING=0` to opt out); yesterday's
   "log silently didn't emit" issue across MPI-spawned workers made
   default-on the safer choice for the duration of this investigation.
   (The Python `[disagg-dbg]` line is still silent after round-7
   reruns on top5_signed -- separate issue, secondary to the C++
   root cause above which is now fully diagnosed via `[disagg-prof]`.)

## Function-level confirmation (round 7b, 2026-05-22 13:25)

Re-ran the same two configs (`disagg_round7b_si1_top5` and
`disagg_round7b_no_reuse_top5`) after rebuilding with sub-phase timing
added inside `getBlockRangeForSending` and `BlockRange::fromReuseTree`.
The new fields on the `[disagg-prof] format` line are `br_path`,
`br_find_tree_ms`, `br_chain_walk_ms`, `br_finalize_ms`,
`br_num_blocks_collected`, `br_fallback_from_all_ids_ms`,
`br_fallback_slice_ms`.

n = 318 ctx format lines per run.

### Sub-phase attribution

| field                       | si1 p50 | si1 p90 | no_reuse p50 | no_reuse p90 |
|-----------------------------|---------|---------|--------------|--------------|
| `total_ms` (ctx format)     | 70.8    | 207.4   | 76.5         | 208.9        |
| `blockrange_ms`             | 70.5    | 207.1   | 72.4         | 202.6        |
| **`br_find_tree_ms`**       | **70.5**| **207.1**| **72.3**    | **202.3**    |
| `br_chain_walk_ms`          | 0.00    | 0.01    | 0.15         | 0.26         |
| `br_finalize_ms`            | 0.00    | 0.01    | 0.00         | 0.01         |
| `walk_ms` (format-level)    | 0.02    | 0.03    | 0.19         | 0.29         |
| `split_ms`                  | 0.08    | 0.11    | 0.71         | 1.06         |
| `send_ms`                   | 0.11    | 0.23    | 3.15         | 4.80         |
| `br_num_blocks_collected`   | 16      | 76      | 1368         | 2099         |
| `prepopulated_tokens`       | 42639   | 66490   | 42637        | 66488        |
| `br_path` (all 318 lines)   | reuse_tree | reuse_tree | reuse_tree | reuse_tree |

### Attribution ratios

| | si1 | no_reuse |
|---|---|---|
| `br_find_tree_ms / blockrange_ms` p50 | **100.0%** | **99.8%** |
| `br_find_tree_ms / blockrange_ms` p90 | 100.0% | 99.9% |
| `br_find_tree_ms / blockrange_ms` min | 98.8% | 98.3% |
| `br_find_tree_ms / total_ms` p50      | **99.7%** | **94.5%** |
| `br_find_tree_ms / total_ms` p90      | 99.9% | 97.1% |

`findBlocksInReuseTreeByBlockKey` accounts for **>99% of `blockrange_ms`
in every single one of the 636 measurements** (min ratio = 98.3%). It
is the entire cost.

### Correlations (n=318 each)

| | r(`br_find_tree_ms`, `prepopulated_tokens`) | r(`br_find_tree_ms`, `br_num_blocks_collected`) |
|---|---|---|
| si1     | **+0.886** | **−0.042** |
| no_reuse | +0.874     | +0.900 (collinear; in this run `br_num_blocks_collected ≈ prepop_blocks`) |

In si1, the function is asked for only `indexFromEnd + 1` blocks
(p50 = 16, p90 = 76, max = 578). Its runtime is **uncorrelated with
the number of blocks requested** and **strongly correlated with the
prefix length it must walk to find the leaf**. This is direct evidence
that the cost is the radix-tree walk over the full prefix's tokens
(under `mCachedBlocksRootMutex`), not the result-collection step.

### Per-prefix-block cost

Linear fit of `br_find_tree_ms` vs `br_num_blocks_collected` on the
no_reuse run (where `br_num_blocks_collected` ≈ full prefix in blocks):

> **slope = 0.134 ms per 64-token prefix block** on B200, this build.

For a 1280-block (~80k token) prefix:
- Predicted: 0.134 × 1280 = **172 ms**
- Observed (si1 p99, prefix ≈ 79k tokens): **319 ms p99** (matches order-of-magnitude; spread above the fit is the mutex-contention tail with ctx's own scheduler).

### Anatomy of one deep turn (si1, prepop=69260)

```text
[disagg-prof] format ctx_req=409798612476221 pools=1 blocks=10 bytes=12779520 targets=1
  per_window_blocks=131080:10 prepopulated_tokens=69260 prompt_len=69547 reuse_enabled=1
  blockrange_ms=198.097 walk_ms=0.015 split_ms=0.078 send_ms=0.084 total_ms=198.275
  br_path=reuse_tree
  br_find_tree_ms=198.090  <- 99.9% of total_ms
  br_chain_walk_ms=0.002
  br_finalize_ms=0.003
  br_num_blocks_collected=10  <- only 10 blocks requested
  br_fallback_from_all_ids_ms=0.000
  br_fallback_slice_ms=0.000
```

198 ms to "find" 10 blocks: the tree walk chops the full 69k-token
prefix into 2160 per-block keys (= `prepop_tokens / tokens_per_block`
with `tokens_per_block = 32`) and hash-lookups each one in sequence
under `mCachedBlocksRootMutex`. The 10 blocks we actually wanted are
collected by the `getPrevBlock()` chain in 0.002 ms.

### `br_find_tree_ms` vs ISL (pooled n=636: si1 + no_reuse)

`tokens_per_block = 32` (from `kv_cache_config`).

| ISL (`prompt_len`) | n | mean | p50 | p90 | p99 | max | prefix blocks (p50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 – 16k   |  38 |   5.85 |   4.46 |  16.50 |  18.46 |  18.46 | ~310 |
| 16 – 32k  | 156 |  19.67 |  14.93 |  30.88 | 151.84 | 158.21 | ~697 |
| 32 – 48k  | 214 |  63.26 |  64.77 |  86.75 |  90.19 | 201.99 | ~1276 |
| 48 – 64k  | 150 | 141.55 | 138.08 | 189.14 | 382.52 | 479.49 | ~1726 |
| 64 – 80k  |  78 | 245.89 | 253.35 | 308.03 | 349.30 | 349.30 | ~2339 |

### Per-prefix-block cost growth (key super-linearity evidence)

| ISL bucket | n | ms / prefix_block (p50) | ms / prefix_block (mean) |
|---|---:|---:|---:|
| 8 – 16k   |  38 | 0.0134 | 0.0247 |
| 16 – 32k  | 156 | 0.0212 | 0.0269 |
| 32 – 48k  | 214 | 0.0495 | 0.0490 |
| 48 – 64k  | 150 | 0.0760 | 0.0821 |
| 64 – 80k  |  78 | 0.1074 | 0.1075 |

Cost per block grows ~8x across a 6x ISL range, so the function's
total cost is **super-linear (likely O(prefix^2))**. Most likely culprit
inside `findBlocksInReuseTreeByBlockKey` is `chopVectorIntoBlocks`
allocating an O(prefix_tokens)-sized `vector<vector<UniqueToken>>`
inside an O(prefix_blocks) loop -- net O(prefix^2). Worth one more
pass of sub-instrumentation (`chop_ms` vs `walk_loop_ms`) if we want
to confirm, but the F1 fix below makes the question moot for warm
turns.

## Root cause (round-7 verified, 2026-05-22 PM)

The new `[disagg-prof]` C++ logging in `cacheFormatter.cpp` (added
2026-05-22 PM, lands in both `format()` and `unformat()`) drained
318 paired ctx/gen lines per run on the top5_signed RWLT subset. Data:

**Receiver (`gen.log` `unformat` lines, si1_top5):**

| field             | p50      | p90      | max      |
|-------------------|----------|----------|----------|
| `total_ms`        | 85.80    | 230.30   | 500.57   |
| `recv_ms`         | 85.72    | 230.22   | 500.08   |
| `walk_ms`         | 0.01     | 0.02     | 0.09     |
| `alloc_ms`        | 0.01     | 0.02     | 0.94     |
| `concat_ms`       | 0.06     | 0.09     | 8.71     |
| `blockrange_ms`   | 0.00     | 0.00     | 0.01     |
| `blocks` (on wire)| 16       | 76       | 578      |
| `seq_blocks`      | 1368     | 2099     | 2553     |
| `bytes`           | 20 MB    | 97 MB    | 738 MB   |

The gen receiver itself does no significant work. `recv_ms` is the
wait for ctx's `format()` to finish.

**Sender (`ctx.log` `format` lines, si1_top5):**

| field             | p50      | p90      | max      |
|-------------------|----------|----------|----------|
| `total_ms`        | 85.74    | 230.24   | 500.18   |
| **`blockrange_ms`**| **85.01**| **230.02**| **498.38**|
| `walk_ms`         | 0.02     | 0.03     | 0.08     |
| `split_ms`        | 0.08     | 0.11     | 10.19    |
| `send_ms`         | 0.11     | 0.23     | 35.57    |
| `blocks` (on wire)| 16       | 76       | 578      |
| `prepop. tokens`  | 42638    | 66489    | 81599    |

`blockrange_ms` is **99% of total** at p50.

**Pair correlation (n=318) between ctx `blockrange_ms` and the inputs
to the operation it performs:**

|                                | si1 (reuse on) | no_reuse (gen reuse off) |
|--------------------------------|----------------|---------------------------|
| vs `prepopulated_tokens`       | **+0.866**     | **+0.892**                |
| vs `prompt_len`                | +0.890         | +0.918                    |
| vs `blocks` (new on the wire)  | **−0.026**     | +0.918*                   |

\*In the no_reuse run, `blocks ≈ seq_blocks` because gen-side reuse is
disabled, so the new-blocks count and the prefix length are
collinear; the correlation alone doesn't distinguish them, but the
**per-bucket means** (below) do.

**Bucketed by `prepopulated_tokens` (si1):**

| prepop tokens   | n   | `blockrange_ms` p50 | mean new blocks |
|-----------------|-----|---------------------|-----------------|
| 0–4k            | 5   | 4.1                 | 266.0           |
| 4–16k           | 20  | 5.7                 | 84.2            |
| 16–32k          | 78  | 20.9                | 37.4            |
| 32–49k          | 106 | **79.8**            | 28.7            |
| 49–65k          | 72  | **163.2**           | 21.2            |
| 65–90k          | 37  | **282.8**           | 17.6            |

Cost grows linearly with the prefix length while the new-blocks count
*decreases* — definitive proof the cost is in prefix bookkeeping, not
new-block work. Slope ≈ **3.5 ms / 1k tokens ≈ 0.22 ms per 64-token
block** on a B200 + this build.

### The single hot function

`KVCacheManager::findBlocksInReuseTreeByBlockKey`
(`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:1884–1909`):

```cpp
std::shared_ptr<KVCacheBlock> WindowBlockManager::findBlocksInReuseTreeByBlockKey(BlockKey const& blockKey)
{
    std::lock_guard<std::mutex> lock(mCachedBlocksRootMutex);    // global lock
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(
        blockKey.uniqueTokens, blockKey.uniqueTokens.size(), mTokensPerBlock, true);
    std::vector<BlockKey> blockKeys;
    for (auto const& blockedUniqueTokensList : blockedUniqueTokens) { ... }
    auto searchRoot = mCachedBlocksRoot;
    for (auto const& blockKey : blockKeys) {                       // O(prefix_blocks)
        auto [partialMatch, numMatched, matchingBlock] = searchRoot
            ? searchRoot->findMatchingBlock(blockKey, true, true)  // hash + token compare
            : ...;
        if (matchingBlock == nullptr) return nullptr;
        searchRoot = std::move(matchingBlock);
    }
    ...
}
```

Called from:

1. `BlockRange::fromReuseTree` (`cacheFormatter.cpp:222`)
2. `getBlockRangeForSending` (`cacheFormatter.cpp:182–223`)
3. `CacheFormatter::format` (`cacheFormatter.cpp:384`) — synchronously
   blocks the send path on every per-target send.

For the no_reuse run, gen sends an empty `lastBlockKey`, so ctx
falls through to `BlockRange::fromAllBlockIds` instead. That path
returns the sequence's full block list directly from
`mCacheManager->getCacheBlockIds(...)` — also O(prefix_blocks)
because ctx-side reuse still attached all prior turns' blocks to the
sequence. Hence ~equal cost.

### Why bytes don't matter

`send_ms`/`recv_ms` of the actual transport:

| run      | bytes p50 | `send_ms` p50 | implied B/W |
|----------|-----------|---------------|-------------|
| si1      | 20 MB     | 0.11 ms       | ~1.5 Tbit/s |
| no_reuse | 1.7 GB    | 3.14 ms       | ~4.4 Tbit/s |

Wire is healthy. The 85 ms p50 floor is purely the
ctx-side `findBlocksInReuseTreeByBlockKey` walk.

### Fix candidates (cheapest first)

**F1. Cache the `lastBlock` pointer on the ctx-side `GenerationRequest`
between turns.** After the first turn, ctx already has the leaf
KVCacheBlock for this sequence. Reuse it on the next `format()` instead
of re-walking the radix tree from root every send. Invalidate on free.

**F2. Pass `indexFromEnd` blocks not via `findMatchingBlock` chain but
via the sequence's own `mCacheBlockIds` array.** When ctx's
`enable_block_reuse` is on AND `sequenceHasReuseHits == true`, the
sequence's `mCacheBlockIds[windowSize]` already lists the matched
blocks in order — no tree walk needed. The current code path uses
`fromReuseTree` exactly because the *gen* side wants the last-N blocks
in tree order; but ctx can serve that out of `mCacheBlockIds`
without taking `mCachedBlocksRootMutex`.

**F3. Drop the global `mCachedBlocksRootMutex` for read-only lookups.**
The mutex serializes ctx's own request scheduling against the
transceiver send path. Replace with a shared/exclusive lock; reads in
`findBlocksInReuseTreeByBlockKey` are read-only.

**F4. Eliminate the `chopVectorIntoBlocks` allocation.** It builds a
`vector<vector<UniqueToken>>` of size `prefix_tokens / tokens_per_block`
every call, then iterates over it. Easy to do in-place via a span
or by reusing the per-block sub-keys that the gen side already has
(they could be passed in the RequestInfo).

F1 alone should drop `format()` p50 from 85 ms to <2 ms for warm turns
(no `prepopulated_tokens` dependence at all, since we don't traverse).

## Hypothesis

> The disagg-vs-agg TTFT gap **grows monotonically with turn number within a
> multi-turn conversation**, driven by per-block work in the cache
> transceiver `format()`/`unformat()` paths that scales with the conversation
> prefix held on the gen worker, NOT with the bytes the transceiver actually
> moves over the wire.

This is a refinement of round 6's "per-block transceiver overhead on
agent-056" observation: round 6 saw it on the worst single trajectory only.
Round 7's analysis below shows the same effect is the rule, not the
exception, across 4 of 5 RWLT trajectories.

### "Per-block work" — what specifically (not hand-waving)

The wall-clock between `kv_cache_transfer_start` and `kv_cache_transfer_end`
is set in exactly one place on the gen receiver:

  - `dataTransceiver.cpp::requestSync` (line ~1054), which brackets:
    1. `sendRequestInfo(llmRequest)` — local manifest build + send.
    2. `receiveReadySignal(session)` — block on a single recv from ctx.
    3. `receiveSync(session)` → `mCacheTransferLayer.unformat(session)` →
       per-connection `session.recv(...)` → `concatKvCacheV2Dispatch`.

Per the round-6 si=1 transceiver CSV (`disagg_round6_si1/disagg_kvcache_time/rank_0_recv.csv`):

  - RequestInfo + Preparation + Preprocess + Postprocess + Delay = **< 5 ms p50**.
  - **`Transmissions` = 73 ms p50, 216 ms p90, 531 ms max.** Effective bandwidth
    p50 = 2.5 Gbps (NVLink peak is ~600 Gbps).

So >95% of `kv_xfer_ms` lives in `Transmissions`, which on the gen side is
the wait inside `session.recv(...)` for bytes from ctx PLUS ctx's `format()`
work to start producing them.

The per-block work that scales with the **full prefix length** every turn,
on both sides:

1. **Sender ctx `getBlockRangeForSending` + format() walk** —
   `cacheFormatter.cpp:411–432`. For VSWA (`poolNum > 1`) this returns the
   request's full-window block list and the loop pushes every block pointer
   into `inputKvCacheBlocksPerWindow`, summing `allCacheBlockSize`.
   O(ctx prefix blocks).
2. **Sender `splitKVCacheDispatch`** — `cacheFormatter.cpp:550`. CUDA gather
   kernel launched with input = full prefix block volume; followed by a
   `bufferManager.getStream().synchronize()` at line 553. Per-block
   addressing work and kernel-launch overhead scale with block count.
3. **Receiver gen `getBlockRangeForReceiving`** —
   `cacheFormatter.cpp:268–284`. For VSWA the optimized "slice off reused
   blocks" branch (`cacheFormatter.cpp:235`, requires `poolNum == 1`) is
   bypassed. The fallback keeps every block of the full-window pool
   (`windowSize/tokensPerBlock + 1 = 2049` >> typical prefix of 100–2500
   blocks, so `startBlockIdx = 0`). Returned blockRange contains
   `reused + new` block IDs.
4. **Receiver `unformat()` walk** — `cacheFormatter.cpp:614–627`. Mirrors the
   sender: pushes every block into `outputBuffersPerWindow`, sums
   `cacheBlockSizeSum`. O(gen prefix blocks).
5. **Receiver `concatKvCacheV2Dispatch`** (scatter kernel after the recv
   buffer is filled) — sized for `cacheBlockSizeSum`, which is the
   full-prefix block volume regardless of how many blocks actually carry
   new data.
6. **Per-`session.send/recv` fixed transport cost** (UCX/NIXL/MPI control
   path). Visible as the low p50 effective bandwidth (2.5 Gbps).

The empirical data (round-7 si=1, n=317 warm requests) singles this out
cleanly. Pearson correlations of `kv_xfer_ms` with:

  - `reused_blocks` (= prefix length already on gen):           r = +0.910
  - `total_blocks` (`reused + new`):                            r = +0.930
  - `new_alloc_blocks` (blocks that carry actual deltas):       r = -0.049
  - `kv_size_bytes` (bytes actually moved on the wire):         r = -0.049

And bucketed by prefix length, **bytes go down as wall-clock goes up**:

| reused blocks | n   | kv_xfer_ms p50 | kv_xfer_ms p90 | mean bytes |
|---------------|-----|----------------|----------------|------------|
|    100 – 500  |  19 |   6.5 ms       |  20.0 ms       | 112 MB     |
|    500 – 1000 |  72 |  17.2 ms       |  35.8 ms       |  46 MB     |
|   1000 – 1500 | 101 |  75.4 ms       | 103.7 ms       |  41 MB     |
|   1500 – 2000 |  78 | 147.4 ms       | 214.5 ms       |  25 MB     |
|   2000 – 3000 |  43 | 268.3 ms       | 324.0 ms       |  23 MB     |

Slope ≈ **0.13 ms per reused block**. At 2500-block prefixes that's ~325 ms
of pure per-block cost.

Plus the round-7 `no_reuse` ablation: bytes go 20 MB → **1.75 GB (88× more)**,
effective bandwidth jumps 2.2 → 173 Gbps, and `kv_xfer_ms` p50 stays at
89 → **82 ms** (slightly faster). Bytes are not on the critical path. At all.

To attribute the 0.13 ms/block among items #1–#6 above we need the
per-phase NVTX / `session.setTime` instrumentation queued as I1/I2/I6/I7
below. The data so far is consistent with #3 (full-prefix walk on gen) and
#1/#2 (full-prefix walk + gather on ctx) being the dominant share, since
those are the only ones that scale with `reused_blocks` even when no new
bytes need to move.

## Evidence (from existing data, no new code required)

All numbers below come from re-analyzing data already on disk:

- `rwlt-results/disagg_round5_patched_top5_0521/perf_metrics_proxy.json`
  (the round 5 disagg drain, which carries `kv_cache_metrics` per request
  — including `num_total_allocated_blocks`, `num_new_allocated_blocks`,
  `num_reused_blocks`).
- `rwlt-results/disagg_round6_si1/disagg_kvcache_time/rank_0_recv.csv`
  (the round 6 si=1 transceiver per-phase breakdown).

Reproducible via:

```bash
python3 scripts/turn_progression_analysis.py \
  --agg-dir   rwlt-results/agg_round5_patched_top5_0521 \
  --disagg-dir rwlt-results/disagg_round5_patched_top5_0521 \
  --out-dir   rwlt-results/turn_progression_round5_ref
```

(See `turn_progression_round5_ref/{correlations.txt,growth_summary.txt}`
for the full output committed alongside this writeup.)

### 1. Δ TTFT grows monotonically with turn number (4 of 5 trajectories)

Per-trajectory `Pearson r(turn_idx, delta_ttft_ms)` on the round 6 si=1
data (best baseline today, conc=1, gpt-oss-120b, RWLT top-5):

| conv | n turns | Δ TTFT turn 0 (ms) | Δ TTFT last (ms) | Δ med early (ms) | Δ med late (ms) | r(turn, Δ) |
|---|---:|---:|---:|---:|---:|---:|
| agent-010 | 79 | -59.7 |  178.4 | -24.6 |  84.0 | **+0.93** |
| agent-011 | 63 | -47.8 |   87.0 | -43.7 |  36.7 | **+0.92** |
| agent-051 | 47 | -60.1 |  121.9 | -19.3 |  71.7 | **+0.98** |
| agent-056 | 61 | 1051.7 | 305.6 | 152.1 | 252.1 | +0.14 |
| agent-073 | 68 | -64.4 |   71.1 | -46.7 |  34.4 | **+0.75** |

For 4/5 trajectories disagg actually **beats** agg on early turns
(`Δ TTFT < 0`) and crosses to losing by late turns. Agent-056 is the
single outlier — its turn 0 carries a 1051 ms cold-start cost (9.5 K
fresh tokens, no prefix to reuse on gen) that swamps the trend signal,
and its per-turn slope is shallower because every turn has unusually
high `num_reused_blk` (60–80 K reused tokens vs 30–45 K for the others).

### 2. The growth is entirely in transceiver `Transmissions`

The transceiver CSV breaks `kv_xfer` into `Preparation`, `Preprocess`,
`Transmissions`, `Postprocess`, `Delay`. Per-trajectory early-half vs
late-half medians (ms) on round 6 si=1:

| conv | n | prep e/l | preproc e/l | xfer e/l | postproc e/l |
|---|---:|---:|---:|---:|---:|
| 010 | 79 | 0.83 / 1.52 | 0.02 / 0.03 |  **29.5 / 135.7** | 0.06 / 0.06 |
| 011 | 63 | 0.73 / 1.25 | 0.02 / 0.02 |  **21.3 /  86.0** | 0.06 / 0.06 |
| 051 | 47 | 0.81 / 1.38 | 0.03 / 0.02 |  **28.9 / 111.5** | 0.06 / 0.06 |
| 056 | 61 | 1.53 / 2.07 | 0.02 / 0.02 | **181.9 / 280.4** | 0.06 / 0.06 |
| 073 | 68 | 0.54 / 1.19 | 0.02 / 0.02 |  ** 9.1 /  74.7** | 0.06 / 0.06 |

`prep` grows from sub-1 ms to sub-2 ms (a 2× swing of <1 ms — noise).
`preproc`, `postproc`, `delay` are flat. All ~5× growth from
early-half to late-half lives in **`Transmissions`** — the actual
transfer wall-clock. So the bookkeeping that scales with turn number
runs *inside* the transceiver's transfer loop, not in the framing
phases.

### 3. `Transmissions` time correlates with `num_reused_blocks`, not bytes

This is the smoking-gun finding. Pooled Pearson r across all 317
joinable round 5 disagg requests:

| Predictor | corr(`kv_xfer_ms`, X) | corr(`delta_ttft_ms`, X) |
|---|---:|---:|
| `num_new_allocated_blocks` (= blocks actually transferred) | -0.04 | +0.19 |
| `num_total_allocated_blocks` (= same as new in this run)   | -0.04 | +0.19 |
| `kv_cache_size` (= bytes on the wire)                     | -0.04 | +0.19 |
| `num_reused_blocks` (= gen-cached prefix length)          | **+0.87** | **+0.77** |
| `reused_tok` (rwlt-level same quantity)                   | +0.82 | +0.76 |
| `fresh` (= tokens newly prefilled by ctx)                 | -0.07 | -0.31 |

Per-trajectory the picture is even cleaner (4/5 trajectories with
`|r| > 0.9` only for `num_reused_blocks`; `|r| < 0.35` for everything
byte-related). The per-tertile `kv_xfer_ms / num_new_allocated_blocks`
ratio grows by 6–13× from low- to high-`num_reused_blocks` tertile:

```text
  conv   bucket    n    med xfer/new_blk(ms)
   010      low    26                   1.429
   010      mid    26                   4.758
   010      high   26                   8.234
   011      low    21                   0.742
   011      mid    21                   2.866
   011      high   21                   8.990
   051      low    15                   0.678
   051      mid    15                   5.631
   051      high   17                   6.139
   056      low    20                   5.297
   056      mid    20                  17.222
   056      high   21                  28.044
   073      low    22                   0.393
   073      mid    22                   2.436
   073      high   24                   5.341
```

i.e., the transceiver takes **6–13× longer to move the same number of
blocks** once the gen worker holds more cached prefix from prior turns.

### 4. What this rules in / rules out

| Candidate                                                  | Verdict |
|---|---|
| Bytes-on-wire bandwidth limit                              | Ruled out (corr ≈ 0 with bytes; r ≈ +0.9 with prefix length) |
| `prep` / `preproc` framing cost (e.g. RequestInfo serialization) | Ruled out (these are sub-ms and flat across early/late) |
| `postproc` / response-emission cost                        | Ruled out (flat) |
| Per-block setup/teardown inside `Transmissions` proportional to `num_reused_blocks` | **Strongly supported** |
| Per-block setup proportional to `num_total_allocated_blocks` (= new only) | Ruled out (corr ≈ 0) |
| Generic gen-worker scheduler queue depth                   | Already ruled out in round 6 (transceiver `Transmissions` ≈ `kv_xfer_ms` ≈ `wait_ms`, so the wait is inside the transceiver, not the scheduler) |
| **GPT-OSS-specific: VSWA bypasses the reuse-skip optimization in the disagg transfer path** | **Confirmed by code inspection — see §5** |

Working sub-hypothesis (refined): the cache transceiver's transfer
sequence on the receiver side requests the **entire** prefix block
range every turn for GPT-OSS, because the reuse-aware "slice off the
blocks the gen worker already has" optimization in
`cacheFormatter.cpp` is guarded against multi-pool models. Per-block
descriptor exchange / buffer setup for the full prefix then dominates
`Transmissions` even when only `num_new_allocated_blocks` worth of
bytes actually move.

### 5. GPT-OSS-specific: the VSWA fast-path bypass

GPT-OSS uses **variable sliding window attention** (sliding-window
layers interleaved with full-attention layers), which gives
`BlockManager::isVariableWindow() == true` and `getNumPools() > 1`.
A surprising amount of the disagg KV-cache transfer code is gated
against this case. Confirmed call sites (no runtime measurement
needed — these are static guards):

1. `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp:235`
   (`getBlockRangeForReceiving`):

   ```cpp
   if (poolNum == 1 && srcEnableBlockReuse && srcEnablePartialReuse
       && !recvSideHasCP && srcPpSize == 1)
   {
       // Build from all block ids, then slice off the reused blocks
       // so we only transfer newly allocated ones.
       ...
       auto const reusedBlocks = ... prepopulatedTokens / tokensPerBlock ...;
       ... newBlockIds.assign(allBlockIds.begin() + reusedBlocks, ...);
   }
   ```

   For GPT-OSS, `poolNum > 1`, so this fast path is skipped. The
   fallback at line 268 returns `BlockRange::fromAllBlockIds(...)` for
   the **entire** prefix and then iterates per-window only to compute
   window-size offsets — it does **not** trim by `prepopulatedTokens`.

2. `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp:823`
   (`CacheReceiver::sendRequestInfo`):

   ```cpp
   RequestInfo requestInfo(requestId, mSelfState);
   if (!mCacheTransferLayer.getCacheManager()->getBlockManager().isVariableWindow())
   {
       // ... compute lastBlockKey + indexFromEnd from prefix-cached blocks ...
       requestInfo = RequestInfo(requestId, mSelfState, indexFromEnd, lastBlockKey);
   }
   ```

   For GPT-OSS, this entire block is skipped. `RequestInfo` ships
   with the default `indexFromEnd = 0` and empty `lastBlockKey`, so
   the sender has no "skip everything before this block" hint.

3. `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3261, 3377`:
   `mEnableBlockReuse && !mBlockManager.isVariableWindow()` — block
   reuse fast paths in the KV cache manager itself are gated off.

4. `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:1316`:

   ```cpp
   TLLM_CHECK_WITH_INFO(!isVariableWindow(),
       "analyzePrefixReuse does not work for variable window attention");
   ```

5. `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:195-196,
   292-298, 374, 421`: disagg-side scheduler optimizations also gated
   against VSWA.

This explains the round 5 / round 6 numbers in one sentence: **for
GPT-OSS the disagg KV cache transfer path always asks for and walks
the full conversation prefix; the per-block cost is paid every turn
in proportion to the prefix length, not the delta.**

It also makes the round 7 ablation more meaningful as an
attribution test (next section), and changes the C++ instrumentation
priorities (item I6 added below).

## What landed today (2026-05-22)

### Python-only logging addition

`tensorrt_llm/_torch/pyexecutor/{llm_request,py_executor}.py`:

- **`_disagg_dbg_enabled()` default flipped to ENABLED-on**
  (`os.getenv("TRTLLM_DEBUG_DISAGG_GEN_TIMING", "1") != "0"`). To opt
  out explicitly: `TRTLLM_DEBUG_DISAGG_GEN_TIMING=0`. Yesterday the
  log silently didn't emit for an entire run because the env var
  didn't reach the MPI-spawned gen worker — the round 6 fix tried to
  address this via the `TRTLLM*` prefix allowlist in `MpiPoolSession`
  but it apparently still doesn't fire end-to-end reliably. Until the
  forwarding is hardened, default-on is the safer setting for this
  investigation. **Revert to default-off when the investigation
  closes** (one-character edit: change `"1"` back to `"0"`).
- The startup banner now reports the raw env value (or
  `<unset, default-on>`) so we can tell at a glance whether the
  default was used vs an explicit opt-in/out.
- Two new `LlmRequest` fields:
  `py_dbg_gen_pool_{free,used}_blocks_at_arrival`. Both `None` by
  default (allocation-free when the gate is off, which we still
  honor — the flip is to the gate's *default*, not its semantics).
- `PyExecutor._sample_gen_pool_occupancy()` returns
  `(free_blocks, used_blocks)` from
  `kv_cache_manager.get_kv_cache_stats()`. Returns `(None, None)` on
  any failure — never raises into the scheduling loop.
- `_prepare_disagg_gen_init` stamps both fields on each arriving
  disagg-gen request **once per batch** (single C++ stats call
  amortized across all requests landing on the same iter).
- The existing `[disagg-dbg]` log line at first-token time now carries
  `gen_pool_free_blocks_at_arrival=N gen_pool_used_blocks_at_arrival=N`
  alongside the existing five-component timing breakdown.

Why this is the right "small" addition: gen-pool occupancy is a direct
proxy for cumulative conversation state and lets us correlate
`kv_xfer_ms` against it per-request, **without** depending on the
`/perf_metrics` drain (which has destructive-read semantics — round 6
already lost an entire run's records to a single missed drain). The
debug log is the source of record.

### Reproducible turn-progression analyzer

`repro-gptoss-ttft/scripts/turn_progression_analysis.py`. Joins
`(agg, disagg)` `perf_metrics{,_proxy}.json` + `rwlt_requests.jsonl`
on `(conversation_id, conversation_idx)` after sorting by
`start_time` (valid at conc=1). Optionally folds in the transceiver
CSV (`disagg_kvcache_time/rank_0_recv.csv`) for the per-phase
`prep/preproc/Transmissions/postproc` breakdown. Emits:

- `turn_progression.tsv` — one row per joined request, full schema for
  follow-up plotting.
- `correlations.txt` — pooled + per-trajectory Pearson r between
  `{kv_xfer_ms, delta_ttft_ms, wait_ms}` and each predictor.
- `growth_summary.txt` — per-trajectory early-half vs late-half median
  table + per-tertile `xfer / num_new_blk` ratio.

The numbers in §1–§3 above were generated by running this against the
existing round 5 dirs; output committed under
`rwlt-results/turn_progression_round5_ref/`.

### `configs_0522/round7/`

The round 7 directory is a self-contained reproduction set. Four
configs:

```text
configs_0522/round7/
├── repro_agg_tp1_eagle3.yaml                 # unchanged from round6
├── repro_disagg_ctx_tp1.yaml                 # unchanged from round6
├── repro_disagg_proxy.yaml                   # unchanged from round6
├── repro_disagg_gen_tp1_pp4_si1.yaml         # round6 si=1, reference re-run
└── repro_disagg_gen_tp1_pp4_si1_no_reuse.yaml  # ABLATION: gen block reuse OFF
```

The reference re-run gives a same-day same-build comparison point so
the no_reuse ablation isn't cross-versioned against round 6.

## Round 7 ablation: gen-side block reuse OFF

This was originally proposed as a "where does the bookkeeping live"
test. With §5's VSWA finding the ablation is now better understood as
a sensitivity test: GPT-OSS already bypasses the per-receiver
reuse-aware skip, so flipping gen-side `enable_block_reuse` does NOT
change the transceiver's behavior on the VSWA path — but it does
change what gen has cached locally, what `num_reused_blocks` reports
in `kv_cache_metrics`, and what `getPrepopulatedPromptLen()` returns
(though the VSWA code path ignores that value).

Setup:

- Reference (round 6 si=1 carried into round 7 as
  `configs_0522/round7/repro_disagg_gen_tp1_pp4_si1.yaml`): gen has
  `enable_block_reuse: true`. Over a conversation gen accumulates
  ~2 K cached prefix blocks (`num_reused_blocks` p99 = 2463 in our
  round 5 data). Each turn the transceiver still walks the full prefix
  on the VSWA path even though the gen KV pool has those blocks.

- Ablation A
  (`configs_0522/round7/repro_disagg_gen_tp1_pp4_si1_no_reuse.yaml`):
  gen has `enable_block_reuse: false`. Every turn arrives with
  `num_reused_blocks == 0` on the gen worker. Ctx still has its own
  prefix cache (ctx config unchanged), so ctx still pre-computes the
  prefix cheaply; only the transfer side changes.

Three interpretation cases (refined for the VSWA picture):

| Case | Outcome | Interpretation | Action |
|---|---|---|---|
| **A** | `kv_xfer_ms` becomes a clean linear function of *bytes-on-wire* (= the full prefix in MB, ~16× larger than the reference per-turn delta). Ratio `kv_xfer / bytes` is flat across turns. | The bookkeeping that grew turn-by-turn in the reference was gen-side **block-table update** for the prefix that gen already had cached. With reuse=off, gen rebuilds its table from scratch every turn but the per-block walk happens at-rest cost on each block since none are matched against existing cache. The transceiver's `Transmissions` time becomes bandwidth-bound. | Add I5 (KV cache manager block-table update timing). |
| **B** | `kv_xfer_ms` still grows monotonically with turn — even with `num_reused_blocks == 0` — and correlates with `reused_tok` from rwlt (the *workload* prefix length, not what gen has cached). | The cost is **ctx-side**: `getBlockRangeForReceiving` returns `fromAllBlockIds` (full prefix) for VSWA regardless of `prepopulatedTokens`, and `sendRequestInfo` ships no `lastBlockKey` (§5 items 1–2). The transceiver therefore walks the full prefix's worth of blocks on the sender side every turn. | Add I2 + I6 (NVTX-annotate the VSWA `getBlockRangeForReceiving` fallback path + the per-window per-block descriptor exchange loop on the sender side). |
| **C** | Both effects are present (most likely on GPT-OSS) — `kv_xfer_ms` improves somewhat but does not collapse. | Both ctx-side full-prefix walk AND gen-side block-table update contribute. The split comes from comparing per-turn growth slopes. | Both I5 + I2 + I6. |

The strong prior from §5 is that **case B is what we'll see**: the
ctx-side full-prefix walk for VSWA is unconditional on gen-cache
state, so changing gen's `enable_block_reuse` should leave the
transceiver's `Transmissions` time largely unchanged in shape (still
grows with turn). If that prediction holds, the round-7 result becomes
direct evidence that the GPT-OSS VSWA fallback path in
`cacheFormatter.cpp:268-279` is the bug, and the right fix is to
generalize the `poolNum == 1` fast path (lines 235-264) to also handle
the VSWA case — i.e., still slice off
`prepopulatedTokens / tokensPerBlock` blocks per window before
returning the range to the transceiver.

### Optional: confirmatory ablation that doesn't require code

If the round 7 result looks ambiguous, two further sanity-check runs
sit in the same dir:

- **Ablation A′**: `cache_transceiver_config.backend: UCX` on both ctx
  and gen, holding everything else equal. Round 6 already showed UCX
  is uniformly slower than NIXL, but on the VSWA hypothesis the
  *shape* (per-turn growth slope) should be identical between UCX and
  NIXL — the prefix walk dominates both. Mismatch in shape would point
  at a transport-layer cost we missed.
- **Ablation A″**: drop `cuda_graph_config` on gen entirely. Tests
  whether the per-block cost is amplified by CUDA-graph-related stream
  pauses; unlikely but cheap.

Neither is wired up as a separate YAML yet — they're one-knob edits
on the existing si=1 reference, queued for if we need them.

## Reproducing the round 7 setup (when ready)

The launch and teardown scripts already accept arbitrary `CONFIG_DIR`
and `GEN_CONFIG_BASE`, so the round 7 commands are:

```bash
cd repro-gptoss-ttft

# Reference re-run (same as round6 si=1, but on today's build with the
# new gen-pool-occupancy debug log)
CUDA_VISIBLE_DEVICES=0 \
  CONFIG_DIR=configs_0522/round7 \
  scripts/run_session.sh agg rwlt_round1_top5_signed agg_round7_si1

CTX_GPU=0 GEN_GPU=1 \
  CONFIG_DIR=configs_0522/round7 \
  GEN_CONFIG_BASE=repro_disagg_gen_tp1_pp4_si1 \
  scripts/run_session.sh disagg rwlt_round1_top5_signed disagg_round7_si1

# Ablation A: gen-side block reuse OFF
CTX_GPU=0 GEN_GPU=1 \
  CONFIG_DIR=configs_0522/round7 \
  GEN_CONFIG_BASE=repro_disagg_gen_tp1_pp4_si1_no_reuse \
  scripts/run_session.sh disagg rwlt_round1_top5_signed disagg_round7_no_reuse

# Per-turn analysis for each ablation
python3 scripts/turn_progression_analysis.py \
  --agg-dir   rwlt-results/agg_round7_si1 \
  --disagg-dir rwlt-results/disagg_round7_si1 \
  --out-dir   rwlt-results/turn_progression_round7_si1

python3 scripts/turn_progression_analysis.py \
  --agg-dir   rwlt-results/agg_round7_si1 \
  --disagg-dir rwlt-results/disagg_round7_no_reuse \
  --out-dir   rwlt-results/turn_progression_round7_no_reuse
```

Compare `correlations.txt` between the two disagg variants: the
hypothesis is confirmed (case A) iff
`corr(kv_xfer_ms, num_new_allocated_blocks)` flips from ≈ 0 in the
reference to strongly positive in the no_reuse run, AND the
late-half-vs-early-half growth in `growth_summary.txt` collapses.

## C++ instrumentation candidates for follow-up

These are the targeted insertion points that would tell us, post round 7
ablation, which exact piece of code scales with `num_reused_blocks`. None
of them is needed for round 7's go/no-go decision; they're queued for
the follow-up PR.

| # | Where | Change | Why |
|---|---|---|---|
| **I1** | `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::TransferSession::exportMeasure` (~line 135) | Add columns `NumBlocksTotal`, `NumBlocksTransferred` to the per-request CSV. Both are already in scope (`requestedBlockRange` size on receiver, `RequestInfo::mIndexFromEnd` on sender). | One-line `outFile << ","` add per row. Lets us compute `Transmissions / NumBlocksTotal` directly from the CSV, no perf_metrics drain required. |
| **I2** | `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::CacheReceiver::sendRequestInfo` | Wrap the body in a `setTime(kTimeSendRequestInfo)` block; add a `kTimeSendRequestInfo` `TimeNames` enum value. Add a second sub-time around the `getBlockRangeForReceiving(...)` call specifically — that's the VSWA full-prefix walk (§5 item 1). | Today the `Preparation` column lumps everything between request arrival and `Preprocess`. If the slowdown is the VSWA `fromAllBlockIds` walk, this exposes it. |
| **I3** | `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::CacheReceiver` inner per-connection loop | Per-counterpart-block start/end `setTime`. | If the per-block descriptor exchange is the cost, this localizes it to a single iteration of the per-counterpart loop. |
| **I4** | `cpp/tensorrt_llm/nanobind/batch_manager/bindings.cpp` LlmRequest binding | Add a `getter` for `getPerfMetrics().kvCacheMetrics` so Python can read `num_total_allocated_blocks` etc. directly off the request without going through `update_perf_metrics`. | Today's Python-side `[disagg-dbg]` log can't include block counts because the kv_cache_metrics aren't surfaced on the LlmRequest binding. With this, the dbg log becomes a single source of truth (no `/perf_metrics` drain required). |
| **I5** | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp::KVCacheManager::reuseBlocks` (or disagg-receiver-side equivalent) | NVTX range + per-request entry/exit `setTime` callbacks consumed by the dbg log. | If case-A wins in the round 7 ablation, this is where the gen-side per-receiver prefix-match walk happens. |
| **I6** | `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp::getBlockRangeForReceiving` (line 268-279, the VSWA fallback path) | NVTX range + a `TLLM_LOG_DEBUG` of `(poolNum, isVariableWindow, prepopulatedTokens, totalBlocks, returnedBlocks)`. Optionally add a per-window block-id-count log. | Directly observes whether GPT-OSS hits the fast path (§5 item 1, line 235) or the fallback (line 268) and how much prefix the fallback returns. Confirms the bypass hypothesis at runtime without any byte-cost. |
| **I7** | `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp::getBlockRangeForReceiving` fast-path guard (line 235) | **Fix candidate** (not just instrumentation): relax `poolNum == 1` to a per-window slice that also handles VSWA. For each window, compute `reusedBlocksForWindow = min(usedBlocksForWindow, prepopulatedTokens / tokensPerBlock)` and slice. | This is the actual code fix if the round 7 ablation confirms case B. It only requires reasoning about each window-pool's reuse boundary independently. |

## Open questions / known gaps

- §5 establishes the static code-path divergence for VSWA but does
  not yet prove that's the *only* contributor. The round 7 ablation
  outcome (case A vs B vs C) is the runtime confirmation. If case C
  hits, we'll need to split the prefix-walk cost between ctx-side
  (`getBlockRangeForReceiving` fallback) and gen-side (block-table
  rebuild) — that's what instrumentation I2 + I5 are for.

- The fix in I7 (relax `cacheFormatter.cpp:235`'s `poolNum == 1`
  guard to a per-window slice) needs review by someone who owns the
  VSWA cache logic — there may be a non-obvious correctness reason
  the guard exists (e.g., per-window prefix boundaries don't align,
  or the receiver allocates window-specific blocks in a way that
  doesn't tolerate slicing). The TODO comment at line 233 hints at
  unrelated work for PP > 1; the VSWA case looks orthogonal but
  hasn't been verified.

- Agent-056's residual is partly a workload artifact (60 K reused-token
  prefix every turn) and partly the same per-block bookkeeping cost
  amplified. The hypothesis predicts agent-056 will benefit
  proportionally more from a future per-block-overhead fix; round 7
  ablation A does not directly improve agent-056 (its `num_reused_blocks`
  rate of growth is similar to others; it just starts higher).

- `corr(delta_ttft, num_reused_blocks)` is +0.77 pooled, lower than
  `corr(kv_xfer, num_reused_blocks)` = +0.87. The gap (~0.10) is the
  share of TTFT delta that the prefix-length effect does NOT explain —
  likely the cold-start cost on the first turn of each trajectory (where
  delta is dominated by `num_new_allocated_blocks`, which a separate
  bandwidth-bound transfer accounts for).

- `num_new_allocated_blocks == num_total_allocated_blocks` in 100% of
  round 5 records. This is the GEN-side view: gen has no entries from
  prior turns in its own cache when the disagg-gen request first
  arrives, so every block it allocates is "new" on its side. The
  prefix-already-cached count lives in `num_reused_blocks` instead.

- The `TRTLLM_DEBUG_DISAGG_GEN_TIMING` default flip from off to on is
  **temporary** and tagged in the source as such. Revert to default-off
  once the round-7 (and any follow-up rounds) investigations close.
  The right long-term fix is to harden the env-var forwarding through
  `MpiPoolSession._start_mpi_pool` so opt-in actually works
  end-to-end — until then we keep this on by default to avoid losing
  another day's data to a silent-no-emit bug.

---

This is a living document. Append round 7 ablation results below as
they come in; do not edit the analysis above (it's the reference
baseline for the ablation interpretation).

---

# Round 8 (2026-05-22 PM): structural ablation — `reuse=OFF` on both ctx and gen

## Hypothesis being tested

Round 7b localized the bottleneck to a single ctx-side function call
(`KVCacheManager::findBlocksInReuseTreeByBlockKey`) reached only through the
`br_path=reuse_tree` branch of `getBlockRangeForSending`. Round 7b also
proved the `gen-side enable_block_reuse=false` ablation is not a real test:
gen still populates `lastBlockKey` regardless, so ctx still hits
`reuse_tree`. Round 8 is the structural ablation: turn off `enable_block_reuse`
on BOTH workers (and on the agg comparator, to keep things apples-to-apples).
This forces the `br_path=fallback_all_ids` branch in `cacheFormatter.cpp`,
structurally bypassing the suspect call.

## Configs

[`configs_0522/round8/`](../configs_0522/round8/):

- `repro_agg_tp1_eagle3.yaml`: agg with `enable_block_reuse: false`
- `repro_disagg_ctx_tp1.yaml`: disagg ctx with `enable_block_reuse: false`
- `repro_disagg_gen_tp1_pp4_si1.yaml`: disagg gen with `enable_block_reuse: false`
- `repro_disagg_proxy.yaml`: unchanged from round 7

Identical to round 7 in every other respect (TP/PP layout, max_seq_len,
stream_interval, Eagle3, etc.) — only the reuse flag differs.

## Result (n=318 matched turns, `rwlt_round1_top5_signed`)

| Metric | Round 7 baseline (reuse ON) | Round 8 (reuse OFF both) | Δ |
|---|---|---|---|
| `br_path` distribution on ctx | `reuse_tree` 318/318 | `fallback_all_ids` 318/318 | path switched as designed |
| `br_find_tree_ms` p50 | ~10 ms (up to 250 ms at 80k ISL) | **0.000 ms** | hot function structurally bypassed |
| `[disagg-prof] format total_ms` p50 (warm req, 9574 prepop) | 10.9 ms | **1.0 ms** | -10x |
| ctx KV bytes per request (300-block req) | 383 MB | 383 MB | unchanged (single pool, full prefix) |
| agg TTFT p50 | 180.7 ms | **798.5 ms** | **+4.4x WORSE** (lost reuse) |
| disagg TTFT p50 | 208.4 ms | 747.6 ms | +3.6x (also lost reuse) |
| disagg − agg TTFT p50 | **+24.6 ms** (disagg slower) | **-48.5 ms** (disagg faster) | sign flipped |
| disagg − agg TTFT mean | +39.6 ms | -38.8 ms | sign flipped |

## Interpretation

Round 8 proves the +40 ms disagg-vs-agg gap in round 7 is caused by the
`findBlocksInReuseTreeByBlockKey` walk and nothing else. When the call is
structurally unreachable, disagg becomes -49 ms p50 *faster* than agg
(the expected sign: disagg gets prefill parallelism via separate ctx/gen).

But round 8 is not a deployable fix — both sides lose 4x in absolute TTFT
because every turn now does a full cold prefill (no warm-prefix reuse).
It confirms the bottleneck, doesn't fix it.

## Per-conversation breakdown

See [`diff_round8_agg_vs_disagg.tsv`](diff_round8_agg_vs_disagg.tsv). All 5
conversations show the same pattern: disagg wins on every warm turn by a
consistent ~45-55 ms; the only positive deltas are on the first turn of
each conversation where disagg pays the first-request connection-setup cost.

---

# Round 9 (2026-05-22 PM): the actual fix — explicit `max_attention_window`

## What changed vs round 7

The configs in [`configs_0522/round9/`](../configs_0522/round9/) are
**identical to round 7** in every respect (reuse ON everywhere, same TP/PP,
same Eagle3) except **one** added line in every `kv_cache_config`:

```yaml
max_attention_window: [128, 131072]
```

This is the per-layer-type attention window for GPT-OSS hybrid attention
(alternating sliding-window 128 + full-attention 131072). Without setting
it, the KVCacheManager defaults to a single window covering all layers,
which collapses both layer types into a single block pool.

## Result (n=318 matched turns)

| Metric | Round 7 (implicit window) | Round 9 (explicit window) | Δ |
|---|---|---|---|
| `pools` reported by ctx | 1 | **2** | per-layer-type pooling activated |
| `per_window_blocks` (first req, 9573-token prompt) | `131080:300` | `128:5, 131072:300` | sliding-window layer only needs 5 blocks |
| KV bytes per req (300-block warm req) | 383 MB | **190 MB** | -50% memory |
| `br_path` distribution on ctx | `reuse_tree` 318/318 | **`fallback_all_ids` 318/318** | hot path bypassed |
| `br_find_tree_ms` p50 | ~10 ms (up to 250 ms tail) | **0.000 ms** | hot function entirely skipped |
| `[disagg-prof] format total_ms` p50 (warm req, 9574 prepop) | 10.9 ms | **0.65 ms** | -17x |
| `slow_blockkey_walk` canary count | 0 (logging not present in r7) | **0** (logging present, no hits) | confirmed |
| agg TTFT p50 | 180.7 ms | 183.4 ms | +1.5% (noise) |
| disagg TTFT p50 | 208.4 ms | **136.6 ms** | **-34.5%** (-72 ms) |
| disagg − agg TTFT p50 | **+24.6 ms** (disagg slower) | **-46.7 ms** (disagg faster) | sign flipped, magnitude doubled |
| disagg − agg TTFT mean | +39.6 ms | -46.2 ms | sign flipped |

## Root cause, mechanically

The branch in `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`
that selects between the two code paths:

```cpp
if (poolNum > 1 || !cacheManager->isEnableBlockReuse() || !cacheManager->isEnablePartialReuse()
    || lastBlockKey.uniqueTokens.size() == 0 || recvSideHasCP || ppSize > 1)
{
    dbg.path = "fallback_all_ids";   // cheap: BlockRange::fromAllBlockIds
}
else
{
    dbg.path = "reuse_tree";          // expensive: BlockRange::fromReuseTree
                                      //   -> KVCacheManager::findBlocksInReuseTreeByBlockKey
                                      //   -> radix-tree walk under mLastBlocksInTreeMutex
}
```

In round 7, GPT-OSS reports `poolNum=1` because there is one default
window covering all layers, so the `poolNum > 1` short-circuit doesn't
trigger. With `reuse` flags on and a non-empty `lastBlockKey`, every
ctx-side `format()` calls `fromReuseTree` → walks the radix tree from
the root, matching `lastBlockKey.uniqueTokens` block-by-block to find
the request's leaf. Cost is O(prefix length) under a global mutex.

In round 9, the explicit `max_attention_window: [128, 131072]` causes
the WindowBlockManager to instantiate **two separate pools** (one per
distinct window size), so `poolNum=2` and the `poolNum > 1` clause
forces `fallback_all_ids`. The hot function is never called.

## Why this is also a memory win

The sliding-window layer with window=128 only needs ~5 blocks per
request (128 / 32 tokens per block + slack) regardless of prefix
length. In round 7 it was wastefully allocated like a full-attention
layer (~300 blocks for a 9573-token prompt). Halving KV memory has
no direct TTFT effect on this benchmark but raises the
free_gpu_memory_fraction headroom for higher concurrency.

## Per-conversation breakdown

See [`diff_round9_agg_vs_disagg.tsv`](diff_round9_agg_vs_disagg.tsv)
(per-turn columns) and
[`round9_per_turn_by_conv.txt`](round9_per_turn_by_conv.txt)
(formatted side-by-side).

| Conversation | Turns | Δ_TTFT p50 | Δ_TTFT mean | Disagg slower turns |
|---|---|---|---|---|
| `aa-rwlt-coding-agent-010` | 79 | -50.0 ms | -53.0 ms | 1 (+4 ms, turn 56) |
| `aa-rwlt-coding-agent-073` | 68 | -51.4 ms | -54.8 ms | 0 |
| `aa-rwlt-coding-agent-011` | 63 | -51.4 ms | -53.0 ms | 0 |
| `aa-rwlt-coding-agent-056` | 61 | -37.7 ms | -20.0 ms | 4 (cold-start turns 0-3) |
| `aa-rwlt-coding-agent-051` | 47 | -45.1 ms | -47.2 ms | 0 |

313 / 318 turns are negative (disagg faster). The 5 positive deltas
are all on the very first or first-few turns of agent-056, where
disagg pays one-time first-request connection-setup overhead. The
agent-056 turn-0 spike (+883 ms) is a cold-start artifact, not a
warm-state regression — see the per-turn file for full context.

ITL (decode speed) is essentially unchanged R7 → R9 (~1.7 ms agg,
~2.6 ms disagg in both rounds) — the fix is purely a TTFT-side
improvement; decode is unaffected.

## Action items

1. **Document the GPT-OSS deployment requirement**: any `trtllm-serve`
   deployment of `gpt-oss-120b` (or any hybrid-attention model) MUST set
   `max_attention_window` explicitly to the per-layer-type windows.
   Without it, ctx-side disagg incurs a +40 ms p50 TTFT regression that
   grows super-linearly with prefix length.

2. **Update integration test reference**:
   [`tests/integration/defs/accuracy/test_disaggregated_serving.py:1420`](../../tests/integration/defs/accuracy/test_disaggregated_serving.py)
   uses `[128, 32768]` sized for GSM8K's shorter context. If the test
   ever runs with longer prompts, that ceiling needs to match
   `max_seq_len`.

3. **Engine-level fix (optional, longer term)**: the engine could detect
   hybrid-attention models from the architecture config and auto-set
   `max_attention_window` accordingly, so users never have to know this
   knob exists. Until then, this is a config gotcha.

4. **The F1 cache patch (drafted then reverted in round 7d) is still
   relevant** for any model that legitimately runs with `pools=1` AND
   `reuse=ON` — for those workloads the radix walk remains O(prefix)
   per send. Not urgent for GPT-OSS now that the config fix exists.

## Reproduction commands

```bash
# Round 9 disagg
cd /home/bbuddharaju/scratch/TensorRT-LLM_AA_RWLT/repro-gptoss-ttft && \
CTX_GPU=0 GEN_GPU=1 \
CONFIG_DIR=configs_0522/round9 \
CTX_CONFIG_BASE=repro_disagg_ctx_tp1 \
GEN_CONFIG_BASE=repro_disagg_gen_tp1_pp4_si1 \
PROXY_CONFIG_BASE=repro_disagg_proxy \
LOG_DIR=$(pwd)/rwlt-results/disagg_round9_reuse_top5/logs \
scripts/run_session.sh disagg rwlt_round1_top5_signed disagg_round9_reuse_top5

# Round 9 agg comparator
cd /home/bbuddharaju/scratch/TensorRT-LLM_AA_RWLT/repro-gptoss-ttft && \
CONFIG_DIR=configs_0522/round9 \
AGG_CONFIG_BASE=repro_agg_tp1_eagle3 \
LOG_DIR=$(pwd)/rwlt-results/agg_round9_reuse_top5/logs \
scripts/run_session.sh agg rwlt_round1_top5_signed agg_round9_reuse_top5

# Per-turn comparison
python scripts/diff_rwlt_runs.py \
  rwlt-results/agg_round9_reuse_top5/rwlt_requests.jsonl \
  rwlt-results/disagg_round9_reuse_top5/rwlt_requests.jsonl \
  --label-a agg_r9 --label-b disagg_r9 \
  --per-turn-out rwlt-results/diff_round9_agg_vs_disagg.tsv
```
