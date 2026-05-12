<!-- SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Next-step performance plan

Post-M7 status doc for the TinyLlama-1.1B prefill megakernel on L40S
(NVBug 5615248). `RESULTS.md` holds the canonical benchmark numbers;
this doc captures the *direction*: what test infrastructure exists,
what regressions to guard against, and which design changes have the
biggest expected impact next.

## 1. Where we are

Latest end-to-end status, after the M6 tile sweep, M7 bench/report, and
the `grid_gemm` 2D-tile bug fix:

| Knob | Value | Notes |
|------|------:|-------|
| Tile | `BM=128, BN=64, BK=32, NSTAGES=3, WR=4, WC=2` | Legitimate sweep winner (see §2 for the bogus one). |
| Block size | 256 threads (8 warps) | `__launch_bounds__(256, 2)` requests 2 blocks/SM. |
| Grid | 284 = 142 SMs × 2 blocks/SM | Confirmed by the runtime diagnostic. |
| SMEM/block | 45,056 B (44 KB) | 2 blocks/SM use 88 KB; under sm_89's 99 KB cap. |
| Bench (TTFT) | _<fill in from latest bench run; record in RESULTS.md>_ ms | Compare against the 17.18 ms M5 baseline. |
| TRT target | 2.31 ms | Practical HBM floor for this workload. |

`MEGAKERNEL_REPORT_OCC=1` confirms 2 blocks/SM is active. All 12 tests
in `tests/test_gemm_pipeline.py`, `tests/test_modules.py`, and
`tests/test_numerics.py` pass on this configuration.

## 2. Test infrastructure

Three layers; run them top-down when triaging regressions.

| Layer | File | What it validates | Wallclock |
|-------|------|-------------------|----------:|
| GEMM primitive | `tests/test_gemm_pipeline.py` | `block_gemm` matches `torch.matmul(bf16)` across 6 shapes covering qkv / o_proj / gate_up / down_proj / small. | ~5 s |
| Per-stage | `tests/test_modules.py` | RMSNorm, RoPE, attention, SiLU each match HF eager on synthetic inputs, ULP-tight thresholds. | ~10 s |
| End-to-end | `tests/test_numerics.py` | Full forward vs HF `LlamaForCausalLM` eager: BF16 noise-floor thresholds + 50-prompt top-1 token match. | ~80 s |

**Important rule: `tests/sweep.py` does NOT validate numerics.** It
only rebuilds and times the kernel. The first sweep round crowned
`64_128_32_3_2_4` at "11.10 ms" — those numbers were measuring a
kernel that did half the GEMM math, because `grid_gemm` (the
megakernel's inner GEMM driver) hard-coded `m_block_off = 0` and
looped only over N-tiles. With `BM=64 < seq_len=128`, rows 64–127 of
every GEMM output were never written. The bug is fixed (the loop now
iterates the full `m_tiles × n_tiles` space) and `sweep.py`'s
docstring carries an explicit warning. **Always pair a sweep with
`test_numerics.py` before promoting a config to the default.**

### Optional dev knobs

| Env var / flag | Purpose |
|---------------|---------|
| `MEGAKERNEL_REPORT_OCC=1` | One-line dump of `max_blocks_per_sm`, SMEM bytes, grid size at every kernel launch. Useful when iterating on tile sizes or `__launch_bounds__` to confirm the compiler actually fit N blocks/SM — register pressure can silently demote 2→1. |
| `MEGAKERNEL_DEBUG=1` (build-time) | Compiles `-G`, enables the per-layer residual dump path in `bindings.cpp`, consumed by `tests/debug_per_layer.py` for layer-by-layer BF16 diff vs HF eager. |
| `MEGAKERNEL_{BM,BN,BK,NSTAGES,WARP_ROWS,WARP_COLS}` | Compile-time tile knobs honored by `setup.py`. Drive the sweep harness. |
| `MEGAKERNEL_NUM_BLOCKS` | Override the persistent grid size. Launcher auto-clamps to `num_sms × cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)`, so over-requesting is safe. |

### Static guardrails added during M7

- `gemm_pipeline.cuh` now has
  `static_assert(sizeof(GemmSmem) <= MEGAKERNEL_SMEM_CAP_BYTES)`
  (99 KB on sm_89, 227 KB on sm_90+). Catches the `BK=64, NSTAGES=3`
  case that silently reported "0.032 ms" in the M6 sweep because the
  kernel exited without doing work.

## 3. Performance gap analysis

| Component | Time | vs HBM floor | Bound |
|-----------|-----:|-------------:|-------|
| L40S BF16 peak (363 TF) | 0.72 ms | — | compute floor |
| Weight HBM read (≈2 GB / 864 GB/s) | 2.3 ms | 1.0× | memory floor |
| TRT (cuBLAS) | 2.31 ms | 1.0× | HBM-saturating |
| `torch.compile` reduce-overhead | 4.33 ms | 1.9× | mostly HBM + launch overhead |
| Megakernel post-M7 | _<fill in>_ | — | TC pipeline + tile parallelism |
| Megakernel M5 baseline | 17.18 ms | 7.5× | as above, no per-GEMM tuning |

The remaining gap to TRT is **structural**, not a tile knob:

1. **Tensor Core pipeline depth.** At `BLOCK_SIZE=256` we have 8 warps
   per block, each issuing ~2 outstanding MMAs in steady state. With 2
   blocks/SM that's ~32 outstanding MMAs/SM. L40S sm_89 has 4 TC
   partitions/SM, each happy with ~8 deep MMA pipelines = 32 to
   saturate. We're right at the edge. Going deeper requires **more
   warps/SM**, i.e. smaller blocks with more blocks/SM (Phase B below).

2. **Single tile profile for every GEMM.** The same `(BM, BN, BK)`
   compiles for qkv (N=2560), o_proj (N=2048), gate_up (N=11264),
   down_proj (N=2048), lm_head (N=32000). With current defaults
   `BM=128, BN=64`, per-GEMM tile counts are:

   | GEMM      | Tiles | Of 284 blocks | SM utilization |
   |-----------|------:|-------------:|---------------:|
   | qkv       | 40    | 14%          | low            |
   | o_proj    | 32    | 11%          | low            |
   | gate_up   | 176   | 62%          | medium         |
   | down_proj | 32    | 11%          | low            |
   | lm_head   | 500   | 100% × 2 waves | high         |

   Four of five GEMMs leave most of the device idle.

3. **Grid-sync overhead.** Each stage boundary is a `cg::this_grid().sync()`.
   22 layers × ~12 stage syncs + a few epilogue syncs ≈ 270 syncs.
   At maybe 50–100 ns per cooperative sync, that's ~15–30 μs total —
   not the bottleneck, but it sets a floor on how short the kernel
   can ever be.

## 4. Next-step plan, ordered by impact / effort

### A. Per-GEMM tile (high impact, medium effort)

Template `block_gemm` / `grid_gemm` on `(BM, BN, BK, NSTAGES, WR, WC)`
so each call site picks a shape-appropriate tile. The `grid_gemm` 2D-tile
bug fix already makes `BM < seq_len` regimes work; now we can use them.

| GEMM | Suggested tile | Rationale |
|------|---------------|-----------|
| qkv (M=128, K=2048, N=2560)       | `BM=64, BN=128` | 2×20 = 40 tiles → 80 tiles (post-fix). Doubles SM utilization. |
| o_proj (M=128, K=2048, N=2048)    | `BM=64, BN=64`  | 4× more tiles (16 → 128); largest gain since this GEMM was 11%-utilized. |
| gate_up (M=128, K=2048, N=11264)  | `BM=128, BN=128` | Already 62% utilized; bigger tile improves TC efficiency. |
| down_proj (M=128, K=5632, N=2048) | `BM=64, BN=64`  | Same logic as o_proj; high K means more pipelining benefit. |
| lm_head (M=128, K=2048, N=32000)  | `BM=128, BN=128` | N-parallelism is abundant; big tile wins on TC efficiency. |

**Estimated win**: 1.3–1.5× on the small-N GEMMs, no harm on the big
ones. Combined: roughly **1.3–1.5×** on the layer body.

**Implementation cost**: ~100 LOC, mostly in `gemm_pipeline.cuh` (turn
the `kBM`/`kBN`/... constants into template parameters) and the
`grid_gemm` callers in `tinyllama_megakernel.cu`. The SMEM budget
needs care: if the union-max tile (`BM=128, BN=128, BK=32, NSTAGES=3`)
is ~56 KB, two of those don't fit in 99 KB and we'd silently demote
to 1 block/SM for the gate_up/lm_head GEMMs. Two options:

1. **Union-max SMEM, single launch_bounds** — accept that some GEMMs
   only get 1 block/SM and lose Phase-1's gain on them.
2. **Per-tile SMEM via `cudaFuncSetAttribute` on multiple kernel
   entry points** — but we only have one persistent kernel, so this
   means splitting the megakernel into per-stage kernels, which loses
   the whole design.

Option 1 is the pragmatic path. Net expected: still ~1.3× on small
GEMMs, neutral on big.

### B. `BLOCK_SIZE=128` with 4 blocks/SM (medium impact, medium effort)

Halve the block size to 128 threads (4 warps), set
`__launch_bounds__(128, 4)`. 4 blocks × 128 = 512 threads/SM, well
under L40S's 1536-thread/SM ceiling and 65,536/512 = 128 regs/thread
budget. SMEM/block at current tile drops to ~25 KB so 4 fit in 100 KB
(needs careful tile choice to stay under 99 KB).

**Wins**:

- 4× more independent blocks per SM → better stall hiding across cp.async
  waits.
- Total grid = 568 blocks; even small GEMMs spread further (still
  ≥1 wave for most).
- More resident warps/SM → deeper MMA pipeline saturation (addresses
  the TC pipeline depth bottleneck from §3).

**Cost**: every stage must drop the implicit `kWarpsPerBlock = 8`
assumption. The good news is that RMSNorm and attention already scale
via `__shared__ float scratch[kWarpsPerBlock + 1]` etc., so the only
hard-coded site is the GEMM warp grid (`WR × WC = kWarpsPerBlock`).
For a 128-thread block, `WR=2, WC=2` is the natural mapping for a
64×64 tile; for 128×128 it stays 2×2 with each warp owning 4×4 MMA
fragments.

**Estimated additional 1.3–1.8× on top of A.**

### C. Hybrid: megakernel orchestration + cuBLAS for the GEMMs (large impact, design pivot)

The persistent megakernel design has its biggest payoff when launch
overhead dominates — i.e. **decode** (M=1, dozens of tiny ops).
For **prefill at seq=128**, cuBLAS is already at the HBM floor (TRT
2.31 ms ≈ 870 GB/s, 100% BW). A hybrid would:

1. Keep the megakernel for orchestration: per-layer RMSNorm, RoPE,
   attention, SiLU, and residual all stay in one persistent kernel.
2. Cut to a host-launched `cublasGemmEx` (or `cublasLtMatmul`) for
   qkv / o_proj / gate_up / down_proj / lm_head.

Achievable target: **~2.5–3 ms range.** Beats TRT only if the
non-GEMM ops can be hidden inside the cuBLAS calls via streams.

**Trade-off**: gives up the "single dispatch" property that motivates
the megakernel design. Justified only if the goal is *purely beat TRT
on this workload*; not if the goal is to study/showcase the megakernel
pattern itself.

### D. Diagnostics that would unblock further tuning

- **Build with `-Xptxas -v`** and capture regs/thread and stack usage.
  Currently we infer occupancy at runtime; the static numbers would tell
  us exactly how much register headroom remains before spilling.
- **Per-stage NVTX ranges** inside `tinyllama_megakernel.cu`
  (`nvtxRangePush/Pop` around each stage; gate behind
  `MEGAKERNEL_NVTX=1`). The current `analyze_nsys.py` shows that the
  entire 17.2 ms is inside the megakernel but doesn't break it down
  further. Per-stage timing would identify whether attention,
  gate_up, or lm_head dominates.
- **Per-GEMM roofline** against L40S 363 TF / 864 GB/s — tells us
  whether each GEMM is compute-bound or memory-bound. cuBLAS hits
  ~30% peak (113 TF) on this shape; that's the realistic upper bound
  for a single-launch persistent design.

## 5. Recommended order

1. **A — per-GEMM tile** (biggest, lowest risk, no design change).
2. **D — diagnostics** (cheap; makes the next iteration data-driven).
3. **B — `BLOCK_SIZE=128`** (touches every stage; do AFTER A so the
   tile-knob search and the warp-count change don't interact).
4. **C — hybrid** (only if A+B don't close enough of the gap and the
   spec is "beat TRT" rather than "ship a clean megakernel").

If the goal is to close the gap to TRT *as a megakernel*, expected
landing zone after A+B is **~3–4 ms** (1.3–1.7× off TRT). Going past
TRT requires C.

## 6. Regression guards

Before promoting any new default in `setup.py`, run:

```bash
cd nvbugs_5615248/megakernel
pip install -e . --no-build-isolation
pytest tests/test_gemm_pipeline.py tests/test_modules.py tests/test_numerics.py -v
MEGAKERNEL_REPORT_OCC=1 python tests/bench.py --use-synthetic --warmup 50 --iters 200
```

The first command catches GEMM and per-stage drift. The second catches
end-to-end accumulation. The third confirms occupancy AND that the
new config doesn't regress latency. If `max_blocks_per_sm` in the
occupancy line is lower than expected (e.g. 1 when you set
`__launch_bounds__(_, 2)`), suspect register pressure first — that's
usually fixed by a smaller tile or by allowing fewer outstanding WMMA
fragments per warp.
