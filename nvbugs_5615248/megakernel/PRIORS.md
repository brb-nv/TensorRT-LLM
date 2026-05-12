# M0 — Prior art synthesis for an L40S TinyLlama-1.1B prefill megakernel

This document distills what we need from the three reference implementations of fused
LLM forward passes, and explicitly lays out which design choices change for
**L40S (sm_89, Ada Lovelace)**.

## Sources

1. **Hazy Research — "Megakernels and the future of AI inference" / "No Bubbles" (2025)**
   ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)).
   Original public articulation of the persistent single-kernel forward pass.
   Targets Llama-1B *decode* on H100. Introduces the **task-graph + interpreter**
   formulation: every layer becomes a list of "ops" that are scheduled onto a
   persistent grid, with explicit barriers between data-dependent ops.

2. **Lucebox `megakernel/`** for Qwen3.5-0.8B
   ([Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/tree/main/megakernel)).
   Targets Ampere (sm_86) and Turing (sm_75) for **decode**. Prefill on
   Ampere/Ada (`prefill.cu`) is **NOT a single-dispatch megakernel** — it is
   cuBLAS GEMMs glued to a few standalone CUDA kernels (RMSNorm, RoPE, attn,
   SiLU). Their `prefill_megakernel.cu` is single-dispatch but is
   **Blackwell-only** because it relies on TMA + `wgmma`. So for our L40S
   target, **the prefill megakernel does not yet exist in the public domain.**

3. **AlpinDale/qwen_megakernel** and **Infatoshi/MegaQwen** — earlier
   Qwen-family fused-kernel attempts. Both target decode, both use the
   persistent-grid + cooperative-sync pattern, both BF16 on sm_80+.

## Cross-cutting design themes that we MUST inherit

### 1. Persistent grid sized to "1 block per SM"

Every reference launches a fixed number of blocks equal to (or bounded by)
the SM count. On L40S that is **142 blocks**. Each block stays resident for
the entire forward pass and processes a fraction of every layer's work.

Launch path is `cudaLaunchCooperativeKernel`, which requires every block to
be **co-resident** at launch time. If our register / SMEM footprint forces
fewer than `gridDim.x` blocks per device, the kernel deadlocks at the first
`grid.sync()`. **Mitigation**: query
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` at launch and clamp grid
size to `min(num_sms, max_active_blocks * num_sms)`. Lucebox does exactly this
(`kernel.cu` calls it the "resident-block ceiling"); we will too.

### 2. Cooperative grid sync between layers, NEVER inside a per-token loop

Lucebox notes verbatim: *"`grid.sync()` inside loops will deadlock silently"*.
The safe pattern is:

```text
for layer in [0..N):
  Stage 1 (RMSNorm + QKV)   ; cg::this_grid().sync()
  Stage 2 (RoPE)            ; cg::this_grid().sync()
  Stage 3 (attention)       ; cg::this_grid().sync()
  Stage 4 (O proj + resid)  ; cg::this_grid().sync()
  Stage 5 (RMSN + gate_up)  ; cg::this_grid().sync()
  Stage 6 (SiLU * up)       ; cg::this_grid().sync()
  Stage 7 (down + resid)    ; cg::this_grid().sync()
```

Each stage reads from one global activation buffer and writes to another,
flipped between layers (double-buffered).

### 3. Weights stream from HBM, not SMEM-resident

TinyLlama-1.1B BF16 is ~2.2 GB. We have ~14 MB of total L1/SMEM across the
whole device. Weights live in HBM; each GEMM tile fetches its
slab via `cp.async`. **Activations** between layers (~512 KB at M=128) fit
comfortably in a small set of global "scratch" buffers, *not* in SMEM —
SMEM doesn't survive `grid.sync()` (it's per-block), so all cross-block /
cross-stage data must live in HBM.

### 4. Register-resident accumulators

MMA accumulators stay in registers across the full K reduction. FP32 accumulation
into BF16 outputs. Final BF16 cast happens only on the spill to SMEM/HBM. This
is critical for sm_89: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

### 5. Compile-time architecture branch

Lucebox uses `-DTARGET_SM=86 / 75 / 120 / 121` to switch BF16 vs FP16,
TMA vs `cp.async`, etc. We do the same with `-DTARGET_SM=89` baked in.
(We deliberately do NOT inherit their FP16 path — L40S supports BF16
natively and TinyLlama is trained in BF16.)

## What changes on L40S (sm_89, Ada)

| Concern                       | Hopper (sm_90+, what Hazy targets)        | Ada / Ampere (what we target) |
|-------------------------------|-------------------------------------------|-------------------------------|
| Bulk async loads from global  | TMA (`cp.async.bulk.tensor`, `mbarrier`)  | `cp.async.{ca,cg}.shared.global` 16B + `cp.async.commit_group` / `wait_group` |
| Tensor-core issue             | `wgmma.mma_async` (warpgroup, 128 threads)| `mma.sync.aligned.m16n8k16` (warp, 32 threads), accumulator in registers |
| Cross-block shared memory     | DSMEM (distributed shared memory)         | None — all cross-block via HBM |
| Async copy / compute overlap  | Built into wgmma + TMA                    | Manual via `cp.async` pipeline depth + `__pipeline_wait_prior` (CUDA driver) or raw PTX |
| Max dynamic SMEM per block    | 228 KB (H100)                             | **99 KB on L40S** — must `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` |
| BF16 throughput               | 989 TFLOPS (H100 SXM)                     | **~362 TFLOPS** |
| HBM bandwidth                 | 3.35 TB/s (H100 SXM)                      | **~864 GB/s** |
| Memory ratio (FLOPS:BW)       | ~295                                      | **~419** |

Two implications of that last row:

- **L40S is *more* compute-rich relative to bandwidth than H100.**
  Our megakernel's bandwidth ceiling is *worse* relative to compute.
  That makes weight-reuse across the K-reduction critically important.
- **L40S has no TMA**: every memory operation has to be issued explicitly
  by threads via `cp.async`. That means we burn more *instruction issue
  slots* on memory than a Hopper kernel does. Mitigation: maximize the
  bytes-per-instruction by always using 16-byte (`int4`) `cp.async`.

## Per-stage SMEM layout (initial picks, will refine in M3/M4)

For BM=128, BN=128, BK=32, 3-stage `cp.async` ring:

```
A tile (M×K):  128 × 32 × 2B = 8 KB  × 3 stages = 24 KB
B tile (K×N):   32 × 128 × 2B = 8 KB × 3 stages = 24 KB
mbarrier slots:                 ~1 KB
                                ------
                                ~49 KB  (well under 99 KB Ada cap)
```

For Stage 3 (prefill attention):

```
Q tile (heads × M × head_dim):   8 × 128 × 64 × 2B = 128 KB   ← exceeds 99 KB!
```

So we cannot materialize all 32 Q-heads at once. Tile by head-groups: 4 Q-heads × M × head_dim
= 64 KB. KV tile: 4 K-heads (the GQA group) × M × head_dim = 64 KB, but we can amortize
across the 8-fold head-sharing.

The Stage 3 SMEM design is the most delicate part of the kernel and will be
revisited in M4. For M3 (GEMM-only primitive) the simpler stage-1 layout is fine.

## "Lessons" we are accepting on faith from Lucebox

- **`grid.sync()` inside per-token loops deadlocks silently.** Only between layers.
- **Register pressure kills silently.** Spills to local memory, no error, perf cliff.
  Initial target: ≤ 96 registers/thread, `__launch_bounds__(256, 1)`.
- **`S_TILE=16` was too aggressive on Ampere; `S_TILE=8` was the sweet spot.**
  We will start with conservative tiles and grow.

## Bottom-line for our project

We are extending the Lucebox/Hazy persistent-megakernel idea **into a regime that
does not yet have a public implementation**: a true single-dispatch *prefill*
megakernel on Ada. The Lucebox Blackwell prefill megakernel is the closest
prior reference and its key trick (TMA-driven weight streaming) does not
translate to L40S. We replace it with a `cp.async` 3-stage pipeline that
issues 16-byte vector loads from HBM into a small ring of SMEM tiles. The
rest of the architecture (persistent grid, per-stage barriers, register-resident
accumulators, double-buffered HBM activations) carries over verbatim.

Expected risk: getting `cp.async` pipelining tuned on Ada is harder than the
TMA path on Hopper, so M3 ("GEMM pipeline primitive") is where the most
schedule risk lives. We allocate 1 week to it.
