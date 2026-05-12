# TinyLlama-1.1B Prefill Megakernel (L40S, NVBug 5615248)

A persistent single-dispatch CUDA megakernel that runs the entire TinyLlama-1.1B
BF16 prefill forward pass (embedding + 22 layers + RMSNorm + LM-head) in
**one** `cudaLaunchCooperativeKernel`, targeting an L40S (sm_89, Ada Lovelace)
at the **NVBug 5615248** workload point (batch=1, seqlen=100 padded to 128).

The driving perf target is to beat the TRT engine baseline of **2.31 ms** from
[POST_V5_DIVERGENCE.md](../trtllm_bench/POST_V5_DIVERGENCE.md) and ultimately
close the 4.93 ms TTFT gap between TRT-LLM's PyT backend and its TRT backend
for this workload.

This is **v1: out-of-tree only**. TRT-LLM integration (paged KV-cache write,
PyExecutor fast-path, thop op registration) is a Phase-2 follow-up — see
[the plan](../../../.cursor/plans/tinyllama_l40s_prefill_megakernel_2d2922ff.plan.md)
under "Phase 2 (deferred): TRT-LLM integration".

## Status

Implemented in milestones M0-M7 per the plan. **Has not yet been built or run
on hardware** — this branch was developed without GPU access. See the per-milestone
sections below for what each milestone delivers and what it expects the user to
exercise on an L40S.

## Quick start (on an L40S node)

```bash
cd nvbugs_5615248/megakernel
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers pytest    # torch BEFORE the extension build
pip install -e . --no-build-isolation

# M1: scaffold + pack roundtrip + reference (no GPU compute, no _C needed)
pytest tests/test_pack_roundtrip.py -v

# M3: GEMM primitive sanity vs torch.matmul
pytest tests/test_gemm_pipeline.py -v

# M5: end-to-end numerics with synthetic weights
pytest tests/test_numerics.py -v

# Optional: full reference-vs-HF check (requires HF download)
LUCEBOX_RUN_HF_TEST=1 pytest tests/test_reference_vs_hf.py -v

# Bench: M2 baselines + M7 final
sudo nvidia-smi -lgc 2520,2520     # lock L40S clocks (optional but recommended)
python tests/bench.py --backend all --iters 1000

# M6 tile sweep
python tests/sweep.py --warmup 50 --iters 200
```

## Architecture (recap from PRIORS.md)

```
Host: cudaLaunchCooperativeKernel(grid=142, block=256, dyn_smem=~56 KB)
 |
 V
 Stage 0  : token_embed -> residual                                              grid.sync()
 |
 +-- for L in 0..22:
 |     Stage 1a : RMSNorm(residual)            -> act_a              grid.sync()
 |     Stage 1b : act_a @ W_qkv                -> qkv_buf  (cp.async + WMMA)  grid.sync()
 |     Stage 2  : RoPE in-place on Q,K of qkv_buf                              grid.sync()
 |     Stage 3  : causal prefill attention      -> attn_buf                      grid.sync()
 |     Stage 4  : residual += attn_buf @ W_o   (cp.async + WMMA)                grid.sync()
 |     Stage 5a : RMSNorm(residual)            -> act_a                          grid.sync()
 |     Stage 5b : act_a @ W_gate_up            -> gate_up_buf                    grid.sync()
 |     Stage 6  : SiLU(gate) * up              -> silu_buf                       grid.sync()
 |     Stage 7  : residual += silu_buf @ W_down                                  grid.sync()
 |
 Final  : RMSNorm(residual) -> act_a; logits = act_a @ W_lm_head
```

Persistent grid of 142 blocks × 256 threads, one block per SM. Every stage
distributes work by `block_id` round-robin so it scales naturally to any
grid size. `cg::this_grid().sync()` between every stage — **never inside**
the layer loop (per Lucebox: causes silent deadlocks on Ada).

GEMM tile: BM=128, BN=128, BK=32 with a 3-stage `cp.async` ring, 4×2 warp grid,
WMMA m16n16k16 → mma.sync m16n8k16 BF16/FP32 under the hood. Override via
`MEGAKERNEL_{BM,BN,BK,NSTAGES,WARP_ROWS,WARP_COLS}` env vars at build time.

## File layout

```
megakernel/
├── PRIORS.md              -- M0: prior art synthesis + L40S-specific risks
├── README.md              -- this file
├── RESULTS.md             -- M7: empty template, fill in once you've run on L40S
├── setup.py               -- PyTorch CUDAExtension build, arch auto-detect
├── csrc/
│   ├── common.cuh         -- TinyLlama-1.1B constants + MegakernelParams
│   ├── pipeline.cuh       -- cp.async + mma.sync PTX primitives (sm_80+)
│   ├── gemm_pipeline.cuh  -- block_gemm / block_gemm_add (cp.async + WMMA)
│   ├── gemm_pipeline.cu   -- standalone M3 launcher
│   ├── stages.cuh         -- M4 non-GEMM stages (embed, RMSNorm, RoPE, attn, SiLU)
│   ├── tinyllama_megakernel.cu  -- M5 the persistent megakernel + launcher
│   └── bindings.cpp       -- pybind11 + Torch op registration
├── python/lucebox_tinyllama/
│   ├── __init__.py        -- Python API surface
│   ├── pack_weights.py    -- M1 HF state_dict -> packed BF16 blob
│   └── reference.py       -- M1 PyT-eager oracle for numerics
└── tests/
    ├── test_pack_roundtrip.py     -- M1 (no GPU compute)
    ├── test_reference_vs_hf.py    -- M1 (HF download required)
    ├── test_gemm_pipeline.py      -- M3
    ├── test_numerics.py           -- M5 end-to-end
    ├── bench.py                   -- M2 + M7 latency bench
    └── sweep.py                   -- M6 tile sweep harness
```

## Caveats & known limitations of v1

- **Untested on hardware.** The kernel is structurally correct and matches the
  PTX/WMMA patterns from documented Ampere/Ada megakernels (Lucebox decode,
  AlpinDale/qwen_megakernel), but the developer authoring this branch did not
  have an L40S to compile against. Expect compile errors and 1-2 numerical bugs
  on the first build; iterate via `tests/test_gemm_pipeline.py` (smallest unit)
  → `tests/test_numerics.py` (full kernel).
- **No KV-cache write.** v1 is logits-only. Decode handoff is Phase 2.
- **No batching, no variable seqlen.** Fixed at compile-time MEGAKERNEL_SEQ_LEN
  (default 128). Rebuild for other seqlens.
- **Stage 3 attention is the simple "one (head, query_row) per work-unit"
  algorithm**, not FlashAttention. At M=128 the FA win is small and FA's SMEM
  footprint conflicts with the GEMM SMEM budget. M6 has room to revisit.
- **Register pressure may force spills.** WMMA fragments + accumulators come to
  ~128 FP32/thread of accumulator state alone (8 per-warp wmma fragments × 16
  FP32/thread/fragment) — total reg usage likely 200-250. The `__launch_bounds__(256, 1)`
  hint asks for 1 block/SM minimum; if the compiler refuses, drop BLOCK_SIZE
  or reduce wmma-per-warp via a finer warp grid (e.g. WARP_COLS=4).

## How v1 perf is expected to break

Concrete guesses to validate on first run:

1. **First build fails** because of a `mma.h` include path or `cooperative_groups.h`
   linkage — straightforward fix.
2. **First run gives numerical garbage** because the WMMA fragment row/col layout
   doesn't match the HBM B layout (row-major weights vs col-major B fragment).
   Fix by toggling `wmma::row_major` <-> `wmma::col_major` for `b_frag` in
   `gemm_pipeline.cuh::warp_mma_one_stage`.
3. **First clean run is slower than 2.31 ms** because the GEMM is not yet pipelined
   tight enough (only 3 cp.async stages, large idle gaps between commit_group and
   wait_group). Sweep via `tests/sweep.py` to find the per-shape sweet spot.
4. **`cudaLaunchCooperativeKernel` returns `cudaErrorInvalidValue`** because
   register pressure forced 0 blocks/SM. Mitigation in `launch_megakernel`:
   we clamp grid_size to `num_sms * max_active_blocks_per_sm`, but if that's 0
   we need to back off `BLOCK_SIZE` (try 128) or `WARP_ROWS*WARP_COLS` (try 4
   warps -> only 4-warp configs supported).
5. **Stage 3 (attention) is the slowest stage** because only the first 64 threads
   write the output (one per head_dim). M6 should rewrite phase 3 to use all 256
   threads (8 threads per d, 8-way reduction).

## Phase 2 (TRT-LLM integration, deferred)

When v1 lands sub-2 ms standalone, follow these steps to land into TRT-LLM:

1. Move `tinyllama_megakernel.cu` + headers under
   [cpp/tensorrt_llm/thop/](../../cpp/tensorrt_llm/thop/).
2. Wrap as a `TORCH_LIBRARY_FRAGMENT(trtllm, m)` op
   (template: [cublasScaledMM.cpp](../../cpp/tensorrt_llm/thop/cublasScaledMM.cpp) L325).
3. Register the fake op in [cpp_custom_ops.py](../../tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py).
4. Hook into [modeling_llama.py](../../tensorrt_llm/_torch/models/modeling_llama.py)
   `LlamaForCausalLM.forward` with a guarded fast-path (sm_89 + TinyLlama-1.1B
   shape + prefill + batch=1 + bf16 → call the megakernel op; else fall through).
5. **The hard part**: extend the megakernel to write into TRT-LLM's paged KV
   cache layout so decode can proceed seamlessly. This is the main reason
   Phase 2 is a separate project.
