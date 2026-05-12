# Results — TinyLlama-1.1B Prefill Megakernel on L40S (NVBug 5615248)

> M7 deliverable. Template — fill in once you run on hardware. The plan calls
> for promoting one row from this file into the top of
> [POST_V5_DIVERGENCE.md](../trtllm_bench/POST_V5_DIVERGENCE.md) alongside the
> PyT-v5 and TRT rows once you have a clean number.

## Reproducer

```bash
cd nvbugs_5615248/megakernel
sudo nvidia-smi -lgc 2520,2520       # lock L40S clocks
python tests/bench.py --backend all --iters 1000 --warmup 100
```

Three TTFT baselines are gathered separately from the existing `trtllm-bench`
runs on this branch:

```bash
bash nvbugs_5615248/trtllm_bench/run_multirun_pytorch.sh \
     nvbugs_5615248/trtllm_bench/optimized_v5_pyt_fresh
bash nvbugs_5615248/trtllm_bench/run_multirun_trt.sh \
     nvbugs_5615248/trtllm_bench/optimized_v5_trt_fresh
```

## Headline (target form)

Workload: TinyLlama-1.1B-Chat-v1.0, batch=1, seqlen=128 (ISL=100 padded), BF16,
single L40S, locked at 2520 MHz.

| Method                              | mean (ms) | p50 (ms) | p99 (ms) | vs TRT |
|-------------------------------------|----------:|---------:|---------:|-------:|
| HF eager                            |    TODO   |   TODO   |   TODO   |  TODO  |
| HF `torch.compile(reduce-overhead)` |    TODO   |   TODO   |   TODO   |  TODO  |
| TRT engine `enqueueV3`              |    2.31   |     —    |     —    |  1.00× |
| PyT-backend `_forward_step` (v5)    |    5.91   |     —    |     —    |  2.56× |
| **lucebox megakernel v1**           |   **TODO**|   TODO   |   TODO   | **TODO** |

(The 2.31 ms TRT and 5.91 ms PyT rows are copied from
[POST_V5_DIVERGENCE.md](../trtllm_bench/POST_V5_DIVERGENCE.md), measured at
beam=10 ISL=100 OSL=20 — the per-prefill-iter cost rows. They are the
right comparison point for our standalone prefill bench.)

## M6 tile sweep summary (fill in after running)

| BM  | BN  | BK | NSTAGES | WARP_ROWS x WARP_COLS | mean (ms) | notes |
|----:|----:|---:|--------:|----------------------:|----------:|-------|
| 128 | 128 | 32 |       3 |                   4x2 |    TODO   | plan default |
| 128 | 128 | 32 |       2 |                   4x2 |    TODO   | |
| 128 | 128 | 64 |       3 |                   4x2 |    TODO   | bigger K-tile |
| 128 | 128 | 16 |       3 |                   4x2 |    TODO   | smaller K-tile |
| 128 | 128 | 32 |       3 |                   2x4 |    TODO   | transposed warp grid |
| 128 |  64 | 32 |       3 |                   4x2 |    TODO   | smaller N-tile |
|  64 | 128 | 32 |       3 |                   2x4 |    TODO   | smaller M-tile, 2 m-blocks |

## Per-stage profile (ncu, fill in after profiling)

Use `nsys nvtx` to identify the wall-clock breakdown, then `ncu --kernel-name
tinyllama_megakernel` on the dominant region. Expected:

| Stage                                    | per-layer (us) | x 22 layers (ms) | % of total |
|------------------------------------------|---------------:|-----------------:|-----------:|
| 1a pre-attn RMSNorm                      |        TODO    |        TODO      |    TODO    |
| 1b QKV GEMM (K=2048 N=2560)              |        TODO    |        TODO      |    TODO    |
| 2  RoPE                                  |        TODO    |        TODO      |    TODO    |
| 3  causal attention (M=128 keys)         |        TODO    |        TODO      |    TODO    |
| 4  O-proj GEMM (K=2048 N=2048) + residual|        TODO    |        TODO      |    TODO    |
| 5a post-attn RMSNorm                     |        TODO    |        TODO      |    TODO    |
| 5b gate_up GEMM (K=2048 N=11264)         |        TODO    |        TODO      |    TODO    |
| 6  SiLU * up                             |        TODO    |        TODO      |    TODO    |
| 7  down GEMM (K=5632 N=2048) + residual  |        TODO    |        TODO      |    TODO    |
| grid.sync overhead (combined)            |          —     |        TODO      |    TODO    |
| **Total**                                |                |     **TODO**     |   100 %    |

## Go / no-go decision for Phase 2

The plan says:

> Anything > 2.31 ms means we've lost to TRT and the project should pivot to
> per-layer fusion.

State of v1: **TODO** (fill in `result < 2.31 ms` → promote to Phase 2, or
`result > 2.31 ms` → write retro and pivot).

If we're in the win region, the Phase-2 work to land into TRT-LLM is:
1. KV-cache write into TRT-LLM's paged layout from inside the megakernel
2. thop op + fake-op registration + `LlamaForCausalLM.forward` fast-path
3. CI / numerical regression coverage in `tests/unittest/_torch/`

Per the plan, that's another ~2-3 weeks of focused work.
