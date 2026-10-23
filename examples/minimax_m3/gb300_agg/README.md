<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M3 on GB300 (SM103): aggregated serving for agentic traffic

One aggregated TP2 worker per GB300 node serves MiniMax-M3 NVFP4 with an Eagle3 draft
(3 draft tokens) to multi-turn agentic traffic (AgentX).

| File | Role |
|---|---|
| `agg_agentx_c45.yaml` ... `agg_agentx_c20.yaml` | Worker config for 45, 35, 30, 25 and 20 concurrent conversations. They differ only in `max_batch_size` and the CUDA graph batch sizes. |
| `../rubin/setup_fa4.sh` (applies `3rdparty/patches/fa4_fp8_rescale0.patch`) | FA4 for the FP8 dense-layer prefill; prepend its directory to `PYTHONPATH` in the worker. |

## Setup

1. Build and install TensorRT-LLM from this branch, then `pip install -r requirements.txt`. That installs
   flash-attn-4 4.0.0b19, nvidia-cutlass-dsl 4.8.0 and quack-kernels 0.5.0, which the FA4 dense prefill needs.
2. Run `bash examples/minimax_m3/rubin/setup_fa4.sh <dir>` in the same environment. It copies the installed
   `flash_attn` package to `<dir>/flash_attn` and patches it; site-packages is left unchanged.
3. Set `speculative_model` in the worker config to the Eagle3 draft checkpoint.
4. Start one worker per node (no other job on the node):
   `trtllm-serve <MiniMax-M3 NVFP4 checkpoint> --config agg_agentx_c<N>.yaml`.

## Worker environment

Set these in the worker process:

| Variable | Value | Why |
|---|---|---|
| `PYTHONPATH` | `<dir>` (from `setup_fa4.sh`) first | The dense layers' FP8 prefill runs the patched FA4. Without it they keep fmha_sm100. |
| `TRTLLM_ENABLE_PDL` | `1` | Programmatic dependent launch. |
| `TRTLLM_TORCH_COMPILE_CONTEXT_ONLY` | `1` | |
| `TRTLLM_SERVER_DISABLE_GC`, `TRTLLM_WORKER_DISABLE_GC` | `1` | No Python GC pauses while serving. |
| `TLLM_NUMA_AWARE_WORKER_AFFINITY` | `1` | |
| `NCCL_GRAPH_MIXING_SUPPORT` | `0` | |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | |
| `TRTLLM_SERVE_ENABLE_MSGSPEC` | `1` | |
| `TRTLLM_SERVE_TOOL_PARSER` | `minimax_m3` | |

Some container images set `PYTHONPATH` in their own environment, which replaces a value exported from the
launcher (for example through `srun --export`). Prepend `<dir>` in the command the container runs, so it lands
in front of the image's entries.

For throughput measurements with a fixed speculative acceptance, also set
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1.78` (1.78 accepted draft tokens per step). Leave it unset for
accuracy runs.

The MiniMax-M3 optimizations need no variable; each is on by default on SM103 and has its own kill switch:

| Switch | Turns off |
|---|---|
| `TRTLLM_M3_BLACKWELL_RUBIN_OPTS=0` | Every Rubin optimization enabled on SM100/SM103 (FA4 dense prefill included). |
| `TRTLLM_MSA_DISABLE_DENSE_FA4=1` | FA4 dense prefill only. |
| `M3AX_STACK_A=0` | All of the following. |
| `M3AX_MSA_PREWARM_VARIANTS=0` | Startup compilation of the MSA fmha_sm100 kernel variants. Off, a variant first reached while serving compiles then (a step stall of up to ~70 s). |
| `M3AX_SPARSE_DECODE_PREWARM=0` | Startup compilation of the Triton sparse-decode specializations. |
| `M3_O8_SPARSE_TABLES=0` | The TP2 sparse-decode launch table. |
| `M3_L1_FMHA=0` | The full-attention decode split for long contexts. |

The measured configs ran with these defaults. Three Rubin optimizations are opt-in on SM100/SM103 and were off:
the TF32 CTX router (`TRTLLM_CTX_ROUTER_TF32_FIXED_ARCHS`), the tensor-core generation router
(`TRTLLM_M3_ROUTER_TC_GEN_ARCHS`) and the CTX CLC MXFP8 GEMM (`TRTLLM_CTX_CLC_GEMM_ARCHS`). The work-balanced
indexer proxy plan (`M3_O12A_PROXY_SPLIT=1`) is off by default as well.

## Startup checks

The server log shows, for each TP rank N:

- `[M3AX msa-prewarm] stack-a parts rank=N master=1 o8=1 o12a=0 prewarm=1 sd_prewarm=1`
- `[M3AX msa-prewarm] prewarm done rank=N ... failed=0 ... missing_after=0`, with 14 reachable variants at TP2
- `[M3AX msa-prewarm] sparse-decode prewarm done rank=N ... failed=0`
- `[M3AX l1-fmha] rank=N ... enabled=1 ... table=a18b0007 entries=95`
- `MiniMax-M3 dense prefill: FA4 (flash_attn.cute, ...)`. `FA4 unavailable` or
  `FA4 lacks the FP8 rescale_threshold 0.0 change` means the dense prefill fell back to fmha_sm100; check
  `PYTHONPATH`.

## Measured

AgentX (`semianalysis_cc_traces_weka_062126`), 60-minute windows, one GB300 node per point, forced acceptance
1.78. P90 interactivity is 1 / P90 per-request inter-token latency; throughput counts input and output tokens.

| Config | P90 interactivity (tok/s/user) | Total tok/s per GPU |
|---|---|---|
| `agg_agentx_c45.yaml` | 68.9 | 114,869 |
| `agg_agentx_c35.yaml` | 86.0 | 101,297 |
| `agg_agentx_c30.yaml` | 98.5 | 89,036 |
| `agg_agentx_c25.yaml` | 115.3 | 81,127 |
| `agg_agentx_c20.yaml` | 136.0 | 66,413 |
