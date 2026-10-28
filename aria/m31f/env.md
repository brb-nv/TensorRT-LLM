# ARIA env switches on this branch

Each switch is read once, at import or construction time. "Set by" lists the recipes (in [recipes/](recipes/)) whose
recipe sets the switch: S6 = all `c*-tp2-s6*-full`, S5 = `c48/c52-tp2-s5-full`, S4 = `c32/c64/c80/c96-tp4-s4-full`,
S3 = `c1/c4/c8/c16/c32-tp4-s3*-full` and the two-tray `c1/c2-tp8-full`, stock = `c1-tp4-base-full`, `c4-tp4-base-full-r2`.

## Switches added on this branch (over 705ea94d)

| Variable | Default | Effect | Where | Set by |
|---|---|---|---|---|
| `TRTLLM_ARIA_NO_MIX` | `0` | Never-mix: context chunks and decode rows do not share a step | `pyexecutor/scheduler/scheduler_v2.py` | S3, S4, S5, S6 (`1`) |
| `TRTLLM_ARIA_NO_MIX_DECODE_INTERVAL` | `0` | After a context step, this many decode-only steps before context is admitted again | same | S3, S4, S5, S6 (`1`) |
| `TRTLLM_ARIA_NO_MIX_MIN_GEN` | `1` | Never-mix splits a step only when at least this many generation requests are scheduled | same | none (default) |
| `TRTLLM_ARIA_FAST_SAMPLER` | `0` | Fused fast sampler for the DSpark rejection sampling path | `pyexecutor/sampler/ops/aria_fast_sampler.py`, `flashinfer.py` | S5, S6 (`1`) |
| `TRTLLM_ARIA_DECODE_FUSION` | `0` | MoE top-k finalize + shared-expert add as one kernel on decode (bitwise equal) | `models/modeling_minimaxm3.py` | S5, S6 (`1`) |
| `TRTLLM_ARIA_DECODE_FUSION_MAX_ROWS` | `256` | Row cap for the above | same | none |
| `TRTLLM_ARIA_PREFILL` | `0` | cuBLASLt MXFP8 GEMM for large eager (prefill) row counts | `modules/linear.py` | S5, S6 (`1`) |
| `TRTLLM_ARIA_PREFILL_GEMM_MIN_M` | `1024` | Row threshold for the above | same | none |
| `TRTLLM_ARIA_PREFILL2` | `0` | Smaller k2q work items in the MSA NVFP4 sparse prefill | `attention/backends/fmha/msa_prefill.py` | S5, S6 (`1`) |
| `TRTLLM_ARIA_PREFILL2_MSA_TARGET_Q` | `512` | Work-item query cap for the above | same | none |
| `TRTLLM_ARIA_AR_NVFP4` | `0` | H39: the routed experts' NVFP4 input (linear SF) comes from the pre-MoE AllReduce+RMSNorm epilogue | `models/modeling_minimaxm3.py`, C++ `allreduceOp.cpp` | S6 (`1`) |
| `TRTLLM_ARIA_AR_NVFP4_MAX_ROWS` | `512` | Row cap for the above | same | none |
| `TRTLLM_ARIA_MOE_AR` | `0` | H39: MoE finalize + shared add + AR + residual + RMSNorm (+ MXFP8) in one kernel | `models/modeling_minimaxm3.py`, C++ `moeAllReduceFusionKernels.cu` | S6 (`1`) |
| `TRTLLM_ARIA_MOE_AR_MAX_ROWS` | `128` | Row cap (also bounded by the AR workspace) | same | none |
| `TRTLLM_ARIA_LOWM_GEMM` | `0` | H36: fixed swap-AB CuTeDSL tactic for 33-256-row MXFP8 decode GEMMs | `custom_ops/torch_custom_ops.py` | S6 (`1`) |
| `TRTLLM_KV_LAYERWISE_ONBOARD` | **on** (anything but `0`) | wheel-1: layer-wise host-KV onboarding overlapping the forward | `pyexecutor/kv_cache/kv_layerwise_onboard.py`, C++ KV v2 | `0`: S4, S5, S6, all `repro-` of S3/stock (TP4 and TP8); `1`: `c44/c52-tp2-s6lw-full` |
| `TRTLLM_KV_ONBOARD_CTAS` | unset (Little's-Law grid, floor of 32 CTAs on C2C) | wheel-1: onboard copy grid override | C++ `batchedPageCopy.cu` | none of the original recipes; `12` in `repro-` of S3/stock (TP4 and TP8) |
| `TRTLLM_M3_ASYNC_OFFLOAD` | `0` | wheel-1: async offload headroom per v2 pool | `pyexecutor/py_executor.py` | none |
| `TRTLLM_M3_ASYNC_OFFLOAD_TOKENS` | max_num_tokens | Headroom size for the above | same | none |
| `TRTLLM_KV_ONBOARD_PREFETCH` | `0` | wheel-1: prefetch onboarding for up to n not-yet-admitted requests | same | none |
| `TRTLLM_KV_ONBOARD_PREFETCH_EARLY_FETCH` | `1` | Activate arrivals right after launch (only with PREFETCH) | same | none |
| `TRTLLM_KV_ONBOARD_PREFETCH_DEFER_TOKENS` | unset | Hold big-onboard arrivals out one iteration (only with PREFETCH) | same | none |

The following wheel-1 change has **no switch**: the MXFP8 next-layer QKV input from the boundary AllReduce
(`cae9c789c`). It applies whenever the QKV projection is a CUTLASS MXFP8 linear and the AllReduce strategy is
neither UB nor MNNVL. It is bitwise identical in activations.

## Pre-existing switches the recipes use

| Variable | Default | Effect | Set by |
|---|---|---|---|
| `TRTLLM_M3_MSA_V1_DECODE` | `0` | MSA-v1 Q8KV4 sparse decode (`kernels/msa_v1_decode.py`, needs MSA) | S4, S5, S6 (`1`) |
| `MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT` | set to `quad4` by TRT-LLM when MSA-v1 decode is on | MSA's V-scale layout (the msa-quad4 patch) | S4, S5, S6 (`quad4`) |
| `TORCH_EXTENSIONS_DIR` | `~/.cache/torch_extensions` | MSA JIT cache | S4, S5, S6 (`/tmp/aria-msa-jit`) |
| `TRTLLM_ENABLE_PDL` | `1` in some paths | Programmatic dependent launch | all (`1`) |
| `TRTLLM_SERVER_DISABLE_GC`, `TRTLLM_WORKER_DISABLE_GC` | `0` | Disable Python GC | all (`1`) |
| `NCCL_GRAPH_MIXING_SUPPORT` | NCCL default | - | all (`0`) |

Config fields that act as switches: `sparse_attention_config.indexer_kv_dtype: nvfp4` (S4, S5, S6; S3/stock use
`fp8`) and `kv_cache_config.host_cache_size` (178 GB per rank at TP2, 170 GB at TP4 and TP8).

## Per-recipe matrix (original recipes)

| Recipe | NO_MIX | NM_INT | FAST_SAMP | PREFILL | PREFILL2 | DEC_FUS | AR_NVFP4 | MOE_AR | LOWM | KV_LW | MSA_V1 | V_SCALE | TORCH_EXT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c1-tp8-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c2-tp8-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c1-tp4-s3-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c1-tp4-base-full` | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `c4-tp4-s3-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c4-tp4-base-full-r2` | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `c8-tp4-s3-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c16-tp4-s3-full-r2` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c20-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c24-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c28-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c32-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c32-tp4-s3-full` | 1 | 1 | - | - | - | - | - | - | - | - | - | - | - |
| `c32-tp4-s4-full` | 1 | 1 | - | - | - | - | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c36-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c40-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c44-tp2-s6lw-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | quad4 | /tmp/aria-msa-jit |
| `c44-tp2-s6-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c48-tp2-s5-full` | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c52-tp2-s6-full-r2` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c52-tp2-s6lw-full` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | quad4 | /tmp/aria-msa-jit |
| `c52-tp2-s5-full` | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c64-tp4-s4-full` | 1 | 1 | - | - | - | - | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c80-tp4-s4-full` | 1 | 1 | - | - | - | - | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |
| `c96-tp4-s4-full` | 1 | 1 | - | - | - | - | - | - | - | 0 | 1 | quad4 | /tmp/aria-msa-jit |

Columns: NO_MIX/NM_INT = `TRTLLM_ARIA_NO_MIX`/`_DECODE_INTERVAL`, FAST_SAMP, PREFILL, PREFILL2, DEC_FUS = the
`TRTLLM_ARIA_*` switches above, KV_LW = `TRTLLM_KV_LAYERWISE_ONBOARD`, MSA_V1 = `TRTLLM_M3_MSA_V1_DECODE`, V_SCALE =
`MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT`, TORCH_EXT = `TORCH_EXTENSIONS_DIR`. "-" means unset (the default).
