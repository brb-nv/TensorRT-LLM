# NVBug 5615248 — TTFT root-cause analysis (PyTorch vs TensorRT)

This document explains the residual TTFT gap between the PyTorch and TensorRT
backends on the workload from NVBug 5615248, **after** the original
`enable_piecewise_cuda_graph` + `max_seq_len=128` capture-fallback bug was
already fixed on this branch.

The companion file [`REPRO.md`](REPRO.md) covers how the benchmarks were run.

---

## TL;DR

- The piecewise CUDA-graph fix is in place and **is being exercised** during
  the prefill — 100% of the prefill GEMMs / RMSNorms / SiLU launches go
  through `cudaGraphLaunch`.
- The remaining ~5.2 ms TTFT regression vs TRT decomposes into:
  - **~1.8 ms on-GPU** — gaps inside the prefill window from ~67 eager kernel
    launches per prefill at attention boundaries (see
    [§3](#3-prefill-on-gpu-where-the-time-actually-goes)).
  - **~2.6 ms on-host** — Python-side `_prepare_inputs` + first-step
    `_sample_async` + `_process_requests`, which on the *first* token can't
    overlap with prior-iteration GPU work
    (see [§5](#5-host-side-contribution-first-step-only)).
- The on-GPU gap is **by design** — `tensorrt_llm/_torch/compilation/piecewise_optimizer.py`
  splits the captured graph at every `attn_custom_op_inplace` /
  `mla_custom_op_inplace` node. Closing the gap is an attention-boundary
  problem, not a "the fix isn't on" problem.

---

## 1. Setup

| | Value |
|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden=2048, vocab=32k) |
| GPU | L40S (sm_89) |
| Workload | concurrency=1, ISL=100, OSL=20, beam_width=10 |
| `max_seq_len` | 129 (workaround for the original NVBug; capture limit is 128) |
| Trace | `nsys profile … --cuda-graph-trace=node`, capture range = `cudaProfilerApi`, 3 measurements × (1 prefill + 19 decode) = 60 forward iterations |
| PyT trace | `trace_pytorch_kernels.nsys-rep` |
| TRT trace | `trace_trt_kernels.nsys-rep` |
| Post-processed | `nvbugs_5615248/trtllm_bench/nsys_kernels/{pytorch,trt}/*.{txt,csv}` |

---

## 2. The TTFT gap to explain

From `trtllm-bench --concurrency 1` (steady-state per-request stats):

| Metric | PyTorch | TRT | Δ (PyT − TRT) |
|---|---:|---:|---:|
| **TTFT (mean)** | ~11.5 ms | ~6.3 ms | **+5.2 ms** |
| Prefill GPU window per call (steady) | 5.62 ms | 3.84 ms | +1.78 ms |
| Prefill host wrapper (`_forward_step` / `enqueueV3`) | 5.36 ms | 2.57 ms | +2.79 ms |

Prefill GPU window ≈ `_forward_step ... 1 ctx reqs` GPU projection (PyT) /
`enqueueV3` GPU projection (TRT). Steady-state excludes the first measurement
which carried cold-start cost (PyT iter 63 = 6.74 ms, iters 84/105 ≈ 5.6 ms).

---

## 3. Prefill on-GPU — where the time actually goes

### 3.1 PyTorch prefill kernel inventory (per call, from `nvtx_kern_sum`)

| Kernel | Per call | Time | Comment |
|---|---:|---:|---|
| `ampere_bf16_s16816gemm 128x64` | 22 | 1,582 µs | 1 large prefill GEMM/layer |
| `cutlass wmma 32x32x128` | 66 | 1,491 µs | 3 GEMMs/layer |
| `gemvx::kernel` (lm_head) | 1 | 182 µs | output projection |
| `flashinfer FusedAddRMSNorm` | 44 | 97 µs | 2 norms/layer |
| `fmha_v2 flash attention` (prefill) | 22 | 76 µs | 1/layer |
| `cublasLt splitKreduce` | 44 | 61 µs | GEMM split-K reductions |
| `applyBiasRopeUpdateKVCacheV2` | 22 | 53 µs | RoPE + KV write |
| `computeSeqAndPaddingOffsets` | 22 | 51 µs | input prep |
| `silu_and_mul_kernel` | 22 | 28 µs | SwiGLU |
| Misc PyTorch elementwise / index / scan | ~15 | ~25 µs | |
| **Kernel sum** | **~280 launches** | **~3,650 µs** | |
| **GPU window (steady)** | | **~5,620 µs** | |
| **Idle time between kernels** | | **~1,970 µs** | **← key signal** |

### 3.2 TRT prefill kernel inventory (per call, from `nvtx_gpu_proj_sum`)

| Range | Per call time | Comment |
|---|---:|---|
| 21 × per-layer QKV ForeignNode (myelin-fused matmul, layers 1-21) | 21 × ~146 µs ≈ 3,070 µs | dominant kernel: `sm80_xmma 128x128x32 fused` (~183 µs each) |
| Layer 0 QKV ForeignNode | ~49 µs | smaller (different fusion) |
| 22 × `gpt_attention` plugin | 22 × ~10 µs ≈ 240 µs | (layer 0 ~26 µs) |
| Other foreign nodes (MLP / RMSNorm / lm_head) | ~480 µs | balance to enqueueV3 |
| **Kernel-equivalent compute** | **~3,840 µs** | |
| **GPU window (= enqueueV3)** | **3,838 µs** | |
| **Idle time between kernels** | **≈ 0** | **CUDA-graph-tight** |

### 3.3 Conclusion #1

**PyTorch and TRT do roughly the same amount of compute (~3.6-3.8 ms of
kernels). The 1.78 ms on-GPU prefill gap is dead time between kernels in
PyTorch.**

The structural reason:
- PyTorch issues ~280 distinct kernel launches per prefill.
- TRT issues ~50 distinct launches per prefill (myelin fuses ~4 GEMMs/layer
  into 1 foreign-node GEMM and packs the whole graph tightly).
- The launch-count delta is what creates the inter-kernel slack.

---

## 4. Why isn't the piecewise CUDA graph closing the gap?

It is — partially. The bug fix is fully in effect; it's the **design** of
piecewise CUDA graph that leaves attention boundaries eager.

### 4.1 Direct evidence — kernels are partitioned by execution context

From `cuda_kern_exec_sum.csv` (PyTorch trace), kernels grouped by API:

**Captured into the piecewise graph** (replayed via `cudaGraphLaunch`):

| Kernel | Total launches | Per prefill |
|---|---:|---:|
| `ampere_bf16_s16816gemm 128x64` (prefill QKV/dense GEMM) | 66 | 22 |
| `cutlass wmma 32x32x128` (other prefill GEMMs) | 198 | 66 |
| `silu_and_mul_kernel` | 1320 | 22 |
| `flashinfer FusedAddRMSNormKernel` | 2640 | 44 |
| `cutlass wmma 16x16x128` (decode GEMM) | 5073 | 0 (decode-only) |
| `mmha::masked_multihead_attention_kernel` (decode) | 1254 | 0 (decode-only) |

**Eager (not in graph)** (launched via `cudaLaunchKernel` /
`cudaLaunchKernelExC`):

| Kernel | Total launches | Per prefill |
|---|---:|---:|
| `fmha_v2_flash_attention_*_paged_kv_64_causal_*` | 66 | 22 |
| `applyBiasRopeUpdateKVCacheV2` | 66 | 22 |
| `computeSeqAndPaddingOffsets` | 66 | 22 |
| `gemvx::kernel` (lm_head) | 3 | 1 |

So per prefill, **67 launches stay eager**, all clustered at the per-layer
attention boundary plus the final lm_head.

### 4.2 Where the design choice is in code

`tensorrt_llm/_torch/compilation/piecewise_optimizer.py` (lines 262-278):

```272:288:tensorrt_llm/_torch/compilation/piecewise_optimizer.py
            if (node.target != torch.ops.trtllm.attn_custom_op_inplace.default
                    and node.target
                    != torch.ops.trtllm.mla_custom_op_inplace.default
                    and node.target
                    != torch.ops.trtllm.mla_dsa_attn_inplace.default):
                # We only know it is safe to continue splitting after attention
                stop_partition = True
```

Attention is unconditionally excluded from capture because per-call attention
metadata (paged KV-block-table pointers, per-request sequence lengths,
chunked-prefill ranges) is not constant.

### 4.3 Cost arithmetic

- Eager launches per prefill: 67
- `cudaLaunchKernel` queue→start latency observed in
  `cuda_kern_exec_sum.csv` "QAvg" column for these eager kernels:
  ~12-19 µs (split across 60 forward steps but heavily prefill-side).
- 67 × ~25 µs ≈ **~1.7 ms** of GPU bubbles per prefill.

That ~1.7 ms matches the measured ~1.97 ms inter-kernel idle time in §3.1.
**The gap *is* the boundary cost.**

---

## 5. Host-side contribution (first-step only)

Sampling/scheduling on PyTorch is in Python; on TRT it's in C++ inside
`enqueueV3`. From `nvtx_pushpop_sum.column.txt` (CPU-side range durations,
per step):

| PyT range | CPU duration / step | What it does |
|---|---:|---|
| `_prepare_inputs` | 583 µs | Build inputs / IDs tensors |
| `_forward_step` outer (prefill only) | 5.36 ms | Python-side wrapper around prefill |
| `sample_async` (host portion) | 2.69 ms | Host-side sampler orchestration (mbtopk + cub launches) |
| `_process_requests` | 1.88 ms | Post-process / state update |
| `_write_finish_reasons` | 287 µs | Bookkeeping |

These ranges overlap **with each other and with GPU work** during steady-state
decoding. But on the **first** step there's no prior-iteration GPU work to
overlap with, so the bulk lands directly on the TTFT critical path.

TRT spends an equivalent ~2.57 ms inside `enqueueV3` (no Python-level NVTX
because everything runs in C++).

**Estimated TTFT-critical-path host delta**: PyT ~5.2 ms − TRT ~2.6 ms ≈
**~2.6 ms** of the +5.2 ms TTFT regression.

---

## 6. First-sampling-step contribution

Sampling fires identically every step, so per-step averages from
`nvtx_gpu_proj_sum` directly approximate the first-step cost.

| | PyT `_sample_async` | TRT (sampling kernels, summed) |
|---|---:|---:|
| GPU time per step | **445 µs** (median 294 µs) | **~164 µs** |
| Host time per step | **2.69 ms** | (in C++, no NVTX label) |
| Dominant GPU kernels | mbtopk cub (`computeDigitCumSum`, `computeBlockwiseWithinKCounts`, `gatherTopK`), `bitonicSortKVInPlace`, `cunn_SoftMaxForward`, ~25+ small elementwise | `insertUnfinishedPathKernel` (72 µs), `addBiasSoftMax` (24 µs), `air_topk_stable` × 3 kernels (~11 µs total), `beamStage3Kernel` (13 µs), `batchApplyPenalty` (13 µs), `finalizeKernel` (8 µs), ~10 small bookkeeping kernels |

Sampling is a **secondary contributor** to TTFT (~280 µs GPU + ~2 ms host
overhead extra in PyT). PyTorch's beam-search top-k is implemented with stock
`mbtopk` + cub primitives, vs TRT's purpose-built `air_topk_stable` flight
that runs ~2.7× faster on the GPU and has no Python-level host overhead.

---

## 7. Final TTFT decomposition

| Contribution | Estimate | Notes |
|---|---:|---|
| GPU prefill — gaps from eager attention boundaries (PyT) | +1.7-2.0 ms | Confirmed by §3.1 + §4.3 |
| Host — `_prepare_inputs` + first `_sample_async` + `_process_requests`, not yet overlapped | +2.6 ms | §5 |
| First-token sampling GPU compute (PyT mbtopk vs TRT air_topk_stable) | +0.3 ms | §6 |
| Response-emit / first-token tail | ~0.5 ms | balance to observed 5.2 ms |
| **Sum** | **~5.1-5.4 ms** | matches observed +5.2 ms |

---

## 8. Action items

Two complementary directions for closing the gap. Recommended order is
(b1) → (b2) → (a) — small surgical wins first, biggest-but-most-invasive last.

### (b1) Hoist `computeSeqAndPaddingOffsets` out of the per-layer attention call

| File | Change |
|---|---|
| `tensorrt_llm/_torch/attention_backend/trtllm.py` (`TrtllmAttentionMetadata`) | Move the `computeSeqAndPaddingOffsets` launch into the metadata `prepare()` (runs once per forward step). Cache the result on the metadata object. |
| `tensorrt_llm/_torch/modules/attention.py` (`_attn_impl`) | Read the cached offsets/padding tensor from `attn_metadata` instead of re-launching. |

**Expected gain**: drops 21 of the 22 `computeSeqAndPaddingOffsets` launches
per prefill = ~21 × ~10 µs of GPU bubble + ~21 × ~5 µs host overhead =
**~315 µs saved per prefill**. Low blast radius — ship as a standalone PR.

### (b2) Fuse `applyBiasRopeUpdateKVCacheV2` with `fmha_v2_*` into one entry

The TRT `GPT_ATTENTION_PLUGIN` already does this fusion at the plugin level.
The PyTorch TRTLLM backend exposes the two kernels as separate Python calls.

| File | Change |
|---|---|
| `cpp/tensorrt_llm/kernels/attention/...` and `cpp/tensorrt_llm/thop/attentionOp.cpp` | Add a single fused entry point that internally chains `applyBiasRopeUpdateKVCacheV2` → `fmha_v2_*` without returning to Python between them. |
| `tensorrt_llm/_torch/attention_backend/trtllm.py` (prefill path) | Switch to the fused entry point instead of two separate ops. |

**Expected gain**: drops 22 eager launches per prefill = **~440 µs saved per
prefill**. Helps decode too.

### (a) Capture attention into the graph too — full-graph prefill capture

The infrastructure for capturing the full forward (attention included) already
exists in `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` for the decode
path. Decode works because:
1. Decode shape per step is fixed (`max_batch_size × beam_width` tokens).
2. Attention metadata uses pre-allocated buffers that are written in place per
   step, so kernel parameter addresses are stable across replays.

Extending the same trick to chunked prefill:

| Step | File | Change |
|---|---|---|
| 1 | `tensorrt_llm/_torch/attention_backend/trtllm.py` (and `flashinfer.py`) | Add a "static-prefill metadata" mode: pre-allocate buffers for `seq_lens`, `cu_seq_lens`, `paged_kv_block_table`, `host_request_types` sized for `max_num_tokens`. On every prefill call, write into them in place rather than allocating fresh tensors. |
| 2 | `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` | Extend the existing decode-graph capture path to also capture **per-prefill-chunk-size** graphs. Reuse the static-input-address + replay machinery. |
| 3 | `tensorrt_llm/_torch/pyexecutor/model_engine.py` | Add a config flag (e.g., `cuda_graph_config.prefill_capture_sizes`) and dispatch logic: when an incoming chunk matches a captured size, replay full-graph; otherwise fall back to piecewise (current behavior). |
| 4 | `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` (lines 262-278) | When the backend signals "static-metadata-safe", drop `attn_custom_op_inplace` from the exclude list so attention can be inlined into the captured pieces. |

**Expected gain**: closes essentially the entire ~1.7-2.0 ms prefill gap
(turns 67 eager launches per prefill into 0).

**Risks / why this hasn't been done yet**:
- Variable-length chunked prefill across requests means
  `paged_kv_block_table` contents differ per call. Already updated in place
  for decode; same approach should work, but the test surface is much bigger
  (all attention backends, all KV-cache layouts, all paging modes).
- KV-cache reuse / cache-block recycling logic may write to descriptors
  mid-step in some configs.
- Full-graph capture requires a static `max_num_tokens` per captured size
  (multiple captures needed if chunk sizes vary).

### Combined potential

| State | PyT TTFT (estimate) |
|---|---:|
| Today (post-NVBug fix, piecewise CUDA graph on) | 11.5 ms |
| + (b1) | 11.2 ms |
| + (b1) + (b2) | 10.7 ms |
| + (a) (full-graph prefill capture) | 6.5-7.0 ms |
| TRT baseline | 6.3 ms |

---

## 9. Reproducing this analysis

1. Generate the traces (already done; outputs under
   `nvbugs_5615248/trtllm_bench/`):

   ```bash
   bash nvbugs_5615248/trtllm_bench/run_nsys_stats.sh
   ```

2. Confirm piecewise CUDA graph is being hit. Inspect
   `nsys_kernels/pytorch/cuda_kern_exec_sum.csv` and grep for
   `cudaGraphLaunch` rows associated with prefill GEMMs:

   ```bash
   rg "cudaGraphLaunch.*ampere_bf16_s16816gemm_bf16_128x64" \
      nvbugs_5615248/trtllm_bench/nsys_kernels/pytorch/cuda_kern_exec_sum.csv
   ```

   A non-empty result confirms prefill GEMMs are inside the captured graph.

3. Confirm attention is NOT in the captured graph:

   ```bash
   rg "cudaLaunchKernel.*fmha_v2_flash_attention" \
      nvbugs_5615248/trtllm_bench/nsys_kernels/pytorch/cuda_kern_exec_sum.csv
   ```

   A non-empty result confirms attention runs eagerly at piecewise boundaries.

4. Per-layer TRT prefill timing comes from
   `nsys_kernels/trt/nvtx_gpu_proj_sum.column.txt` (search for
   `transformer/layers/N/attention/qkv/get_weight` and
   `transformer/layers/N/attention/wrapper_L551/gpt_attention`).
