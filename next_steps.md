# TRTLLM-gen MoE FP8 block-scale LoRA — next steps

Status of branch `feat/trtllmgen-moe-fp8-block-scale-lora`. The feature is
**structurally complete end-to-end in code** but **unbuilt / unvalidated** (no
GPU or C++ build in the authoring environment; testing was deferred by design).
This doc is the handoff for the build-node validation pass.

Scope: routed-expert MoE LoRA (`moe_h_to_4h` / `moe_gate` / `moe_4h_to_h`) on the
**TRTLLM-gen backend, FP8 block-scale base weights only**, via TRT-LLM's native
`torch.ops.trtllm` path (no new FlashInfer dependency). BF16 and FP4 are deferred.

## What landed (commit-by-commit, bottom-up)

1. `trtllm-gen batched GEMM: thread Mn (per-element) bias` — `KernelRunner.{h,cpp}`:
   `biasType` / `fusedBiasShuffleMode` options + `permutedIdxToBiasRowIdx`;
   non-LoRA config selection unchanged (Mn kernels excluded unless `biasType==Mn`).
2. `Add native BGMV MoE LoRA kernels + torch ops` — `cpp/tensorrt_llm/kernels/bgmvMoe/`
   (`moeBgmvKernels.{cuh,h,cu}`) + `thop/bgmvMoeOp.cpp` (`torch.ops.trtllm.bgmv_moe_{shrink,expand}`).
3. `Add routed-MoE LoRA delta builders` — `moe_lora_delta.py`
   (`bgmv_moe_gemm1/gemm2_lora_delta`, `fill_w_ptr`).
4. `shared marker/activity helper + allow trtllm-gen FP8 block-scale` — `moe_lora.py`;
   `validation.py` relaxed to allow `moe_backend='TRTLLM'` for FP8 block-scale.
5. `blockScaleMoe runner: fuse gemm1_lora_delta as Mn bias` — `blockScaleMoe/runner.{h,cu}`.
6. `fp8 block-scale MoE thop: gemm1_lora_delta path + fp8_block_scale_moe_lora op` —
   additive list-returning LoRA op (autotuner path untouched).
7. `TRTLLMGen MoE LoRA: marker/activity, scheduler threading, eager run_moe flow`.
8. `cuda-graph reservation hook + MoE guide note`.
9. `BGMV MoE LoRA: switch to TRT-LLM per-adapter pointer model (op extension)` —
   `w_ptr[slice, adapter]`; per-expert offset (compile-time `feat_in*feat_out`)
   added in-kernel; dropped `lora_stride`. Deliberate divergence from FlashInfer's ABI.
10. `wire eager + cuda-graph BGMV input extraction` — `_build_moe_lora_bgmv_inputs`
    (both modes) + `CudaGraphLoraParams.get_moe_slot_inputs_device`.

Data flow at runtime (FP8 block-scale, `TRTLLMGenFusedMoE.run_moe`):
`lora_params` → `_build_moe_lora_bgmv_inputs` (per-adapter `w_ptr`, per-token
`lora_ids`, uniform `rank`) → `bgmv_moe_gemm1_lora_delta` → `[T,k,2I]` bf16 delta
→ `torch.ops.trtllm.fp8_block_scale_moe_lora` (delta fused as `Mn` GEMM1 bias;
returns `[out, expanded_idx_to_permuted_idx, activation, activation_scale]`) →
dequant activation → `bgmv_moe_gemm2_lora_delta` → add FC2 delta to output.

## Known gaps

### Blocking (must resolve before anything runs)
- **trtllm-gen `Mn`-bias FP8 block-scale cubins are NOT in the bundled export.**
  `KernelMetaInfo.h` (`TLLM_GEN_COMMIT 71d2730e`) has 1323 `biasFp32M_` and **0**
  `biasFp32Mn`. The `Mn` runner (`run_moe_lora`) throws "No kernel found" until the
  export is bumped to a version with FP8 block-scale GEMM1 `Mn`-bias (bf16 bias,
  `Shuffle` fused-bias mode) kernels. Source: the merged `Mn` changes at
  `/home/scratch.bbuddharaju_gpu/trtllm-gen`.
- **Nothing has been compiled.** All C++ (batched-GEMM plumbing, blockScaleMoe
  runner, thop, BGMV kernels) was written without a build. Expect compile fixes.

### Needs GPU numerical validation (correct-by-construction, unverified)
- **BGMV kernel correctness** — ported from FlashInfer `csrc/bgmv_moe`; `vec_t`
  replaced by a self-contained `VecT` doing element-wise shared loads
  (correctness-first, not perf-tuned).
- **FC2 activation dequant** (`TRTLLMGenFusedMoE._dequant_activation`) — the FP8
  block-scale post-SwiGLU activation is fp8 + a `[inter/128, padded]` block scale;
  confirm the dequant matches the kernel's finalize.
- **Gate/up slice ordering** — `gemm1_lora_delta` is `concat(gate, up)` (slice 0 =
  `moe_h_to_4h`, slice 1 = `moe_gate`). Confirm this matches the trtllm-gen GEMM1
  output packing (which half is silu-gate vs linear-up); a swap silently produces
  wrong results.
- **`Mn` bias ↔ dequant-scale interaction** on the DeepSeek FP8 path — the delta is
  passed through directly (mirrors FlashInfer). Verify the bias is applied in the
  right (dequantized) space.
- **CUDA-graph replay with changing adapters** — graph-safe by construction (device
  assembly over `CudaGraphLoraParams` address-stable `d_b_ptrs`/`slot_ranks` +
  captured H2D of the refreshed pinned `token_to_slot_host`; uniform rank read from
  pinned host to avoid a capture-time device sync). Confirm on GPU.
- **`reserve_moe_lora_cuda_graph_workspace` is a no-op** — the delta builders
  (`moe_lora_delta.py`) and the `fp8_block_scale_moe_lora` thop allocate scratch
  per call (`torch.zeros`, workspace tensors, `gemm1_lora_delta`). Under torch CUDA
  graph capture these land in the graph memory pool (OK for fixed shapes), but this
  is unverified. If capture complains or replay corrupts, move those allocations
  into pre-reserved, address-stable buffers owned by the reservation hook.

### Scoped-out / follow-ups (deliberate)
- **Varying-rank MoE LoRA on trtllm-gen** — rejected (`_uniform_active_rank` raises).
  BGMV compiles one rank per call; use the Cutlass backend for varying ranks.
- **LoRA-path autotuning** — `fp8_block_scale_moe_lora` uses the default tactic.
- **Compiled BGMV dims** — only the `(rank, hidden/inter)` pairs in
  `moeBgmvKernels.cuh` (`TLLM_FOR_MOE_ALL_WIDE_NARROW`); other dims raise. Add
  model-specific dims as needed. rank ∈ {8,16,32,64}.
- **BF16 / FP4** MoE LoRA on trtllm-gen — deferred (no native BF16 runner; no `Mn`
  FP4 GEMM1 kernel), enforced in `validation.py`.

## Issues to look out for during validation
- **Routing consistency**: LoRA forces precomputed routing (scheduler sets
  `router_logits=None`) so the BGMV delta and the MoE kernel share top-k. Confirm
  `token_selected_experts`/`token_final_scales` are valid for the routing method in
  use (esp. DeepSeekV3 / grouped routing).
- **Scale is folded into weights**: `lora_manager` does `t_out *= scale`
  (alpha/rank) at load; the builders therefore use `scale=1.0`. Do NOT double-apply.
- **Token→seq expansion (eager)**: `_build_moe_lora_bgmv_inputs_eager` expands
  `lora_ids` from `host_request_types`/`prompt_lens_cpu` (context → prompt_len
  tokens, gen → 1; spec decode is relabeled upstream). Assert
  `lora_ids.numel() == num_tokens`; if it fires, the token layout assumption is off.
- **All three modules required**: the SwiGLU FP8 path needs gate+up+down LoRA;
  missing `moe_gate` raises.
- **Bf16 MoE input**: `_run_fp8_block_scale_lora` asserts bf16 activations
  (`x_sf is None`) so the FC1 delta is built pre-quant. Paths that pre-quantize x
  before the MoE would need the raw bf16 x threaded through.
- **TP/EP**: FC2 delta output width is `hidden_size`; validate under TP/EP sharding
  (`slot_start` / `expert_size_per_partition` are passed to the op).

## Validation plan (build node + GPU)
1. **Build** the C++ extension (with the bumped `Mn` cubins present).
2. **CPU plumbing** (no GPU):
   ```
   pytest tests/unittest/_torch/lora/test_moe_lora_validator.py \
          tests/unittest/_torch/lora/test_moe_lora_model_path.py \
          tests/unittest/_torch/lora/test_trtllmgen_moe_lora_build_inputs.py -q
   ```
3. **BGMV ops + delta builders** (GPU; independent of the cubin bump):
   ```
   pytest tests/unittest/_torch/lora/test_bgmv_moe.py \
          tests/unittest/_torch/lora/test_moe_lora_delta.py -q
   ```
4. **C++ unit tests** (if wired): `ctest -R "moeLora|bgmv|blockScaleMoe"`.
5. **Eager end-to-end** (GPU, requires `Mn` cubins):
   ```
   pytest tests/unittest/_torch/lora/ -k "trtllmgen and fp8" -q
   ```
6. **CUDA graph**: rerun (5) under cudagraph; verify replay with a changing adapter set.
7. Compare a small FP8 block-scale MoE + LoRA against a bf16 reference
   (`tensorrt_llm/_torch/peft/lora/moe_layout.py::reference_swiglu_moe_lora`).

## Suggested follow-up PRs (after validation)
- Add the trtllm-gen `Mn` cubin bump (separate prerequisite PR).
- End-to-end eager GPU test (Qwen-style FP8 MoE + reference) and a cudagraph test.
- Pre-reserve BGMV/delta scratch in `reserve_moe_lora_cuda_graph_workspace` if the
  per-call allocations are not graph-safe.
- LoRA-path autotuning for `fp8_block_scale_moe_lora`.
- Varying-rank support (per-rank grouping) if required on trtllm-gen.

## Key files
- Kernels: `cpp/tensorrt_llm/kernels/bgmvMoe/moeBgmvKernels.{cuh,h,cu}`,
  `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.{h,cpp}`,
  `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.{h,cu}`.
- Ops: `cpp/tensorrt_llm/thop/bgmvMoeOp.cpp`, `cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp`,
  `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py` (`fp8_block_scale_moe_lora`).
- Python: `tensorrt_llm/_torch/modules/fused_moe/{moe_lora.py,moe_lora_delta.py,
  fused_moe_trtllm_gen.py,moe_scheduler.py}`,
  `tensorrt_llm/_torch/peft/lora/{validation.py,cuda_graph_lora_params.py}`.
- Tests: `tests/unittest/_torch/lora/{test_bgmv_moe.py,test_moe_lora_delta.py,
  test_trtllmgen_moe_lora_build_inputs.py,test_moe_lora_validator.py}`.
- Docs: `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md`
  (Routed-expert MoE LoRA section).
