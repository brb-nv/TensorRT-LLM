# MiniMax-M3 (MSA) op-fusion opportunities, inspired by DeepSeek-V3

## Implementation status

| # | Optimization | Status | Notes |
|---|--------------|--------|-------|
| 7 | Hoist per-step `fmha_sm100` plans (dedup prefill builds) | **Landed** | Host-only, no numeric change. Commit "MSA: hoist per-step fmha_sm100 plans". |
| 4 | Fuse `index_q_proj` + `index_k_proj` into one GEMM | **Landed** | Numerically identical; load-time row-wise weight concat. Commit "fuse MiniMax-M3 index_q/k projections". |
| 1 | Gemma-norm weight fold → fused norm+quant | Deferred | Fold alone is pure churn; the payoff needs the fused-quant Linear wiring (input_scale attach) which is GPU-only to validate. Do with #2. |
| 2 | AllReduce + residual + RMSNorm(+quant) fusion | Deferred | Multi-GPU (TP>1) only; the quant variant is blocked by gemma norm (#1). Restructures the decoder forward like DSV3 `forward_MoE`/`forward_mlp`. |
| 3 | Fused GEMM + SwiGLU-OAI (dense MLP + shared expert) | Deferred | `silu_and_mul` has no `alpha` gain / `up+1` offset, so this needs a new/extended CUDA/Triton kernel + GPU validation. |
| 5 | Drop redundant post-split `contiguous()` | Deferred | Whether `fmha_sm100` tolerates a strided Q is unverifiable without a GPU; low value, correctness risk. |

Deferred items all require either a new kernel, multi-GPU, or GPU-only
correctness validation, so they are left unimplemented rather than landed blind.
The two landed items are self-contained and numerically safe (see the test plan
at the end).



Scope: op-fusion improvements for the MiniMax-M3 sparse-attention (MSA) codepath
running the mixed-precision "FP4" checkpoint with an FP8 KV cache. Excludes the
fused QK-norm-RoPE kernel (already being implemented separately).

## Checkpoint ground truth (don't assume datatypes)

From `MiniMax-M3-NVFP4/config.json` → `quant_algo: MIXED_PRECISION`,
`kv_cache_quant_algo: null`:

- **MoE routed experts → NVFP4** (21,888 quantized tensors).
- **Attention block + dense/shared MLP → MXFP8**: `q_proj/k_proj/v_proj/o_proj`,
  `index_q_proj`/`index_k_proj`, dense MLP, shared-expert `down_proj`.
- Router gates are in `exclude_modules` (bf16/fp32).
- Attention activations stay **bf16** (model forces `torch_dtype=bf16` unless the
  KV cache itself is fp8/fp4 — see `MiniMaxM3Model.__init__`).
- `num_mtp_modules: 1` (MTP head present; out of scope for this doc).

Geometry: 60 layers (0–2 dense, 3–59 sparse+MoE), hidden 6144, 64 q-heads,
4 kv-heads (GQA group 16), head_dim 128, partial RoPE rotary_dim 64,
128 experts top-4 + 1 shared expert, sparse index_dim 128 / 4 index heads /
topk 16 blocks / block_size 128.

---

## Root blocker: Gemma norms disable the fused norm→quant epilogues

M3 builds **every** RMSNorm with `use_gemma=True` (input_layernorm,
post_attention_layernorm, q/k_norm, index norms, final norm — see
`MiniMaxM3DecoderLayer.__init__` and `MiniMaxM3Attention.__init__`). But the
fused add+RMSNorm+quantize path is explicitly gated off for gemma norms:

```python
# tensorrt_llm/_torch/modules/rms_norm.py (lines ~139-140)
nvfp4_scale = self.nvfp4_scale if self.is_nvfp4 else None
if nvfp4_scale is not None and not self.use_gemma:
    return self._fused_nvfp4_quant(...)
```

Consequence: none of `fused_add_rmsnorm_fp4_quantize` /
`fused_rmsnorm_fp4_quantize` / warp-specialized `fused_add_rms_norm_quant` can
ever fire for M3. Every layer boundary runs plain gemma-RMSNorm → **separate**
activation-quantize kernel → GEMM. DeepSeek-V3 folds all three into one kernel.

**Fix (highest-leverage enabler):** fold Gemma's `(1 + weight)` into the stored
norm weight at load time so the runtime norm is a plain RMSNorm
(`use_gemma=False`). Numerically identical, load-time only, and it unblocks the
entire family of fused norm+quant and AllReduce+norm+quant kernels below.

---

## Concrete fusions, ordered by impact

### 1. AllReduce + residual-add + RMSNorm (+ quant) at every layer boundary

M3 today does an unfused `AllReduce(result)` in `MiniMaxM3MoE.forward`, then a
separate `post_attention_layernorm(hidden_states, residual)` in the decoder
layer, then the next `input_layernorm` — three passes over the hidden state per
boundary. It passes `final_all_reduce_params=None`, so no fusion is possible.

DeepSeek-V3 collapses each boundary into one op:

```python
# tensorrt_llm/_torch/models/modeling_deepseekv3.py (forward_MoE, ~1514)
hidden_states, residual = self.allreduce(
    hidden_states,
    all_reduce_params=AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=self.post_attention_layernorm.weight,
        eps=self.post_attention_layernorm.variance_epsilon,
    ))
```

On the NVFP4 path it uses `RESIDUAL_RMS_NORM_QUANT_NVFP4` (norm output lands
pre-quantized for the next NVFP4 GEMM) and `MoEAllReduce`/`MoEAllReduceParams`
to combine routed-finalize + shared-expert add + residual + next-layer norm in
a single op.

**Action:** adopt the DSV3 `forward_MoE`/`forward_mlp` structure for M3:
- attention `o_proj` AllReduce → `RESIDUAL_RMS_NORM(_QUANT_*)`
- MoE output AllReduce → fused with next `input_layernorm`
- shared+routed combine → `MoEAllReduce`

Removes 2–3 reduction/norm/quant kernels per layer × 60 layers. Depends on the
Gemma weight-fold (or a gemma-aware fused AR op).

### 2. Fused GEMM + SwiGLU-OAI for the dense MLP and shared expert

Routed experts already fuse the activation via `ActivationType.SwigluBias`
(CUTLASS). But the **dense MLP (layers 0–2)** and the **shared expert (all 57
MoE layers)** go through `GatedMLP` with a Python
`partial(_minimax_m3_swiglu_oai, …)` activation, which hits the eager fallback:

```python
# tensorrt_llm/_torch/modules/gated_mlp.py (~157)
elif callable(self.activation):
    return self.activation(x)
```

i.e. `chunk → clamp → clamp → sigmoid → mul → mul → add` as ~6 separate
elementwise kernels, and it **bypasses** the fused GEMM+SwiGLU epilogue paths
(`_can_fuse_gate_up_swiglu*`, which require `activation == F.silu`).

**Action:** extend the fused `swiglu` kernel (already carries `swiglu_limit` /
`quant_scale`) to the swigluoai shape (alpha gain 1.702, `up+1` beta, asymmetric
gate clamp) — the same math the MoE path already fuses — so gate_up GEMM +
activation (+ optional FP4-out) collapse into one kernel for those layers.

### 3. Fuse `index_q_proj` + `index_k_proj` into one GEMM; drop the `torch.cat`

In `_sparse_forward` the index branch runs two separate replicated MXFP8
projections of the same `hidden_states`, then concatenates them just to feed the
fused norm+RoPE:

```python
# tensorrt_llm/_torch/models/modeling_minimaxm3.py
idx_q = self.index_q_proj(hidden_states)      # ~1352, out = 4*128 = 512
idx_k = self.index_k_proj(hidden_states)      # ~1353, out = 128
...
idx_qk = torch.cat([idx_q, idx_k], dim=-1)    # ~1388
```

Both are replicated (`tensor_parallel_mode=None`), same dtype, same input →
merge into one `index_qk_proj` (out = `4*128 + 128 = 640`), then `split`.
Removes one GEMM launch and the `cat` alloc/copy per sparse layer (57 layers),
and the fused output is already contiguous for `fused_qk_norm_rope`.
(The main `qkv_proj` can't join — it is COLUMN-sharded while these are replicated.)

### 4. Quant epilogue on the QK-norm-RoPE kernel (ties into the FP8 KV path)

On the FP8 KV path the cast is currently a separate kernel:
`q_view = q_view.to(torch.float8_e4m3fn)` (in `run_msa_paged_gqa`) and the cache
write does `.to(cache.dtype)` inside `write_msa_main_kv`. DSA fuses
norm+rope+quant so Q (and the K written to cache) come out E4M3 directly,
eliminating those casts. The quant epilogue on the QK-norm-RoPE kernel is the
natural home for this — noted here for completeness since that kernel is owned
separately.

### 5. Drop redundant post-split `contiguous()` copies

Both `_dense_forward` and `_sparse_forward` do `q, k = q.contiguous(),
k.contiguous()` after splitting the fused qkv. `q` is already the leading
contiguous segment and the attention core only needs a `[N, H, D]` view
(last-dim contiguous holds after the split). These are explicit copy kernels on
the hot path that are likely unnecessary — verify and remove.

---

## Existing infra note

There is already a fused **allreduce + Gemma-RMSNorm** custom op in the tree —
`MiniMaxAllReduceRMS` (`torch.ops.trtllm.minimax_allreduce_rms` and a
`minimax_allreduce_rms_qk` variant) with a `MiniMaxRMSNorm` wrapper. In
`modeling_minimaxm3.py`, `MiniMaxRMSNorm` is **defined and never instantiated**
(dead code; M2 uses it for sharded Q/K norm, M3 does not). That op handles the
gemma `(1+w)` scaling but has **no residual arg**, so it is not a drop-in for
the layer-boundary add+norm. Prefer #1 (weight-fold + standard fused AR ops)
over extending this M3-specific op.

---

## Suggested ordering

1. **Gemma-norm weight fold** — the enabler for #1 and the NVFP4 norm+quant path.
2. **DSV3-style AllReduce + residual + norm(+quant) fusions** (#1) — all 60 layers.
3. **Fused SwiGLU-OAI** (#2) — layers 0–2 + all 57 shared experts.
4. **Fused index_q/k projection + cat removal** (#3) and **contiguous cleanup**
   (#5) — localized MSA-path wins.
5. **QK-norm-RoPE quant epilogue** (#4) — folds into the separately-owned kernel.

---

## Duplicated work in `fmha_sm100_plan` (MSA plan building)

There are three logical plans per step, all built by `fmha_sm100_plan`, and all
depend only on `qo_lens / kv_lens / qo_offset / page_size / head counts / topk /
causal` — every one of which is **layer-invariant** within a step:

- **proxy** (indexer max-score): `num_q_heads=num_index_heads(4)`,
  `num_kv_heads=1`, `output_maxscore=True`
- **gqa** (sparse layers 3–59): full q/kv heads, `kv_block_num=topk`
- **dense** (layers 0–2): full q/kv heads, no `kv_block_num`

### Decode — already optimal

`_build_decode_plans` runs once in `prepare()` (outside capture), builds those 3
plans, mirrors them into the graph-stable `_MsaGraphSafePlan` buffers, and every
layer reads them via `msa_decode_proxy_plan` / `msa_decode_gqa_plan` /
`msa_decode_dense_plan`:

```python
# tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/msa_backend.py (~428)
def _build_decode_plans(self) -> None:
    """... The plans are layer-invariant for MiniMax-M3, so they are built once
    per step from the shared sparse geometry, mirrored into CUDA-graph-stable
    buffers, and reused by every layer."""
```

Per decode step: **3 plan builds, reused across all 60 layers.** No per-layer
duplication.

### Prefill / mixed — the duplication

`_build_decode_plans` bails out as soon as the batch has any context request:

```python
# msa_backend.py (~444)
# A decode batch is pure generation (no context requests).
if int(self.num_contexts or 0) > 0:
    return
```

With the metadata plans left `None`, each layer rebuilds its own plan inline:

- Indexer, per sparse layer — `msa_indexer.py`, the `proxy_plan is None` branch
  calls `_proxy_max_score` → `fmha_sm100_plan`.
- Main GQA, per layer — `run_msa_sparse_gqa` in `msa_sparse_gqa.py`, the
  `plan is None` branch → `fmha_sm100_plan`.

Nothing caches the inline-built plan, so per prefill / chunked-prefill / mixed
step:

- 57 sparse layers × (proxy + gqa) = **114 builds**
- 3 dense layers × dense = **3 builds**
- **≈117 plan builds, of which only 3 are unique** — the identical worklist
  (including the 131072-wide `packed_work_info`) is recomputed 57×.

### Fix: hoist prefill/mixed plans into `prepare()`

They're layer-invariant, and prefill runs eagerly (no CUDA-graph capture), so
they don't need the `_MsaGraphSafePlan` stable-buffer machinery — store 3 plain
plan tuples on the metadata once per step and have `run_indexer` /
`run_msa_sparse_gqa` read them, exactly like decode. Takes prefill from
~117 → **3 builds/step**.

Concretely:
- Drop the `num_contexts > 0` early return; build proxy/gqa/dense for the prefill
  geometry too (they only need `qo_offset = kv_len - qo_len`, already given by
  the existing `msa_*_cpu` properties for both phases).
- Store them on the metadata (plain attributes for the eager path; the existing
  graph-safe owners for the captured path).
- In `MsaIndexer.select_blocks` and `run_msa_sparse_gqa`, treat a present
  metadata plan as authoritative; keep inline build only for the standalone-test
  path.

**Below 3/step is not free:** proxy (MQA, maxscore), gqa (topk blocks), and dense
(full page table) are genuinely different worklists. gqa and dense differ only
by `kv_block_num`, but that changes work partitioning, so merging them needs a
`fmha_sm100` API change. The 3-per-step floor is the target; the real win is
killing the ~114 redundant prefill builds.

**Caveat:** mixed batches (chunked prefill + decode in one forward) currently
take the fully-inline path for the decode rows too — verify the hoisted plan
covers the mixed geometry (it should, since `qo_offset` already encodes the
cached prefix per request).

---

## Test plan for the landed changes (#7, #4)

Both changes are numerically identical to the baseline (host-side plan
dedup; a row-wise weight concat + one GEMM instead of two + a cat), so the bar
is "no accuracy change, no crash, and a measurable host/kernel reduction".
Requires a Blackwell (SM100) node with `LLM_MODELS_ROOT` set.

### 1. Unit tests (fast, GPU-gated construction)

```bash
# Fused index_qk_proj construction + shape assertions (updated for the fusion)
pytest -q tests/unittest/_torch/models/test_minimax_m3.py

# MSA backend metadata (plan buffers, proxy score view)
pytest -q tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_backend.py

# VL wrapper still loads text weights through the fused-index path
pytest -q tests/unittest/_torch/models/test_minimax_m3_vl.py
```

Key things these cover: `index_qk_proj` exists with `out_features == 640`,
the old `index_q_proj`/`index_k_proj` modules are gone, and dense layers expose
no index branch.

### 2. Accuracy — the target config (NVFP4 + MSA + FP8 KV)

```bash
pytest -q "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_nvfp4[use_msa=True]"
```

This is the exact path both changes touch: MSA sparse attention, FP8 KV cache,
mixed-precision NVFP4 checkpoint. GSM8K/MMLU scores must match the pre-change
baseline. Also run the Triton control to confirm no cross-path regression:

```bash
pytest -q "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_nvfp4[use_msa=False]"
pytest -q "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_mxfp8[use_msa=True]"
# Prefill-plan hoist under CUDA graph / piecewise:
pytest -q "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_mxfp8_piecewise_cuda_graph"
```

The last one is important for #7: it exercises the mixed decode + graph path,
which is where the eager-vs-graph-safe plan split matters.

### 3. Perf validation (nsys)

Re-run the existing `minimax_m3_fp4_nsys/run_nsys.sh` on the NVFP4+MSA config
and diff against a baseline capture:

- **#7**: in the prefill / context phase, `fmha_sm100_plan` should now appear
  ~3x per step instead of ~117x (look for the plan-build host ranges / CPU
  time in `prepare`). Biggest effect at high concurrency / short ISL / chunked
  prefill.
- **#4**: each sparse layer should show one index projection GEMM instead of
  two, and the `aten::cat` before the index QK-norm+RoPE should be gone.

### What to watch for

- A GSM8K/MMLU delta vs. baseline would most likely point at #4's weight
  concat (wrong row order) — check that `index_qk_proj.weight[:512]` equals the
  old `index_q_proj.weight` and `[512:]` equals `index_k_proj.weight`.
- A crash in prefill/mixed would point at #7 — the eager plan geometry or the
  `use_fp8_kvcache` flag. The inline-build fallback still exists, so temporarily
  forcing `msa_eager_*_plan` to `None` isolates #7.
