# MiniMax-M3 — Elementwise-kernel reduction (all modules)

Analysis of the elementwise kernels across **all** MiniMax-M3 modules
(dense attn / sparse attn / MLP / MoE), with concrete, DeepSeek-V3-aligned
fusion changes.

- **Profiles:**
  - `minimax_m3_fp4_nsys/out/c1/minimax_m3_fp4_c1_msa_log.sqlite` — **primary
    (MSA codepath)**, used for the sparse-attn + per-module split below.
  - `minimax_m3_fp4_nsys/out/c1/minimax_m3_fp4_c1_ind_fused.sqlite` — earlier
    run used for the original MoE/dense analysis (MoE/dense/MLP numbers are
    identical to the MSA run within noise).
- **Setup:** B200, TP4 / EP4, ISL=8192 / OSL=1 (prefill), NVFP4 MIXED_PRECISION
- **Model:** `tensorrt_llm/_torch/models/modeling_minimaxm3.py`
  (`hidden_size=6144`, 60 layers = 3 dense + 57 sparse/MoE)
- All numbers are **per forward step**, device 0, 12 steps. Kernels mapped to
  modules via per-layer NVTX ranges (`layerN.moe`, `layerN.mlp`,
  `layerN.dense_attn`, `layerN.sparse_attn` / `layerN.msa.sparse.*`).

> **Run note:** the `ind_fused` capture had a giant `elementwise_kernel`
> (~2,109/step, **70 ms/step**) inside sparse attention that is **absent** in
> the `msa_log` capture (285/step, 2.9 ms/step). The MSA codepath measured here
> is the lighter/representative one; total device-0 kernels drop 66,660 → 44,772
> across the 12 steps. MoE, dense-attn and MLP are unchanged between runs.

---

## 0. Model & test setup

### Model: MiniMax-M3 NVFP4
- **HF id / checkpoint:** `nvidia/MiniMax-M3-NVFP4`
  - Files: `/home/scratch.trt_llm_data_ci/llm-models/MiniMax-M3-NVFP4/`
  - `architectures = ["MiniMaxM3SparseForConditionalGeneration"]`,
    `model_type = minimax_m3_vl` (published as multimodal; the **text-decoder**
    path is what we profile/eval), `torch_dtype = bfloat16`.
- **Quantization:** `MIXED_PRECISION` — **MXFP8 base layers + NVFP4 routed
  experts**. On the MSA path the **KV cache is FP8**; on the Triton path it stays
  BF16.
- **TRT-LLM model file:** `tensorrt_llm/_torch/models/modeling_minimaxm3.py`
- **Sparse-attention codepath:** **MSA** (`implementation="msa"`, not Triton).
  MSA kernels (`fmha_sm100`, combine/proxy, `k2q` CSR-build,
  `minimaxM3SelectBlocks`) are cloned at `/home/scratch.bbuddharaju_gpu/msa`.

### Architecture (from `config.json` → `text_config`)
| Field | Value |
|---|---|
| `hidden_size` | 6144 |
| `num_hidden_layers` | 60 → **3 dense + 57 sparse/MoE** (`sparse_disable_index_value` / `moe_layer_freq` = `[0,0,0,1,…,1]`) |
| `num_attention_heads` / `num_key_value_heads` | 64 / 4 (GQA), `head_dim=128` |
| RoPE | `rope_theta=5e6`, `partial_rotary_factor=0.5`, `rotary_dim=64` (partial RoPE on 64 of 128) |
| QK norm | `use_qk_norm=True`, `qk_norm_type=per_head`, `use_gemma_norm=True`, `rms_norm_eps=1e-6` |
| Activation | `hidden_act=swigluoai`, `swiglu_alpha=1.702`, `swiglu_limit=7.0` |
| `attention_output_gate` | False |
| `max_position_embeddings` | 1,048,576 (1M) |
| `vocab_size` | 200,064 |
| **MoE** | `num_local_experts=128`, `num_experts_per_tok=4` (top-4), `scoring_func=sigmoid`, `use_routing_bias=True`, `routed_scaling_factor=2.0` |
| Shared expert | `n_shared_experts=1`, `shared_intermediate_size=3072` |
| Dense MLP (first 3 layers) | `dense_intermediate_size=12288` |
| MTP | `num_mtp_modules=1` |

### Sparse attention (`sparse_attention_config`)
`use_sparse_attention=True`, `sparse_block_size=128`, `sparse_topk_blocks=16`,
`sparse_num_index_heads=4`, `sparse_index_dim=128`. Layers 0–2 run **dense**
attention; layers 3–59 run **block-sparse** (indexer selects top-16 128-token
blocks per query).

### Test of interest
```bash
pytest tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_nvfp4[use_msa=True] -s -v
```
Runtime config exercised by this test (`TestMiniMaxM3.test_nvfp4`):
- **TP=4, EP=4** (`tensor_parallel_size=4`, `moe_expert_parallel_size=4`)
- `sparse_attention_config = MiniMaxM3SparseAttentionConfig(implementation="msa")`
- `moe_config = MoeConfig(backend="CUTLASS")`
- `kv_cache_config`: `free_gpu_memory_fraction=0.6`, `enable_block_reuse=False`,
  `dtype="fp8"` (MSA path)
- `max_seq_len=4096`, `trust_remote_code=True`
- Asserts `quant_algo == MIXED_PRECISION`; evaluates **MMLU** + **GSM8K**
- Gated: `skip_less_device(4)`, `skip_less_device_memory(140000)`, **Blackwell-only (SM100+)**

> **Profiling vs test:** the nsys captures analyzed below were taken at
> **ISL=8192 / OSL=1 (prefill)** on **B200, TP4/EP4, NVFP4 MIXED_PRECISION** — the
> same model/parallelism/codepath as the test, but a fixed 8K-token prefill shape
> (the accuracy test itself runs MMLU/GSM8K at `max_seq_len=4096`).

---

## 1. Elementwise-op split per module

Per forward step (device 0), MSA run. "elem" = the `*elementwise*` kernel family
(`elementwise_kernel`, `vectorized_/unrolled_/index_elementwise_kernel`,
`elementwise_kernel_with_index`).

| Module | **elem kern/step** | elem µs/step | total kern/step | total µs/step |
|---|---:|---:|---:|---:|
| **moe** | 228 | **10,781** | 969 | 51,155 |
| **sparse_attn** | **1,368** | 7,860 | 2,337 | 55,553 |
| dense_attn | 45 | 283 | 63 | 1,977 |
| OUTSIDE_NVTX | 71 | 284 | 100 | 1,074 |
| mlp | 6 | 13 | 21 | 1,394 |
| layernorm | 0 | 0 | 241 | 50,030 |
| **TOTAL elem** | **1,718** | **19,221** | — | — |

By **count**, sparse attention dominates (1,368/step, ~80% of all elementwise
launches) — spread over 57 layers as many tiny gather/scatter/reshape kernels.
By **time**, MoE dominates (10.8 ms/step from just 2 large kernels).

Per-module elementwise kernel-type breakdown (per step):

| Module | elementwise_kernel | vectorized_ | index_ | unrolled_ |
|---|---|---|---|---|
| moe | — | 171 @ 3,019µs | — | **57 @ 7,762µs** |
| sparse_attn | 285 @ 2,875µs | 741 @ 2,488µs | 171 @ 1,858µs | 171 @ 639µs |
| dense_attn | 9 @ 111µs | 24 @ 86µs | 6 @ 62µs | 6 @ 23µs |
| mlp | — | 6 @ 13µs | — | — |

---

## 2. Where the elementwise time actually is

### MoE (10.78 ms/step) — two large kernels, ~97%
Confirmed by exact grid-size arithmetic (`num_tokens=8192`, `hidden_size=6144`):

1. **Router fp32 up-cast** — `unrolled_elementwise_kernel`, **136 µs × 57 = 7,762 µs/step**
   - grid `(98304,1,1)` = `8192×6144 / 4` → this is `hidden_states.to(torch.float32)`
     inside `MiniMaxM3Gate.forward`. Reads/writes ~50M elements per layer just to
     feed a tiny `[8192,128]` router GEMM. → **Fix A.**
2. **Shared+routed combine add** — `vectorized_elementwise_kernel`, **48 µs × 57 = ~2,750 µs/step**
   - grid `(49152,1,1)` = `8192×6144 / 8` → `shared_output.add_(routed_output)`. → **Fix B.**

Adjacent fusable (not "elementwise"-named): **`quantize_with_block_size` = 5.8 ms/step
in MoE (3×/layer)** — the fp4 block-quant passes. → **Fix C.**

### Sparse attn (7.86 ms/step) — many tiny launches (1,368/step)
Dominated by launch count, not per-kernel cost. Submodule split (per step,
57 sparse layers): → **Fix D / E.**

| Sparse submodule | elem kern/step | elem µs/step | per-layer |
|---|---:|---:|---:|
| `msa.sparse.fmha` | 741 | 3,465 | ~13 elem/layer |
| `msa.sparse.qkv_index_proj_norm_rope` | 285 | 2,725 | ~5 elem/layer |
| `msa.sparse.indexer` | 228 | 1,337 | ~4 elem/layer |
| `msa.sparse.indexer.proxy_fmha` | 57 | 213 | 1 elem/layer |
| `msa.sparse.o_proj` | 57 | 120 | 1 elem/layer |

### Why this matters beyond busy-time
GPU **idle is 28–30% of wall**, dominated by 10–50 µs and 100–500 µs
python-dispatch gaps. The model launches ~3,730 kernels/step; sparse attn alone
is 2,337 of them. Removing launches (especially the ~1,368 tiny sparse
elementwise) cuts directly into that dispatch-idle — likely a bigger wall-clock
win than the busy-time math alone.

---

## A. Router fp32 cast → DSV3 bf16 router-GEMM op

**Fixes the 136 µs `unrolled_elementwise` × 57 = ~7.78 ms/step.**

> **STATUS: the bf16-weight variant below (A) is a no-go for now** (weight
> fp32→bf16 is lossy and can flip top-4 routing vs SGLang). See
> **A-deep-dive** for what's actually happening and the *safe* ways to remove the
> 136 µs cast without changing routing numerics — that is the recommended path.

### Deep dive: what the "fp32 router" is really doing

Current path, per MoE layer (×57/step):
```339:341:tensorrt_llm/_torch/models/modeling_minimaxm3.py
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router runs in fp32 to match SGLang.
        return torch.nn.functional.linear(hidden_states.to(torch.float32), self.weight)
```
1. `hidden_states.to(torch.float32)` → `unrolled_elementwise_kernel`, **136 µs**
   — casts `[8192, 6144]` bf16 → fp32 (~74 MB read, ~147 MB write). Pure HBM
   movement.
2. `F.linear(hidden_fp32, weight_fp32)` →
   `cutlass3x_sm100_tensorop_s128x128x8**tf32**gemm_f32_...`, **43 µs**. Output is
   only `[8192, 128]`.
3. Routing (`MiniMaxM3MoeRoutingMethod`): sigmoid + `e_score_correction_bias` +
   group top-k + `routed_scaling_factor`, in fp32.

So ~179 µs/layer; the **cast is 136 µs, the matmul only 43 µs.**

**The nuance — the matmul is already TF32, not fp32.** The kernel is a `tf32gemm`:
the tensor cores truncate inputs to TF32 (10 mantissa bits) in the MMA. The
activation data path is:

> bf16 (8 mantissa bits) → cast to fp32 → **TF32 truncation (10 bits) in the MMA**

The activation began as bf16 (8 bits), so upcasting to fp32 then feeding TF32
recovers **no real precision**. The fp32 cast is essentially a no-op for accuracy
on the activation side. Precision that *does* matter comes from the **fp32 weight**
and the **fp32 accumulation** over the 6144 reduction dim — not from the cast.

**Why the bf16 path (A) is a no-go, precisely.** `dsv3_router_gemm_op` uses bf16
activation × **bf16 weight** → fp32. Vs today, the only meaningful loss is the
**weight** (fp32→bf16); the activation was already 8-bit and accumulate stays
fp32. Router picks top-4 of 128, so a perturbed weight can flip selections →
accuracy drift. That weight change is the legitimate blocker — *not* the
activation/cast handling.

### Safe ways to remove the 136 µs cast (recommended), best first

The cast is **separable from the matmul numerics** — `bf16 → fp32` upcast is
exact, so it can be removed without changing a single routing decision.

1. **Fold the fp32 upcast into the producing RMSNorm (best).** The router input is
   `post_attention_layernorm`'s output (already fused into the AllReduce as
   `RESIDUAL_RMS_NORM`). RMSNorm computes in fp32 then downcasts to bf16 — have it
   emit an **fp32 side-output** for the router in the same kernel. Removes the
   136 µs cast (7.8 ms/step) **and is more accurate** than today (router sees the
   true fp32 norm output, not the bf16-rounded-then-upcast value). Dovetails with
   **Fix C**: the same norm becomes one multi-output kernel emitting **bf16**
   (shared expert) + **fp32** (router) + **fp4+scale** (routed experts) — one pass,
   three consumers, zero standalone cast/quant kernels.
2. **Fuse the bf16→fp32 upcast into the GEMM load (bit-identical).** A GEMM that
   loads the current bf16 activation and converts on-chip against the fp32 weight
   (fp32 accumulate) gives **bit-identical** logits minus the HBM round-trip.
   Fallback if a mixed-input convert-on-load cutlass kernel is available.
3. **bf16-weight `dsv3_router_gemm_op` (A, fastest, lossy, gated).** Removes the
   cast *and* speeds the matmul, but changes the weight to bf16 → validate top-4
   match-rate vs SGLang on a calibration set before enabling. The sigmoid/bias/
   top-k stay fp32 regardless.

> **To confirm upstream:** whether SGLang does the router **matmul** in fp32 or
> only the **sigmoid/top-k** in fp32 (many implementations do bf16 matmul + fp32
> scoring). If the latter, even option 3's activation handling matches the
> reference and only the weight precision is in question.

**Recommendation: option 1** — biggest safe win, improves fidelity, and merges
into the **Fix C** norm-fold work. The bf16-weight rewrite (A) below stays gated.

### How DSV3 does it
The gate stores a **bf16** weight and uses a dedicated router GEMM that takes
bf16 input and emits fp32 logits directly (nvjet kernels) — no full-hidden fp32
materialization. The op is generic (reused by afmoe, glm4-moe, nemotron-h) and
only requires the router weight to be bf16.

```python
# modeling_deepseekv3.py  DeepseekV3Gate.forward
logits = torch.ops.trtllm.dsv3_router_gemm_op(
    hidden_states, self.weight.t(), bias=None, out_dtype=torch.float32)
```

### Current MiniMax-M3 (the slow path)
```330:341:tensorrt_llm/_torch/models/modeling_minimaxm3.py
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.float32),
            requires_grad=False,
        )
        ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router runs in fp32 to match SGLang.
        return torch.nn.functional.linear(hidden_states.to(torch.float32), self.weight)
```

### Concrete change (`MiniMaxM3Gate`)
1. Make the weight bf16 (model dtype):
```python
self.weight = nn.Parameter(
    torch.empty((num_experts, hidden_size), dtype=torch.bfloat16),
    requires_grad=False,
)
```
2. Replace `forward`:
```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # bf16 x bf16 -> fp32 logits (no full-hidden fp32 materialization).
    return torch.ops.trtllm.dsv3_router_gemm_op(
        hidden_states, self.weight.t(), bias=None, out_dtype=torch.float32)
```
3. In `load_weights`, store bf16: `self.weight.copy_(w["weight"][:].to(torch.bfloat16))`.

The routing math (sigmoid + `e_score_correction_bias` + `routed_scaling_factor`)
still runs in fp32 on the fp32 logits; only the *input precision* changes (bf16
with fp32 accumulate), matching what afmoe/glm/nemotron ship.
**Removes ~7.78 ms/step + 57 launches** and replaces the 43 µs fp32 tf32 gemm
with a faster bf16 gemm.

> **Validate:** top-4 selection vs SGLang — this is the one place accuracy could shift.

---

## B. `shared_output.add_(routed_output)` → fold into the AllReduce

**Fixes the 48 µs `vectorized_elementwise` × 57 = ~2.75 ms/step.**

### How DSV3 does it — two mechanisms
1. **Best (min-latency):** MoE returns *unfinalized* outputs (`do_finalize=False`)
   and the shared+routed combine + finalize + allreduce + next-norm collapse into
   a single `MoEAllReduce` kernel:
```1559:1569:tensorrt_llm/_torch/models/modeling_deepseekv3.py
                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=shared_output,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)
```
2. **Portable (CUTLASS path):** write the add straight into the preallocated
   NCCL-window buffer the allreduce consumes, so no temporary and the reduce
   reads it in place:
```1202:1224:tensorrt_llm/_torch/models/modeling_deepseekv3.py
            output_tensor = None
            if not self.use_dp and self.mapping.tp_size > 1:
                w, actual_kind = torch.ops.trtllm.allocate_output(
                    shared_output, self.allreduce.output_buffer_kind,
                    self.mapping.tp_group)
                if actual_kind == int(BufferKind.NCCL_WINDOW):
                    output_tensor = w
            ...
                    final_hidden_states = torch.add(shared_output,
                                                    routed_output,
                                                    out=output_tensor)
```

### Current MiniMax-M3
```574:579:tensorrt_llm/_torch/models/modeling_minimaxm3.py
            # In-place add into ``shared_output`` to avoid allocating a
            # temporary (matches DeepSeekV3 / GLM convention).
            result = shared_output.add_(routed_output)

        if self.allreduce is not None:
            result = self.allreduce(result, all_reduce_params=final_all_reduce_params)
```

MiniMax-M3 already defers the allreduce to POST fusion
(`_apply_next_layer_layernorm` → `RESIDUAL_RMS_NORM`).

### Concrete change
- **Full removal (mechanism 1):** add a `MoEAllReduce` module to
  `MiniMaxM3DecoderLayer` (mirror DSV3 lines 1301–1307 / 1559–1569), have
  `MiniMaxM3MoE.forward` return `(shared_output, routed_output)` unreduced when
  POST fusion is on, and pass `shared_expert_output=shared_output` into
  `moe_allreduce`. Folds the combine into the allreduce+`next_layer_layernorm`
  kernel that already runs. Requires the fused MoE to expose an unfinalized path
  (TRTLLM nvfp4 min-latency, single-node + p2p — same gate DSV3 uses at
  lines 1529–1533). MiniMax-M3 currently runs the **CUTLASS** backend
  (profile shows `finalizeKernelVecLoad` + `bmm_E2m1…swiGlu`).
- **Low-effort interim (mechanism 2):** keep `add_` but write it into the
  `allocate_output` NCCL window (removes the temp alloc + lets the reduce read
  in place). Marginal, but portable to the CUTLASS backend.

---

## C. Duplicate fp4 quant → DSV3 norm-fold + fp4 reuse

**Targets ~2–4 ms/step of the 5.79 ms/step `quantize_with_block_size`.**

### How DSV3 does it
The dense-MLP/attention input quant is folded into the producing RMSNorm via
`RESIDUAL_RMS_NORM_QUANT_NVFP4`; the norm returns an `Fp4QuantizedTensor` that
downstream consumers reuse:
```1604:1617:tensorrt_llm/_torch/models/modeling_deepseekv3.py
        if self.fusion_config.PRE_MLP_FUSION:
            if self.mlp.gate_up_proj.has_nvfp4:
                act_fp4, act_sf, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.
                        RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        scale=self.mlp.gate_up_proj.input_scale,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
```
The scale is attached to each norm at load time, self-gating to a no-op unless
the consumer is static-NVFP4:
```2026:2037:tensorrt_llm/_torch/models/modeling_deepseekv3.py
            layer.input_layernorm.nvfp4_scale = _static_nvfp4_input_scale(
                getattr(self_attn, "kv_a_proj_with_mqa", None))
            ...
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, GatedMLP):
                layer.post_attention_layernorm.nvfp4_scale = (
                    _static_nvfp4_input_scale(getattr(mlp, "gate_up_proj", None)))
```
The MoE `forward` is already plumbed to accept a shared `hidden_states_fp4` for
**both** routed and shared experts:
```1165:1180:tensorrt_llm/_torch/models/modeling_deepseekv3.py
        def _compute_shared_output():
            shared_input = (hidden_states_fp4 if
                            (hidden_states_fp4 is not None
                             and self.shared_experts_use_fp4) else
                            hidden_states)
            shared_output = self.shared_experts(shared_input)
            ...
        def _compute_routed_output():
            routed_output = self.compute_routed_output(hidden_states,
                                                       hidden_states_fp4, ...)
```

### Current MiniMax-M3
`_apply_pre_feed_forward_norm` uses plain `RESIDUAL_RMS_NORM` (no quant), and
`MiniMaxM3MoE.forward` passes bf16 `hidden_states` into `self.experts(...)`
**and** `self.shared_experts(...)` separately → each re-quantizes to fp4 (two of
the three `quantize_with_block_size`/layer).

### Concrete change
1. Add a `_static_nvfp4_input_scale`-style hook: at load/`setup_aliases`, set
   `post_attention_layernorm.nvfp4_scale` = the routed-experts' fp4 `input_scale`
   (only when routed experts are static-NVFP4 **and** the shared expert's
   `gate_up_proj.input_scale` matches — assert equality like DSV3 does for
   q_a/kv_a). Reuse `_static_nvfp4_input_scale` / `is_static_nvfp4_input_eligible`.
2. In `_apply_pre_feed_forward_norm`, when that scale is present use
   `AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4` (passing `scale=...`) and
   return an `Fp4QuantizedTensor` (copy DSV3 lines 1604–1617).
3. Thread that fp4 tensor into `MiniMaxM3MoE.forward` and pass it to **both**
   `self.experts(hidden_states_fp4, ...)` and `self.shared_experts(hidden_states_fp4)`
   (`create_moe` experts and `GatedMLP` both accept `Fp4QuantizedTensor`).

Folds the MoE-input quant into the norm and removes the duplicate shared-expert
input quant. **Caveat:** routed vs shared `input_scale` must be equal to share
one fp4 tensor (the assert DSV3 encodes). The remaining quant (shared-expert
post-swiglu, before down_proj) is the same fusion the routed `SwigluBias` kernel
already does internally; matching it on the shared `GatedMLP` path is a separate,
lower-value follow-up.

---

## Priority

| Change | Module | Est. saving | Risk / effort |
|---|---|---|---|
| **A** router bf16 GEMM | moe | ~7.0 ms/step + 57 launches | low; verify routing accuracy vs SGLang |
| **C** norm-fold + fp4 reuse | moe | ~2–4 ms/step + ~114 launches | medium; needs matching routed/shared input_scale |
| **B** combine-add into MoEAllReduce | moe | ~2.75 ms/step + 57 launches | high; needs unfinalized MoE path / TRTLLM nvfp4 |
| **E** batch/hoist MSA gather+k2q prep | sparse_attn | ~2–4 ms/step + **~600–900 launches** | medium/high; biggest launch-count/idle win |
| **D** drop QK/idx `.contiguous()` copies | sparse+dense attn | ~1.5–2 ms/step + ~230 launches | low/medium; needs fused_qk_norm_rope out-variant |

Land **A** first — isolated to `MiniMaxM3Gate`, and the single biggest
elementwise **time** cost. For **wall-clock / GPU-idle**, prioritize **E** — it
removes the largest share of the ~1,718 elementwise launches/step (sparse attn is
80 % of them).

### Priority if A is excluded (router bf16 GEMM is a no-go)

All TRT-LLM-only except where noted. **Suggested sequence: C(+router fp32 side-output) → E → D → B.**

> Even with **A** gated, the **136 µs router cast (7.8 ms/step)** is still
> removable *safely* — fold the `bf16→fp32` upcast into the producing
> `post_attention_layernorm` as an **fp32 side-output** (see A-deep-dive
> option 1). This is bit-safe (actually more accurate) and rides along with the
> **Fix C** norm-fold, so it costs almost nothing extra once C is done. This makes
> C the top item.

1. **E — cut the sparse-attn elementwise count** (~2–4 ms/step + **~600–900
   launches/step**). Biggest wall-clock win: GPU idle is 28–30 % and dominated by
   tiny per-layer dispatch gaps, and these 1,368 elementwise/step are the largest
   source. Hoist step-invariant gather/page-table/row-map prep into
   `MiniMaxM3AttentionMetadata.prepare` (build once, reuse across all 57 layers),
   fuse gather+cast, drop `.contiguous()` on gathered KV.
2. **C — MoE norm-fold + fp4 reuse** (~2–4 ms/step + ~114 launches). Highest
   **busy-time** win without touching the router. Needs matching routed/shared
   `input_scale`.
3. **D — drop QK/idx `.contiguous()` copies** (~1.5–2 ms/step + ~230 launches).
   Lowest effort; best effort-to-payoff ratio. Ideally via a
   `fused_qk_norm_rope` out-variant.
4. **B — combine-add into `MoEAllReduce`** (~2.75 ms/step + 57 launches). High
   effort — needs the unfinalized MoE path (TRTLLM nvfp4 min-latency, single-node
   + p2p), which the current CUTLASS backend doesn't expose. Cheap interim:
   `allocate_output` NCCL-window `torch.add(out=...)`.

Optional deeper follow-up (**needs the MSA submodule**, not required for the
above): **E′** — push the pre-gather into `fmha_sm100`'s native paged layout and
collapse the k2q CSR-build kernels.

---

## Ownership: TRT-LLM vs MSA submodule

The MSA submodule is the external `fmha_sm100` package
(`/home/scratch.bbuddharaju_gpu/msa`, `python/fmha_sm100/`). It owns the sparse
attention forward/combine kernels, the proxy fmha, and the **k2q CSR-build
pipeline** (`cute/src/sm100/build_k2q_csr/build_k2q_csr.cu`, `prepare_k2q_csr.py`
→ `k2q_build_row_map/hist/row_prefix/tile_prefix_smem/scatter`).

**Key fact:** the MSA kernels (fmha, combine, proxy, k2q, `minimaxM3SelectBlocks`)
are *separately named* — **none of them are in the elementwise count**. All 1,368
sparse `*elementwise*` kernels/step are aten/torch ops emitted by **TRT-LLM
Python glue** (`msa_backend.py`, `msa_indexer.py`, `msa_utils.py`, `common.py`,
`modeling_minimaxm3.py`) — the KV/index gather, dtype casts, reshapes, and
`.contiguous()` copies. So reducing the *elementwise* count is overwhelmingly a
TRT-LLM change.

| Fix | Doable in TRT-LLM? | Needs MSA submodule change? |
|---|---|---|
| **A** router bf16 GEMM | ✅ Yes — `MiniMaxM3Gate` + existing `trtllm::dsv3_router_gemm_op` | ❌ No (MoE, unrelated to sparse) |
| **B** combine-add → MoEAllReduce | ✅ Yes — `MiniMaxM3MoE` + decoder + `MoEAllReduce` op | ❌ No |
| **C** norm-fold fp4 quant + reuse | ✅ Yes — RMSNorm/`AllReduce` fusion + `create_moe`/`GatedMLP` | ❌ No |
| **D** drop QK/idx `.contiguous()` | ✅ Yes — copies live in `modeling_minimaxm3.py`; zero-copy via a `trtllm::fused_qk_norm_rope` out-variant (also TRT-LLM) | ⚠️ Optional — only if you instead make `fmha_sm100` accept strided Q/K to skip the copy |
| **E** — reduce sparse elementwise **count** (gather/cast/reshape, hoist step-invariant prep, drop gathered-KV `.contiguous()`, fuse gather+cast, cache page-tables/row-maps across the 57 layers) | ✅ Yes — all in the TRT-LLM sparse glue (`msa_backend.py`, `msa_indexer.py`, `msa_utils.py`, `MiniMaxM3AttentionMetadata.prepare`) | ❌ No for the elementwise ops themselves |
| **E′** — eliminate the pre-gather entirely / collapse k2q kernels | ➖ Partial | ✅ Yes — requires `fmha_sm100` to read native paged KV/index layout and/or fuse the k2q CSR-build pipeline (these are MSA-owned, and are *not* elementwise kernels) |

### Summary
- **A, B, C** — pure TRT-LLM (MoE/model), **no MSA involvement**.
- **D** — TRT-LLM (optional MSA only for the strided-input variant).
- **E** — the elementwise-count reduction (the thing that drives GPU-idle) is
  **TRT-LLM**; MSA changes (**E′**) are a *separate, deeper* optimization that
  removes the TRT-LLM pre-gather and the (non-elementwise) k2q kernels, but is not
  required to cut the 1,368 elementwise launches.

---

## D. Sparse-attn QK/index norm+RoPE — drop the `.contiguous()` copies

**Targets `msa.sparse.qkv_index_proj_norm_rope` = 285 elem/step, 2,725 µs/step**
(also the same pattern in dense attn, 3 layers).

### Observed kernel sequence (`layer3.msa.sparse.qkv_index_proj_norm_rope`, per layer)
```
   45.38us  quantize_with_block_size            # index_qk_proj input quant
  121.63us  device_kernel                       # qkv_proj GEMM
   90.82us  fusedQKNormRopeKernel               # main Q/K norm+partial-RoPE  (fused)
   28.06us  elementwise_kernel  grid=(32768)    # q.contiguous()/k.contiguous() copy
    3.97us  elementwise_kernel
   61.92us  nvjet_..._TNT                        # index_qk_proj GEMM
   28.86us  fusedQKNormRopeKernel               # index Q/K norm+RoPE (fused)
    9.38us  elementwise_kernel
    4.25us  elementwise_kernel
```

The norm+RoPE is already fused (good), but the split+`.contiguous()` on the four
outputs (`q`, `k`, `idx_q`, `idx_k`) generates the trailing elementwise copies:

```1438:1470:tensorrt_llm/_torch/models/modeling_minimaxm3.py
            if fused_qkv is not None:
                q, k, v = fused_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                return q.contiguous(), k.contiguous(), v
            ...
            if fused_idx is not None:
                idx_q, idx_k = fused_idx.split(
                    [self.sparse_num_index_heads * self.sparse_index_dim, self.sparse_index_dim],
                    dim=-1,
                )
                return idx_q.contiguous(), idx_k.contiguous()
```

### Concrete change
- Have `torch.ops.trtllm.fused_qk_norm_rope` write **contiguous, already-split**
  Q/K(/V) and idx_Q/idx_K into pre-allocated output buffers (an out-variant),
  removing the `split(...).contiguous()` copies entirely (4 copies/layer → 0).
- If the MSA attention core (`_forward_attention_core` → `msa_backend`) can accept
  strided/column-sliced tensors, drop the `.contiguous()` and pass the split
  views directly. Removes the 28 µs copy (grid `(32768)` = `8192×512`, the
  4-index-head × 128 idx_q copy) + the small ones.

---

## E. Sparse-attn gather/scatter/k2q prep — batch the tiny MSA host-side ops

**Targets `msa.sparse.fmha` (741 elem/step, 3,465 µs/step) + `msa.sparse.indexer`
(228 elem/step, 1,337 µs/step)** — the largest **count** of elementwise launches
in the whole model, all small (<12 µs).

### Observed (`layer3.msa.sparse.fmha`, per layer) — ~13 tiny elem before the real work
```
    3.46us  unrolled_elementwise_kernel
    3.52us  vectorized_elementwise_kernel
    3.01us  vectorized_elementwise_kernel
    2.40us  vectorized_elementwise_kernel
   11.90us  index_elementwise_kernel        # KV / index gather
    2.94us  unrolled_elementwise_kernel
    3.04us  vectorized_elementwise_kernel
    2.98us  vectorized_elementwise_kernel
    4.74us  elementwise_kernel
    8.58us  index_elementwise_kernel        # gather
   11.01us  vectorized_elementwise_kernel
    1.50us  vectorized_elementwise_kernel
    1.50us  vectorized_elementwise_kernel
    2.27us  k2q_build_row_map_kernel        # k2q remap prep (5 launches)
    6.59us  k2q_hist_kernel
    5.41us  k2q_row_prefix_kernel
    3.39us  k2q_tile_prefix_smem_kernel
    9.98us  k2q_scatter_kernel
  165.44us  ...fwdSparseAttentionForward     # the actual sparse attention
   93.82us  ...fwdcombineSparseAttentionForward
```

These come from the MSA backend host-side prep (KV/index gather, dtype/layout
casts, contiguity, and the k2q remap chain) in
`tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/{msa_backend.py,msa_indexer.py}`.
Per-kernel time is small, but at 57 layers this is ~57 k of the launch count that
drives the 28–30 % dispatch-idle.

### Concrete changes (launch-count reduction; low busy-time but high idle impact)
- **Fuse the KV/index gather with its cast/reshape.** The `index_elementwise_kernel`
  gathers (`_gather_paged_batched`) followed by separate vectorized casts — fold
  the dtype/layout conversion into the gather (single indexed-copy kernel) instead
  of gather → contiguous → cast.
- **Collapse the k2q remap prep** (`k2q_build_row_map` → `hist` → `row_prefix` →
  `tile_prefix_smem` → `scatter`, 5 launches/layer = 285/step): these are a
  histogram+prefix-sum+scatter pipeline that can be a single fused kernel (or at
  least computed once and cached across layers when the block pattern is
  layer-invariant for a given step).
- **Hoist step-invariant prep out of the per-layer loop.** Any gather index /
  row-map that depends only on `attn_metadata` (seq lens, block table), not on
  per-layer activations, should be built once per step in
  `MiniMaxM3AttentionMetadata.prepare` and reused by all 57 sparse layers,
  eliminating 56× redundant launches.
- **Remove `.contiguous()` on gathered K/V** where the sparse attention CUTLASS
  kernel accepts the native paged layout.

> Sparse-attn elementwise is a **launch-overhead** problem, not a bandwidth one:
> optimize for fewer kernels, not faster ones.

---

## Minor: dense attention (3 layers, ~0.28 ms/step)
Same `.contiguous()`/gather pattern as **D/E** but only 3 layers, so the ceiling
is small. The fixes in **D** (drop `q,k = q.contiguous(),k.contiguous()` in
`_dense_forward`) apply verbatim.

## Reproduce
Analysis scripts saved next to the profiles:
- `minimax_m3_fp4_nsys/out/c1/map_kernels2.py <sqlite>` — per-module + sparse-submodule
  elementwise split (auto-detects process / step count).
- `minimax_m3_fp4_nsys/out/c1/seq2.py <sqlite>` — per-NVTX-range kernel sequence dump.
- Originals: `map_kernels.py`, `seq.py` (hardcoded to the `ind_fused` capture).
