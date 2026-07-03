# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-layer diagnostic for the MiniMax-M3 NVFP4 routed-expert path.

Context
-------
The full-model NVFP4 accuracy test
(``TestMiniMaxM3::test_nvfp4`` in
``tests/integration/defs/accuracy/test_llm_api_pytorch.py``) evaluates to
0.000 while the sibling MXFP8 checkpoint passes. Both checkpoints share the
same base model and the same ``MiniMaxM3MoE`` wiring; only the routed-expert
quantization differs (MXFP8 experts vs NVFP4 experts). This module isolates a
single routed-MoE layer so we can tell whether the regression lives in

  * quant-config plumbing (``_set_minimax_m3_moe_quant_config``),
  * NVFP4 expert weight / scale loading, or
  * the fused kernel / activation itself,

instead of paying the multi-GPU, full-model, MMLU-sized evaluation cost.

The test loads ONLY one MoE layer's experts from each real checkpoint (the
full model is far too large for a unit test) and:

  1. asserts the mixed-precision plumbing tagged the layer's experts as NVFP4
     (pure CPU, no GPU required);
  2. asserts the resolved MoE quant method is an NVFP4 method;
  3. loads the real per-expert NVFP4 weights through the production
     ``load_weights`` + ``process_weights_after_loading`` path and asserts the
     loaded scale tensors are non-degenerate (this is where a silent
     scale-loading bug shows up);
  4. runs a forward pass on a fixed input and asserts the output is finite and
     non-zero;
  5. (cross-check) compares the NVFP4 expert output against the known-good
     MXFP8 expert output for the same input + routing. Garbage NVFP4 loading
     shows up as a near-zero cosine similarity here.

The checkpoint paths default to the CI/scratch locations used during
bring-up; override via env vars ``MINIMAX_M3_NVFP4_DIR`` /
``MINIMAX_M3_MXFP8_DIR``. The test skips cleanly when the checkpoints or a GPU
are unavailable.

NOTE: authored on a host without a GPU; expect to run it inside the TRT-LLM
GPU container. If a checkpoint key layout differs from the assumptions in
``_collect_layer_expert_weights`` the helper raises with the observed keys so
the mapping can be adjusted quickly.
"""

import os
import re
from typing import Dict, List, Tuple

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

# First sparse/MoE layer in MiniMax-M3 (layers 0-2 are dense). The NVFP4
# checkpoint stores experts per-layer as ``experts-layer-003.safetensors`` ...
# ``experts-layer-059.safetensors``.
_MOE_LAYER_IDX = 3

_NVFP4_DIR = os.environ.get(
    "MINIMAX_M3_NVFP4_DIR",
    "/home/scratch.bbuddharaju_gpu/random/hf_models/Minimax-M3-NVFP4",
)
_MXFP8_DIR = os.environ.get(
    "MINIMAX_M3_MXFP8_DIR",
    "/home/scratch.trt_llm_data_ci/llm-models/MiniMax-M3-MXFP8",
)

# Expert-weight checkpoint suffixes we know how to map onto the fused-MoE
# VANILLA loader keys (``{expert}.w1|w2|w3.{suffix}``).
_PROJ_ALIASES = {
    "w1": "w1",
    "w3": "w3",
    "w2": "w2",
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}
# Scale/weight tensor suffixes emitted by modelopt NVFP4 / MXFP8 checkpoints.
_TENSOR_SUFFIXES = (
    "weight",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("MiniMax-M3 NVFP4 expert test requires a CUDA device.")


def _require_dir(path: str, label: str):
    if not os.path.isdir(path):
        pytest.skip(f"{label} checkpoint not found at {path}.")


def _safetensors_files(checkpoint_dir: str) -> List[str]:
    return [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".safetensors")
    ]


def _expert_key_regex(layer_idx: int) -> re.Pattern:
    # Matches e.g.
    #   language_model.model.layers.3.block_sparse_moe.experts.17.w1.weight_scale
    #   model.layers.3.block_sparse_moe.experts.17.w1.weight
    return re.compile(
        r"(?:^|\.)layers\." + str(layer_idx) +
        r"\.block_sparse_moe\.experts\.(\d+)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)$")


def _collect_layer_expert_weights(
        checkpoint_dir: str, layer_idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
    """Return ({"{expert}.{w1|w2|w3}.{suffix}": cpu_tensor}, num_experts).

    Only the requested layer's routed-expert tensors are read, so the memory
    footprint is one layer, not the whole model. Raises with a diagnostic if
    no matching keys are found (so an unexpected key layout is obvious).
    """
    from safetensors import safe_open

    pat = _expert_key_regex(layer_idx)
    out: Dict[str, torch.Tensor] = {}
    max_expert = -1
    seen_projs = set()
    for path in _safetensors_files(checkpoint_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                m = pat.search(key)
                if m is None:
                    continue
                expert_id = int(m.group(1))
                proj_raw = m.group(2)
                suffix = m.group(3)
                seen_projs.add(proj_raw)
                proj = _PROJ_ALIASES.get(proj_raw)
                if proj is None or suffix not in _TENSOR_SUFFIXES:
                    continue
                out[f"{expert_id}.{proj}.{suffix}"] = f.get_tensor(key)
                max_expert = max(max_expert, expert_id)

    if not out:
        raise AssertionError(
            f"No routed-expert tensors found for layer {layer_idx} in "
            f"{checkpoint_dir}. Observed projection names: {sorted(seen_projs)}. "
            f"Update _PROJ_ALIASES / _expert_key_regex to match this checkpoint.")
    suffix_counts: Dict[str, int] = {}
    for k in out:
        suffix_counts[k.split(".", 2)[2]] = suffix_counts.get(
            k.split(".", 2)[2], 0) + 1
    print(f"[collect] {os.path.basename(checkpoint_dir.rstrip('/'))} layer "
          f"{layer_idx}: experts={max_expert + 1} projs={sorted(seen_projs)} "
          f"suffix_counts={suffix_counts}")
    return out, max_expert + 1


def _build_model_config(checkpoint_dir: str, moe_backend: str) -> ModelConfig:
    # ModelConfig is frozen after from_pretrained, so pass the single-GPU
    # (tp=1/ep=1, no attention-DP) mapping at construction time rather than
    # mutating it afterward.
    model_config = ModelConfig.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        moe_backend=moe_backend,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1),
    )
    return model_config


def _build_moe_layer(model_config: ModelConfig, layer_idx: int):
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
    from tensorrt_llm._torch.utils import AuxStreamType

    aux_stream_dict = {
        AuxStreamType.MoeShared: torch.cuda.Stream(),
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
    }
    moe = MiniMaxM3MoE(model_config, aux_stream_dict, layer_idx=layer_idx)
    return moe.cuda()


def _load_and_process(experts, weights: Dict[str, torch.Tensor]):
    experts.load_weights([weights])
    # process_weights_after_loading finalizes NVFP4 global scales / bias &
    # swiglu renormalization; the accuracy-relevant scale folding happens here.
    if hasattr(experts, "post_load_weights"):
        experts.post_load_weights()
    experts.process_weights_after_loading()


def _backend_of(experts):
    """Unwrap ConfigurableMoE to the concrete backend that owns the scales.

    ``create_moe`` returns a ``ConfigurableMoE`` wrapper (ENABLE_CONFIGURABLE_MOE
    defaults to 1); it delegates load/forward to ``self.backend`` but only
    proxies a few attributes, so scale tensors live on the backend.
    """
    return getattr(experts, "backend", experts)


def _scale_tensor_names(experts) -> List[str]:
    backend = _backend_of(experts)
    names = []
    for name in [
            "w3_w1_weight_scale",
            "w2_weight_scale",
            "fc31_alpha",
            "fc2_alpha",
            "fc31_scale_c",
            "fc31_input_scale",
            "fc2_input_scale",
    ]:
        if getattr(backend, name, None) is not None:
            names.append(name)
    return names


def _assert_non_degenerate(tensor: torch.Tensor, name: str):
    t = tensor.float()
    assert torch.isfinite(t).all(), f"{name} contains non-finite values."
    assert t.abs().sum() > 0, f"{name} is all-zero (scales not loaded)."
    # A single constant value across a per-block/per-expert scale tensor is a
    # strong signal the real scales were not applied (left at init default).
    if t.numel() > 1:
        assert t.min() != t.max(), (
            f"{name} is constant ({t.min().item()}); real per-block/per-expert "
            f"scales were likely not loaded.")


def _run_experts(moe, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    return moe.experts(
        hidden_states,
        router_logits,
        all_rank_num_tokens=[num_tokens],
        use_dp_padding=False,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("moe_backend", ["TRTLLM", "CUTLASS"])
def test_minimaxm3_nvfp4_expert_layer_plumbing_and_forward(moe_backend):
    """Load one NVFP4 MoE layer and validate plumbing, scales, and forward."""
    _require_dir(_NVFP4_DIR, "MiniMax-M3 NVFP4")

    # ---- Step 1: quant-config plumbing (CPU only) ----
    model_config = _build_model_config(_NVFP4_DIR, moe_backend)
    coarse_key = f"model.layers.{_MOE_LAYER_IDX}.block_sparse_moe.experts"
    quant_dict = getattr(model_config, "quant_config_dict", None)
    assert quant_dict is not None and coarse_key in quant_dict, (
        "_set_minimax_m3_moe_quant_config did not inject the coarse NVFP4 "
        f"expert key {coarse_key!r}. quant_config_dict="
        f"{None if quant_dict is None else list(quant_dict)[:4]}")
    assert quant_dict[coarse_key].quant_algo == QuantAlgo.NVFP4, (
        f"Expected NVFP4 for {coarse_key}, got "
        f"{quant_dict[coarse_key].quant_algo}.")

    _require_gpu()

    # ---- Step 2: resolved quant method is NVFP4 ----
    moe = _build_moe_layer(model_config, _MOE_LAYER_IDX)
    experts = moe.experts
    quant_method_name = type(getattr(experts, "quant_method", None)).__name__
    assert "NVFP4" in quant_method_name, (
        f"Resolved MoE quant method is {quant_method_name!r}, expected an "
        f"NVFP4 method for the routed experts.")
    # MiniMax passes gpt-oss-style swiglu (alpha/beta/limit), so the TRTLLM-Gen
    # path must pick the gpt-oss NVFP4 method, not the plain base method.
    assert int(experts.activation_type) == int(ActivationType.SwigluBias)

    # ---- Step 3: load real NVFP4 expert weights + validate scales ----
    weights, num_experts = _collect_layer_expert_weights(_NVFP4_DIR,
                                                         _MOE_LAYER_IDX)
    _load_and_process(experts, weights)

    backend = _backend_of(experts)
    scale_names = _scale_tensor_names(experts)
    assert scale_names, (
        "No NVFP4 scale tensors present after loading on backend "
        f"{type(backend).__name__}.")
    # Print a compact stats table so one run tells us whether each scale was
    # actually loaded or left at its create_weights default (ones).
    for name in scale_names:
        t = getattr(backend, name).float()
        print(f"[nvfp4 scale] {name}: shape={tuple(t.shape)} "
              f"min={t.min().item():.4g} max={t.max().item():.4g} "
              f"mean={t.mean().item():.4g}")
    # The per-block weight scales must not be left at the all-ones default.
    for name in ("w3_w1_weight_scale", "w2_weight_scale"):
        if getattr(backend, name, None) is not None:
            _assert_non_degenerate(getattr(backend, name), name)

    # ---- Step 4: forward is finite and non-zero ----
    hidden_size = model_config.pretrained_config.hidden_size
    torch.manual_seed(0)
    num_tokens = 16
    hidden_states = torch.randn(num_tokens,
                                hidden_size,
                                dtype=torch.bfloat16,
                                device="cuda")
    router_logits = torch.randn(num_tokens,
                                num_experts,
                                dtype=torch.float32,
                                device="cuda")
    with torch.inference_mode():
        # First call warms up / autotunes the kernel; the TRTLLM-Gen path was
        # observed to return all-zeros on the very first (pre-autotune) call.
        first = _run_experts(moe, hidden_states, router_logits)
        out = _run_experts(moe, hidden_states, router_logits)
    print(f"[forward] backend={moe_backend} first_norm={first.float().norm():.4g} "
          f"warm_norm={out.float().norm():.4g}")
    assert torch.isfinite(out).all(), "NVFP4 expert output has non-finite values."
    assert out.float().abs().sum() > 0, "NVFP4 expert output is all-zero."


def _import_nvfp4_reference():
    """Import the NVFP4 bf16-dequant reference module from the MoE test dir."""
    import sys

    moe_test_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "modules", "moe"))
    if moe_test_dir not in sys.path:
        sys.path.insert(0, moe_test_dir)
    from quantize_utils import NVFP4RefMLPFusedMoE  # noqa: E402
    return NVFP4RefMLPFusedMoE


def _build_nvfp4_reference(moe, model_config, num_experts, weights):
    """Build a bf16 gated-MLP reference that dequantizes the SAME NVFP4 weights.

    Uses the routing method + swiglu params from the real ``MiniMaxM3MoE`` so the
    only difference vs the fused backend is the quantized GEMM/activation kernel.
    """
    NVFP4RefMLPFusedMoE = _import_nvfp4_reference()

    from tensorrt_llm._torch.model_config import ModelConfig as _MC
    from tensorrt_llm.models.modeling_utils import QuantConfig

    pretrained = model_config.pretrained_config
    intermediate_size = getattr(pretrained, "moe_intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = pretrained.intermediate_size

    ref_quant = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    ref_model_config = _MC(pretrained_config=pretrained, quant_config=ref_quant)

    ref = NVFP4RefMLPFusedMoE(
        num_experts=num_experts,
        routing_method=moe.gate.routing_method,
        hidden_size=pretrained.hidden_size,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        model_config=ref_model_config,
        bias=False,
        activation_type=ActivationType.SwigluBias,
        swiglu_alpha=moe.swiglu_alpha_value,
        swiglu_beta=moe.swiglu_beta_value,
        swiglu_limit=moe.swiglu_limit_value,
        swiglu_gptoss_style=True,
    ).cuda()
    ref.load_weights([weights])
    return ref


@pytest.mark.parametrize("moe_backend", ["TRTLLM", "CUTLASS"])
def test_minimaxm3_nvfp4_matches_bf16_reference(moe_backend):
    """Compare the fused NVFP4 expert output against a bf16 dequant reference.

    The reference (``NVFP4RefMLPFusedMoE``) dequantizes the exact same NVFP4
    weights + scales to bf16 and runs the intended gpt-oss/MiniMax gated-MLP
    math (non-interleaved gate/up, alpha/beta/limit). It is the ground truth for
    correctness, so this catches scale, gate/up-ordering, and interleave bugs
    that a "non-zero output" check misses. Run per backend so we learn whether
    CUTLASS and/or TRTLLM-Gen are wrong for this checkpoint.
    """
    _require_dir(_NVFP4_DIR, "MiniMax-M3 NVFP4")
    _require_gpu()

    model_config = _build_model_config(_NVFP4_DIR, moe_backend)
    moe = _build_moe_layer(model_config, _MOE_LAYER_IDX)
    weights, num_experts = _collect_layer_expert_weights(_NVFP4_DIR,
                                                         _MOE_LAYER_IDX)
    _load_and_process(moe.experts, weights)
    ref = _build_nvfp4_reference(moe, model_config, num_experts, weights)

    hidden_size = model_config.pretrained_config.hidden_size
    torch.manual_seed(0)
    num_tokens = 16
    hidden_states = torch.randn(num_tokens,
                                hidden_size,
                                dtype=torch.bfloat16,
                                device="cuda")
    router_logits = torch.randn(num_tokens,
                                num_experts,
                                dtype=torch.float32,
                                device="cuda")

    with torch.inference_mode():
        _run_experts(moe, hidden_states, router_logits)  # warmup / autotune
        fused_out = _run_experts(moe, hidden_states, router_logits).float()
        ref_out = ref(hidden_states, router_logits).float()

    fused_norm = fused_out.norm().item()
    ref_norm = ref_out.norm().item()
    cos = _cosine(fused_out, ref_out)
    rel = (fused_out - ref_out).norm().item() / (ref_norm + 1e-6)
    print(f"[ref-compare] backend={moe_backend} fused_norm={fused_norm:.4g} "
          f"ref_norm={ref_norm:.4g} cosine={cos:.4f} rel_l2={rel:.4f}")

    assert 0 < ref_norm < 1e6, (
        f"bf16 reference norm={ref_norm:.4g} is implausible; reference wiring "
        f"is off, comparison inconclusive.")
    assert torch.isfinite(fused_out).all(), (
        f"{moe_backend} NVFP4 fused output is non-finite.")
    assert cos > 0.9, (
        f"{moe_backend} NVFP4 fused output diverges from the bf16 dequant "
        f"reference: cosine={cos:.4f} rel_l2={rel:.4f} "
        f"(fused_norm={fused_norm:.4g}, ref_norm={ref_norm:.4g}). The routed "
        f"expert path is numerically wrong for this checkpoint.")
