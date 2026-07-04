# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for TRTLLMGenFusedMoE._build_moe_lora_bgmv_inputs (eager path).

Exercises the pointer-table construction, the token->sequence ``lora_ids``
expansion, and the uniform-rank guard without a GPU or the built C++ op. The
extraction places tensors on ``x.device`` (CPU here).
"""

import pytest
import torch

fused_moe_trtllm_gen = pytest.importorskip(
    "tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen")
lora_layer = pytest.importorskip("tensorrt_llm._torch.peft.lora.layer")

TRTLLMGenFusedMoE = fused_moe_trtllm_gen.TRTLLMGenFusedMoE
LoraModuleType = lora_layer.LoraModuleType

_GATE = int(LoraModuleType.MOE_H_TO_4H)
_UP = int(LoraModuleType.MOE_GATE)
_DOWN = int(LoraModuleType.MOE_4H_TO_H)


def _stub(layer_idx=0):
    b = TRTLLMGenFusedMoE.__new__(TRTLLMGenFusedMoE)
    b.layer_idx = layer_idx
    return b


def _module(adapter_sizes, a_ptrs, b_ptrs):
    """Build one module's eager lora_params entry (flat weight_pointers)."""
    num_seqs = len(adapter_sizes)
    wp = []
    for s in range(num_seqs):
        wp += [a_ptrs[s], b_ptrs[s], 0]
    return {
        "adapter_size": torch.tensor(adapter_sizes, dtype=torch.int32),
        "weight_pointers": torch.tensor(wp, dtype=torch.int64),
    }


def _lora_params(layer_idx, host_request_types, prompt_lens, gate, up, down):
    num_seqs = len(host_request_types)
    return {
        "num_seqs": num_seqs,
        "host_request_types": torch.tensor(host_request_types, dtype=torch.int32),
        "prompt_lens_cpu": torch.tensor(prompt_lens, dtype=torch.int32),
        layer_idx: {
            _GATE: gate,
            _UP: up,
            _DOWN: down,
        },
    }


def _build(layer_idx, lora_params, num_tokens):
    x = torch.zeros(num_tokens, 8)
    return TRTLLMGenFusedMoE._build_moe_lora_bgmv_inputs_eager(
        _stub(layer_idx), lora_params, x)


def test_eager_build_inputs_token_expansion_and_ptrs():
    # seq0: context, prompt_len=3, active rank 8. seq1: gen (1 token), inactive.
    gate = _module([8, 0], [0x1110, 0], [0x1111, 0])
    up = _module([8, 0], [0x3330, 0], [0x3331, 0])
    down = _module([8, 0], [0x2220, 0], [0x2221, 0])
    params = _lora_params(0, [0, 1], [3, 5], gate, up, down)

    out = _build(0, params, num_tokens=4)  # 3 (ctx) + 1 (gen)
    assert out is not None
    assert out["rank"] == 8
    assert out["scale"] == 1.0

    # token -> seq: [0,0,0] for the context seq, then -1 for the inactive gen seq.
    assert out["lora_ids"].tolist() == [0, 0, 0, -1]

    # w_ptr shapes: FC1 = [2 slices, num_seqs], FC2 = [1, num_seqs].
    assert tuple(out["fc1_w_ptr_a"].shape) == (2, 2)
    assert tuple(out["fc2_w_ptr_a"].shape) == (1, 2)
    # slice 0 = gate (moe_h_to_4h), slice 1 = up (moe_gate); column A/B per seq.
    assert out["fc1_w_ptr_a"][0].tolist() == [0x1110, 0]
    assert out["fc1_w_ptr_a"][1].tolist() == [0x3330, 0]
    assert out["fc1_w_ptr_b"][0].tolist() == [0x1111, 0]
    assert out["fc2_w_ptr_a"][0].tolist() == [0x2220, 0]
    assert out["fc2_w_ptr_b"][0].tolist() == [0x2221, 0]


def test_eager_build_inputs_no_active_adapter_returns_none():
    # No sequence has an adapter this step -> None (caller runs the normal path).
    gate = _module([0, 0], [0, 0], [0, 0])
    up = _module([0, 0], [0, 0], [0, 0])
    down = _module([0, 0], [0, 0], [0, 0])
    params = _lora_params(0, [1, 1], [1, 1], gate, up, down)
    assert _build(0, params, num_tokens=2) is None


def test_eager_build_inputs_varying_rank_raises():
    # Two active adapters with different ranks -> BGMV uniform-rank guard.
    gate = _module([8, 16], [0x10, 0x20], [0x11, 0x21])
    up = _module([8, 16], [0x30, 0x40], [0x31, 0x41])
    down = _module([8, 16], [0x50, 0x60], [0x51, 0x61])
    params = _lora_params(0, [1, 1], [1, 1], gate, up, down)
    with pytest.raises(NotImplementedError, match="uniform LoRA rank"):
        _build(0, params, num_tokens=2)


def test_eager_build_inputs_requires_all_three_modules():
    # Missing moe_gate (up) -> rejected for the SwiGLU FP8 path.
    gate = _module([8], [0x10], [0x11])
    down = _module([8], [0x50], [0x51])
    params = {
        "num_seqs": 1,
        "host_request_types": torch.tensor([1], dtype=torch.int32),
        "prompt_lens_cpu": torch.tensor([1], dtype=torch.int32),
        0: {
            _GATE: gate,
            _DOWN: down,
        },
    }
    with pytest.raises(NotImplementedError, match="moe_gate"):
        _build(0, params, num_tokens=1)
