# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for MoE LoRA shared-outer detection and kernel flags.

These tests cover the weight-based shared-side detection and flag plumbing the
HF LoRA loader uses, without any GPU or model-engine dependencies.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import MOE_LORA_MODULES
from tensorrt_llm.moe_lora_shared import (
    KERNEL_FLAG_KEYS,
    MODULE_TO_KERNEL_PREFIX,
    all_false_flags,
    detect_shared_sides_from_stacked,
    merge_moe_shared_flags_for_batch,
    shared_sides_to_kernel_flags,
)


def test_module_names_align_with_moe_layout():
    assert set(MODULE_TO_KERNEL_PREFIX) == set(MOE_LORA_MODULES)


def test_all_false_flags_has_six_keys():
    flags = all_false_flags()
    assert set(flags.keys()) == set(KERNEL_FLAG_KEYS)
    assert not any(flags.values())


# ---------- detect_shared_sides_from_stacked ----------


def _shared_stack(num_experts: int, *shape: int) -> torch.Tensor:
    """A stack whose expert slices are all identical."""
    one = torch.randn(*shape)
    return one.unsqueeze(0).expand(num_experts, *([-1] * len(shape))).contiguous()


def test_detect_per_expert_is_not_shared():
    torch.manual_seed(0)
    a = torch.randn(4, 8, 16)
    b = torch.randn(4, 32, 8)
    assert detect_shared_sides_from_stacked(a, b) == (False, False)


def test_detect_shared_a():
    torch.manual_seed(0)
    a = _shared_stack(4, 8, 16)
    b = torch.randn(4, 32, 8)
    assert detect_shared_sides_from_stacked(a, b) == (True, False)


def test_detect_shared_b():
    torch.manual_seed(0)
    a = torch.randn(3, 8, 16)
    b = _shared_stack(3, 32, 8)
    assert detect_shared_sides_from_stacked(a, b) == (False, True)


def test_detect_single_expert_is_not_shared():
    a = torch.randn(1, 8, 16)
    b = torch.randn(1, 32, 8)
    assert detect_shared_sides_from_stacked(a, b) == (False, False)


# ---------- shared_sides_to_kernel_flags ----------


def test_shared_sides_to_kernel_flags_empty():
    assert shared_sides_to_kernel_flags({}) == all_false_flags()


def test_shared_sides_to_kernel_flags_canonical():
    detected = {
        "moe_h_to_4h": (True, False),
        "moe_gate": (True, False),
        "moe_4h_to_h": (False, True),
    }
    assert shared_sides_to_kernel_flags(detected) == {
        "fc1_shared_a": True,
        "fc1_shared_b": False,
        "fc2_shared_a": False,
        "fc2_shared_b": True,
        "gated_shared_a": True,
        "gated_shared_b": False,
    }


def test_shared_sides_to_kernel_flags_ignores_unknown_module():
    assert shared_sides_to_kernel_flags({"attn_q": (True, True)}) == all_false_flags()


# ---------- merge_moe_shared_flags_for_batch ----------


def test_merge_returns_none_for_no_uids():
    assert merge_moe_shared_flags_for_batch([], lambda uid: all_false_flags()) is None


def test_merge_returns_none_when_all_flags_false():
    flags = all_false_flags()
    assert merge_moe_shared_flags_for_batch(["a"], lambda uid: flags) is None


def test_merge_single_uid_with_shared_flags():
    expected = all_false_flags()
    expected["gated_shared_a"] = True
    assert merge_moe_shared_flags_for_batch(["uid-1"], lambda uid: expected) == expected


def test_merge_multiple_uids_matching_flags():
    expected = all_false_flags()
    expected["fc2_shared_b"] = True
    assert merge_moe_shared_flags_for_batch(["a", "b"], lambda uid: expected) == expected


def test_merge_multiple_uids_raises_on_mismatch():
    flags_a = all_false_flags()
    flags_a["fc2_shared_b"] = True
    flags_b = all_false_flags()
    with pytest.raises(ValueError, match="must match across all adapters"):
        merge_moe_shared_flags_for_batch(
            ["a", "b"],
            lambda uid: flags_a if uid == "a" else flags_b,
        )
