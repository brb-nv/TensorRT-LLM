# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the `lora_layout.json` layout metadata parser.

These tests cover the parser surface that the HF LoRA loader uses to detect
routed-expert MoE shared-outer adapters, without any GPU or model-engine
dependencies.
"""

import json
import os
import tempfile

import pytest

from tensorrt_llm._torch.peft.lora.moe_layout import DEFAULT_SHARED_SIDE, MOE_LORA_MODULES
from tensorrt_llm.lora_layout_metadata import (
    KERNEL_FLAG_KEYS,
    LORA_LAYOUT_FILENAME,
    MODULE_TO_KERNEL_PREFIX,
    LoraLayoutError,
    all_false_flags,
    layout_to_kernel_flags,
    load_lora_layout_metadata,
    merge_moe_shared_flags_for_batch,
    parse_lora_layout,
)


def test_module_names_align_with_moe_layout():
    assert set(MODULE_TO_KERNEL_PREFIX) == set(MOE_LORA_MODULES)


# ---------- file-level loader ----------


def test_load_returns_none_when_layout_metadata_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert load_lora_layout_metadata(tmpdir) is None


def test_load_parses_minimal_valid_layout_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "version": 1,
            "moe_shared_outer": {
                "moe_h_to_4h": {"shared_side": "A"},
                "moe_4h_to_h": {"shared_side": "B"},
            },
        }
        with open(os.path.join(tmpdir, LORA_LAYOUT_FILENAME), "w") as f:
            json.dump(payload, f)
        layout = load_lora_layout_metadata(tmpdir)
    assert layout == {"moe_h_to_4h": "A", "moe_4h_to_h": "B"}


def test_load_raises_on_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, LORA_LAYOUT_FILENAME), "w") as f:
            f.write("{ this is not json")
        with pytest.raises(LoraLayoutError, match="Failed to parse"):
            load_lora_layout_metadata(tmpdir)


# ---------- parse_lora_layout schema validation ----------


def test_parse_rejects_non_dict_top_level():
    with pytest.raises(LoraLayoutError, match="must be a JSON object"):
        parse_lora_layout([], source="<test>")


def test_parse_rejects_unsupported_version():
    with pytest.raises(LoraLayoutError, match="unsupported lora_layout version"):
        parse_lora_layout({"version": 2}, source="<test>")


def test_parse_defaults_version_to_1():
    out = parse_lora_layout({"moe_shared_outer": {}}, source="<test>")
    assert out == {}


def test_parse_rejects_non_dict_moe_section():
    with pytest.raises(LoraLayoutError, match="'moe_shared_outer' must be a JSON object"):
        parse_lora_layout({"moe_shared_outer": []}, source="<test>")


def test_parse_rejects_unknown_module():
    with pytest.raises(LoraLayoutError, match="unknown MoE LoRA module"):
        parse_lora_layout(
            {"moe_shared_outer": {"attn_q": {"shared_side": "A"}}},
            source="<test>",
        )


def test_parse_rejects_invalid_shared_side():
    with pytest.raises(LoraLayoutError, match="must be 'A', 'B' or null"):
        parse_lora_layout(
            {"moe_shared_outer": {"moe_h_to_4h": {"shared_side": "C"}}},
            source="<test>",
        )


def test_parse_rejects_non_dict_module_spec():
    with pytest.raises(LoraLayoutError, match="must be a JSON object"):
        parse_lora_layout(
            {"moe_shared_outer": {"moe_h_to_4h": "A"}},
            source="<test>",
        )


def test_parse_accepts_null_shared_side():
    out = parse_lora_layout(
        {"moe_shared_outer": {"moe_h_to_4h": {"shared_side": None}}},
        source="<test>",
    )
    assert out == {"moe_h_to_4h": None}


def test_parse_tolerates_extra_top_level_keys():
    out = parse_lora_layout(
        {
            "version": 1,
            "moe_shared_outer": {"moe_gate": {"shared_side": "A"}},
            "future_field": {"anything": True},
        },
        source="<test>",
    )
    assert out == {"moe_gate": "A"}


# ---------- layout_to_kernel_flags ----------


def test_all_false_flags_has_six_keys():
    flags = all_false_flags()
    assert set(flags.keys()) == set(KERNEL_FLAG_KEYS)
    assert not any(flags.values())


def test_layout_to_kernel_flags_none():
    assert layout_to_kernel_flags(None) == all_false_flags()


def test_layout_to_kernel_flags_empty():
    assert layout_to_kernel_flags({}) == all_false_flags()


def test_layout_to_kernel_flags_default_shared_outer_pattern():
    flags = layout_to_kernel_flags(DEFAULT_SHARED_SIDE)
    assert flags == {
        "fc1_shared_a": True,
        "fc1_shared_b": False,
        "fc2_shared_a": False,
        "fc2_shared_b": True,
        "gated_shared_a": True,
        "gated_shared_b": False,
    }


def test_layout_to_kernel_flags_null_side_yields_no_flag():
    flags = layout_to_kernel_flags({"moe_h_to_4h": None})
    assert flags == all_false_flags()


def test_layout_to_kernel_flags_only_partial_modules():
    flags = layout_to_kernel_flags({"moe_4h_to_h": "B"})
    expected = all_false_flags()
    expected["fc2_shared_b"] = True
    assert flags == expected


# ---------- merge_moe_shared_flags_for_batch ----------


def test_merge_returns_none_for_no_uids():
    assert merge_moe_shared_flags_for_batch([], lambda uid: all_false_flags()) is None


def test_merge_returns_none_when_all_flags_false():
    flags = all_false_flags()
    assert merge_moe_shared_flags_for_batch(["a"], lambda uid: flags) is None


def test_merge_single_uid_with_shared_flags():
    expected = all_false_flags()
    expected["gated_shared_a"] = True
    result = merge_moe_shared_flags_for_batch(["uid-1"], lambda uid: expected)
    assert result == expected


def test_merge_multiple_uids_matching_flags():
    expected = layout_to_kernel_flags(DEFAULT_SHARED_SIDE)
    result = merge_moe_shared_flags_for_batch(
        ["a", "b"],
        lambda uid: expected,
    )
    assert result == expected


def test_merge_multiple_uids_raises_on_mismatch():
    flags_a = all_false_flags()
    flags_a["fc2_shared_b"] = True
    flags_b = all_false_flags()
    with pytest.raises(ValueError, match="must match across all adapters"):
        merge_moe_shared_flags_for_batch(
            ["a", "b"],
            lambda uid: flags_a if uid == "a" else flags_b,
        )
