import logging

import pytest
import torch

from tensorrt_llm._torch.peft.lora.adapter_slot_manager import AdapterSlotManager
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import (
    LoraLayer,
    LoraModuleType,
    filter_unsupported_lora_target_modules,
)


def test_cuda_graph_lora_params_handle_missing_peft_table():
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(1, 2))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=2, output_sizes=[16, 32])}
    params = CudaGraphLoraParams(
        max_batch_size=2, max_lora_size=2, max_rank=8, layer_info=layer_info
    )
    layer_params = params.layer_params[layer_key]

    layer_params.h_b_ptrs[:, 0] = torch.tensor([11, 22], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[:, 0] = torch.tensor([33, 44], dtype=torch.int64)
    layer_params.h_b_ptrs[:, 1] = torch.tensor([55, 66], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[:, 1] = torch.tensor([77, 88], dtype=torch.int64)
    params.slot_ranks_host[:] = torch.tensor([4, 7], dtype=torch.int32)

    params.update_weight_pointers(None, (123, None))

    assert params.slot_ranks_host.tolist() == [4, 0]
    assert layer_params.h_b_ptrs[:, 0].tolist() == [11, 22]
    assert layer_params.h_b_prime_ptrs[:, 0].tolist() == [33, 44]
    assert layer_params.h_b_ptrs[:, 1].tolist() == [0, 0]
    assert layer_params.h_b_prime_ptrs[:, 1].tolist() == [0, 0]


def test_adapter_slot_manager_handles_missing_peft_cache_manager():
    manager = AdapterSlotManager(max_num_adapters=2)
    manager.slot2task[0] = 123
    manager.task2slot[123] = 0

    manager.remove_evicted_slots_in_cpp(None)

    assert manager.get_slot_to_task_mapping() == (123, None)
    assert manager.task2slot[123] == 0


def _make_attention_only_model() -> torch.nn.Module:
    """Tiny model registering only the four attention LoRA targets."""
    model = torch.nn.Module()
    model.attn_lora = LoraLayer(
        [
            LoraModuleType.ATTENTION_Q,
            LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V,
            LoraModuleType.ATTENTION_DENSE,
        ],
        [16, 16, 16, 16],
    )
    return model


def test_filter_lora_target_modules_returns_supported_targets_unchanged():
    targets = ["attn_q", "attn_k", "attn_v", "attn_dense"]
    assert filter_unsupported_lora_target_modules(_make_attention_only_model(), targets) == targets


def test_filter_lora_target_modules_handles_empty_input():
    model = _make_attention_only_model()
    assert filter_unsupported_lora_target_modules(model, []) == []
    assert filter_unsupported_lora_target_modules(model, None) == []


def test_filter_lora_target_modules_drops_per_expert_moe_targets(caplog):
    caplog.set_level(logging.WARNING, logger="tensorrt_llm")
    result = filter_unsupported_lora_target_modules(
        _make_attention_only_model(),
        ["attn_q", "moe_h_to_4h", "moe_4h_to_h"],
    )
    assert result == ["attn_q"]

    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "Dropping LoRA target module(s)" in msg
    assert "moe_h_to_4h" in msg
    assert "moe_4h_to_h" in msg
    assert "attn_q" in msg


def test_filter_lora_target_modules_rejects_unknown_names():
    with pytest.raises(ValueError) as exc:
        filter_unsupported_lora_target_modules(
            _make_attention_only_model(), ["attn_q", "not_a_real_module"]
        )
    assert "not_a_real_module" in str(exc.value)


def test_filter_lora_target_modules_noops_when_no_lora_layers_registered():
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(4, 4)
    targets = ["attn_q", "moe_h_to_4h", "moe_4h_to_h"]
    assert filter_unsupported_lora_target_modules(model, targets) == targets
