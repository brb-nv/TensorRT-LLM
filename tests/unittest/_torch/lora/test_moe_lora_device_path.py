# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the capture-safe *device path* of routed-expert MoE LoRA in
`torch.ops.trtllm.fused_moe`.

The device path (selected by the slot-indexed input schema, or by
`TLLM_MOE_LORA_USE_DEVICE_PATH=1` for the per-request schema) performs the
per-token pointer expansion, problem building, and grouped GEMMs entirely on
the CUDA stream, so it is safe to record into a CUDA graph. These tests cover
the surface the eager-only tests in `test_moe_lora_op.py` cannot reach:

  1. Device-path eager correctness vs. the legacy host path and an fp32
     PyTorch reference (exercises the new on-device kernels).
  2. CUDA-graph capture + replay of the device path (slot-indexed schema),
     verified against the eager result.
  3. Multi-adapter routing within a single capture (token_to_slot mixing two
     adapter slots).
  4. Slot reassignment under replay: the slot -> per-token expansion runs
     on-device fed by captured H2D copies of the stable pinned slot tables, so
     reassigning a slot's adapter in place is reflected on replay WITHOUT
     re-capture (mirroring attention LoRA and the normal decode loop).

They require a CUDA GPU and the built `trtllm::fused_moe` op.
"""

import pytest
import torch

from tensorrt_llm._torch.peft.lora.moe_layout import make_per_expert_lora, reference_swiglu_moe_lora

_TRTLLM_AVAILABLE = hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_moe")

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_AVAILABLE,
    reason="Requires CUDA and built TensorRT-LLM C++ extension (torch.ops.trtllm.fused_moe).",
)


@pytest.fixture(autouse=True)
def _isolate_moe_runner_cache():
    """Give every test a fresh cached FusedMoeRunner and release captured graphs
    / device scratch afterward.

    The MoE LoRA device path keeps persistent per-runner scratch (slot tables,
    pointer arrays, low-rank workspace) on the module-level MoERunner cache, and
    these tests leak the CUDA graphs they capture. Without isolation, a test that
    grows the cached runner's slot tables (e.g. max_lora_size 1 -> 2) while an
    earlier test's captured graph still references the old device scratch
    corrupts that scratch, producing an illegal memory access in the next
    forward (see TRTLLM-12507). Clearing the cache before each test forces a
    fresh runner; clearing + empty_cache afterward releases the leaked graph
    pools so they cannot alias later allocations.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    MoERunner.runner_dict.clear()
    yield
    MoERunner.runner_dict.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# Adapters drawn from N(0, 1) blow up the SwiGLU intermediate at these shapes;
# scale them down so the legitimate output stays O(1)-O(10) and the bf16 noise
# stays well under the tolerance (see the rationale in test_moe_lora_op.py).
_LORA_SCALE = 0.25
_RTOL = 5e-2
_ATOL = 1.0


def _build_base_inputs(
    num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device, seed=0
):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w3_w1 = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device) * 0.02
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1)
    return x, w3_w1, w2, topk_ids.to(torch.int32), topk_scores.to(torch.float32)


def _make_adapter_set(num_experts, rank, hidden_size, inter_size, dtype, device, base_seed):
    """Three scaled per-expert adapters (fc1/gate-side, gated/up-side, fc2)."""

    def _scaled(*args, seed):
        a = make_per_expert_lora(*args, dtype=dtype, device=device, seed=seed)
        a["A"].mul_(_LORA_SCALE)
        a["B"].mul_(_LORA_SCALE)
        return a

    fc1 = _scaled(num_experts, rank, hidden_size, inter_size, seed=base_seed + 0)
    gated = _scaled(num_experts, rank, hidden_size, inter_size, seed=base_seed + 1)
    fc2 = _scaled(num_experts, rank, inter_size, hidden_size, seed=base_seed + 2)
    return {"fc1": fc1, "gated": gated, "fc2": fc2}


def _per_request_kwargs(num_tokens, adapters, rank):
    """Single-request per-request schema covering all tokens with one adapter."""
    fc1, gated, fc2 = adapters["fc1"], adapters["gated"], adapters["fc2"]
    return dict(
        fc1_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        fc1_lora_weight_ptrs=torch.tensor(
            [[fc1["A"].data_ptr(), fc1["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        fc2_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        fc2_lora_weight_ptrs=torch.tensor(
            [[fc2["A"].data_ptr(), fc2["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        gated_lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cpu"),
        gated_lora_weight_ptrs=torch.tensor(
            [[gated["A"].data_ptr(), gated["B"].data_ptr(), 0]], dtype=torch.int64, device="cpu"
        ),
        host_request_types=torch.zeros(1, dtype=torch.int32, device="cpu"),
        host_context_lengths=torch.tensor([num_tokens], dtype=torch.int32, device="cpu"),
        lora_max_low_rank=rank,
    )


def _slot_kwargs(token_to_slot, adapter_sets, rank):
    """Slot-indexed schema. `adapter_sets` is a list (one per slot) of adapter
    dicts; `token_to_slot` maps each token to a slot index. Pinned host tensors
    so the op can dereference them and the addresses stay stable across capture.
    """
    max_lora_size = len(adapter_sets)

    def _ptrs(key):
        return torch.tensor(
            [[a[key]["A"].data_ptr(), a[key]["B"].data_ptr(), 0] for a in adapter_sets],
            dtype=torch.int64,
            device="cpu",
        ).pin_memory()

    ranks = torch.full((max_lora_size,), rank, dtype=torch.int32, device="cpu").pin_memory()
    return dict(
        fc1_slot_lora_ranks=ranks,
        fc1_slot_lora_weight_ptrs=_ptrs("fc1"),
        fc2_slot_lora_ranks=ranks,
        fc2_slot_lora_weight_ptrs=_ptrs("fc2"),
        gated_slot_lora_ranks=ranks,
        gated_slot_lora_weight_ptrs=_ptrs("gated"),
        token_to_slot=token_to_slot.to(torch.int32).cpu().pin_memory(),
        lora_max_low_rank=rank,
    )


def _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, output_dtype, lora_kwargs):
    common = dict(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_scores,
        fc1_expert_weights=w3_w1,
        fc1_expert_biases=None,
        fc2_expert_weights=w2,
        fc2_expert_biases=None,
        output_dtype=output_dtype,
        quant_scales=[],
    )
    common.update(lora_kwargs)
    return torch.ops.trtllm.fused_moe(**common)[0]


def _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters):
    return reference_swiglu_moe_lora(
        x,
        w3_w1,
        w2,
        topk_ids,
        topk_scores,
        fc1_a=adapters["fc1"]["A"],
        fc1_b=adapters["fc1"]["B"],
        gated_a=adapters["gated"]["A"],
        gated_b=adapters["gated"]["B"],
        fc2_a=adapters["fc2"]["A"],
        fc2_b=adapters["fc2"]["B"],
    )


def _reference_multi_slot(x, w3_w1, w2, topk_ids, topk_scores, adapter_sets, token_to_slot):
    """Per-token reference where token t uses adapter_sets[token_to_slot[t]].

    The LoRA delta and base MoE are fully per-token independent, so we evaluate
    the single-adapter reference once per slot over the whole input and gather
    each token's row from the reference computed with its slot's adapter.
    """
    per_slot = [_reference(x, w3_w1, w2, topk_ids, topk_scores, a) for a in adapter_sets]
    out = torch.empty_like(per_slot[0])
    for t in range(x.shape[0]):
        out[t] = per_slot[int(token_to_slot[t].item())][t]
    return out


@requires_cuda_and_op
def test_device_path_eager_matches_host_and_reference(monkeypatch):
    """Per-request schema on the device path (env-var opt-in) must match both
    the legacy host path and the fp32 PyTorch reference. Exercises the on-device
    pointer-expand / problem-builder / grouped-GEMM kernels in eager mode.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapters = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=300
    )
    lora_kwargs = _per_request_kwargs(num_tokens, adapters, rank)

    # Host path (device path env explicitly disabled), fresh runner.
    monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", "0")
    MoERunner.runner_dict.clear()
    try:
        out_host = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(lora_kwargs))
    finally:
        MoERunner.runner_dict.clear()

    # Device path (env opt-in), fresh runner.
    monkeypatch.setenv("TLLM_MOE_LORA_USE_DEVICE_PATH", "1")
    MoERunner.runner_dict.clear()
    try:
        out_device = _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(lora_kwargs))
    finally:
        MoERunner.runner_dict.clear()

    out_ref = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters)

    assert torch.isfinite(out_device).all()
    torch.testing.assert_close(out_device, out_ref, rtol=_RTOL, atol=_ATOL)
    # Host vs device path are different reduction orders but should agree
    # within the same bf16 tolerance.
    torch.testing.assert_close(out_device, out_host, rtol=_RTOL, atol=_ATOL)


def _warmup_and_capture(call_fn, warmup_iters=3):
    """Warm up `call_fn` (so workspaces/tactics are allocated and LoRA scratch
    is sized outside capture), then capture it into a CUDA graph. Returns
    (graph, captured_output)."""
    for _ in range(warmup_iters):
        call_fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call_fn()
    return graph, captured


@requires_cuda_and_op
def test_device_path_cuda_graph_replay_matches_eager():
    """Slot-indexed schema auto-selects the device path; capturing the op into a
    CUDA graph and replaying must reproduce the eager device-path output.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapters = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=400
    )
    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32)
    slot_kwargs = _slot_kwargs(token_to_slot, [adapters], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    out_eager = call().clone()

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_replay = captured.clone()

    out_ref = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapters)
    assert torch.isfinite(out_replay).all()
    # Replay reproduces the captured device-path computation bit-for-bit.
    torch.testing.assert_close(out_replay, out_eager, rtol=0, atol=0)
    torch.testing.assert_close(out_replay, out_ref, rtol=_RTOL, atol=_ATOL)


@requires_cuda_and_op
def test_device_path_cuda_graph_multi_adapter():
    """Two adapter slots routed per-token via token_to_slot, captured + replayed.
    Validates the device path threads each token's slot through the on-device
    pointer expansion correctly.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapter_a = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=500
    )
    adapter_b = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=600
    )
    # Even tokens -> slot 0 (adapter_a), odd tokens -> slot 1 (adapter_b).
    token_to_slot = (torch.arange(num_tokens) % 2).to(torch.int32)
    slot_kwargs = _slot_kwargs(token_to_slot, [adapter_a, adapter_b], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    out_replay = captured.clone()

    out_ref = _reference_multi_slot(
        x, w3_w1, w2, topk_ids, topk_scores, [adapter_a, adapter_b], token_to_slot
    )
    assert torch.isfinite(out_replay).all()
    torch.testing.assert_close(out_replay, out_ref, rtol=_RTOL, atol=_ATOL)


def _set_slot_ptrs_inplace(slot_kwargs, adapters, slot_index=0):
    """Overwrite one slot's (A, B) pointer rows for all three modules in place,
    preserving the pinned-tensor storage (and thus the data_ptr the captured
    H2D reads from)."""
    for kernel, mod in (("fc1", "fc1"), ("fc2", "fc2"), ("gated", "gated")):
        row = torch.tensor(
            [adapters[mod]["A"].data_ptr(), adapters[mod]["B"].data_ptr(), 0],
            dtype=torch.int64,
            device="cpu",
        )
        slot_kwargs[f"{kernel}_slot_lora_weight_ptrs"][slot_index].copy_(row)


@requires_cuda_and_op
def test_device_path_slot_reassignment_reflected_on_replay():
    """In-place slot reassignment must be reflected on replay WITHOUT recapture.

    The slot -> per-token expansion runs on-device (launchMoeLoraSlotExpand)
    fed by captured H2D copies of the stable pinned slot tables. Because both
    the copies and the expansion kernel are recorded into the graph, mutating a
    slot's adapter pointers in place (same pinned storage) and replaying picks
    up the new adapter -- mirroring attention LoRA, and the normal decode loop
    where requests/adapters change across steps under one captured graph.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 16, 128, 256
    num_experts, top_k, rank = 4, 2, 8

    x, w3_w1, w2, topk_ids, topk_scores = _build_base_inputs(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, device
    )
    adapter_a = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=700
    )
    adapter_b = _make_adapter_set(
        num_experts, rank, hidden_size, inter_size, dtype, device, base_seed=800
    )

    token_to_slot = torch.zeros(num_tokens, dtype=torch.int32)
    # One slot whose pointers we will reassign in place.
    slot_kwargs = _slot_kwargs(token_to_slot, [adapter_a], rank)

    def call():
        return _call_fused_moe(x, w3_w1, w2, topk_ids, topk_scores, dtype, dict(slot_kwargs))

    out_ref_a = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapter_a)
    out_ref_b = _reference(x, w3_w1, w2, topk_ids, topk_scores, adapter_b)
    # Sanity: the two adapters produce meaningfully different outputs.
    assert (out_ref_a.float() - out_ref_b.float()).abs().mean().item() > 1e-2

    graph, captured = _warmup_and_capture(call)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.clone(), out_ref_a, rtol=_RTOL, atol=_ATOL)

    # Reassign slot 0 to adapter_b *in place* (same pinned tensor storage), then
    # replay WITHOUT re-capturing. The captured H2D + device slot-expand re-run
    # on replay, so the output must now reflect adapter_b.
    _set_slot_ptrs_inplace(slot_kwargs, adapter_b, slot_index=0)

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.clone(), out_ref_b, rtol=_RTOL, atol=_ATOL)

    # And switching back is likewise reflected, confirming it is not a one-shot.
    _set_slot_ptrs_inplace(slot_kwargs, adapter_a, slot_index=0)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.clone(), out_ref_a, rtol=_RTOL, atol=_ATOL)
