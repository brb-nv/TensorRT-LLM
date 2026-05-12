# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent single-dispatch prefill megakernel for TinyLlama-1.1B on L40S.

Public API
----------
- `pack_weights(state_dict) -> PackedWeights`: HF state_dict -> contiguous BF16 blob
  in the megakernel's expected layout.
- `prefill(input_ids, packed) -> logits`: run the megakernel. Returns the full
  (seq_len, vocab) logits tensor; the last row is what beam search consumes.
- `TinyLlamaReference`: pure-PyTorch oracle (HF-eager-equivalent) that emits
  per-stage hidden states for the numerics harness.
- `gemm_pipeline(A, B) -> C`: the M3 standalone GEMM primitive, exposed for
  unit-testing the cp.async + mma.sync pipeline in isolation.

The kernel is hard-coded for TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden=2048,
GQA 32:4, FFN=5632, vocab=32000) and assumes BF16 weights + activations.
"""

import torch

from .pack_weights import PackedWeights, TINYLLAMA_CONFIG, pack_weights
from .reference import TinyLlamaReference, StageDump

# `prefill`, `gemm_pipeline`, and the per-stage test ops come from the C++
# extension. Import lazily so the Python side is importable even before the
# extension has been built (useful for the reference forward + numerics tests
# during M1/M2).
try:
    from ._C import prefill as _prefill_raw
    from ._C import gemm_pipeline  # noqa: F401
    from ._C import rms_norm        # noqa: F401
    from ._C import rope            # noqa: F401
    from ._C import attention       # noqa: F401
    from ._C import silu_mul        # noqa: F401
    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False

    def _prefill_raw(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "lucebox_tinyllama._C is not built. Run `pip install -e . --no-build-isolation` "
            "from the repository root on a node with an L40S (or other sm_89/86/90) GPU.")

    def _not_built(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "lucebox_tinyllama._C is not built. Run `pip install -e . --no-build-isolation`.")

    gemm_pipeline = _not_built  # type: ignore[assignment]
    rms_norm = _not_built       # type: ignore[assignment]
    rope = _not_built           # type: ignore[assignment]
    attention = _not_built      # type: ignore[assignment]
    silu_mul = _not_built       # type: ignore[assignment]


def _layer_offsets_tensor(pw: PackedWeights) -> torch.Tensor:
    flat = []
    for off in pw.layer_offsets:
        flat.extend([off.input_layernorm, off.qkv_proj, off.o_proj,
                     off.post_attn_norm, off.gate_up, off.down])
    return torch.tensor(flat, dtype=torch.int32, device=pw.blob.device)


def prefill(input_ids: torch.Tensor, pw: "PackedWeights | torch.Tensor",
            layer_offsets: torch.Tensor | None = None,
            embed_offset: int | None = None,
            final_norm_offset: int | None = None,
            lm_head_offset: int | None = None,
            seq_len: int | None = None,
            layer_residual_dump: torch.Tensor | None = None) -> torch.Tensor:
    """Run the TinyLlama-1.1B prefill megakernel.

    Two calling conventions:

      1. High-level (recommended):
            logits = prefill(input_ids, packed_weights)
         where `packed_weights` is a `PackedWeights` from `pack_weights(state_dict)`.

      2. Low-level (matches the raw C++ binding, used by `tests/test_numerics.py`):
            logits = prefill(input_ids, blob, layer_offsets,
                             embed_off, final_norm_off, lm_head_off, seq_len)

    If `layer_residual_dump` is provided, the kernel writes the residual stream
    after each layer L into `layer_residual_dump[L]`; shape must be
    `(num_layers, seq_len, hidden)`, BF16, CUDA. The residual stream is BF16 to
    match HF Llama eager. Used by the per-layer numerics diff harness.

    Returns logits of shape `(seq_len, vocab_size)`, BF16.
    """
    if isinstance(pw, PackedWeights):
        offs = _layer_offsets_tensor(pw)
        return _prefill_raw(input_ids, pw.blob, offs,
                            pw.embed_tokens_offset, pw.final_norm_offset, pw.lm_head_offset,
                            seq_len or input_ids.shape[0],
                            layer_residual_dump)
    return _prefill_raw(input_ids, pw, layer_offsets,
                         embed_offset, final_norm_offset, lm_head_offset,
                         seq_len or input_ids.shape[0],
                         layer_residual_dump)


__all__ = [
    "PackedWeights",
    "TINYLLAMA_CONFIG",
    "pack_weights",
    "TinyLlamaReference",
    "StageDump",
    "prefill",
    "gemm_pipeline",
    "rms_norm",
    "rope",
    "attention",
    "silu_mul",
]
