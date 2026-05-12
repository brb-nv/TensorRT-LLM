# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M5 end-to-end numerics test: megakernel logits match HF Llama eager.

Reference is HF `LlamaForCausalLM` with `attn_implementation="eager"` loaded
from synthetic random weights (no HF download needed). The kernel matches HF
eager at every BF16 cast point inside a layer (residual stream BF16; RMSNorm
casts normalized hidden to BF16 before the weight multiply; attention softmax
probabilities cast to BF16 before V; RoPE uses BF16-quantized cos/sin and
per-term BF16 casts; SiLU(gate) cast to BF16 before multiplying by up; o_proj
and down_proj run as plain GEMMs with explicit BF16+BF16 residual adds).

What's left -- the irreducible BF16-vs-BF16 cross-library noise floor:
- block_gemm uses a different K-summation order than cublasGemmEx (WMMA 16x8x16
  tile traversal vs cublas's larger CTA tiles). For BF16 inputs with FP32
  accumulator, both produce results that round to a final BF16 within ~1 BF16
  ULP of each other per element. At residual magnitude M, 1 ULP ~ M/128.
- attention's Q@K^T and V matmuls similarly differ in summation order.

Through 22 decoder layers the residual magnitude grows from ~0.5 to ~3.2 (with
std=0.02 random weights), and per-layer rounding noise compounds as roughly
sqrt(L) of the local ULP. Empirical drift after L21:
    L21 residual: mean_abs(kernel - HF) ~ 0.10 at magnitude 3.2 (~4 BF16 ULPs).
    final logits: max_abs ~ 0.18, mean_abs ~ 0.023, top-1 disagreement ~ 10%
                  on random prompts (much lower on real text where top-1/top-2
                  gaps are wider).

NORMS, NOT ELEMENTS, are the structural correctness check: the kernel and HF
agree on mean_abs(residual) to ~3e-4 relative through L21, confirming there
is no algorithmic bug. The per-element thresholds below are set generously
above the empirical noise so the test catches real regressions (10x+ drift)
without failing on random rounding.

Skipped if `lucebox_tinyllama._C` is not built or `transformers` is missing.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama import _C_AVAILABLE, TINYLLAMA_CONFIG, pack_weights
from lucebox_tinyllama.reference import TinyLlamaReference

if _C_AVAILABLE:
    from lucebox_tinyllama import prefill

try:
    import transformers  # noqa: F401
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def _make_synthetic_weights(seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = TINYLLAMA_CONFIG
    # Use small std so BF16 doesn't lose too much precision through 22 layers.
    std = 0.02
    sd = {
        "model.embed_tokens.weight": std * torch.randn(cfg.vocab_size, cfg.hidden_size,
                                                       generator=g, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(cfg.hidden_size, dtype=torch.bfloat16),
        "lm_head.weight": std * torch.randn(cfg.vocab_size, cfg.hidden_size,
                                             generator=g, dtype=torch.bfloat16),
    }
    for L in range(cfg.num_layers):
        p = f"model.layers.{L}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(cfg.hidden_size, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_proj.weight"] = std * torch.randn(cfg.q_size, cfg.hidden_size,
                                                                generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.k_proj.weight"] = std * torch.randn(cfg.kv_size, cfg.hidden_size,
                                                                generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.v_proj.weight"] = std * torch.randn(cfg.kv_size, cfg.hidden_size,
                                                                generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.o_proj.weight"] = std * torch.randn(cfg.hidden_size, cfg.q_size,
                                                                generator=g, dtype=torch.bfloat16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(cfg.hidden_size,
                                                                  dtype=torch.bfloat16)
        sd[f"{p}.mlp.gate_proj.weight"] = std * torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                             generator=g, dtype=torch.bfloat16)
        sd[f"{p}.mlp.up_proj.weight"] = std * torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                           generator=g, dtype=torch.bfloat16)
        sd[f"{p}.mlp.down_proj.weight"] = std * torch.randn(cfg.hidden_size, cfg.intermediate_size,
                                                             generator=g, dtype=torch.bfloat16)
    return sd


def _layer_offsets_tensor(pw):
    """Flatten the Python LayerOffsets list into the int32 tensor the kernel reads
    via the `LayerOffsets` struct (6 int32 fields per layer)."""
    flat = []
    for off in pw.layer_offsets:
        flat.extend([off.input_layernorm, off.qkv_proj, off.o_proj,
                     off.post_attn_norm, off.gate_up, off.down])
    return torch.tensor(flat, dtype=torch.int32, device=pw.blob.device)


@pytest.mark.skipif(not _C_AVAILABLE, reason="lucebox_tinyllama._C not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _HF_AVAILABLE, reason="transformers (HF) not installed")
def test_megakernel_matches_reference_random_weights():
    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    sd = _make_synthetic_weights(seed=42)
    pw = pack_weights(sd, device="cuda")
    layer_offsets = _layer_offsets_tensor(pw)

    ref = TinyLlamaReference(sd, device="cuda", dtype=torch.bfloat16, seq_len=SEQ)

    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (SEQ,), dtype=torch.int32, device="cuda")

    out_ref = ref.forward(ids).logits   # (SEQ, V)
    out_meg = prefill(ids, pw.blob, layer_offsets,
                      pw.embed_tokens_offset, pw.final_norm_offset, pw.lm_head_offset,
                      SEQ)               # (SEQ, V)

    diff = (out_ref.float() - out_meg.float()).abs()
    print(f"megakernel vs HF eager: max_abs={diff.max().item():.4f} "
          f"mean_abs={diff.mean().item():.5f}")
    # Element-wise drift thresholds (see module docstring): set ~3x above the
    # empirical BF16-vs-BF16 noise floor (mean ~0.023, max ~0.18) to catch real
    # regressions without flaking on random rounding.
    assert diff.mean().item() < 7e-2, "megakernel logits mean drift exceeds expected floor"
    assert diff.max().item() < 4e-1, "megakernel logits max drift exceeds expected floor"

    # Structural correctness: norm parity. Any algorithmic bug (wrong RoPE,
    # softmax, GEMM transpose, etc.) would create a *systematic* norm
    # divergence. Rounding noise is unbiased and preserves norm. We observe
    # ~3e-4 relative norm agreement empirically; 1e-2 catches real bugs.
    ref_norm = out_ref.float().norm().item()
    meg_norm = out_meg.float().norm().item()
    assert abs(ref_norm - meg_norm) / ref_norm < 1e-2, (
        f"logit norm divergence: ref={ref_norm:.3f} kernel={meg_norm:.3f}")


@pytest.mark.skipif(not _C_AVAILABLE, reason="lucebox_tinyllama._C not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _HF_AVAILABLE, reason="transformers (HF) not installed")
def test_megakernel_top1_matches_reference_50_prompts():
    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    sd = _make_synthetic_weights(seed=42)
    pw = pack_weights(sd, device="cuda")
    layer_offsets = _layer_offsets_tensor(pw)

    ref = TinyLlamaReference(sd, device="cuda", dtype=torch.bfloat16, seq_len=SEQ)

    rng = torch.Generator(device="cpu").manual_seed(0)
    mismatches = 0
    n_trials = 50
    for trial in range(n_trials):
        ids = torch.randint(0, cfg.vocab_size, (SEQ,), generator=rng,
                            dtype=torch.int32).cuda()
        out_ref = ref.forward(ids).logits[-1].argmax().item()
        out_meg = prefill(ids, pw.blob, layer_offsets,
                          pw.embed_tokens_offset, pw.final_norm_offset, pw.lm_head_offset,
                          SEQ)[-1].argmax().item()
        if out_ref != out_meg:
            mismatches += 1
    print(f"top-1 mismatches: {mismatches} / {n_trials}")
    # Empirical mismatch rate with HF eager reference on random prompts is
    # ~10%: random initializations have many near-degenerate top-1/top-2 pairs
    # that flip under the ~0.18 max-abs logit drift. Real prompts have wider
    # gaps and would see <1% disagreement. We cap at 30% (15/50) to detect
    # algorithmic regressions while passing on rounding-noise variation.
    assert mismatches <= 15, (
        f"{mismatches} top-1 disagreements out of {n_trials} prompts "
        f"(>30% suggests an algorithmic bug, not BF16 rounding noise)")
