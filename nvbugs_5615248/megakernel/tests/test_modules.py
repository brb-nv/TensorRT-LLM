# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-level unit tests for each megakernel stage vs the HuggingFace reference.

Each stage's `__device__` implementation in `csrc/stages.cuh` is fused into the
persistent megakernel for production but also exposed via a thin `__global__`
wrapper in `csrc/stage_tests.cu` for unit-testing. This file gates each stage
against HF's exact single-module reference (`LlamaRMSNorm`,
`apply_rotary_pos_emb`, `eager_attention_forward`, `F.silu(gate) * up`).

Tolerances are tight (~1-5 BF16 ULPs) because each test exercises one stage's
worth of rounding, with no 22-layer compounding. An algorithmic bug shows up
as a localized failure here -- the end-to-end test in `test_numerics.py` has
a 50x looser threshold to absorb the cross-library BF16-ULP noise that compounds
through the residual stream.

Skipped if `lucebox_tinyllama._C` is not built or `transformers` is missing.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama import _C_AVAILABLE, TINYLLAMA_CONFIG

if _C_AVAILABLE:
    from lucebox_tinyllama import rms_norm, rope, attention, silu_mul

try:
    import transformers  # noqa: F401
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_PYTEST_SKIPS = (
    pytest.mark.skipif(not _C_AVAILABLE, reason="lucebox_tinyllama._C not built"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _HF_AVAILABLE, reason="transformers (HF) not installed"),
)


def _apply_skips(fn):
    for m in _PYTEST_SKIPS:
        fn = m(fn)
    return fn


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _bf16_ulp(magnitude: float) -> float:
    """1 BF16 ULP at the given magnitude. BF16 has 7 mantissa bits."""
    return magnitude / 128.0


# --------------------------------------------------------------------------
# Stage: RMSNorm
# --------------------------------------------------------------------------
@_apply_skips
def test_rms_norm_matches_hf():
    """Kernel's `stage_rms_norm` vs `LlamaRMSNorm.forward`.

    Both must implement: y = weight_bf16 * float_to_bf16(x_fp32 * rsqrt(var+eps)).
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    torch.manual_seed(0)
    x = torch.randn(SEQ, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(cfg.hidden_size, dtype=torch.bfloat16, device="cuda")

    # Kernel
    y_kernel = rms_norm(x, w)

    # HF reference: build a one-off LlamaRMSNorm with our weight.
    norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_eps).to(device="cuda",
                                                              dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(w)
        y_ref = norm(x)

    max_abs = _max_abs(y_kernel, y_ref)
    mean_abs = _mean_abs(y_kernel, y_ref)
    mag = y_ref.abs().mean().item()
    print(f"rms_norm: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
          f"mag={mag:.3f} ulp={_bf16_ulp(mag):.4e}")

    # RMSNorm is a per-element op: y[i] depends only on x[i,:]. There is no
    # accumulation across rows, so the only drift is the rounding of the FP32
    # mean-of-squares reduction, which is identical between kernel and HF
    # (both reduce in FP32 over the same row of BF16 values, the ordering
    # differs slightly between warp-shuffle and torch's reduction).
    # Expect drift well below 1 ULP per element.
    assert max_abs < 2.0 * _bf16_ulp(mag), (
        f"rms_norm max_abs={max_abs} exceeds 2 BF16 ULPs at magnitude {mag}")
    assert mean_abs < 0.3 * _bf16_ulp(mag), (
        f"rms_norm mean_abs={mean_abs} exceeds 0.3 BF16 ULPs at magnitude {mag}")


# --------------------------------------------------------------------------
# Stage: RoPE
# --------------------------------------------------------------------------
def _pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack HF-shape Q/K/V into the kernel's per-row [Q | K | V] BF16 layout.

    Args:
        q: (seq_len, num_heads, head_dim) BF16
        k: (seq_len, num_kv_heads, head_dim) BF16
        v: (seq_len, num_kv_heads, head_dim) BF16
    Returns:
        (seq_len, q_size + 2*kv_size) BF16, contiguous on CUDA.
    """
    S = q.size(0)
    return torch.cat([q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1)],
                     dim=1).contiguous()


def _unpack_qk(qkv: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `_pack_qkv` for Q and K only (V is unchanged by RoPE)."""
    S = qkv.size(0)
    q = qkv[:, :cfg.q_size].reshape(S, cfg.num_heads, cfg.head_dim)
    k = qkv[:, cfg.q_size:cfg.q_size + cfg.kv_size].reshape(
        S, cfg.num_kv_heads, cfg.head_dim)
    return q, k


@_apply_skips
def test_rope_matches_hf():
    """Kernel's `stage_rope` vs HF `apply_rotary_pos_emb`.

    HF's rotary embedding computes cos/sin in FP32 then casts to BF16; the
    rotation `q_embed = (q * cos) + (rotate_half(q) * sin)` is then three
    BF16 ops per element. The kernel uses FP32 sincosf then casts cos/sin to
    BF16 to match the HF cast point and replicates the per-term BF16 cast.
    """
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    from transformers import LlamaConfig

    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    # HF needs a LlamaConfig to build the rotary embedding module.
    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    rotary = LlamaRotaryEmbedding(config=hf_cfg).to("cuda")

    torch.manual_seed(1)
    q = torch.randn(SEQ, cfg.num_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda")
    k = torch.randn(SEQ, cfg.num_kv_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda")
    v = torch.randn(SEQ, cfg.num_kv_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda")

    # Kernel path: pack -> rope -> unpack
    qkv_packed = _pack_qkv(q, k, v)
    qkv_rot = rope(qkv_packed)
    q_kernel, k_kernel = _unpack_qk(qkv_rot, cfg)

    # HF path: apply_rotary_pos_emb expects (B, H, S, D)
    pos_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
    cos, sin = rotary(q.unsqueeze(0).transpose(1, 2), pos_ids)
    q_hf = q.unsqueeze(0).transpose(1, 2)         # (1, H_q, S, D)
    k_hf = k.unsqueeze(0).transpose(1, 2)         # (1, H_kv, S, D)
    q_ref, k_ref = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    q_ref = q_ref.transpose(1, 2).squeeze(0)       # (S, H_q, D)
    k_ref = k_ref.transpose(1, 2).squeeze(0)       # (S, H_kv, D)

    q_max = _max_abs(q_kernel, q_ref)
    q_mean = _mean_abs(q_kernel, q_ref)
    k_max = _max_abs(k_kernel, k_ref)
    k_mean = _mean_abs(k_kernel, k_ref)
    mag = q_ref.abs().mean().item()
    ulp = _bf16_ulp(mag)
    print(f"rope: q_max={q_max:.4e} q_mean={q_mean:.4e} k_max={k_max:.4e} "
          f"k_mean={k_mean:.4e} mag={mag:.3f} ulp={ulp:.4e}")

    # RoPE is element-wise on Q,K with 3 BF16 casts per output element. The
    # remaining drift comes from FP32 sincosf vs HF's vectorized cos/sin --
    # different but both ~1 ULP precise. Allow up to 3 ULPs to absorb the
    # cos/sin precision difference plus the BF16 cast point alignment.
    assert q_max < 3.0 * ulp, f"rope Q max_abs={q_max} > 3 ULPs ({3*ulp})"
    assert k_max < 3.0 * ulp, f"rope K max_abs={k_max} > 3 ULPs ({3*ulp})"
    assert q_mean < 0.5 * ulp, f"rope Q mean_abs={q_mean} > 0.5 ULPs"
    assert k_mean < 0.5 * ulp, f"rope K mean_abs={k_mean} > 0.5 ULPs"


# --------------------------------------------------------------------------
# Stage: prefill attention
# --------------------------------------------------------------------------
@_apply_skips
def test_attention_matches_hf_eager():
    """Kernel's `stage_attention` vs HF `eager_attention_forward`.

    Both compute causal prefill attention with GQA (32 query heads, 4 KV heads,
    head_dim=64). The kernel performs the Q@K^T dot product in FP32 then casts
    to BF16 (matching HF's BF16 cublas matmul result), scales by 1/sqrt(D) in
    BF16, masks (causal), runs softmax in FP32, casts probabilities to BF16
    before the V matmul (matching HF's `softmax(...).to(bf16)`), and outputs BF16.
    """
    from transformers.models.llama.modeling_llama import eager_attention_forward

    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    torch.manual_seed(2)
    # Q,K,V already-rotated (the kernel's `stage_attention` does NOT do RoPE).
    q = torch.randn(SEQ, cfg.num_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(SEQ, cfg.num_kv_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda") * 0.5
    v = torch.randn(SEQ, cfg.num_kv_heads, cfg.head_dim,
                    dtype=torch.bfloat16, device="cuda") * 0.5

    # Kernel
    qkv_packed = _pack_qkv(q, k, v)
    out_kernel = attention(qkv_packed)   # (SEQ, kHidden=num_heads*head_dim)

    # HF eager: shapes are (B, H, S, D). Build a fake `module` shim that
    # `eager_attention_forward` reads `num_key_value_groups` from.
    class _Shim:
        num_key_value_groups = cfg.num_heads // cfg.num_kv_heads
        training = False

    q_hf = q.unsqueeze(0).transpose(1, 2)       # (1, H_q, S, D)
    k_hf = k.unsqueeze(0).transpose(1, 2)       # (1, H_kv, S, D)
    v_hf = v.unsqueeze(0).transpose(1, 2)       # (1, H_kv, S, D)
    # Causal mask: -inf above the diagonal, 0 on/below. HF expects
    # shape (B, 1, S, S) FP32 added to attn_weights.
    minus_inf = torch.finfo(torch.float32).min
    mask = torch.triu(torch.full((SEQ, SEQ), minus_inf, device="cuda"),
                      diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)        # (1, 1, S, S)
    scale = cfg.head_dim ** -0.5
    out_hf, _ = eager_attention_forward(
        _Shim(), q_hf, k_hf, v_hf, attention_mask=mask, scaling=scale,
        dropout=0.0)
    # eager_attention_forward returns (B, S, H, D) after its internal transpose.
    out_ref = out_hf.squeeze(0).reshape(SEQ, cfg.num_heads * cfg.head_dim)

    max_abs = _max_abs(out_kernel, out_ref)
    mean_abs = _mean_abs(out_kernel, out_ref)
    mag = out_ref.abs().mean().item()
    ulp = _bf16_ulp(mag)
    print(f"attention: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
          f"mag={mag:.3f} ulp={ulp:.4e}")

    # Attention has TWO matmuls (Q@K^T over K=64, then P@V over t=128) plus
    # softmax. Each matmul output is rounded to BF16 at a different point than
    # cublas does (different K-summation order). Expect up to 5 ULPs of drift
    # per output element, well-localized to attention.
    assert max_abs < 5.0 * ulp, f"attention max_abs={max_abs} > 5 ULPs ({5*ulp})"
    assert mean_abs < 0.8 * ulp, f"attention mean_abs={mean_abs} > 0.8 ULPs"


# --------------------------------------------------------------------------
# Stage: SwiGLU (silu(gate) * up)
# --------------------------------------------------------------------------
@_apply_skips
def test_silu_mul_matches_hf():
    """Kernel's `stage_silu_mul` vs HF MLP's `F.silu(gate) * up`.

    HF computes `act_fn(gate_proj(x)) * up_proj(x)` where `act_fn = F.silu`.
    Each multiply on BF16 produces BF16 (FP32 intermediate, BF16 cast).
    The kernel mirrors this by casting silu(gate) to BF16 before the multiply.
    """
    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    torch.manual_seed(3)
    gate = torch.randn(SEQ, cfg.intermediate_size,
                       dtype=torch.bfloat16, device="cuda")
    up = torch.randn(SEQ, cfg.intermediate_size,
                     dtype=torch.bfloat16, device="cuda")
    gate_up = torch.cat([gate, up], dim=1).contiguous()

    out_kernel = silu_mul(gate_up)
    # HF: F.silu(gate) is BF16-out (PyTorch upcasts internally for the exp);
    # then * up is another BF16 op.
    out_ref = F.silu(gate) * up

    max_abs = _max_abs(out_kernel, out_ref)
    mean_abs = _mean_abs(out_kernel, out_ref)
    mag = out_ref.abs().mean().item()
    ulp = _bf16_ulp(mag)
    print(f"silu_mul: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
          f"mag={mag:.3f} ulp={ulp:.4e}")

    # Per-element op. Two BF16 cast points per output (silu output, * up
    # output). Drift comes from FP32 silu approximation (we use FP32 expf;
    # HF uses torch's BF16 SiLU which upcasts to FP32 internally).
    assert max_abs < 2.0 * ulp, f"silu_mul max_abs={max_abs} > 2 ULPs ({2*ulp})"
    assert mean_abs < 0.3 * ulp, f"silu_mul mean_abs={mean_abs} > 0.3 ULPs"
