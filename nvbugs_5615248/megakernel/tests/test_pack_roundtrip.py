# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack -> unpack roundtrip test for M1.

Validates that `pack_weights()` lays out an HF state_dict in the megakernel's
expected K-major orientation by reading slices of the blob and comparing
against the original tensors.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama.pack_weights import TINYLLAMA_CONFIG, pack_weights


def _synthetic_state_dict(seed: int = 0) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = TINYLLAMA_CONFIG
    sd = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size,
                                                 generator=g, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(cfg.hidden_size, generator=g, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.hidden_size,
                                       generator=g, dtype=torch.bfloat16),
    }
    for L in range(cfg.num_layers):
        p = f"model.layers.{L}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(cfg.hidden_size, generator=g,
                                                         dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(cfg.q_size, cfg.hidden_size,
                                                          generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(cfg.kv_size, cfg.hidden_size,
                                                          generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(cfg.kv_size, cfg.hidden_size,
                                                          generator=g, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(cfg.hidden_size, cfg.q_size,
                                                          generator=g, dtype=torch.bfloat16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(cfg.hidden_size, generator=g,
                                                                   dtype=torch.bfloat16)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                       generator=g, dtype=torch.bfloat16)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                     generator=g, dtype=torch.bfloat16)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(cfg.hidden_size, cfg.intermediate_size,
                                                       generator=g, dtype=torch.bfloat16)
    return sd


def _slice(blob: torch.Tensor, offset: int, shape: tuple) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return blob.narrow(0, offset, n).view(*shape)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_pack_roundtrip_embed_and_norm():
    sd = _synthetic_state_dict()
    pw = pack_weights(sd, device=DEVICE)
    cfg = pw.cfg

    embed = _slice(pw.blob, pw.embed_tokens_offset, (cfg.vocab_size, cfg.hidden_size))
    torch.testing.assert_close(embed.cpu(), sd["model.embed_tokens.weight"], rtol=0, atol=0)

    fn = _slice(pw.blob, pw.final_norm_offset, (cfg.hidden_size,))
    torch.testing.assert_close(fn.cpu(), sd["model.norm.weight"], rtol=0, atol=0)


def test_pack_roundtrip_layer0_qkv_and_gate_up():
    sd = _synthetic_state_dict()
    pw = pack_weights(sd, device=DEVICE)
    cfg = pw.cfg

    # QKV block: stored as [H, Q+2KV] (i.e. transposed and concatenated)
    qkv = _slice(pw.blob, pw.layer_offsets[0].qkv_proj, (cfg.hidden_size, cfg.qkv_size))
    q = qkv[:, :cfg.q_size]                                 # (H, Q)
    k = qkv[:, cfg.q_size:cfg.q_size + cfg.kv_size]        # (H, KV)
    v = qkv[:, cfg.q_size + cfg.kv_size:]                  # (H, KV)
    # HF originals are [out, in] = [Q, H], so we must transpose for compare.
    torch.testing.assert_close(q.cpu(), sd["model.layers.0.self_attn.q_proj.weight"].t().contiguous(),
                                rtol=0, atol=0)
    torch.testing.assert_close(k.cpu(), sd["model.layers.0.self_attn.k_proj.weight"].t().contiguous(),
                                rtol=0, atol=0)
    torch.testing.assert_close(v.cpu(), sd["model.layers.0.self_attn.v_proj.weight"].t().contiguous(),
                                rtol=0, atol=0)

    gu = _slice(pw.blob, pw.layer_offsets[0].gate_up, (cfg.hidden_size, cfg.gate_up_size))
    gate = gu[:, :cfg.intermediate_size]
    up = gu[:, cfg.intermediate_size:]
    torch.testing.assert_close(gate.cpu(), sd["model.layers.0.mlp.gate_proj.weight"].t().contiguous(),
                                rtol=0, atol=0)
    torch.testing.assert_close(up.cpu(), sd["model.layers.0.mlp.up_proj.weight"].t().contiguous(),
                                rtol=0, atol=0)


def test_pack_sizes_match_expectations():
    sd = _synthetic_state_dict()
    pw = pack_weights(sd, device=DEVICE)
    n_bytes = pw.n_bytes()
    # TinyLlama-1.1B is ~2.2 GB BF16 plus padding from offset alignment.
    assert 2.0e9 < n_bytes < 2.6e9, f"unexpected packed size {n_bytes / 1e9:.2f} GB"
