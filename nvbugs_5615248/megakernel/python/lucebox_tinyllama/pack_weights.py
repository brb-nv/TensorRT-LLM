# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack a HuggingFace TinyLlama-1.1B state_dict into the megakernel weight blob.

The kernel expects ONE contiguous BF16 tensor with a fixed offset layout. The
offsets are also exposed as a `PackedWeights` namedtuple so Python can index
into sub-tensors for unit tests.

Memory layout (BF16, K-major i.e. the cuBLAS-default `[K, N]` orientation that
the megakernel's `cp.async` tiling assumes; values in elements):

    embed_tokens         [vocab,   hidden]              vocab x hidden
    for L in [0..22):
      input_layernorm    [hidden]                       hidden
      qkv_proj_weight    [hidden, q_size+2*kv_size]     hidden x 2560
      o_proj_weight      [hidden, hidden]               hidden x 2048
      post_attn_norm     [hidden]                       hidden
      gate_up_weight     [hidden, 2*intermediate]       hidden x 11264
      down_weight        [intermediate, hidden]         5632 x 2048
    model.norm.weight    [hidden]                       hidden
    lm_head              [hidden, vocab]                hidden x 32000

Each weight is contiguous. Q, K, V are concatenated along the N (output) axis
to give a single 2560-column matrix that's loaded by one fused GEMM. Similarly
for gate/up -> 11264 columns. All linear weights are stored in `[K, N]`
orientation (PyTorch stores `nn.Linear.weight` as `[out, in]` so we **transpose
once on pack**). RoPE cos/sin tables are NOT in the blob -- the kernel
recomputes them from `position_id` (cheap).

This packing matches the natural orientation the GEMM kernel reads:
`A` is loaded as `[M, K]` (activations, with M=seq_len, K=in_features) and
`B` is loaded as `[K, N]` (the packed weight). The output is `[M, N]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple

import torch


@dataclass(frozen=True)
class TinyLlamaConfig:
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64
    num_layers: int = 22
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    max_position_embeddings: int = 2048

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim  # 2048

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim  # 256

    @property
    def qkv_size(self) -> int:
        return self.q_size + 2 * self.kv_size  # 2560

    @property
    def gate_up_size(self) -> int:
        return 2 * self.intermediate_size  # 11264


TINYLLAMA_CONFIG = TinyLlamaConfig()


class LayerOffsets(NamedTuple):
    input_layernorm: int
    qkv_proj: int
    o_proj: int
    post_attn_norm: int
    gate_up: int
    down: int


@dataclass
class PackedWeights:
    """One large BF16 blob plus the offset table the kernel needs."""
    blob: torch.Tensor                       # 1-D BF16 tensor on CUDA
    embed_tokens_offset: int                 # element offset
    layer_offsets: List[LayerOffsets] = field(default_factory=list)
    final_norm_offset: int = 0
    lm_head_offset: int = 0
    cfg: TinyLlamaConfig = TINYLLAMA_CONFIG

    def n_elements(self) -> int:
        return self.blob.numel()

    def n_bytes(self) -> int:
        return self.blob.numel() * self.blob.element_size()


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pack_weights(state_dict: Dict[str, torch.Tensor],
                 cfg: TinyLlamaConfig = TINYLLAMA_CONFIG,
                 *,
                 device: str | torch.device = "cuda",
                 align_elems: int = 128) -> PackedWeights:
    """Pack a HuggingFace TinyLlama state_dict into the megakernel blob.

    Args:
        state_dict: HF-style flat dict. Keys we read:
            model.embed_tokens.weight                                  [V, H]
            model.layers.{L}.input_layernorm.weight                    [H]
            model.layers.{L}.self_attn.{q,k,v}_proj.weight             [{Q,KV,KV}, H]
            model.layers.{L}.self_attn.o_proj.weight                   [H, Q]
            model.layers.{L}.post_attention_layernorm.weight           [H]
            model.layers.{L}.mlp.{gate,up,down}_proj.weight            [I, H] / [I, H] / [H, I]
            model.norm.weight                                          [H]
            lm_head.weight                                             [V, H]
        device: target device for the blob.
        align_elems: each per-tensor offset is rounded up to this elem boundary
            (128 BF16 elems = 256 B, which matches the kernel's 16-B vector
            `cp.async` width and gives natural alignment for `mma.sync` rows).

    Returns:
        `PackedWeights` whose `.blob` is one big contiguous CUDA BF16 tensor.
    """
    device = torch.device(device)

    # ---- compute offsets ----
    offset = 0
    embed_offset = offset
    offset += cfg.vocab_size * cfg.hidden_size
    offset = _round_up(offset, align_elems)

    layer_offsets: List[LayerOffsets] = []
    for _ in range(cfg.num_layers):
        ln1 = offset; offset += cfg.hidden_size; offset = _round_up(offset, align_elems)
        qkv = offset; offset += cfg.hidden_size * cfg.qkv_size; offset = _round_up(offset, align_elems)
        op = offset; offset += cfg.hidden_size * cfg.hidden_size; offset = _round_up(offset, align_elems)
        ln2 = offset; offset += cfg.hidden_size; offset = _round_up(offset, align_elems)
        gu = offset; offset += cfg.hidden_size * cfg.gate_up_size; offset = _round_up(offset, align_elems)
        dn = offset; offset += cfg.intermediate_size * cfg.hidden_size; offset = _round_up(offset, align_elems)
        layer_offsets.append(LayerOffsets(ln1, qkv, op, ln2, gu, dn))

    final_norm_offset = offset
    offset += cfg.hidden_size
    offset = _round_up(offset, align_elems)

    lm_head_offset = offset
    offset += cfg.hidden_size * cfg.vocab_size
    offset = _round_up(offset, align_elems)

    total = offset
    blob = torch.empty(total, dtype=torch.bfloat16, device=device)

    # ---- copy in ----
    def _copy(src: torch.Tensor, dst_offset: int, expected_numel: int):
        src = src.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        assert src.numel() == expected_numel, (
            f"size mismatch at offset {dst_offset}: got {src.numel()} expected {expected_numel}")
        blob.narrow(0, dst_offset, expected_numel).copy_(src.view(-1))

    _copy(state_dict["model.embed_tokens.weight"], embed_offset,
          cfg.vocab_size * cfg.hidden_size)

    for L in range(cfg.num_layers):
        prefix = f"model.layers.{L}"
        off = layer_offsets[L]

        _copy(state_dict[f"{prefix}.input_layernorm.weight"], off.input_layernorm, cfg.hidden_size)

        # QKV pack: HF stores each as [out, in]. We want a single [in, out_qkv] block
        # so the GEMM reads activations [M, in] x weights [in, out_qkv]. So:
        #   1. transpose each from [out, in] -> [in, out]
        #   2. concatenate along the N (out) axis to [in, q + 2*kv]
        wq = state_dict[f"{prefix}.self_attn.q_proj.weight"].t().contiguous()  # [H, Q]
        wk = state_dict[f"{prefix}.self_attn.k_proj.weight"].t().contiguous()  # [H, KV]
        wv = state_dict[f"{prefix}.self_attn.v_proj.weight"].t().contiguous()  # [H, KV]
        wqkv = torch.cat([wq, wk, wv], dim=1).contiguous()                     # [H, Q+2KV]
        _copy(wqkv, off.qkv_proj, cfg.hidden_size * cfg.qkv_size)

        # O proj: HF [out=H, in=Q] -> [Q=H, H]. Q==H for TinyLlama (32*64=2048).
        wo = state_dict[f"{prefix}.self_attn.o_proj.weight"].t().contiguous()  # [H, H]
        _copy(wo, off.o_proj, cfg.hidden_size * cfg.hidden_size)

        _copy(state_dict[f"{prefix}.post_attention_layernorm.weight"], off.post_attn_norm,
              cfg.hidden_size)

        # gate_up pack: same trick as qkv. HF gate/up are [I, H] -> [H, I] each, cat to [H, 2I].
        wg = state_dict[f"{prefix}.mlp.gate_proj.weight"].t().contiguous()  # [H, I]
        wu = state_dict[f"{prefix}.mlp.up_proj.weight"].t().contiguous()    # [H, I]
        wgu = torch.cat([wg, wu], dim=1).contiguous()                       # [H, 2I]
        _copy(wgu, off.gate_up, cfg.hidden_size * cfg.gate_up_size)

        # down: HF [out=H, in=I] -> [I, H]
        wd = state_dict[f"{prefix}.mlp.down_proj.weight"].t().contiguous()  # [I, H]
        _copy(wd, off.down, cfg.intermediate_size * cfg.hidden_size)

    _copy(state_dict["model.norm.weight"], final_norm_offset, cfg.hidden_size)

    lm_head = state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"])
    lm_head = lm_head.t().contiguous()  # [H, V]
    _copy(lm_head, lm_head_offset, cfg.hidden_size * cfg.vocab_size)

    return PackedWeights(
        blob=blob,
        embed_tokens_offset=embed_offset,
        layer_offsets=layer_offsets,
        final_norm_offset=final_norm_offset,
        lm_head_offset=lm_head_offset,
        cfg=cfg,
    )
