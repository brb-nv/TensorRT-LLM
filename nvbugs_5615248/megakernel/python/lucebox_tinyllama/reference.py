# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyLlama-1.1B reference using HuggingFace `LlamaForCausalLM` directly.

This is the numerical oracle for the megakernel. We use HF's own implementation
(in eager attention mode) rather than a hand-rolled rewrite, so the test gates
on bit-perfect parity with HF -- not parity with an approximation of HF.

The reference exposes per-layer post-decoder-layer hidden states via a forward
hook, which `tests/debug_per_layer.py` consumes for divide-and-conquer diffs.

Why eager rather than SDPA: HF eager's attention path is
  attn_weights = softmax(q @ k.T * scale + mask, dim=-1, dtype=fp32).to(bf16)
  attn_output = attn_weights @ v
which has well-defined BF16 quantization at known points. SDPA backends (math,
mem-efficient, flash) each implement the same math differently and produce
different last-bit results; eager is the canonical reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .pack_weights import TINYLLAMA_CONFIG, TinyLlamaConfig


@dataclass
class StageDump:
    """One layer's worth of intermediates. Only `s7_post_down` (= the decoder
    layer's output / next layer's input) is captured by the HF hook; the other
    fields are kept for backward-compat with old debug scripts and set to None."""
    s1_qkv: Optional[torch.Tensor] = None
    s2_q_rotated: Optional[torch.Tensor] = None
    s2_k_rotated: Optional[torch.Tensor] = None
    s2_v: Optional[torch.Tensor] = None
    s3_attn: Optional[torch.Tensor] = None
    s4_post_o: Optional[torch.Tensor] = None
    s5_gate_up: Optional[torch.Tensor] = None
    s6_silu_mul: Optional[torch.Tensor] = None
    s7_post_down: Optional[torch.Tensor] = None


@dataclass
class ReferenceOutput:
    """End-to-end reference output."""
    logits: torch.Tensor                     # (seq_len, vocab)
    s0_emb_normed: torch.Tensor              # (seq_len, hidden), kept for compat
    layers: List[StageDump] = field(default_factory=list)


def _build_llama_config(cfg: TinyLlamaConfig):
    """Build a HF LlamaConfig matching the TinyLlama-1.1B-Chat-v1.0 shape."""
    from transformers import LlamaConfig
    return LlamaConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.rms_eps,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )


class TinyLlamaReference:
    """Thin wrapper around HF `LlamaForCausalLM` (eager attention) configured for
    TinyLlama-1.1B. Accepts a HF-style state_dict and runs prefill on a 1-D
    `input_ids` tensor of length `seq_len`.
    """

    def __init__(self, weights: Dict[str, torch.Tensor], device: torch.device = "cuda",
                 dtype: torch.dtype = torch.bfloat16, seq_len: Optional[int] = None):
        from transformers import LlamaForCausalLM
        self.cfg = TINYLLAMA_CONFIG
        self.device = torch.device(device)
        self.dtype = dtype
        self.seq_len = seq_len or self.cfg.max_position_embeddings

        hf_cfg = _build_llama_config(self.cfg)
        # Force eager attention so the test is gated against a well-defined
        # numerical path (not whatever SDPA backend the runtime selects).
        hf_cfg._attn_implementation = "eager"
        model = LlamaForCausalLM(hf_cfg)
        # `weights` uses HF naming conventions, so load_state_dict just works.
        model.load_state_dict({k: v.detach() for k, v in weights.items()})
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, dump: bool = False) -> ReferenceOutput:
        """Run prefill. `input_ids` shape `(seq_len,)` int32/int64 on CUDA."""
        assert input_ids.ndim == 1, "reference is batch=1 only"

        ids = input_ids.to(self.device, dtype=torch.long).unsqueeze(0)  # (1, S)

        # Per-layer hook: capture each decoder layer's output (= residual stream
        # after the layer == input to the next layer). The HF decoder layer
        # returns its output as the first element of a tuple (or directly).
        captured: List[torch.Tensor] = []
        hooks = []
        if dump:
            def make_hook():
                def hook(_module, _inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured.append(h.detach().clone().squeeze(0))
                return hook
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(make_hook()))

        try:
            out = self.model(ids).logits.squeeze(0)  # (S, V), BF16
        finally:
            for h in hooks:
                h.remove()

        ref_out = ReferenceOutput(
            logits=out,
            s0_emb_normed=torch.empty(0),
            layers=[StageDump(s7_post_down=h) for h in captured] if dump else [],
        )
        return ref_out


def load_hf_tinyllama_weights(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                              dtype: torch.dtype = torch.bfloat16,
                              device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Convenience: load TinyLlama-1.1B weights from HuggingFace.

    Returns a flat state_dict keyed by HF parameter names. Heavy import lives
    inside the function so the rest of this file does not require `transformers`
    when running on a node without it installed.
    """
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    return {k: v.detach().contiguous() for k, v in hf.state_dict().items()}
