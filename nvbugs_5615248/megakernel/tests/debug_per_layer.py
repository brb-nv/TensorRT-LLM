# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer divide-and-conquer diff between the megakernel and the PyT reference.

Run as a script (not a pytest):

    python tests/debug_per_layer.py

Prints a table

    L   max_abs   mean_abs    p99_abs    (kernel - reference)

for L in [0, num_layers). The first L where max_abs > BF16 noise floor
(~5e-3 over 1-3 layers, growing to ~5e-2 over 22) is the first stage to blame.

Once we know "layer L is where it first diverges", we can drill in further by
comparing the kernel's residual at L-1 (which agrees) plus the reference's
sub-stages for layer L (s1_qkv, s2_q_rotated, s3_attn, s4_post_o, s5_gate_up,
s6_silu_mul, s7_post_down) to pinpoint the bad sub-stage.
"""
from __future__ import annotations

import torch

from lucebox_tinyllama import (
    TINYLLAMA_CONFIG, TinyLlamaReference, pack_weights, prefill,
)

# Reuse the EXACT synthetic-weight generator from test_numerics.py so the diff
# numbers reproduce what pytest sees (seed=42, std=0.02).
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from test_numerics import _make_synthetic_weights  # noqa: E402


@torch.no_grad()
def main() -> None:
    assert torch.cuda.is_available(), "needs CUDA"
    cfg = TINYLLAMA_CONFIG
    SEQ = 128

    sd = _make_synthetic_weights(seed=42)
    pw = pack_weights(sd, device="cuda")

    ref = TinyLlamaReference(sd, device="cuda", dtype=torch.bfloat16, seq_len=SEQ)

    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (SEQ,), dtype=torch.int32, device="cuda")

    # --- Reference: capture per-layer s7_post_down (BF16 residual after layer L).
    out_ref = ref.forward(ids, dump=True)
    ref_layers = [out_ref.layers[L].s7_post_down for L in range(cfg.num_layers)]

    # --- Kernel: per-layer residual dump via the kernel's debug hook (BF16).
    dump = torch.empty((cfg.num_layers, SEQ, cfg.hidden_size),
                       dtype=torch.bfloat16, device="cuda")
    logits = prefill(ids, pw, layer_residual_dump=dump, seq_len=SEQ)
    torch.cuda.synchronize()

    # --- Final logits diff (sanity check we reproduce the failing test).
    diff_logits = (out_ref.logits.float() - logits.float()).abs()
    print(f"final logits: max_abs={diff_logits.max().item():.4f}  "
          f"mean_abs={diff_logits.mean().item():.5f}\n")

    # --- Per-layer table (everything compared in FP32 after upcast from BF16).
    print(f"{'L':>3}  {'max_abs':>8}  {'mean_abs':>9}  {'p99_abs':>8}  "
          f"{'ker_norm':>8}  {'ref_norm':>8}")
    print("-" * 64)
    for L in range(cfg.num_layers):
        r = ref_layers[L].to(torch.float32)
        k = dump[L].to(torch.float32)
        d = (r - k).abs()
        p99 = torch.quantile(d.flatten(), 0.99).item()
        print(f"{L:3d}  {d.max().item():8.4f}  {d.mean().item():9.5f}  "
              f"{p99:8.4f}  {k.abs().mean().item():8.4f}  "
              f"{r.abs().mean().item():8.4f}")


if __name__ == "__main__":
    main()
