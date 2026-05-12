# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1 oracle test: TinyLlamaReference forward matches HF eager logits.

Requires:
  - `transformers` installed
  - access to TinyLlama-1.1B-Chat-v1.0 weights (HF hub or local cache)

Tolerance is BF16-loose (max-abs ~ 1e-1). The reference keeps the residual
stream in FP32 (matching the megakernel), while HF eager keeps it in BF16, so
the two diverge by ~one residual-cast worth of error per layer (~0.05 max-abs
on the final logits for a 22-layer model). Other differences come from RMSNorm
cast-up vs fused, attention SDPA backend choice, etc.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama.reference import TinyLlamaReference, load_hf_tinyllama_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(os.environ.get("LUCEBOX_RUN_HF_TEST") != "1",
                     reason="set LUCEBOX_RUN_HF_TEST=1 to actually pull TinyLlama from HF")
def test_reference_logits_match_hf_eager():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("LUCEBOX_TINYLLAMA_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa").cuda().eval()

    prompt = "The quick brown fox jumps over"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()  # (1, S)
    seq_len = ids.shape[1]

    # Pad to seqlen=128 to match the kernel's compile-time seq_len.
    pad_to = 128
    pad_token = tok.pad_token_id or 0
    if seq_len < pad_to:
        pad = torch.full((1, pad_to - seq_len), pad_token, dtype=ids.dtype, device=ids.device)
        ids_padded = torch.cat([ids, pad], dim=1)
    else:
        ids_padded = ids[:, :pad_to]

    with torch.no_grad():
        hf_out = hf(ids_padded).logits  # (1, S, V)

    state_dict = {k: v.detach().contiguous() for k, v in hf.state_dict().items()}
    del hf
    torch.cuda.empty_cache()
    ref = TinyLlamaReference(state_dict, device="cuda", dtype=torch.bfloat16, seq_len=pad_to)
    out = ref.forward(ids_padded.squeeze(0))
    ref_logits = out.logits  # (S, V)

    # Compare only the valid (non-pad) positions to avoid spuriously large diffs
    # from the embedding of the pad token cascading through 22 layers.
    diff = (ref_logits[:seq_len].float() - hf_out.squeeze(0)[:seq_len].float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"reference vs HF eager: max_abs={max_abs:.4f} mean_abs={mean_abs:.5f}")
    assert max_abs < 1e-1, f"reference forward drifted from HF eager: max_abs={max_abs}"
