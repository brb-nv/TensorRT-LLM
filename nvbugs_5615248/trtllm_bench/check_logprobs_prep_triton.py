#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone numerical check for ``beam_logprobs_prep_triton``.

Compares the hand-written Triton kernel against the original unfused
PyTorch sequence on CUDA, for shapes covering the production regime
(B=1, K=10, V=32k for Llama) plus some larger batches and a tight V.

Run from the repo root inside the TRT-LLM container:
    python3 nvbugs_5615248/trtllm_bench/check_logprobs_prep_triton.py
"""

from __future__ import annotations

import sys

import torch

from tensorrt_llm._torch.pyexecutor._beam_search_kernels import (  # noqa: E402
    beam_logprobs_prep_triton,
)


def _eager_reference(
    logits: torch.Tensor,           # [B, K_in, V]
    finished_mask: torch.Tensor,    # [B, K_in], bool
    cum_log_probs: torch.Tensor,    # [B, K_in]
) -> torch.Tensor:
    logprobs = torch.log_softmax(logits, dim=-1)
    finished_e = finished_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    logprobs = torch.where(finished_e, float("-inf"), logprobs)
    logprobs = logprobs.clone()
    logprobs[..., 0] = torch.where(finished_mask, 0, logprobs[..., 0])
    logprobs = logprobs + cum_log_probs.unsqueeze(-1)
    return logprobs


def _check_one(
    *, B: int, K: int, V: int, dtype: torch.dtype, seed: int = 42
) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(B, K, V, device="cuda", dtype=dtype, generator=g)
    cum_log_probs = torch.randn(B, K, device="cuda", dtype=torch.float32, generator=g)
    finished_mask = (
        torch.rand(B, K, device="cuda", generator=g) < 0.3
    )

    expected = _eager_reference(logits, finished_mask, cum_log_probs)
    got = beam_logprobs_prep_triton(logits, finished_mask, cum_log_probs)

    assert got.shape == expected.shape, f"shape mismatch: got {got.shape} vs {expected.shape}"
    assert got.dtype == logits.dtype, f"dtype mismatch: got {got.dtype} vs {logits.dtype}"

    # -inf positions must agree exactly.
    inf_eq = (
        torch.isinf(got)
        & torch.isinf(expected)
        & (torch.signbit(got) == torch.signbit(expected))
    )
    finite_mask = ~torch.isinf(expected)
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 5e-3, 5e-3
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2

    torch.testing.assert_close(
        got[finite_mask].to(torch.float32),
        expected[finite_mask].to(torch.float32),
        atol=atol, rtol=rtol,
        msg=f"finite-position mismatch (B={B},K={K},V={V},dtype={dtype})",
    )
    n_inf_expected = torch.isinf(expected).sum().item()
    n_inf_match = inf_eq.sum().item()
    assert n_inf_match == n_inf_expected, (
        f"-inf positions mismatch (B={B},K={K},V={V},dtype={dtype}): "
        f"got {n_inf_match} matching, expected {n_inf_expected}"
    )

    print(
        f"  OK  B={B:2d} K={K:2d} V={V:6d} dtype={str(dtype):20s}  "
        f"-inf positions: {n_inf_expected:8d}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping (Triton kernel is CUDA-only).")
        sys.exit(0)

    print("Path B (Triton) numerical equivalence vs eager reference")
    print("=" * 72)

    cases = [
        # (B, K, V) — small + production + larger batch
        (1, 1, 32),         # smoke test
        (1, 10, 32000),     # production-ish (Llama-2 vocab)
        (1, 10, 128256),    # Llama-3 vocab
        (2, 3, 100),        # unit-test regime
        (4, 8, 50000),      # multi-batch + medium V
        (1, 10, 1024),      # small V (tests last-block masking)
    ]

    for B, K, V in cases:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            _check_one(B=B, K=K, V=V, dtype=dtype)

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
