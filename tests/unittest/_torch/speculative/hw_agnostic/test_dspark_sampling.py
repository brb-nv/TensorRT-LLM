# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block verification and temperature-only sampling under rejection sampling."""

import math
from unittest.mock import patch

import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import custom as fused_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.speculative.block_verification import block_verify
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

VOCAB, K = 3, 3


def _prefix_model(generator, sharpness):
    """A next-token distribution for every prefix of length 0..K."""
    num_prefixes = sum(VOCAB**m for m in range(K + 1))
    logits = sharpness * torch.randn(num_prefixes, VOCAB, generator=generator, dtype=torch.float64)
    return torch.softmax(logits, -1)


def _rows(model, prefixes):
    ids = torch.zeros(prefixes.shape[0], dtype=torch.int64)
    for m in range(prefixes.shape[1]):
        ids = ids * VOCAB + prefixes[:, m]
    return model[ids + sum(VOCAB**m for m in range(prefixes.shape[1]))]


def test_block_verification_is_lossless():
    """Blocks verified, then completed by sampling the target, are distributed as the target's own samples."""
    generator = torch.Generator().manual_seed(0)
    target, draft = _prefix_model(generator, 1.5), _prefix_model(generator, 1.0)
    num = 200_000
    drafts = torch.empty(num, 0, dtype=torch.int64)
    draft_rows = []
    for _ in range(K):
        draft_rows.append(_rows(draft, drafts))
        drafts = torch.cat([drafts, torch.multinomial(draft_rows[-1], 1, generator=generator)], 1)
    target_rows = [_rows(target, drafts[:, :j]) for j in range(K + 1)]
    tokens, num_accepted = block_verify(
        drafts,
        torch.stack(draft_rows, 1).float(),
        torch.stack(target_rows, 1).float(),
        torch.rand(num, K, generator=generator, dtype=torch.float64),
        lambda weights: torch.multinomial(weights, 1, generator=generator).squeeze(-1),
    )

    sequences = torch.zeros(num, K + 1, dtype=torch.int64)
    for m in range(K + 1):
        continued = torch.multinomial(_rows(target, sequences[:, :m]), 1, generator=generator)
        sequences[:, m] = torch.where(
            m < num_accepted.long(), tokens[:, m].long(), continued.squeeze(-1)
        )
    grid = torch.cartesian_prod(*[torch.arange(VOCAB)] * (K + 1))
    expected = torch.full((grid.shape[0],), float(num), dtype=torch.float64)
    for m in range(K + 1):
        expected *= _rows(target, grid[:, :m]).gather(-1, grid[:, m : m + 1]).squeeze(-1)
    ids = (sequences * VOCAB ** torch.arange(K, -1, -1)).sum(-1)
    observed = torch.bincount(ids, minlength=grid.shape[0]).double()
    big = expected >= 5  # pool the rare sequences
    observed = torch.cat([observed[big], observed[~big].sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    stat = ((observed - expected) ** 2 / expected).sum().item()
    dof = observed.numel() - 1
    critical = dof * (1 - 2 / (9 * dof) + 3.719 * math.sqrt(2 / (9 * dof))) ** 3  # p = 1e-4
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


def test_temperature_only_sampling_runs_no_filter():
    """No filter reaches the FlashInfer chain; the fused op gets its disable sentinels."""
    logits, params = torch.randn(2, VOCAB), SpecWorkerBase._temperature_only_params(torch.ones(2))
    with patch.object(
        flashinfer_sampling, "sampling_batch_spec_dec_one_model_for_rejection"
    ) as chain:
        SpecWorkerBase._sample_from_logits_with_probs(AdvancedSamplingMode.NO_TOPK, logits, *params)
    assert chain.call_args.args[2:] == (None, None)
    with patch.object(fused_sampling, "fused_sample_from_logits_with_probs") as fused:
        SpecWorkerBase._sample_from_logits_with_probs(AdvancedSamplingMode.FULL, logits, *params)
    _, _, top_ks, top_ps, min_ps = fused.call_args.args
    assert (
        (top_ks == torch.iinfo(torch.int32).max).all()
        and (top_ps == 1).all()
        and (min_ps == 0).all()
    )
