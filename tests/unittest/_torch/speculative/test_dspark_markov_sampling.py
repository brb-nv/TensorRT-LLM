# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark ancestral drafting under rejection sampling: one Gumbel-max kernel per block position."""

import math
import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.dspark import DSparkWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(
    rows, vocab, num_pos, *, same_rows, rank=16, temperature=0.8, seed=0, dtype=torch.float32
):
    generator = torch.Generator().manual_seed(seed)
    first = 1 if same_rows else rows
    drafter = types.SimpleNamespace(
        has_markov_head=True,
        markov_w1=(0.3 * torch.randn(vocab, rank, generator=generator)).to(torch.bfloat16).cuda(),
        markov_w2=(0.3 * torch.randn(vocab, rank, generator=generator)).to(torch.bfloat16).cuda(),
    )
    base = (2.0 * torch.randn(first, num_pos, vocab, generator=generator)).expand(rows, -1, -1)
    anchors = torch.randint(vocab, (first,), generator=generator, dtype=torch.int32).expand(rows)
    meta = types.SimpleNamespace(
        request_temperatures=torch.full((rows,), temperature).cuda(),
        request_seeds=torch.full((rows,), 42, dtype=torch.int64).cuda(),
        request_offsets=(torch.arange(rows, dtype=torch.int64) * 512 + (1 << 62)).cuda(),
        rng_draft_slot=lambda step: num_pos + 1 + step,
        batch_slot_ids=torch.arange(rows, dtype=torch.int64).cuda(),
        draft_probs=torch.zeros(rows + 1, num_pos, vocab).cuda(),
        draft_probs_last_dim=0,
        dummy_slot_row=rows,
        is_all_greedy_sample=False,
        wants_advanced_draft_sampling=True,
    )
    worker = DSparkWorker.__new__(DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = None
    return worker, drafter, meta, base.to(dtype).contiguous().cuda(), anchors.contiguous().cuda()


def _draft(worker, drafter, meta, base, anchors):
    logits = base.clone()
    tokens, vocab = worker._sample_block_ancestrally(
        drafter, logits, {"first_prev_tokens": anchors}, meta, 0, base.shape[0]
    )
    assert vocab == base.shape[-1] == meta.draft_probs_last_dim
    return tokens, logits


def _chi_square_passes(observed, expected):
    big = expected >= 5  # pool the rare cells
    observed = torch.cat([observed[big], observed[~big].sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    stat = ((observed - expected) ** 2 / expected).sum().item()
    dof = observed.numel() - 1
    return stat < dof * (1 - 2 / (9 * dof) + 3.719 * math.sqrt(2 / (9 * dof))) ** 3  # p = 1e-4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [24, 4500])  # one and three vocab chunks per row
def test_ancestral_drafts_follow_the_drawn_tokens_and_store_their_proposal(vocab, dtype):
    """Position k is drawn from q_k = softmax((base_k + bias(d_{k-1})) / T) in float32, d_{k-1} the token drawn
    before it, and q_k is the proposal stored at the request's slot; the drafter's logits are left as they are."""
    rows = 100_000
    worker, drafter, meta, base, anchors = _setup(rows, vocab, 2, same_rows=True, dtype=dtype)
    tokens, logits = _draft(worker, drafter, meta, base, anchors)
    assert torch.equal(logits, base)

    prev = anchors
    for k in range(2):
        bias = dspark_markov_step_bias(prev, drafter.markov_w1, drafter.markov_w2)
        biased = base[:, k].float() + bias.float()
        q = torch.softmax(biased / meta.request_temperatures[:, None], -1)
        torch.testing.assert_close(meta.draft_probs[:rows, k], q, atol=1e-6, rtol=1e-5)
        # Position 0 shares one distribution; position 1 pools q_1 over the drawn d_0.
        observed = torch.bincount(tokens[:, k].long(), minlength=vocab).double()
        assert _chi_square_passes(observed.cpu(), q.double().sum(0).cpu())
        prev = tokens[:, k]


def test_ancestral_drafts_replay_in_a_graph_and_skip_padding_rows():
    """A graph replay draws what the eager call draws, a new offset draws anew, and padding rows (uninitialized
    logits) draw legal tokens with finite proposals."""
    rows, vocab = 12, 200064
    worker, drafter, meta, base, anchors = _setup(rows, vocab, 7, same_rows=False, rank=256)
    padding = [5, 9]
    meta.batch_slot_ids[padding] = meta.dummy_slot_row
    base[padding] = float("nan")
    eager, _ = _draft(worker, drafter, meta, base, anchors)
    eager_probs = meta.draft_probs.clone()
    assert ((eager >= 0) & (eager < vocab)).all() and torch.isfinite(eager_probs).all()

    logits = torch.empty_like(base)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        logits.copy_(base)
        worker._sample_block_ancestrally(
            drafter, logits, {"first_prev_tokens": anchors}, meta, 0, rows
        )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits.copy_(base)
        tokens, _ = worker._sample_block_ancestrally(
            drafter, logits, {"first_prev_tokens": anchors}, meta, 0, rows
        )
    graph.replay()
    real = [r for r in range(rows) if r not in padding]
    assert torch.equal(tokens, eager) and torch.equal(meta.draft_probs[real], eager_probs[real])
    meta.request_offsets += 512
    graph.replay()
    assert (tokens[real] != eager[real]).float().mean() > 0.5
