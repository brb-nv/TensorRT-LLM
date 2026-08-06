# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

# The only shape trtllm::fp32_router_gemm is instantiated for (MiniMax-M3).
NUM_EXPERTS = 128
HIDDEN_SIZE = 6144
MAX_TOKENS = 32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="fp32_router_gemm requires an SM90+ GPU")


@pytest.fixture
def no_tf32():
    """The kernel accumulates in true fp32, so the reference must too."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("num_tokens", list(range(1, MAX_TOKENS + 1)))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_router_gemm(num_tokens, dtype, no_tf32):
    torch.manual_seed(24)
    device = torch.device("cuda")

    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
    weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE),
                         dtype=torch.float32,
                         device=device)

    logits = torch.ops.trtllm.fp32_router_gemm(x, weight)
    assert logits.shape == (num_tokens, NUM_EXPERTS)
    assert logits.dtype == torch.float32

    # The kernel widens bf16 to fp32 and accumulates in fp32, so it matches an
    # fp32 reference over the same widened inputs. Logits here have magnitude
    # ~sqrt(6144), so the only residual difference is summation order.
    ref = torch.nn.functional.linear(x.float(), weight)
    torch.testing.assert_close(logits, ref, rtol=1e-3, atol=5e-2)


def test_fp32_router_gemm_zero_tokens():
    device = torch.device("cuda")
    x = torch.empty(0, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE),
                         dtype=torch.float32,
                         device=device)
    logits = torch.ops.trtllm.fp32_router_gemm(x, weight)
    assert logits.shape == (0, NUM_EXPERTS)


def test_fp32_router_gemm_rejects_unsupported():
    device = torch.device("cuda")
    weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE),
                         dtype=torch.float32,
                         device=device)

    # Above the instantiated token range.
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp32_router_gemm(
            torch.randn(MAX_TOKENS + 1,
                        HIDDEN_SIZE,
                        dtype=torch.bfloat16,
                        device=device), weight)

    # A bf16 weight is the dsv3_router_gemm_op case, not this one.
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp32_router_gemm(
            torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
            weight.to(torch.bfloat16))

    # A hidden dim the kernel is not instantiated for.
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp32_router_gemm(
            torch.randn(4, 4096, dtype=torch.bfloat16, device=device),
            torch.randn((NUM_EXPERTS, 4096),
                        dtype=torch.float32,
                        device=device))


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_fp32_router_gemm_matches_gate_forward(num_tokens, no_tf32):
    """The wired MiniMaxM3Gate must agree with its own F.linear fallback."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate

    torch.manual_seed(24)
    device = torch.device("cuda")
    gate = MiniMaxM3Gate(hidden_size=HIDDEN_SIZE,
                         num_experts=NUM_EXPERTS,
                         top_k=8,
                         routed_scaling_factor=1.0).to(device)
    with torch.no_grad():
        gate.weight.normal_()
    assert gate._use_fp32_router_gemm

    x = torch.randn(num_tokens,
                    HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                    device=device)
    logits = gate(x)

    gate._use_fp32_router_gemm = False
    ref = gate(x)
    torch.testing.assert_close(logits, ref, rtol=1e-3, atol=5e-2)
