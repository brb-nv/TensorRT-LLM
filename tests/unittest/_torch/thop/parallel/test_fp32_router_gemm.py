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

# MiniMax-M3's router shape, the only one the kernel instantiates.
NUM_EXPERTS = 128
HIDDEN_SIZE = 6144

# The kernel sums each thread's 48 products serially and then reduces across 128
# threads through a 7-level butterfly, so ~55 fp32 roundings can accumulate. 128
# leaves headroom without letting a real bug through: the summation order differs
# from cuBLAS's, so a bitwise or near-bitwise match is not available to test for.
ACCUM_ROUNDINGS = 128


def _reference(input: torch.Tensor, weight: torch.Tensor):
    """fp64 ground truth and a per-element bound on fp32 accumulation error.

    Args:
        input: bf16 activation, ``[num_tokens, hidden_size]``.
        weight: fp32 router weight, ``[num_experts, hidden_size]``.

    Returns:
        ``(logits, tolerance)``, both ``[num_tokens, num_experts]`` float64. The
        tolerance scales with the magnitude actually summed rather than with the
        (heavily cancelled) result, which is what fp32 error actually tracks.
    """
    a = input.double()
    b = weight.double()
    logits = a @ b.t()
    magnitude = a.abs() @ b.abs().t()
    tolerance = ACCUM_ROUNDINGS * torch.finfo(torch.float32).eps * magnitude
    return logits, tolerance


def _assert_close(actual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
    expected, tolerance = _reference(input, weight)
    error = (actual.double() - expected).abs()
    worst = (error / tolerance).max().item()
    assert worst <= 1.0, (
        f"max error {error.max().item():.3e} exceeds tolerance "
        f"{tolerance.flatten()[error.argmax()].item():.3e} ({worst:.2f}x)"
    )


@pytest.mark.parametrize("num_tokens", list(range(1, 17)))
def test_fp32_router_gemm(num_tokens):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32, device=device)

    logits = torch.ops.trtllm.fp32_router_gemm_op(input, weight.t(), None, torch.float32)

    assert logits.dtype == torch.float32
    assert logits.shape == (num_tokens, NUM_EXPERTS)
    _assert_close(logits, input, weight)


@pytest.mark.parametrize(
    "num_tokens,num_experts",
    [(1, 64), (17, NUM_EXPERTS)],
    ids=["unsupported_num_experts", "unsupported_num_tokens"],
)
def test_fp32_router_gemm_cublas_fallback(num_tokens, num_experts):
    """Shapes with no instantiated kernel must still compute the right answer.

    The fallback hands cuBLASLt a pre-cast fp32 activation. Handing it the bf16
    activation alongside the fp32 weight instead would not raise; it would return
    silent garbage.
    """
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weight = torch.randn((num_experts, HIDDEN_SIZE), dtype=torch.float32, device=device)

    logits = torch.ops.trtllm.fp32_router_gemm_op(input, weight.t(), None, torch.float32)

    assert logits.shape == (num_tokens, num_experts)
    _assert_close(logits, input, weight)


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)],
    ids=["fp32_activation", "bf16_weight"],
)
def test_fp32_router_gemm_rejects_wrong_dtypes(input_dtype, weight_dtype):
    """Both operands are validated, so a mismatch fails loudly rather than being cast."""
    device = torch.device("cuda")
    input = torch.randn(1, HIDDEN_SIZE, dtype=input_dtype, device=device)
    weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=weight_dtype, device=device)

    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp32_router_gemm_op(input, weight.t(), None, torch.float32)
