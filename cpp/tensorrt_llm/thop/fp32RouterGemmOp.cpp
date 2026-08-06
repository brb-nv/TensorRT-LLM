/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/fp32RouterGemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

// The kernel is templated on the token count, so dispatch it from the runtime
// value. Unlike trtllm::dsv3_router_gemm_op there is no cuBLAS fallback: the
// op is narrow by construction and a caller that reaches it with an
// unsupported shape has a wiring bug, which should be loud rather than
// silently slow.
constexpr int kFp32RouterGemmMaxTokens = 32;

template <typename InputT, int kNumExperts, int kHiddenDim, int kBegin, int kEnd>
struct Fp32LoopUnroller
{
    static void unroll(int numTokens, float* output, InputT const* mat_a, float const* mat_b, cudaStream_t stream)
    {
        if (numTokens == kBegin)
        {
            tensorrt_llm::kernels::invokeFp32RouterGemm<InputT, kBegin, kNumExperts, kHiddenDim>(
                output, mat_a, mat_b, stream);
        }
        else
        {
            Fp32LoopUnroller<InputT, kNumExperts, kHiddenDim, kBegin + 1, kEnd>::unroll(
                numTokens, output, mat_a, mat_b, stream);
        }
    }
};

template <typename InputT, int kNumExperts, int kHiddenDim, int kEnd>
struct Fp32LoopUnroller<InputT, kNumExperts, kHiddenDim, kEnd, kEnd>
{
    static void unroll(int numTokens, float* output, InputT const* mat_a, float const* mat_b, cudaStream_t stream)
    {
        TORCH_CHECK(numTokens == kEnd, "fp32_router_gemm supports num_tokens in [1, ", kEnd, "], got ", numTokens);
        tensorrt_llm::kernels::invokeFp32RouterGemm<InputT, kEnd, kNumExperts, kHiddenDim>(
            output, mat_a, mat_b, stream);
    }
};

} // namespace

torch::Tensor fp32RouterGemm(torch::Tensor const& mat_a, torch::Tensor const& mat_b)
{
    // Only the shape the kernel is instantiated for; see fp32RouterGemm.cu.
    constexpr int64_t kNumExperts = 128;
    constexpr int64_t kHiddenDim = 6144;

    TORCH_CHECK(mat_a.is_cuda() && mat_b.is_cuda(), "fp32_router_gemm expects CUDA tensors");
    TORCH_CHECK(mat_a.device() == mat_b.device(), "fp32_router_gemm expects both operands on the same device");
    TORCH_CHECK(mat_a.dim() == 2, "mat_a must be [num_tokens, hidden_dim], got ", mat_a.sizes());
    TORCH_CHECK(mat_b.dim() == 2, "mat_b must be [num_experts, hidden_dim], got ", mat_b.sizes());
    TORCH_CHECK(mat_a.is_contiguous() && mat_b.is_contiguous(), "fp32_router_gemm expects contiguous operands");
    TORCH_CHECK(mat_b.scalar_type() == torch::kFloat32, "fp32_router_gemm expects a float32 mat_b (router weight), got ",
        mat_b.scalar_type());
    TORCH_CHECK(mat_a.scalar_type() == torch::kFloat32 || mat_a.scalar_type() == torch::kBFloat16,
        "fp32_router_gemm expects a float32 or bfloat16 mat_a, got ", mat_a.scalar_type());
    TORCH_CHECK(mat_a.size(1) == mat_b.size(1), "mat_a and mat_b must share hidden_dim, got ", mat_a.size(1), " and ",
        mat_b.size(1));
    TORCH_CHECK(mat_b.size(0) == kNumExperts && mat_a.size(1) == kHiddenDim,
        "fp32_router_gemm is instantiated for (num_experts, hidden_dim) == (", kNumExperts, ", ", kHiddenDim,
        "), got (", mat_b.size(0), ", ", mat_a.size(1), ")");

    int64_t const numTokens = mat_a.size(0);
    TORCH_CHECK(numTokens >= 0 && numTokens <= static_cast<int64_t>(kFp32RouterGemmMaxTokens),
        "fp32_router_gemm supports num_tokens in [0, ", kFp32RouterGemmMaxTokens, "], got ", numTokens);

    auto output = torch::empty({numTokens, kNumExperts}, mat_a.options().dtype(torch::kFloat32));
    if (numTokens == 0)
    {
        return output;
    }

    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 90, "fp32_router_gemm requires SM90 or newer");

    auto const stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
    auto* outputPtr = output.data_ptr<float>();
    auto const* matBPtr = mat_b.data_ptr<float>();
    if (mat_a.scalar_type() == torch::kBFloat16)
    {
        Fp32LoopUnroller<__nv_bfloat16, kNumExperts, kHiddenDim, 1, kFp32RouterGemmMaxTokens>::unroll(
            static_cast<int>(numTokens), outputPtr,
            reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()), matBPtr, stream);
    }
    else
    {
        Fp32LoopUnroller<float, kNumExperts, kHiddenDim, 1, kFp32RouterGemmMaxTokens>::unroll(
            static_cast<int>(numTokens), outputPtr, mat_a.data_ptr<float>(), matBPtr, stream);
    }
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp32_router_gemm(Tensor mat_a, Tensor mat_b) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp32_router_gemm", &tensorrt_llm::torch_ext::fp32RouterGemm);
}
