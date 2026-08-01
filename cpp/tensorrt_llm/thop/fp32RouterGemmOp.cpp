/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/cublasScaledMM.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
// Only num_tokens 1..16 are instantiated, to bound compile time.
constexpr int kMaxNumTokens = 16;

// Supported (hidden_dim, num_experts) pairs.
constexpr int kHiddenDim6144 = 6144; // MiniMax-M3
constexpr int kNumExperts128 = 128;  // MiniMax-M3

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller
{
    static void unroll(
        int num_tokens, float* output, __nv_bfloat16 const* input, float const* weights, cudaStream_t stream)
    {
        if (num_tokens == kBegin)
        {
            tk::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, kBegin, kNumExperts, kHiddenDim, float>(
                output, input, weights, stream);
        }
        else
        {
            LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim>::unroll(num_tokens, output, input, weights, stream);
        }
    }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim>
{
    static void unroll(
        int num_tokens, float* output, __nv_bfloat16 const* input, float const* weights, cudaStream_t stream)
    {
        if (num_tokens == kEnd)
        {
            tk::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim, float>(
                output, input, weights, stream);
        }
        else
        {
            throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
        }
    }
};
} // namespace

//! \brief Router GEMM with a bf16 activation and an fp32 weight.
//!
//! Routers that must run in fp32 (MiniMax-M3, matching SGLang) would otherwise
//! upcast the activation and run a skinny SGEMM. The specialized kernel does the
//! bf16 -> fp32 conversion while loading registers, so no fp32 copy of the
//! activation is ever materialized.
//!
//! Both operand dtypes are checked, unlike ``dsv3_router_gemm_op``, which validates
//! only ``mat_a`` and then reinterpret_casts ``mat_b``: here the two differ by
//! construction, so a wrong ``mat_b`` dtype would otherwise be read as fp32 garbage.
//!
//! \param mat_a bf16 activation, [num_tokens, hidden_dim], row-major.
//! \param mat_b fp32 router weight, [hidden_dim, num_experts], column-major.
//! \param bias unsupported by the kernel; forces the cuBLAS fallback.
//! \param out_dtype fp32 output. Defaults to fp32 because that is the only output
//! the kernel produces; anything else forces the cuBLAS fallback.
th::Tensor fp32_router_gemm_op(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> const& out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.scalar_type() == torch::kBFloat16, "fp32_router_gemm_op expects a bf16 activation, got ",
        mat_a.scalar_type());
    TORCH_CHECK(mat_b.scalar_type() == torch::kFloat32, "fp32_router_gemm_op expects an fp32 weight, got ",
        mat_b.scalar_type());

    int const num_tokens = mat_a.sizes()[0];
    int const hidden_dim = mat_a.sizes()[1];
    int const num_experts = mat_b.sizes()[1];
    auto const out_dtype_ = out_dtype.value_or(torch::kFloat32);

    TORCH_CHECK(mat_b.sizes()[0] == hidden_dim);
    TORCH_CHECK(mat_a.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1); // Column-major

    th::Tensor out = th::empty({mat_a.sizes()[0], mat_b.sizes()[1]}, mat_a.options().dtype(out_dtype_));
    TORCH_CHECK(out.strides()[1] == 1); // Row-major

    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

    bool const shape_ok
        = num_tokens >= 1 && num_tokens <= kMaxNumTokens && out_dtype_ == torch::kFloat32 && !bias.has_value();

    if (shape_ok && hidden_dim == kHiddenDim6144 && num_experts == kNumExperts128)
    {
        LoopUnroller<1, kMaxNumTokens, kNumExperts128, kHiddenDim6144>::unroll(num_tokens,
            reinterpret_cast<float*>(out.mutable_data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            reinterpret_cast<float const*>(mat_b.data_ptr()), stream);
    }
    else // fallback to cublas, can be slow
    {
        // cuBLASLt cannot mix a bf16 A with an fp32 B, so the fallback has to
        // materialize the fp32 activation that this op exists to avoid.
        th::Tensor const mat_a_fp32 = mat_a.to(torch::kFloat32);
        cublas_mm_out(mat_a_fp32, mat_b, bias, out);
    }

    return out;
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp32_router_gemm_op(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp32_router_gemm_op", &tensorrt_llm::torch_ext::fp32_router_gemm_op);
}
