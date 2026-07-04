/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * torch.ops.trtllm bindings for the routed-expert MoE LoRA BGMV kernels. These
 * are the native building blocks the FP8 block-scale LoRA delta builders drive
 * (see tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py). No FlashInfer
 * dependency: the kernels live under tensorrt_llm/kernels/bgmvMoe/.
 */

#include "tensorrt_llm/kernels/bgmvMoe/moeBgmvKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
// Reinterpret an int64 pointer-table tensor as a device T** array.
template <typename T>
T** asDevicePtrTable(at::Tensor const& wPtr)
{
    return reinterpret_cast<T**>(static_cast<int64_t*>(wPtr.data_ptr()));
}
} // namespace

// y[num_slices, num_pairs, rank] += x @ lora_a[lora_id, expert]. y is accumulated
// in place (caller zero-inits). x is [num_tokens, feat_in] (FC1) or
// [num_pairs, feat_in] (FC2, per_pair_input=true). w_ptr is [num_slices, max_loras]
// per-adapter base pointers (each -> that adapter's [num_experts, feat_out, feat_in]
// contiguous bank); the per-expert offset is added inside the kernel.
void bgmvMoeShrinkOp(at::Tensor& y, at::Tensor const& x, at::Tensor const& wPtr, at::Tensor const& sortedTokenIds,
    at::Tensor const& expertIds, at::Tensor const& loraIndices, bool perPairInput)
{
    CHECK_INPUT(sortedTokenIds, at::ScalarType::Long);
    CHECK_INPUT(expertIds, at::ScalarType::Long);
    CHECK_INPUT(loraIndices, at::ScalarType::Long);
    CHECK_INPUT(wPtr, at::ScalarType::Long);
    CHECK_TH_CUDA(y);
    CHECK_TH_CUDA(x);
    TORCH_CHECK(y.dim() == 3, "bgmv_moe_shrink: y must be 3D [num_slices, num_pairs, rank].");
    TORCH_CHECK(x.dim() == 2, "bgmv_moe_shrink: x must be 2D.");
    TORCH_CHECK(wPtr.dim() == 2, "bgmv_moe_shrink: w_ptr must be 2D [num_slices, max_loras].");
    TORCH_CHECK(y.scalar_type() == x.scalar_type(), "bgmv_moe_shrink: y and x must share dtype.");

    int64_t const numSlices = y.size(0);
    int64_t const numPairs = sortedTokenIds.size(0);
    int64_t const numTokens = loraIndices.size(0);
    int64_t const featIn = x.size(1);
    int64_t const featOut = y.size(2);
    int64_t const maxLoras = wPtr.size(1);
    TORCH_CHECK(wPtr.size(0) == numSlices, "bgmv_moe_shrink: w_ptr slice dim mismatch.");

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    bool ok = false;
    switch (x.scalar_type())
    {
    case at::ScalarType::Half:
        ok = tensorrt_llm::kernels::bgmv_moe::bgmvMoeShrink<half>(static_cast<half*>(y.data_ptr()),
            static_cast<half const*>(x.data_ptr()), asDevicePtrTable<half>(wPtr),
            static_cast<int64_t const*>(sortedTokenIds.data_ptr()), static_cast<int64_t const*>(expertIds.data_ptr()),
            static_cast<int64_t const*>(loraIndices.data_ptr()), featIn, featOut, numPairs, numSlices, maxLoras,
            numTokens, perPairInput, stream);
        break;
    case at::ScalarType::BFloat16:
        ok = tensorrt_llm::kernels::bgmv_moe::bgmvMoeShrink<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(y.data_ptr()), static_cast<__nv_bfloat16 const*>(x.data_ptr()),
            asDevicePtrTable<__nv_bfloat16>(wPtr), static_cast<int64_t const*>(sortedTokenIds.data_ptr()),
            static_cast<int64_t const*>(expertIds.data_ptr()), static_cast<int64_t const*>(loraIndices.data_ptr()),
            featIn, featOut, numPairs, numSlices, maxLoras, numTokens, perPairInput, stream);
        break;
    default: TORCH_CHECK(false, "bgmv_moe_shrink: unsupported x dtype (expected float16 or bfloat16).");
    }
    TORCH_CHECK(ok, "bgmv_moe_shrink: dimension pair (feat_in=", featIn, ", feat_out=", featOut,
        ") was not compiled. See moeBgmvKernels.cuh dimension list.");
}

// Project shrink output through LoRA-B. finalize=true: routing-weighted per-token
// combine, y is [num_tokens, total_feat_out]. finalize=false: per-pair unweighted
// store, y is [num_pairs, total_feat_out]. y is float32 and zero-initialized by
// the caller.
void bgmvMoeExpandOp(at::Tensor& y, at::Tensor const& x, at::Tensor const& wPtr, at::Tensor const& sortedTokenIds,
    at::Tensor const& expertIds, at::Tensor const& topkWeights, at::Tensor const& loraIndices,
    at::Tensor const& sliceStartLoc, int64_t firstFeatOut, bool finalize)
{
    CHECK_INPUT(sortedTokenIds, at::ScalarType::Long);
    CHECK_INPUT(expertIds, at::ScalarType::Long);
    CHECK_INPUT(loraIndices, at::ScalarType::Long);
    CHECK_INPUT(wPtr, at::ScalarType::Long);
    CHECK_INPUT(sliceStartLoc, at::ScalarType::Long);
    CHECK_INPUT(topkWeights, at::ScalarType::Float);
    CHECK_INPUT(y, at::ScalarType::Float);
    CHECK_TH_CUDA(x);
    TORCH_CHECK(y.dim() == 2, "bgmv_moe_expand: y must be 2D and float32.");
    TORCH_CHECK(x.dim() == 3, "bgmv_moe_expand: x must be 3D [num_slices, num_pairs, rank].");
    TORCH_CHECK(wPtr.dim() == 2, "bgmv_moe_expand: w_ptr must be 2D [num_slices, max_loras].");
    TORCH_CHECK(firstFeatOut > 0, "bgmv_moe_expand: first_feat_out must be positive.");

    int64_t const numSlices = x.size(0);
    int64_t const numPairs = sortedTokenIds.size(0);
    int64_t const numTokens = loraIndices.size(0);
    int64_t const featIn = x.size(2);
    int64_t const totalFeatOut = y.size(1);
    int64_t const maxLoras = wPtr.size(1);
    TORCH_CHECK(wPtr.size(0) == numSlices, "bgmv_moe_expand: w_ptr slice dim mismatch.");

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    bool ok = false;
    switch (x.scalar_type())
    {
    case at::ScalarType::Half:
        ok = tensorrt_llm::kernels::bgmv_moe::bgmvMoeExpand<half>(static_cast<float*>(y.data_ptr()),
            static_cast<half const*>(x.data_ptr()), asDevicePtrTable<half>(wPtr),
            static_cast<int64_t const*>(sortedTokenIds.data_ptr()), static_cast<int64_t const*>(expertIds.data_ptr()),
            static_cast<int64_t const*>(loraIndices.data_ptr()), static_cast<float const*>(topkWeights.data_ptr()),
            static_cast<int64_t const*>(sliceStartLoc.data_ptr()), featIn, firstFeatOut, numPairs, numSlices,
            maxLoras, totalFeatOut, numTokens, finalize, stream);
        break;
    case at::ScalarType::BFloat16:
        ok = tensorrt_llm::kernels::bgmv_moe::bgmvMoeExpand<__nv_bfloat16>(static_cast<float*>(y.data_ptr()),
            static_cast<__nv_bfloat16 const*>(x.data_ptr()), asDevicePtrTable<__nv_bfloat16>(wPtr),
            static_cast<int64_t const*>(sortedTokenIds.data_ptr()), static_cast<int64_t const*>(expertIds.data_ptr()),
            static_cast<int64_t const*>(loraIndices.data_ptr()), static_cast<float const*>(topkWeights.data_ptr()),
            static_cast<int64_t const*>(sliceStartLoc.data_ptr()), featIn, firstFeatOut, numPairs, numSlices,
            maxLoras, totalFeatOut, numTokens, finalize, stream);
        break;
    default: TORCH_CHECK(false, "bgmv_moe_expand: unsupported x dtype (expected float16 or bfloat16).");
    }
    TORCH_CHECK(ok, "bgmv_moe_expand: dimension pair (feat_in=", featIn, ", feat_out=", firstFeatOut,
        ") was not compiled. See moeBgmvKernels.cuh dimension list.");
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "bgmv_moe_shrink(Tensor(a!) y, Tensor x, Tensor w_ptr, Tensor sorted_token_ids, "
        "Tensor expert_ids, Tensor lora_indices, bool per_pair_input) -> ()");
    m.def(
        "bgmv_moe_expand(Tensor(a!) y, Tensor x, Tensor w_ptr, Tensor sorted_token_ids, "
        "Tensor expert_ids, Tensor topk_weights, Tensor lora_indices, Tensor slice_start_loc, "
        "int first_feat_out, bool finalize) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("bgmv_moe_shrink", &tensorrt_llm::torch_ext::bgmvMoeShrinkOp);
    m.impl("bgmv_moe_expand", &tensorrt_llm::torch_ext::bgmvMoeExpandOp);
}
