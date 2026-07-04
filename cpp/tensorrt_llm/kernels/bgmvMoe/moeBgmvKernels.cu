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
 * Dispatch + explicit instantiations for the BGMV MoE LoRA kernels. Ported from
 * FlashInfer csrc/bgmv_moe/moe_bgmv_ops.cu (Apache-2.0); the TVM-FFI entry
 * points are replaced by the raw-pointer entry points declared in
 * moeBgmvKernels.h (wrapped by thop/bgmvMoeOp.cpp).
 */

#include "tensorrt_llm/kernels/bgmvMoe/moeBgmvKernels.cuh"
#include "tensorrt_llm/kernels/bgmvMoe/moeBgmvKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace bgmv_moe
{

namespace
{
inline constexpr uint64_t packU32(uint32_t a, uint32_t b)
{
    return (uint64_t(a) << 32) | uint64_t(b);
}
} // namespace

// Explicit instantiations of the templated host dispatch over all compiled
// (narrow, wide) dimension pairs, for half and bf16.
TLLM_FOR_MOE_ALL_WIDE_NARROW(TLLM_INST_MOE_BGMV_TWOSIDE, half, half, half)
TLLM_FOR_MOE_ALL_WIDE_NARROW(TLLM_INST_MOE_BGMV_TWOSIDE, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16)

template <typename T, bool PER_PAIR_INPUT>
static bool launchMoeShrink(T* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, uint32_t featIn, uint32_t featOut, int64_t numPairs, int64_t numSlices,
    int64_t numExperts, int64_t numTokens, int64_t loraStride, cudaStream_t stream)
{
    switch (packU32(featIn, featOut))
    {
#define TLLM_CASE_MOE_SHRINK(in_T, out_T, W_T, narrow, wide)                                                           \
    case packU32(wide, narrow):                                                                                        \
        moeBgmvShrinkSliced<wide, narrow, in_T, out_T, W_T, PER_PAIR_INPUT>(Y, X, wPtr, sortedTokenIds, expertIds,     \
            loraIndices, numPairs, numSlices, numExperts, numTokens, loraStride, 1.0f, stream);                        \
        return true;
        TLLM_FOR_MOE_ALL_WIDE_NARROW(TLLM_CASE_MOE_SHRINK, T, T, T)
#undef TLLM_CASE_MOE_SHRINK
    default: return false;
    }
}

template <typename T, bool FINALIZE>
static bool launchMoeExpand(float* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, float const* topkWeights, int64_t const* sliceStartLoc, uint32_t featIn,
    uint32_t featOut, int64_t numPairs, int64_t numSlices, int64_t numExperts, int64_t totalFeatOut, int64_t numTokens,
    int64_t loraStride, cudaStream_t stream)
{
    switch (packU32(featIn, featOut))
    {
#define TLLM_CASE_MOE_EXPAND(in_T, out_T, W_T, narrow, wide)                                                           \
    case packU32(narrow, wide):                                                                                        \
        moeBgmvExpandSliced<narrow, wide, in_T, W_T, FINALIZE>(Y, X, wPtr, sortedTokenIds, expertIds, loraIndices,     \
            topkWeights, sliceStartLoc, numPairs, numSlices, numExperts, totalFeatOut, wide, numTokens, loraStride,    \
            1.0f, stream);                                                                                             \
        return true;
        TLLM_FOR_MOE_ALL_WIDE_NARROW(TLLM_CASE_MOE_EXPAND, T, T, T)
#undef TLLM_CASE_MOE_EXPAND
    default: return false;
    }
}

template <typename T>
bool bgmvMoeShrink(T* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, int64_t featIn, int64_t featOut, int64_t numPairs, int64_t numSlices,
    int64_t numExperts, int64_t numTokens, int64_t loraStride, bool perPairInput, cudaStream_t stream)
{
    return perPairInput ? launchMoeShrink<T, true>(Y, X, wPtr, sortedTokenIds, expertIds, loraIndices,
               static_cast<uint32_t>(featIn), static_cast<uint32_t>(featOut), numPairs, numSlices, numExperts,
               numTokens, loraStride, stream)
                        : launchMoeShrink<T, false>(Y, X, wPtr, sortedTokenIds, expertIds, loraIndices,
                              static_cast<uint32_t>(featIn), static_cast<uint32_t>(featOut), numPairs, numSlices,
                              numExperts, numTokens, loraStride, stream);
}

template <typename T>
bool bgmvMoeExpand(float* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, float const* topkWeights, int64_t const* sliceStartLoc, int64_t featIn,
    int64_t featOut, int64_t numPairs, int64_t numSlices, int64_t numExperts, int64_t totalFeatOut, int64_t numTokens,
    int64_t loraStride, bool finalize, cudaStream_t stream)
{
    return finalize ? launchMoeExpand<T, true>(Y, X, wPtr, sortedTokenIds, expertIds, loraIndices, topkWeights,
               sliceStartLoc, static_cast<uint32_t>(featIn), static_cast<uint32_t>(featOut), numPairs, numSlices,
               numExperts, totalFeatOut, numTokens, loraStride, stream)
                    : launchMoeExpand<T, false>(Y, X, wPtr, sortedTokenIds, expertIds, loraIndices, topkWeights,
                          sliceStartLoc, static_cast<uint32_t>(featIn), static_cast<uint32_t>(featOut), numPairs,
                          numSlices, numExperts, totalFeatOut, numTokens, loraStride, stream);
}

// Public entry-point instantiations (half and bf16).
template bool bgmvMoeShrink<half>(half*, half const*, half**, int64_t const*, int64_t const*, int64_t const*, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, cudaStream_t);
template bool bgmvMoeShrink<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16**, int64_t const*,
    int64_t const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool,
    cudaStream_t);
template bool bgmvMoeExpand<half>(float*, half const*, half**, int64_t const*, int64_t const*, int64_t const*,
    float const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool,
    cudaStream_t);
template bool bgmvMoeExpand<__nv_bfloat16>(float*, __nv_bfloat16 const*, __nv_bfloat16**, int64_t const*,
    int64_t const*, int64_t const*, float const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool, cudaStream_t);

} // namespace bgmv_moe
} // namespace kernels

TRTLLM_NAMESPACE_END
