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

// Router GEMM for a checkpoint that keeps the gate weight in fp32.
//
// dsv3RouterGemm.cu templates one dtype over both operands, so a model whose
// router weight is fp32 (MiniMax-M3) cannot use it and falls back to an fp32
// cast of the activation plus a cuBLAS SGEMM. This kernel splits the operand
// dtypes: fp32 weight, bf16 or fp32 activation, fp32 accumulation and output.
//
// Derived from the DeepSeek-V3 router GEMM in
// kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu. The experts-per-block and
// token-group blocking below, and the per-shape geometry in
// invokeFp32RouterGemm, follow the fp32 router GEMM in vLLM (Apache-2.0),
// csrc/libtorch_stable/fp32_router_gemm.cu, added in
// https://github.com/vllm-project/vllm/pull/48335.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include "tensorrt_llm/kernels/fp32RouterGemm.h"

#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Load VPT weight values. The weight is always fp32, so VPT is set by the
// activation dtype: 4 (one float4) for an fp32 activation, 8 (two float4) for
// bf16, keeping both operand loads at the same 16-byte granularity.
template <int VPT>
__device__ __forceinline__ void loadWeight(float const* ptr, float* dst);

template <>
__device__ __forceinline__ void loadWeight<4>(float const* ptr, float* dst)
{
    float4 const v = *reinterpret_cast<float4 const*>(ptr);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
}

template <>
__device__ __forceinline__ void loadWeight<8>(float const* ptr, float* dst)
{
    float4 const v0 = *reinterpret_cast<float4 const*>(ptr);
    float4 const v1 = *reinterpret_cast<float4 const*>(ptr + 4);
    dst[0] = v0.x;
    dst[1] = v0.y;
    dst[2] = v0.z;
    dst[3] = v0.w;
    dst[4] = v1.x;
    dst[5] = v1.y;
    dst[6] = v1.z;
    dst[7] = v1.w;
}

// Load VPT activation values and widen to fp32.
template <typename T, int VPT>
__device__ __forceinline__ void loadActivation(T const* ptr, float* dst);

template <>
__device__ __forceinline__ void loadActivation<float, 4>(float const* ptr, float* dst)
{
    float4 const v = *reinterpret_cast<float4 const*>(ptr);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
}

template <>
__device__ __forceinline__ void loadActivation<__nv_bfloat16, 8>(__nv_bfloat16 const* ptr, float* dst)
{
    uint4 const v = *reinterpret_cast<uint4 const*>(ptr);
    __nv_bfloat16 const* bf16Ptr = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        dst[i] = __bfloat162float(bf16Ptr[i]);
    }
}

// InputT is the activation dtype; the weight and the output are fp32.
//
// Each block owns kEPB expert columns. kTGroups > 1 splits the tokens across
// groups of kBlockSize threads within the block: every group scans the same
// weight K-slices, so group 0 misses to DRAM and the rest hit L1 and weight
// traffic stays 1x, while the per-thread accumulator count drops by kTGroups.
// That matters because at kNumTokens 16 the 32 fp32 accumulators push the
// kernel to 128 registers per thread and one block per SM.
template <typename InputT, int kBlockSize, int kNumTokens, int kEPB, int kNumExperts, int kHiddenDim, int kTGroups = 1>
__global__ __launch_bounds__(kBlockSize* kTGroups, 1) void fp32RouterGemmKernel(
    float* out, InputT const* mat_a, float const* mat_b)
{
    constexpr int VPT = 16 / sizeof(InputT);
    constexpr int kElemsPerKIteration = VPT * kBlockSize;
    constexpr int kIterations = kHiddenDim / kElemsPerKIteration;
    static_assert(kHiddenDim % kElemsPerKIteration == 0);
    static_assert(kNumTokens % kTGroups == 0);
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = kBlockSize / kWarpSize; // per token group
    constexpr int kMG = kNumTokens / kTGroups;        // tokens per group

    int const eBase = blockIdx.x * kEPB;
    int const tid = threadIdx.x % kBlockSize;
    int const m0 = (threadIdx.x / kBlockSize) * kMG;
    int const warpId = tid / kWarpSize;
    int const laneId = tid % kWarpSize;

    float acc[kMG][kEPB] = {};
    __shared__ float smReduction[kNumTokens][kEPB][kNumWarps];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

#pragma unroll
    for (int ki = 0; ki < kIterations; ++ki)
    {
        int const kBase = ki * kElemsPerKIteration + tid * VPT;

        float bFloat[kEPB][VPT];
#pragma unroll
        for (int e = 0; e < kEPB; ++e)
        {
            loadWeight<VPT>(mat_b + static_cast<size_t>(eBase + e) * kHiddenDim + kBase, bFloat[e]);
        }

#pragma unroll
        for (int mIdx = 0; mIdx < kMG; ++mIdx)
        {
            float aFloat[VPT];
            loadActivation<InputT, VPT>(mat_a + static_cast<size_t>(m0 + mIdx) * kHiddenDim + kBase, aFloat);
#pragma unroll
            for (int e = 0; e < kEPB; ++e)
            {
#pragma unroll
                for (int k = 0; k < VPT; ++k)
                {
                    acc[mIdx][e] += aFloat[k] * bFloat[e][k];
                }
            }
        }
    }

    // Warp-level butterfly reduction.
#pragma unroll
    for (int m = 0; m < kMG; ++m)
    {
#pragma unroll
        for (int e = 0; e < kEPB; ++e)
        {
            float sum = acc[m][e];
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 16);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 8);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
            if (laneId == 0)
            {
                smReduction[m0 + m][e][warpId] = sum;
            }
        }
    }

    __syncthreads();

    // Parallel finalize: one thread per (token, expert) output.
    for (int idx = threadIdx.x; idx < kNumTokens * kEPB; idx += kBlockSize * kTGroups)
    {
        int const m = idx / kEPB;
        int const e = idx % kEPB;
        float finalSum = 0.F;
#pragma unroll
        for (int w = 0; w < kNumWarps; ++w)
        {
            finalSum += smReduction[m][e][w];
        }
        out[m * kNumExperts + eBase + e] = finalSum;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputT, int kBlockSize, int kEPB, int kNumTokens, int kNumExperts, int kHiddenDim, int kTGroups = 1>
void launchFp32RouterGemm(float* output, InputT const* mat_a, float const* mat_b, cudaStream_t stream)
{
    static_assert(kNumExperts % kEPB == 0);
    tensorrt_llm::common::launchWithPdlWhenEnabled("fp32RouterGemm",
        fp32RouterGemmKernel<InputT, kBlockSize, kNumTokens, kEPB, kNumExperts, kHiddenDim, kTGroups>,
        /*grid=*/kNumExperts / kEPB, /*block=*/kBlockSize * kTGroups, /*dynamicShmSize=*/0, stream, output, mat_a,
        mat_b);
}

} // namespace

template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a, float const* mat_b, cudaStream_t stream)
{
    // MiniMax-M3 (kNumExperts 128, kHiddenDim 6144), bf16 activation. The
    // legacy 128/1 geometry from dsv3RouterGemm only fills 128 blocks and pays
    // the same accumulator register cliff, so the block size and token grouping
    // are picked per token count from a B300 sweep:
    //   even kNumTokens in [6, 10] : 384 threads, 2 token groups
    //   even kNumTokens >= 12      : 192 threads, 2 token groups
    //   kNumTokens <= 5 or odd     : 384 threads, 1 token group
    // Only applied on Blackwell, where it was measured; earlier architectures
    // and fp32 activation keep the legacy geometry.
    if constexpr (std::is_same_v<InputT, __nv_bfloat16> && kNumExperts == 128 && kHiddenDim == 6144)
    {
        if (tensorrt_llm::common::getSMVersion() < 100)
        {
            launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(output, mat_a, mat_b, stream);
            return;
        }
        if constexpr (kNumTokens >= 12 && kNumTokens % 2 == 0)
        {
            launchFp32RouterGemm<InputT, 192, 1, kNumTokens, kNumExperts, kHiddenDim, 2>(output, mat_a, mat_b, stream);
        }
        else if constexpr (kNumTokens >= 6 && kNumTokens % 2 == 0)
        {
            launchFp32RouterGemm<InputT, 384, 1, kNumTokens, kNumExperts, kHiddenDim, 2>(output, mat_a, mat_b, stream);
        }
        else
        {
            launchFp32RouterGemm<InputT, 384, 1, kNumTokens, kNumExperts, kHiddenDim>(output, mat_a, mat_b, stream);
        }
    }
    else
    {
        launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(output, mat_a, mat_b, stream);
    }
}

// Explicit instantiations: kNumTokens 1..32 for (kNumExperts, kHiddenDim) ==
// (128, 6144) (MiniMax-M3), for both activation dtypes.
#define INSTANTIATE_FP32_ROUTER_GEMM(T, M, E, H)                                                                       \
    template void invokeFp32RouterGemm<T, M, E, H>(float*, T const*, float const*, cudaStream_t);

#define INSTANTIATE_FP32_ROUTER_GEMM_ALL_TOKENS(T, E, H)                                                               \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 1, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 2, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 3, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 4, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 5, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 6, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 7, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 8, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 9, E, H)                                                                           \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 10, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 11, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 12, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 13, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 14, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 15, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 16, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 17, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 18, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 19, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 20, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 21, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 22, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 23, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 24, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 25, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 26, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 27, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 28, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 29, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 30, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 31, E, H)                                                                          \
    INSTANTIATE_FP32_ROUTER_GEMM(T, 32, E, H)

INSTANTIATE_FP32_ROUTER_GEMM_ALL_TOKENS(float, 128, 6144)
INSTANTIATE_FP32_ROUTER_GEMM_ALL_TOKENS(__nv_bfloat16, 128, 6144)

#undef INSTANTIATE_FP32_ROUTER_GEMM_ALL_TOKENS
#undef INSTANTIATE_FP32_ROUTER_GEMM

} // namespace kernels

TRTLLM_NAMESPACE_END
