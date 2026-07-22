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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeRouterGemm.h"
#include <algorithm>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Shared-memory tiled GEMM computing logits[M, N] = act[M, K] @ weight[N, K]^T.
// See moeRouterGemm.h for the precision contract.
//
// Each block computes a [BM, BN] output tile and each thread a [TM, TN]
// micro-tile. Staging tiles are stored K-major so the global activation load is
// fully coalesced. The bf16 activation is widened to fp32 in shared memory and
// accumulated in fp32.
//
// Split-K parallelism: the reduction axis K is partitioned into
// ``gridDim.z`` chunks of ``k_tile`` elements (a multiple of BK). Block
// ``blockIdx.z`` reduces the sub-range ``[z * k_tile, (z + 1) * k_tile)`` into
// fp32 registers and, when ``split_k`` is set, atomically adds its partial tile
// into the (pre-zeroed) output. This is what keeps the GPU busy for the decode
// shape (small M and N, large K), where the non-split grid collapses to a
// single block. When ``gridDim.z == 1`` the epilogue writes results directly
// and no atomics or output pre-zeroing are needed.
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void moe_router_gemm_kernel(float* __restrict__ out, T const* __restrict__ act,
    float const* __restrict__ weight, int M, int N, int K, int k_tile, bool split_k)
{
    // K-major staging tiles: As[k][m], Bs[k][n].
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    constexpr int kThreads = (BM / TM) * (BN / TN);
    constexpr int kThreadsPerRow = BN / TN;

    int const block_row = blockIdx.x * BM; // token (M) offset of this tile
    int const block_col = blockIdx.y * BN; // expert (N) offset of this tile

    // Split-K: this block reduces the sub-range [k_begin, k_end) of the K axis.
    int const k_begin = blockIdx.z * k_tile;
    int const k_end = min(K, k_begin + k_tile);

    int const tid = threadIdx.x;
    int const thread_col = tid % kThreadsPerRow; // which N micro-column
    int const thread_row = tid / kThreadsPerRow; // which M micro-row

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = k_begin; k0 < k_end; k0 += BK)
    {
        // Load activation tile [BM, BK] (bf16 -> fp32), stored K-major in smem.
#pragma unroll
        for (int idx = tid; idx < BM * BK; idx += kThreads)
        {
            int const r = idx / BK; // row within tile (token)
            int const c = idx % BK; // col within tile (k)
            int const gr = block_row + r;
            int const gc = k0 + c;
            float v = 0.0f;
            if (gr < M && gc < K)
            {
                v = static_cast<float>(act[static_cast<int64_t>(gr) * K + gc]);
            }
            As[c * BM + r] = v;
        }
        // Load weight tile [BN, BK] (fp32), stored K-major in smem.
#pragma unroll
        for (int idx = tid; idx < BN * BK; idx += kThreads)
        {
            int const r = idx / BK; // row within tile (expert)
            int const c = idx % BK; // col within tile (k)
            int const gr = block_col + r;
            int const gc = k0 + c;
            float v = 0.0f;
            if (gr < N && gc < K)
            {
                v = weight[static_cast<int64_t>(gr) * K + gc];
            }
            Bs[c * BN + r] = v;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                a_frag[i] = As[kk * BM + thread_row * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                b_frag[j] = Bs[kk * BN + thread_col * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int const gr = block_row + thread_row * TM + i;
        if (gr >= M)
        {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int const gc = block_col + thread_col * TN + j;
            if (gc < N)
            {
                int64_t const idx = static_cast<int64_t>(gr) * N + gc;
                // Multiple K-splits accumulate into the same (pre-zeroed) output
                // element, so the partials must be combined atomically. The
                // single-split path writes directly to avoid atomic traffic.
                if (split_k)
                {
                    atomicAdd(&out[idx], acc[i][j]);
                }
                else
                {
                    out[idx] = acc[i][j];
                }
            }
        }
    }
}

template <typename T>
void invokeMoeRouterGemm(float* output, T const* act, float const* weight, int num_tokens, int num_experts,
    int hidden_dim, cudaStream_t stream)
{
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int kThreads = (BM / TM) * (BN / TN); // 256

    int const m_tiles = (num_tokens + BM - 1) / BM;
    int const n_tiles = (num_experts + BN - 1) / BN;
    int const mn_tiles = m_tiles * n_tiles;
    int const bk_steps = (hidden_dim + BK - 1) / BK; // number of BK-chunks along K

    // The (M, N) grid alone under-fills the GPU for the decode router shape
    // (e.g. M=64, N=128 -> a single block). Split the K axis so the launch
    // covers ~2 waves of SMs, then combine the partials in the epilogue. The
    // split is capped at bk_steps (one BK-chunk per block) and clamped to >= 1.
    int const sm_count = tensorrt_llm::common::getMultiProcessorCount();
    int const target_blocks = 2 * sm_count;
    int desired_k_tiles = (target_blocks + mn_tiles - 1) / mn_tiles;
    desired_k_tiles = std::min(std::max(desired_k_tiles, 1), bk_steps);

    // Distribute the BK-chunks evenly; each split covers steps_per_tile whole
    // BK-chunks so every block's K sub-range stays BK-aligned.
    int const steps_per_tile = (bk_steps + desired_k_tiles - 1) / desired_k_tiles;
    int const k_tile = steps_per_tile * BK;
    int const k_tiles = (hidden_dim + k_tile - 1) / k_tile;
    bool const split_k = k_tiles > 1;

    // Split-K accumulates into the output atomically, so it must start zeroed.
    if (split_k)
    {
        tensorrt_llm::common::check_cuda_error(
            cudaMemsetAsync(output, 0, sizeof(float) * static_cast<size_t>(num_tokens) * num_experts, stream));
    }

    dim3 const grid(m_tiles, n_tiles, k_tiles);
    dim3 const block(kThreads);
    moe_router_gemm_kernel<T, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, act, weight, num_tokens, num_experts, hidden_dim, k_tile, split_k);
}

template void invokeMoeRouterGemm<__nv_bfloat16>(
    float*, __nv_bfloat16 const*, float const*, int, int, int, cudaStream_t);

template void invokeMoeRouterGemm<half>(float*, half const*, float const*, int, int, int, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
