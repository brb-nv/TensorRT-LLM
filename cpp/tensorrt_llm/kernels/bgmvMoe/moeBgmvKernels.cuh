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
 * Multi-LoRA MoE BGMV (batched gather matrix-vector) kernels. Ported from
 * FlashInfer's csrc/bgmv_moe/ (Apache-2.0) for the routed-expert MoE LoRA
 * delta builders used by the TRTLLM-gen FP8 block-scale path. The FlashInfer
 * `vec_t` dependency is replaced by the self-contained `VecT` helper below and
 * the TVM-FFI dispatch is replaced by a torch custom op (see thop/bgmvMoeOp.cpp).
 *
 * Two kernels:
 *   1. Shrink: y[slice, pair, rank] += x[token|pair] @ lora_a[expert, lora_id]
 *   2. Expand: y[token|pair, feat] += shrink_out[pair, rank] @ lora_b[expert, lora_id] * w
 */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace bgmv_moe
{

namespace cg = cooperative_groups;

// BGMV MoE kernel tuning parameters (from FlashInfer kernel_config.h).
// Target: H100/H200 (sm_90, 228 KB shared memory per SM). Also supports
// sm_80 with reduced pipeline depth.
struct MoeShrinkKernelConfig
{
    static constexpr int tx = 32;       // threads per warp (x-dimension)
    static constexpr int ty = 4;        // number of warps (y-dimension)
    static constexpr int vec_size = 8;  // elements per vectorized load
    static constexpr int rank_tile = 8; // rank elements per block (8x X reuse)

    // Multi-pair decode path: PPB=4 pairs per block for decode, PPB=1 for
    // prefill (grid already saturates the GPU).
    static constexpr int pairs_per_block_prefill = 1;
    static constexpr int pairs_per_block_decode = 4;
    static constexpr int decode_threshold = 32;

    // Pipeline depth: 3 stages on sm_90 decode, 2 stages otherwise.
    static constexpr int num_stages_default = 2;
    static constexpr int num_stages_extended = 3;
};

struct MoeExpandKernelConfig
{
    static constexpr int tz = 4;
    static constexpr int vec_size = 8;
};

// Minimal fixed-width vector helper replacing FlashInfer's vec_t. Only the
// contiguous element load + float conversion used by the BGMV kernels is
// needed. alignas keeps the shared-memory reads naturally aligned; the async
// global->shared copies are done separately via cuda::memcpy_async with an
// explicit aligned_size_t and do not go through this type.
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size <= 16 ? sizeof(T) * vec_size : 16) VecT
{
    T vals[vec_size];

    __device__ __forceinline__ void load(T const* ptr)
    {
#pragma unroll
        for (int i = 0; i < vec_size; ++i)
        {
            vals[i] = ptr[i];
        }
    }

    __device__ __forceinline__ T operator[](int i) const
    {
        return vals[i];
    }
};

// ============================================================
// MoE BGMV Shrink Sliced Kernel
//
// Optimizations:
//   1. RANK_TILE tiling — reuse X tile across RANK_TILE weight rows
//   2. Multi-pair — PPB pairs per block (PPB=4 decode, PPB=1 prefill)
//   3. Deep pipeline — NUM_STAGES async pipeline stages (3 decode, 2 prefill)
//
// Uses dynamic shared memory so large configurations compile for all archs.
// The host wrapper calls cudaFuncSetAttribute on sm_80+ to raise the limit.
// ============================================================
template <int feat_in, int feat_out, int RANK_TILE, int PAIRS_PER_BLOCK, int NUM_STAGES,
    size_t vec_size, size_t X_copy_size, size_t W_copy_size, int tx, int ty, typename in_T, typename out_T,
    typename W_T, bool PER_PAIR_INPUT = false>
__global__ void moeBgmvShrinkSlicedKernel(out_T* __restrict__ Y, in_T const* __restrict__ X,
    W_T** __restrict__ w_ptr, int64_t const* __restrict__ sorted_token_ids,
    int64_t const* __restrict__ expert_ids, int64_t const* __restrict__ lora_indices, int64_t num_pairs,
    int64_t max_loras, int64_t num_tokens, float scale)
{
    // Per-adapter pointer model (TRT-LLM PEFT layout): w_ptr[slice, lora_id]
    // points to that adapter's [num_experts, feat_out, feat_in]-contiguous LoRA-A
    // bank, so the per-expert offset is a compile-time constant feat_in*feat_out.
    // (This deliberately diverges from FlashInfer's per-expert base + adapter
    // stride model; see moeBgmvKernels.h.)
    constexpr int64_t kExpertStride = static_cast<int64_t>(feat_in) * feat_out;
    int const slice_id = blockIdx.z;
    int const pair_block_idx = blockIdx.x;
    int const rank_tile_idx = blockIdx.y;
    int const j0 = rank_tile_idx * RANK_TILE;
    int const p0 = pair_block_idx * PAIRS_PER_BLOCK;

    auto block = cg::this_thread_block();
    constexpr size_t tile_size = tx * ty * vec_size;
    constexpr size_t num_tiles = (feat_in + tile_size - 1) / tile_size;

    // Per-pair metadata
    in_T const* X_tok[PAIRS_PER_BLOCK];
    W_T const* W_base[PAIRS_PER_BLOCK];
    bool pair_valid[PAIRS_PER_BLOCK];

#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
    {
        int const pair_idx = p0 + pp;
        if (pair_idx < num_pairs)
        {
            int64_t const token_idx = sorted_token_ids[pair_idx];
            if (token_idx >= 0 && token_idx < num_tokens)
            {
                int64_t const eid = expert_ids[pair_idx];
                int64_t const lid = lora_indices[token_idx];
                if (lid >= 0)
                {
                    // PER_PAIR_INPUT: input row = pair (FC2 activation) vs token (FC1).
                    int64_t const in_row = PER_PAIR_INPUT ? static_cast<int64_t>(pair_idx) : token_idx;
                    X_tok[pp] = X + in_row * feat_in;
                    W_base[pp] = w_ptr[slice_id * max_loras + lid] + eid * kExpertStride + j0 * feat_in;
                    pair_valid[pp] = true;
                    continue;
                }
            }
        }
        X_tok[pp] = nullptr;
        W_base[pp] = nullptr;
        pair_valid[pp] = false;
    }

    bool any_valid = false;
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
    {
        any_valid |= pair_valid[pp];
    }
    if (!any_valid)
    {
        return;
    }

    // Dynamic shared memory layout
    extern __shared__ char smem[];
    constexpr size_t x_elems = NUM_STAGES * PAIRS_PER_BLOCK * tile_size;
    constexpr size_t w_elems = NUM_STAGES * PAIRS_PER_BLOCK * RANK_TILE * tile_size;
    in_T* X_shared = reinterpret_cast<in_T*>(smem);
    W_T* W_shared = reinterpret_cast<W_T*>(smem + x_elems * sizeof(in_T));
    float* y_warpwise = reinterpret_cast<float*>(smem + x_elems * sizeof(in_T) + w_elems * sizeof(W_T));

    auto pipe = cuda::make_pipeline();
    size_t const toff = (threadIdx.y * tx + threadIdx.x) * vec_size;

    float y_acc[PAIRS_PER_BLOCK][RANK_TILE];
#pragma unroll
    for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
    {
#pragma unroll
        for (int r = 0; r < RANK_TILE; ++r)
        {
            y_acc[pp][r] = 0.f;
        }
    }

    VecT<in_T, vec_size> x_vec;
    VecT<W_T, vec_size> w_vec;

    // Prologue: fill pipeline
    constexpr size_t pro = (num_tiles < NUM_STAGES) ? num_tiles : NUM_STAGES;
#pragma unroll
    for (size_t t = 0; t < pro; ++t)
    {
        size_t const s = t % NUM_STAGES;
        size_t const tb = t * tile_size;
        pipe.producer_acquire();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (pair_valid[pp] && tb + toff < feat_in)
            {
                cuda::memcpy_async(X_shared + (s * PAIRS_PER_BLOCK + pp) * tile_size + toff, X_tok[pp] + tb + toff,
                    cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
#pragma unroll
                for (int r = 0; r < RANK_TILE; ++r)
                {
                    if (j0 + r < feat_out)
                    {
                        cuda::memcpy_async(W_shared + ((s * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff,
                            W_base[pp] + r * feat_in + tb + toff, cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
                    }
                }
            }
        }
        pipe.producer_commit();
    }

    // Main loop
    for (size_t t = pro; t < num_tiles; ++t)
    {
        size_t const cs = (t - pro) % NUM_STAGES;
        pipe.consumer_wait();
        block.sync();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (!pair_valid[pp])
            {
                continue;
            }
            x_vec.load(X_shared + (cs * PAIRS_PER_BLOCK + pp) * tile_size + toff);
#pragma unroll
            for (int r = 0; r < RANK_TILE; ++r)
            {
                if (j0 + r < feat_out)
                {
                    w_vec.load(W_shared + ((cs * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff);
                    float sum = 0.f;
#pragma unroll
                    for (size_t i = 0; i < vec_size; ++i)
                    {
                        sum += float(w_vec[i]) * float(x_vec[i]) * scale;
                    }
#pragma unroll
                    for (size_t off = tx / 2; off > 0; off /= 2)
                    {
                        sum += __shfl_down_sync(0xffffffff, sum, off);
                    }
                    if (threadIdx.x == 0)
                    {
                        y_warpwise[pp * RANK_TILE * ty + r * ty + threadIdx.y] = sum;
                    }
                }
            }
        }
        block.sync();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (!pair_valid[pp])
            {
                continue;
            }
#pragma unroll
            for (int r = 0; r < RANK_TILE; ++r)
            {
                if (j0 + r < feat_out)
                {
                    float v = 0.f;
                    for (int w = 0; w < ty; ++w)
                    {
                        v += y_warpwise[pp * RANK_TILE * ty + r * ty + w];
                    }
                    y_acc[pp][r] += v;
                }
            }
        }
        block.sync();
        pipe.consumer_release();

        // Load next tile
        size_t const ls = t % NUM_STAGES;
        size_t const tb = t * tile_size;
        pipe.producer_acquire();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (pair_valid[pp] && tb + toff < feat_in)
            {
                cuda::memcpy_async(X_shared + (ls * PAIRS_PER_BLOCK + pp) * tile_size + toff, X_tok[pp] + tb + toff,
                    cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
#pragma unroll
                for (int r = 0; r < RANK_TILE; ++r)
                {
                    if (j0 + r < feat_out)
                    {
                        cuda::memcpy_async(W_shared + ((ls * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff,
                            W_base[pp] + r * feat_in + tb + toff, cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
                    }
                }
            }
        }
        pipe.producer_commit();
    }

    // Epilogue: drain remaining pipeline stages
    for (size_t t = (num_tiles > pro ? num_tiles - pro : 0); t < num_tiles; ++t)
    {
        size_t const cs = t % NUM_STAGES;
        size_t const ts = t * tile_size;
        pipe.consumer_wait();
        block.sync();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (!pair_valid[pp])
            {
                continue;
            }
            x_vec.load(X_shared + (cs * PAIRS_PER_BLOCK + pp) * tile_size + toff);
#pragma unroll
            for (int r = 0; r < RANK_TILE; ++r)
            {
                if (j0 + r < feat_out)
                {
                    w_vec.load(W_shared + ((cs * PAIRS_PER_BLOCK + pp) * RANK_TILE + r) * tile_size + toff);
                    float sum = 0.f;
#pragma unroll
                    for (size_t i = 0; i < vec_size; ++i)
                    {
                        sum += float(w_vec[i]) * float(x_vec[i]) * scale;
                    }
#pragma unroll
                    for (size_t off = tx / 2; off > 0; off /= 2)
                    {
                        sum += __shfl_down_sync(0xffffffff, sum, off);
                    }
                    if (threadIdx.x == 0)
                    {
                        if (t == num_tiles - 1)
                        {
                            sum = (ts + threadIdx.y * tx * vec_size < feat_in) ? sum : 0.f;
                        }
                        y_warpwise[pp * RANK_TILE * ty + r * ty + threadIdx.y] = sum;
                    }
                }
            }
        }
        block.sync();
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (!pair_valid[pp])
            {
                continue;
            }
#pragma unroll
            for (int r = 0; r < RANK_TILE; ++r)
            {
                if (j0 + r < feat_out)
                {
                    float v = 0.f;
                    for (int w = 0; w < ty; ++w)
                    {
                        v += y_warpwise[pp * RANK_TILE * ty + r * ty + w];
                    }
                    y_acc[pp][r] += v;
                }
            }
        }
        block.sync();
        pipe.consumer_release();
    }

    // Write results
    if (block.thread_rank() == 0)
    {
#pragma unroll
        for (int pp = 0; pp < PAIRS_PER_BLOCK; ++pp)
        {
            if (!pair_valid[pp])
            {
                continue;
            }
#pragma unroll
            for (int r = 0; r < RANK_TILE; ++r)
            {
                if (j0 + r < feat_out)
                {
                    Y[slice_id * num_pairs * feat_out + (p0 + pp) * feat_out + j0 + r]
                        += static_cast<out_T>(y_acc[pp][r]);
                }
            }
        }
    }
}

// ============================================================
// MoE BGMV Expand Sliced Kernel
//
// FINALIZE=true: combine a token's experts (output row = token, *topk_w, atomicAdd).
// FINALIZE=false: per-pair, unweighted plain store (output row = pair). Both early-return
// skipped pairs into a caller-pre-zeroed Y; only the final write differs (if constexpr).
// ============================================================
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz, typename in_T, typename W_T,
    bool FINALIZE = true>
__global__ void moeBgmvExpandSlicedKernel(float* __restrict__ Y, in_T const* __restrict__ X,
    W_T** __restrict__ w_ptr, int64_t const* __restrict__ sorted_token_ids,
    int64_t const* __restrict__ expert_ids, int64_t const* __restrict__ lora_indices,
    float const* __restrict__ topk_weights, int64_t const* __restrict__ slice_start_loc, int64_t num_pairs,
    int64_t max_loras, int64_t total_feat_out, int32_t current_feat_out, int64_t num_tokens, float scale)
{
    // Per-adapter pointer model: w_ptr[slice, lora_id] is that adapter's
    // [num_experts, feat_out, feat_in]-contiguous LoRA-B bank; per-expert offset
    // is the compile-time constant feat_in*feat_out. (Diverges from FlashInfer;
    // see moeBgmvKernels.h.)
    constexpr int64_t kExpertStride = static_cast<int64_t>(feat_in) * feat_out;
    size_t pair_idx = blockIdx.x;
    size_t tile_idx = blockIdx.y;
    int64_t token_idx = sorted_token_ids[pair_idx];
    if (token_idx < 0 || token_idx >= num_tokens)
    {
        return;
    }
    int64_t lora_id = lora_indices[token_idx];
    if (lora_id < 0)
    {
        return;
    }
    int slice_id = blockIdx.z;
    int64_t expert_id = expert_ids[pair_idx];
    int64_t col_offset = slice_start_loc[slice_id];
    W_T const* W = w_ptr[slice_id * max_loras + lora_id] + expert_id * kExpertStride;
    auto block = cg::this_thread_block();
    VecT<in_T, vec_size> x_vec;
    x_vec.load(X + slice_id * num_pairs * feat_in + pair_idx * feat_in + threadIdx.x * vec_size);
    VecT<W_T, vec_size> w_vec;
    w_vec.load(W + (tile_idx * tz * ty) * feat_in + block.thread_rank() * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
    {
        sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
    cg::thread_block_tile<tx> g = cg::tiled_partition<tx>(block);
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2)
    {
        sum += g.shfl_down(sum, offset);
    }
    sum = g.shfl(sum, 0);
    if (threadIdx.x == 0)
    {
        int out_col = col_offset + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y;
        if constexpr (FINALIZE)
        {
            float topk_w = topk_weights[pair_idx];
            atomicAdd(Y + token_idx * total_feat_out + out_col, sum * topk_w);
        }
        else
        {
            Y[pair_idx * total_feat_out + out_col] = sum;
        }
    }
}

// ============================================================
// Host-side dispatch: Shrink
// ============================================================
template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T, bool PER_PAIR_INPUT>
void moeBgmvShrinkSliced(out_T* __restrict__ Y, in_T const* __restrict__ X, W_T** __restrict__ w_ptr,
    int64_t const* sorted_token_ids, int64_t const* expert_ids, int64_t const* lora_indices, int64_t num_pairs,
    int64_t num_slices, int64_t max_loras, int64_t num_tokens, float scale, cudaStream_t stream)
{
    constexpr int cfg_tx = MoeShrinkKernelConfig::tx;
    constexpr int cfg_ty = MoeShrinkKernelConfig::ty;
    constexpr int RT = MoeShrinkKernelConfig::rank_tile;
    constexpr int gy = (feat_out + RT - 1) / RT;
    constexpr size_t fvs = MoeShrinkKernelConfig::vec_size;

    // Runtime: detect sm_90+ for extended shared memory / deeper pipeline.
    int dev;
    cudaGetDevice(&dev);
    int sm_major = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
    bool const extended = (sm_major >= 9);
    bool const decode = (num_pairs <= MoeShrinkKernelConfig::decode_threshold);

    int const ppb = (extended && decode) ? MoeShrinkKernelConfig::pairs_per_block_decode
                                         : MoeShrinkKernelConfig::pairs_per_block_prefill;
    int const nstg = (extended && decode) ? MoeShrinkKernelConfig::num_stages_extended
                                          : MoeShrinkKernelConfig::num_stages_default;

#define TLLM_BGMV_MOE_LAUNCH(PPB, NSTG, VS)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        constexpr size_t ts = cfg_tx * cfg_ty * (VS);                                                                  \
        constexpr size_t shmem = (NSTG) * (PPB) * ts * sizeof(in_T) + (NSTG) * (PPB) * RT * ts * sizeof(W_T)           \
            + (PPB) * RT * cfg_ty * sizeof(float);                                                                     \
        auto kfn = &moeBgmvShrinkSlicedKernel<feat_in, feat_out, RT, (PPB), (NSTG), (VS), (VS) * sizeof(in_T),         \
            (VS) * sizeof(W_T), cfg_tx, cfg_ty, in_T, out_T, W_T, PER_PAIR_INPUT>;                                     \
        if constexpr (shmem > 48 * 1024)                                                                               \
        {                                                                                                              \
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);                       \
        }                                                                                                              \
        dim3 g((int) ((num_pairs + (PPB) -1) / (PPB)), gy, num_slices);                                                \
        kfn<<<g, dim3(cfg_tx, cfg_ty), shmem, stream>>>(                                                               \
            Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, num_pairs, max_loras, num_tokens, scale);        \
    } while (0)

#define TLLM_BGMV_MOE_DISPATCH(VS)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (ppb == 4 && nstg == 3)                                                                                     \
        {                                                                                                              \
            TLLM_BGMV_MOE_LAUNCH(4, 3, VS);                                                                            \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_BGMV_MOE_LAUNCH(1, 2, VS);                                                                            \
        }                                                                                                              \
    } while (0)

    if constexpr (feat_in % (fvs * cfg_tx) == 0)
    {
        TLLM_BGMV_MOE_DISPATCH(fvs);
    }
    else if constexpr (feat_in % (fvs / 2 * cfg_tx) == 0)
    {
        TLLM_BGMV_MOE_DISPATCH(fvs / 2);
    }
    else if constexpr (feat_in % (fvs / 4 * cfg_tx) == 0)
    {
        TLLM_BGMV_MOE_DISPATCH(fvs / 4);
    }
    else if constexpr (feat_in % cfg_tx == 0)
    {
        TLLM_BGMV_MOE_DISPATCH(1);
    }

#undef TLLM_BGMV_MOE_DISPATCH
#undef TLLM_BGMV_MOE_LAUNCH
}

// ============================================================
// Host-side dispatch: Expand
// ============================================================
template <int feat_in, int feat_out, typename in_T, typename W_T, bool FINALIZE>
void moeBgmvExpandSliced(float* __restrict__ Y, in_T const* __restrict__ X, W_T** __restrict__ w_ptr,
    int64_t const* sorted_token_ids, int64_t const* expert_ids, int64_t const* lora_indices,
    float const* topk_weights, int64_t const* slice_start_loc, int64_t num_pairs, int64_t num_slices,
    int64_t max_loras, int64_t total_feat_out, int32_t current_feat_out, int64_t num_tokens, float scale,
    cudaStream_t stream)
{
    constexpr size_t vec_size = MoeExpandKernelConfig::vec_size;
    constexpr int tz = MoeExpandKernelConfig::tz;
    static_assert(feat_in % vec_size == 0);
    constexpr int tx = feat_in / vec_size;

    if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0)
    {
        constexpr int ty = 32 / tx;
        moeBgmvExpandSlicedKernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T, FINALIZE>
            <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(Y, X, w_ptr,
                sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc, num_pairs, max_loras,
                total_feat_out, current_feat_out, num_tokens, scale);
    }
    else if constexpr (16 % tx == 0 && feat_out % (16 / tx * tz) == 0)
    {
        constexpr int ty = 16 / tx;
        moeBgmvExpandSlicedKernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T, FINALIZE>
            <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(Y, X, w_ptr,
                sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc, num_pairs, max_loras,
                total_feat_out, current_feat_out, num_tokens, scale);
    }
    else if constexpr (8 % tx == 0 && feat_out % (8 / tx * tz) == 0)
    {
        constexpr int ty = 8 / tx;
        moeBgmvExpandSlicedKernel<feat_in, feat_out, vec_size, tx, ty, tz, in_T, W_T, FINALIZE>
            <<<dim3(num_pairs, feat_out / (ty * tz), num_slices), dim3(tx, ty, tz), 0, stream>>>(Y, X, w_ptr,
                sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc, num_pairs, max_loras,
                total_feat_out, current_feat_out, num_tokens, scale);
    }
}

// ============================================================
// Compiled (narrow=rank, wide=hidden/intermediate) dimension pairs.
// narrow = LoRA rank (8, 16, 32, 64); wide % 32 == 0. Keep in sync with the
// wide dims used by supported MoE models (see FlashInfer moe_bgmv_config.h).
// ============================================================
// clang-format off
#define TLLM_FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, narrow)                                                             \
    f(in_T, out_T, W_T, narrow, 384) f(in_T, out_T, W_T, narrow, 736) f(in_T, out_T, W_T, narrow, 768)                 \
        f(in_T, out_T, W_T, narrow, 1024) f(in_T, out_T, W_T, narrow, 1344) f(in_T, out_T, W_T, narrow, 1472)          \
            f(in_T, out_T, W_T, narrow, 1536) f(in_T, out_T, W_T, narrow, 1856) f(in_T, out_T, W_T, narrow, 2048)      \
                f(in_T, out_T, W_T, narrow, 2112) f(in_T, out_T, W_T, narrow, 2688) f(in_T, out_T, W_T, narrow, 2816)  \
                    f(in_T, out_T, W_T, narrow, 2880) f(in_T, out_T, W_T, narrow, 2944)                                \
                        f(in_T, out_T, W_T, narrow, 3072) f(in_T, out_T, W_T, narrow, 4096)                            \
                            f(in_T, out_T, W_T, narrow, 5120) f(in_T, out_T, W_T, narrow, 5888)                        \
                                f(in_T, out_T, W_T, narrow, 7168) f(in_T, out_T, W_T, narrow, 8192)                    \
                                    f(in_T, out_T, W_T, narrow, 10240) f(in_T, out_T, W_T, narrow, 14336)              \
                                        f(in_T, out_T, W_T, narrow, 16384) f(in_T, out_T, W_T, narrow, 28672)

#define TLLM_FOR_MOE_ALL_WIDE_NARROW(f, in_T, out_T, W_T)                                                               \
    TLLM_FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 8) TLLM_FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 16)                        \
        TLLM_FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 32) TLLM_FOR_MOE_ALL_WIDE(f, in_T, out_T, W_T, 64)
// clang-format on

// Explicit-instantiation helpers. Shrink is [wide -> narrow] (feat_in=wide,
// feat_out=rank); expand is [narrow -> wide] (feat_in=rank, feat_out=wide).
#define TLLM_INST_MOE_BGMV_SHRINK_SLICED(feat_in, feat_out, in_T, out_T, W_T)                                          \
    template void moeBgmvShrinkSliced<feat_in, feat_out, in_T, out_T, W_T, false>(out_T*, in_T const*, W_T**,          \
        int64_t const*, int64_t const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, float, cudaStream_t);     \
    template void moeBgmvShrinkSliced<feat_in, feat_out, in_T, out_T, W_T, true>(out_T*, in_T const*, W_T**,           \
        int64_t const*, int64_t const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, float, cudaStream_t);

#define TLLM_INST_MOE_BGMV_EXPAND_SLICED(feat_in, feat_out, in_T, W_T)                                                 \
    template void moeBgmvExpandSliced<feat_in, feat_out, in_T, W_T, true>(float*, in_T const*, W_T**, int64_t const*,  \
        int64_t const*, int64_t const*, float const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, int32_t,    \
        int64_t, float, cudaStream_t);                                                                                \
    template void moeBgmvExpandSliced<feat_in, feat_out, in_T, W_T, false>(float*, in_T const*, W_T**, int64_t const*, \
        int64_t const*, int64_t const*, float const*, int64_t const*, int64_t, int64_t, int64_t, int64_t, int32_t,    \
        int64_t, float, cudaStream_t);

#define TLLM_INST_MOE_BGMV_TWOSIDE(in_T, out_T, W_T, narrow, wide)                                                     \
    TLLM_INST_MOE_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T)                                                   \
    TLLM_INST_MOE_BGMV_EXPAND_SLICED(narrow, wide, in_T, W_T)

} // namespace bgmv_moe
} // namespace kernels
} // namespace tensorrt_llm
