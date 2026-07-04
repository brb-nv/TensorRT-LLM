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
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace bgmv_moe
{

// Multi-LoRA MoE BGMV shrink: y[slice, pair, rank] += x @ lora_a[lora_id, expert].
//
// T is the activation / LoRA-weight element type (half or __nv_bfloat16).
//
// Per-adapter pointer model (TRT-LLM PEFT layout, NOT FlashInfer's): wPtr is a
// device [numSlices, maxLoras] int64 table of per-(slice, adapter) LoRA base
// pointers (reinterpreted to T**), where each pointer addresses that adapter's
// [numExperts, feat_out, feat_in]-contiguous per-expert bank. The kernel adds
// the per-expert offset (a compile-time feat_in*feat_out) internally, so there
// is no adapter stride and no separate per-expert pointer table. This diverges
// from FlashInfer's csrc/bgmv_moe (per-expert base + adapter stride) because
// TRT-LLM stores each adapter's MoE LoRA weights as its own contiguous
// [numExperts, rank, feat] tensor (see lora_manager.py) and exposes per-adapter
// pointers (CudaGraphLoraParams.h_b_ptrs / eager weight_pointers). Rank is
// uniform across adapters in a call (encoded by the compiled feat dims).
// perPairInput selects the FC2 path where the input row is the routed pair
// (gathered post-activation) instead of the token (FC1 path).
//
// Returns false if the (featIn, featOut) pair was not compiled (see the
// dimension list in moeBgmvKernels.cuh).
template <typename T>
bool bgmvMoeShrink(T* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, int64_t featIn, int64_t featOut, int64_t numPairs, int64_t numSlices,
    int64_t maxLoras, int64_t numTokens, bool perPairInput, cudaStream_t stream);

// Multi-LoRA MoE BGMV expand: project shrink output through LoRA-B (per-adapter
// pointer model, same wPtr [numSlices, maxLoras] convention as bgmvMoeShrink).
// finalize=true does the routing-weighted per-token atomic combine (Y is
// [numTokens, totalFeatOut]); finalize=false does a per-pair unweighted store
// (Y is [numPairs, totalFeatOut]). Y is always float32 and must be
// zero-initialized by the caller. Returns false if (featIn, featOut) was not
// compiled.
template <typename T>
bool bgmvMoeExpand(float* Y, T const* X, T** wPtr, int64_t const* sortedTokenIds, int64_t const* expertIds,
    int64_t const* loraIndices, float const* topkWeights, int64_t const* sliceStartLoc, int64_t featIn,
    int64_t featOut, int64_t numPairs, int64_t numSlices, int64_t maxLoras, int64_t totalFeatOut, int64_t numTokens,
    bool finalize, cudaStream_t stream);

} // namespace bgmv_moe
} // namespace kernels

TRTLLM_NAMESPACE_END
