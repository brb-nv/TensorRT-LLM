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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Router GEMM for an fp32 gate weight: activation(InputT) x weight(fp32) -> fp32.
///
/// Distinct from kernels::dsv3MinLatencyKernels::invokeRouterGemm, which
/// templates a single dtype over both operands and so cannot serve a
/// checkpoint that keeps the router weight in fp32 (MiniMax-M3). The
/// activation may be bf16 or fp32; the weight and the output are always fp32.
///
/// mat_a is [kNumTokens, kHiddenDim] row-major, mat_b is [kNumExperts,
/// kHiddenDim] row-major (expert-major, i.e. an nn.Linear weight as stored),
/// output is [kNumTokens, kNumExperts] row-major.
///
/// Instantiated for (kNumExperts, kHiddenDim) == (128, 6144) and
/// kNumTokens in [1, 32]; see fp32RouterGemm.cu.
template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a, float const* mat_b, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
