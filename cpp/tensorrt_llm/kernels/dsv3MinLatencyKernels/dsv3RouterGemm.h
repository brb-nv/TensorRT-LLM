/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv3MinLatencyKernels
{

//! \brief Skinny router GEMM: [kNumTokens, kHiddenDim] x [kHiddenDim, kNumExperts] -> fp32.
//!
//! \tparam T activation element type.
//! \tparam WeightT weight element type. Trails the non-type parameters so it can default to \p T,
//! which keeps the DeepSeek call sites and their explicit instantiations unchanged.
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim, typename WeightT = T>
void invokeRouterGemm(float* output, T const* mat_a, WeightT const* mat_b, cudaStream_t stream);

} // namespace kernels::dsv3MinLatencyKernels

TRTLLM_NAMESPACE_END
