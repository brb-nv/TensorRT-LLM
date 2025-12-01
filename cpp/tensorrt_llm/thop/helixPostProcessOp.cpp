/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/helixKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace torch_ext
{

template <typename T, typename Fn>
inline torch::Tensor helix_post_process_impl(
    torch::Tensor const& gathered_o, torch::Tensor const& gathered_stats, double scale, int cp_dim, Fn fn)
{
    CHECK_TH_CUDA(gathered_o);
    CHECK_CONTIGUOUS(gathered_o);
    CHECK_TH_CUDA(gathered_stats);
    CHECK_CONTIGUOUS(gathered_stats);
    auto tokens_dim = 1 - (cp_dim == 0 ? 0 : 1);
    auto heads_dim = 2 - (cp_dim == 2 ? 1 : 0);

    TORCH_CHECK(gathered_o.dim() == 4,
        "gathered_o must be 4D tensor [cp_size, "
        "num_tokens, num_heads, kv_lora_rank] "
        "where cp_size is at index cp_dim");
    TORCH_CHECK(gathered_stats.dim() == 4,
        "gathered_stats must be 4D tensor [cp_size, num_tokens, num_heads, 2] "
        "where cp_size is at index cp_dim");

    auto const cp_size = gathered_stats.sizes()[cp_dim];
    auto const num_tokens = gathered_stats.sizes()[tokens_dim];
    auto const num_heads = gathered_stats.sizes()[heads_dim];
    auto const kv_lora_rank = gathered_o.sizes()[3];

    // check remaining input tensor dimensions
    TORCH_CHECK(gathered_o.sizes()[cp_dim] == cp_size, "gathered_o cp_dim must match cp_size");
    TORCH_CHECK(gathered_o.sizes()[tokens_dim] == num_tokens, "gathered_o tokens_dim must match num_tokens");
    TORCH_CHECK(gathered_o.sizes()[heads_dim] == num_heads, "gathered_o heads_dim must match num_heads");

    TORCH_CHECK(gathered_stats.sizes()[3] == 2, "gathered_stats last dimension must be 2");

    // Check data types
    TORCH_CHECK(
        gathered_o.scalar_type() == at::ScalarType::Half || gathered_o.scalar_type() == at::ScalarType::BFloat16,
        "gathered_o must be half or bfloat16");
    TORCH_CHECK(gathered_stats.scalar_type() == at::ScalarType::Float, "gathered_stats must be float32");

    // Check alignment requirements for gathered_o (16-byte aligned for async
    // memcpy)
    TORCH_CHECK(reinterpret_cast<uintptr_t>(gathered_o.data_ptr()) % 16 == 0, "gathered_o must be 16-byte aligned");

    // Check that kv_lora_rank * sizeof(data_type) is a multiple of 16
    size_t data_type_size = torch::elementSize(gathered_o.scalar_type());
    TORCH_CHECK((kv_lora_rank * data_type_size) % 16 == 0, "kv_lora_rank * sizeof(data_type) must be a multiple of 16");

    // Create output tensor
    std::vector<int64_t> output_shape = {num_tokens, num_heads * kv_lora_rank};
    torch::Tensor output = torch::empty(output_shape, gathered_o.options());

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(gathered_o.get_device());

    tensorrt_llm::kernels::HelixPostProcParams<T> params{reinterpret_cast<T*>(output.mutable_data_ptr()),
        reinterpret_cast<T const*>(gathered_o.data_ptr()), reinterpret_cast<float2 const*>(gathered_stats.data_ptr()),
        static_cast<int>(cp_size), static_cast<int>(num_tokens), static_cast<int>(num_heads),
        static_cast<int>(kv_lora_rank), cp_dim};
    fn(params, stream);

    if (scale != 1.0)
    {
        output *= scale;
    }

    return output;
}

template <int version>
inline torch::Tensor helix_post_process_impl_version(
    torch::Tensor const& gathered_o, torch::Tensor const& gathered_stats, double scale, int64_t cp_dim)
{
    if constexpr (version == 2)
    {
        TORCH_CHECK(cp_dim >= 0 && cp_dim <= 2, "cp_dim must be 0, 1, or 2 for version 2");
        if (gathered_o.scalar_type() == at::ScalarType::Half)
        {
            return helix_post_process_impl<__half>(
                gathered_o, gathered_stats, scale, int(cp_dim), tensorrt_llm::kernels::helixPostProcess<__half>);
        }
        else if (gathered_o.scalar_type() == at::ScalarType::BFloat16)
        {
            return helix_post_process_impl<__nv_bfloat16>(gathered_o, gathered_stats, scale, int(cp_dim),
                tensorrt_llm::kernels::helixPostProcess<__nv_bfloat16>);
        }
        else
        {
            TLLM_THROW("helix_post_process only supports half and bfloat16 tensors.");
        }
    }
    else
    {
        TLLM_THROW("version must be 2");
    }
}

TORCH_LIBRARY_FRAGMENT(helix, m)
{
    m.def(
        "helixPostProcess(Tensor gathered_o, Tensor gathered_stats, float "
        "scale, int cp_dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(helix, CUDA, m)
{
    m.impl("helixPostProcess", &helix_post_process_impl_version<2>);
}

} // namespace torch_ext
