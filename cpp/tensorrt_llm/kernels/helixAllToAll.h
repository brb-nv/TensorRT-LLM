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
#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// Field and memory constants
// ============================================================================

constexpr int FIFO_DEPTH = 4;
constexpr int FIFO_ENTRY_BYTES = 128 * 1024;
constexpr int FIFO_ENTRY_128B_COUNT = FIFO_ENTRY_BYTES / 128;
constexpr int FIFO_TOTAL_BYTES = FIFO_ENTRY_BYTES * FIFO_DEPTH;
constexpr int FIFO_TOTAL_U64 = FIFO_TOTAL_BYTES / sizeof(uint64_t);
constexpr int BYTES_PER_128B_BLOCK = 128;
constexpr int UINT64_PER_128B_BLOCK = 16;

// Block organization constants
constexpr int MAX_GROUP_COUNT_PER_BLOCK = 8; // Max warps per block
constexpr int WARP_SIZE = 32;
constexpr uint32_t WARP_MASK = 0xffffffff;

// ============================================================================
// Structure declarations and definitions
// ============================================================================

struct HelixPairInfo
{
    int senderRank;
    int receiverRank;
    int channel;
    int runChannelCount;
};

// 256-byte aligned for optimal performance (matches TensorRT-LLM)
#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

struct ALIGN_256 FifoInfo
{
    volatile int64_t head;
    volatile int64_t tail;
};

struct HelixFieldInfo
{
    uint8_t* dataPtr;
    int elementCount; // Number of elements (e.g., kv_lora_rank for field 0, 1 for
                      // field 1)
    int elementSize;  // Size of each element in bytes (2 for half, 8 for float2)
    int stride;       // Stride between rows in bytes
};

struct HelixAllToAllParams
{
    HelixFieldInfo sendFields[2];
    HelixFieldInfo recvFields[2];
    int entryCount; // Number of entries per peer rank to process
    uint64_t* workspace;
    int workspaceStrideInU64;
    int cpRank;
    int cpSize;
    int channelCount; // use 0 to auto-compute
    int maxChannelCount;
};

namespace fused
{

struct HelixFusedFieldInfo
{
    uint8_t* dataPtr;
    int numTokens;
    int numHeads;
    int headDim;
    int elementSize; // Size of each element in bytes (2 for half, 8 for float2)
    // note: the kernel assumes the data is contiguous,
    // thus the helper sizes and strides are set automatically.
    int rankSize;   // size in bytes for a single rank (numHeads * headDim *
                    // elementSize)
    int rankStride; // Stride between ranks in bytes.
    int headStride; // Stride between heads in bytes.
};

// TODO this should use TRT-LLM or torch data types instead
enum class DataType : uint8_t
{
    FP8,
    FP16,
    BF16,
};

struct HelixFusedAllToAllParams
{
    HelixFusedFieldInfo sendFields[2];
    HelixFusedFieldInfo recvField;
    DataType postProcDataType;
    uint64_t* workspace;
    int workspaceStrideInU64;
    int cpRank;
    int cpSize;
    int channelCount; // use 0 to auto-compute
    int maxChannelCount;
};

} // namespace fused

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Ceiling division: compute ceil(a / b) for integers
 */
template <typename T>
inline constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

/**
 * Align value up to nearest multiple of alignment
 */
template <typename T>
inline constexpr T align_up(T value, T alignment)
{
    return ceil_div(value, alignment) * alignment;
}

// ============================================================================
// Workspace Management Functions
// ============================================================================

/**
 * Compute number of channels for communication based on cpSize.
 *
 * @param cpSize Number of context parallel ranks
 * @param smCount Number of SMs available (0 = auto-detect)
 * @return Number of channels to use
 */
inline int computeHelixMaxChannelCount(int cpSize, int smCount = 0)
{
    if (smCount == 0)
    {
        int deviceId = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));
    }

    int blockCountPerChannel = ceil_div(cpSize, MAX_GROUP_COUNT_PER_BLOCK);
    blockCountPerChannel *= 2; // for send and recv

    int preferredChannel = smCount / blockCountPerChannel;
    return std::max(preferredChannel, 1); // at least one channel
}

/**
 * Compute the workspace size required per rank for the all-to-all operation.
 *
 * @param cpSize Number of context parallel ranks
 * @return Size in bytes
 */
inline size_t computeHelixWorkspaceSizePerRank(int cpSize)
{
    static int maxChannelCount = 0;
    if (maxChannelCount == 0)
    {
        maxChannelCount = computeHelixMaxChannelCount(cpSize);
    }

    // FIFO buffers: cpSize * channelCount pairs
    size_t fifoSize = static_cast<size_t>(FIFO_TOTAL_BYTES) * cpSize * maxChannelCount;

    // Sender and receiver FIFO info structures
    size_t senderInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;
    size_t receiverInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;

    return fifoSize + senderInfoSize + receiverInfoSize;
}

namespace fused
{

/**
 * Compute number of channels for fused all-to-all operation.
 *
 * @param cpSize Number of context parallel ranks
 * @param smCount Number of SMs available (0 = auto-detect)
 * @return Number of channels to use
 */
inline int computeHelixFusedMaxChannelCount(int cpSize, int smCount = 0)
{
    if (smCount == 0)
    {
        int deviceId = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));
    }

    int blockCountPerChannel = 1;
    blockCountPerChannel *= 2; // for send and recv

    int preferredChannel = smCount / blockCountPerChannel;
    return std::max(preferredChannel, 1); // at least one channel
}

/**
 * Compute the workspace size required per rank for the fused all-to-all
 * operation.
 *
 * @param cpSize Number of context parallel ranks
 * @return Size in bytes
 */
inline size_t computeHelixFusedWorkspaceSizePerRank(int cpSize)
{
    static int maxChannelCount = 0;
    if (maxChannelCount == 0)
    {
        maxChannelCount = computeHelixFusedMaxChannelCount(cpSize);
    }

    // FIFO buffers: cpSize * channelCount pairs
    size_t fifoSize = static_cast<size_t>(FIFO_TOTAL_BYTES) * cpSize * maxChannelCount;

    // Sender and receiver FIFO info structures
    size_t senderInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;
    size_t receiverInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;

    return fifoSize + senderInfoSize + receiverInfoSize;
}

} // namespace fused

/**
 * Initialize workspace memory for a given rank.
 * Should be called once during setup.
 *
 * @param workspace Pointer to workspace memory (per-rank view)
 * @param cpSize Number of context parallel ranks
 * @param stream CUDA stream for asynchronous operations
 */
void initializeHelixWorkspace(uint64_t* workspace, int cpSize, cudaStream_t stream);

/**
 * Launch the helix all-to-all kernel.
 *
 * @param params Kernel parameters including field info and workspace
 * @param allowVariableField1 Whether to allow variable field 1
 * @param stream CUDA stream for kernel launch
 */
void launchHelixAllToAll(HelixAllToAllParams const& params, bool allowVariableField1, cudaStream_t stream);

namespace fused
{

/**
 * Initialize workspace memory for a given rank for the fused all-to-all
 * operation. Should be called once during setup.
 *
 * @param workspace Pointer to workspace memory (per-rank view)
 * @param cpSize Number of context parallel ranks
 * @param stream CUDA stream for asynchronous operations
 */
void initializeHelixFusedWorkspace(uint64_t* workspace, int cpSize, cudaStream_t stream);

/**
 * Launch the helix fused all-to-all kernel.
 *
 * @param params Kernel parameters including field info and workspace
 * @param stream CUDA stream for kernel launch
 */
void launchHelixFusedAllToAll(HelixFusedAllToAllParams& params, cudaStream_t stream);

} // namespace fused

} // namespace kernels
} // namespace tensorrt_llm
