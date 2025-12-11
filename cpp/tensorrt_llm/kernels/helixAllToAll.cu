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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/helixAllToAll.h"

#include <cstdint>
#include <cstdio>
#include <unordered_map>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// Inlined utilities from cudaAsyncOps.cuh
// (These may be refactored to a shared header in the future)
// ============================================================================

static __device__ __forceinline__ uint32_t __helix_as_ptr_smem(void const* __ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
}

static __device__ __forceinline__ uint64_t __helix_as_ptr_gmem(void const* __ptr)
{
    return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
}

__device__ __forceinline__ void helix_mbarrier_init(uint64_t* addr, uint32_t const& count)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__helix_as_ptr_smem(addr)), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ uint64_t helix_mbarrier_arrive_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint64_t state;
    asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(__helix_as_ptr_smem(addr)), "r"(txCount)
        : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool helix_mbarrier_try_wait_parity(uint64_t* addr, uint32_t const& phaseParity)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint32_t waitComplete;
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)
        : "r"(__helix_as_ptr_smem(addr)), "r"(phaseParity)
        : "memory");
    return static_cast<bool>(waitComplete);
#else
    return false;
#endif
}

template <int COPY_SIZE = 4>
__device__ __forceinline__ void helix_ldgsts(int* dstShm, int const* srcMem, bool predGuard)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) predGuard),
        "r"(__helix_as_ptr_smem(dstShm)), "l"(__helix_as_ptr_gmem(srcMem)), "n"(COPY_SIZE));
#endif
}

__device__ __forceinline__ void helix_cp_async_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void helix_cp_async_wait_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void helix_cp_async_bulk_g2s(void* dstMem, void const* srcMem, int copySize, uint64_t* smemBar)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(__helix_as_ptr_smem(dstMem)), "l"(__helix_as_ptr_gmem(srcMem)), "r"(copySize), "r"(__helix_as_ptr_smem(smemBar))
        : "memory");
#endif
}

__device__ __forceinline__ void helix_cp_async_bulk_s2g(void* dstMem, void const* srcMem, int copySize)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(__helix_as_ptr_gmem(dstMem)), "r"(__helix_as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#endif
}

__device__ __forceinline__ void helix_cp_async_bulk_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void helix_cp_async_bulk_wait_group_read()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void helix_initSmemBar(uint64_t* smemBar, int laneId)
{
    if (laneId == 0)
    {
        helix_mbarrier_init(smemBar, WARP_SIZE);
    }
    __syncwarp();
}

__device__ __forceinline__ void helix_smemBarWait(uint64_t* smemBar, uint32_t* phaseParity)
{
    while (!helix_mbarrier_try_wait_parity(smemBar, *phaseParity))
    {
    }
    *phaseParity = 1 - *phaseParity;
}

// ============================================================================
// Inlined LL128Proto class from ll128Proto.cuh
// (This may be refactored to a shared header in the future)
// ============================================================================

class HelixLL128Proto
{
public:
    static constexpr uint32_t INITIALIZED_VALUE = 0xFFFFFFFFU;

    template <bool USE_FINISH>
    static __device__ __forceinline__ int checkDataReceivedInShm(uint8_t* sharedMemoryBase, uint64_t step,
        int countIn128Bytes, int fifoEntry128ByteIndexBase, int loaded128ByteCount, int laneId)
    {
        // return value should be how many package already been received.
        // 0 means no data received, -1 means has received finish package(should be the very first 128 Byte).
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int totalValidCount = 0;
        for (int idxBase = loaded128ByteCount; idxBase < countIn128Bytes; idxBase += WARP_SIZE)
        {
            int idx = idxBase + laneId;
            bool valid = false;
            bool finish = false;
            if (idx < countIn128Bytes)
            {
                int indexInFifoEntry = fifoEntry128ByteIndexBase + idx;
                uint64_t value
                    = aligned128BytesShm[idx * HELIX_UINT64_PER_128B_BLOCK + indexInFifoEntry % HELIX_UINT64_PER_128B_BLOCK];
                if (USE_FINISH)
                {
                    finish = (value == (step & (1ULL << 63ULL)));
                    valid = (value == step) || finish;
                }
                else
                {
                    valid = (value == step);
                }
            }
            __syncwarp();
            unsigned validMask = __ballot_sync(WARP_MASK, valid);
            // here we check valid in order, if previous valid is not true, we ignore the current valid.
            int validCount = (validMask == WARP_MASK) ? WARP_SIZE : (__ffs(~validMask) - 1);
            if (USE_FINISH)
            {
                unsigned finishedMask = __ballot_sync(WARP_MASK, finish);
                // finish should be the very first 128 Byte.
                if (finishedMask & 0x1)
                {
                    return -1;
                }
            }
            totalValidCount += validCount;

            if (validCount != WARP_SIZE)
            {
                break;
            }
        }
        return totalValidCount;
    }

    static __device__ __forceinline__ void protoPack(
        uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        // for LL128 15 * 128 Bytes will be packed to 16 * 128 Bytes, each 16 threads is used for one 15 * 128 bytes.
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % HELIX_UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = step;
            uint64_t tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            if (halfLaneId == 15)
            {
                tailInnerIndex = tailFlagInnerIndex;
            }
            int targetTailIndex = tailOffsetIn128Bytes * HELIX_UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * HELIX_UINT64_PER_128B_BLOCK + idxFromFifoEntry % HELIX_UINT64_PER_128B_BLOCK;
                tailValue = aligned128BytesShm[flagIndex];
                aligned128BytesShm[flagIndex] = step;
            }
            aligned128BytesShm[targetTailIndex] = tailValue;
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void protoUnpack(uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes,
        int fifoEntry128ByteIndexBase, int loaded128ByteCount, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % HELIX_UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = 0;
            int tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            int targetTailIndex = tailOffsetIn128Bytes * HELIX_UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (halfLaneId < 15)
            {
                tailValue = aligned128BytesShm[targetTailIndex];
            }
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * HELIX_UINT64_PER_128B_BLOCK + idxFromFifoEntry % HELIX_UINT64_PER_128B_BLOCK;
                aligned128BytesShm[flagIndex] = tailValue;
            }
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void rearm(
        uint32_t* u32FifoPtr, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        // LL128 don't need rearm
    }

    static __device__ __host__ __forceinline__ int computeProtoTransfer128ByteAlignedSize(
        int compact128ByteSizeBeforeProto)
    {
        // each 15 * 128 byte need one tail 128 byte
        int tail128ByteSize = (compact128ByteSizeBeforeProto + 15 * 128 - 1) / (15 * 128) * 128;
        return compact128ByteSizeBeforeProto + tail128ByteSize;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

__host__ __device__ inline int getFieldSize(HelixFieldInfo const& fieldInfo)
{
    return fieldInfo.elementCount * fieldInfo.elementSize;
}

__host__ __device__ inline uint8_t* getPtr(HelixFieldInfo const& fieldInfo, int blockIdx)
{
    return fieldInfo.dataPtr + blockIdx * fieldInfo.stride;
}

__device__ __forceinline__ void waitG2sAllFields(uint64_t* smemBar, uint32_t* phaseParity)
{
    helix_cp_async_wait_group<0>();
    helix_smemBarWait(smemBar, phaseParity);
}

// Align size to 128 bytes
__host__ __device__ __forceinline__ int align128(int size)
{
    return helix_align_up(size, HELIX_BYTES_PER_128B_BLOCK);
}

// ============================================================================
// G2S (Global to Shared) Operations
// ============================================================================

__device__ __forceinline__ void g2sField(
    HelixFieldInfo const& fieldInfo, int dataIndex, uint8_t* shmemBase, int shmemOffset, uint64_t* smemBar, int laneId)
{
    int copySize = getFieldSize(fieldInfo);
    if (copySize > 0 && laneId == 0)
    {
        uint8_t* srcPtr = getPtr(fieldInfo, dataIndex);
        uint8_t* dstPtr = shmemBase + shmemOffset;
        helix_cp_async_bulk_g2s(dstPtr, srcPtr, copySize, smemBar);
    }
}

template <bool ALLOW_VARIABLE_FIELD1>
__device__ __forceinline__ int g2sAllFields(
    HelixFieldInfo const* fieldInfo, int dataIndex, uint8_t* shmemBase, uint64_t* smemBar, int laneId)
{
    int totalSize = 0;

    // Load field 0 (variable size half)
    g2sField(fieldInfo[0], dataIndex, shmemBase, 0, smemBar, laneId);
    int field0Size = getFieldSize(fieldInfo[0]);
    totalSize += field0Size;

    // Load field 1 (single float2)
    if constexpr (ALLOW_VARIABLE_FIELD1)
    {
        g2sField(fieldInfo[1], dataIndex, shmemBase, totalSize, smemBar, laneId);
        totalSize += getFieldSize(fieldInfo[1]);
    }
    else
    {
        helix_ldgsts<8>(reinterpret_cast<int*>(shmemBase + totalSize),
            reinterpret_cast<int const*>(getPtr(fieldInfo[1], dataIndex)), laneId == 0);
        helix_cp_async_commit_group();
    }

    return totalSize;
}

// ============================================================================
// S2G (Shared to Global) Operations
// ============================================================================

__device__ __forceinline__ void s2gField(
    HelixFieldInfo const& fieldInfo, int dataIndex, uint8_t* shmemBase, int shmemOffset, int laneId)
{
    int copySize = getFieldSize(fieldInfo);
    if (copySize > 0 && laneId == 0)
    {
        uint8_t* srcPtr = shmemBase + shmemOffset;
        uint8_t* dstPtr = getPtr(fieldInfo, dataIndex);
        helix_cp_async_bulk_s2g(dstPtr, srcPtr, copySize);
    }
}

template <bool ALLOW_VARIABLE_FIELD1>
__device__ __forceinline__ void s2gAllFields(
    HelixFieldInfo const* fieldInfo, int dataIndex, uint8_t* shmemBase, int laneId)
{
    int offset = 0;

    // Store field 0 (variable size half)
    s2gField(fieldInfo[0], dataIndex, shmemBase, offset, laneId);
    int field0Size = getFieldSize(fieldInfo[0]);
    offset += field0Size;

    // Store field 1 (single float2)
    if constexpr (ALLOW_VARIABLE_FIELD1)
    {
        s2gField(fieldInfo[1], dataIndex, shmemBase, offset, laneId);
        offset += getFieldSize(fieldInfo[1]);
    }
    else
    {
        if (laneId == 0)
        {
            auto* srcPtr = reinterpret_cast<float2*>(reinterpret_cast<uint8_t*>(shmemBase) + offset);
            auto* dstPtr = reinterpret_cast<float2*>(getPtr(fieldInfo[1], dataIndex));
            dstPtr[0] = srcPtr[0];
        }
    }
    helix_cp_async_bulk_commit_group();
}

// ============================================================================
// Workspace FIFO Operations
// ============================================================================

__device__ __forceinline__ uint64_t* getFifoBasePtr(HelixAllToAllParams const& params, HelixPairInfo const& pairInfo)
{
    // FIFO is physically located at receiver rank
    int mappedMemoryRank = pairInfo.receiverRank;
    int rankInsideMappedMemory = pairInfo.senderRank;

    auto* mappedMemory = params.workspace + mappedMemoryRank * params.workspaceStrideInU64;
    // Navigate to the right FIFO: [peer_rank][channel]
    size_t fifoOffset = rankInsideMappedMemory * params.maxChannelCount * HELIX_FIFO_TOTAL_U64;
    fifoOffset += pairInfo.channel * HELIX_FIFO_TOTAL_U64;

    return mappedMemory + fifoOffset;
}

__device__ __forceinline__ FifoInfo* getSenderFifoInfo(HelixAllToAllParams const& params, HelixPairInfo const& pairInfo)
{
    // SenderSideFifoInfo is physically located at sender rank
    int mappedMemoryRank = pairInfo.senderRank;
    int rankInsideMappedMemory = pairInfo.receiverRank;

    auto* mappedMemory = reinterpret_cast<uint8_t*>(params.workspace + mappedMemoryRank * params.workspaceStrideInU64);
    size_t fieldOffset = static_cast<size_t>(HELIX_FIFO_TOTAL_BYTES) * params.cpSize * params.maxChannelCount;
    mappedMemory += fieldOffset;
    mappedMemory += rankInsideMappedMemory * params.maxChannelCount * sizeof(FifoInfo);
    mappedMemory += pairInfo.channel * sizeof(FifoInfo);

    return reinterpret_cast<FifoInfo*>(mappedMemory);
}

__device__ __forceinline__ FifoInfo* getReceiverFifoInfo(
    HelixAllToAllParams const& params, HelixPairInfo const& pairInfo)
{
    // ReceiverSideFifoInfo is physically located at receiver rank
    int mappedMemoryRank = pairInfo.receiverRank;
    int rankInsideMappedMemory = pairInfo.senderRank;

    auto* mappedMemory = reinterpret_cast<uint8_t*>(params.workspace + mappedMemoryRank * params.workspaceStrideInU64);
    size_t fieldOffset = static_cast<size_t>(HELIX_FIFO_TOTAL_BYTES) * params.cpSize * params.maxChannelCount;
    fieldOffset += sizeof(FifoInfo) * params.cpSize * params.maxChannelCount;
    mappedMemory += fieldOffset;
    mappedMemory += rankInsideMappedMemory * params.maxChannelCount * sizeof(FifoInfo);
    mappedMemory += pairInfo.channel * sizeof(FifoInfo);

    return reinterpret_cast<FifoInfo*>(mappedMemory);
}

__device__ __forceinline__ void startWorkspaceS2G(
    uint64_t* fifoEntry, uint8_t* shmemBase, int send128ByteCount, int fifo128ByteOffset, int laneId)
{
    int copyByteCount = send128ByteCount * HELIX_BYTES_PER_128B_BLOCK;
    if (laneId == 0)
    {
        helix_cp_async_bulk_s2g(
            fifoEntry + fifo128ByteOffset * HELIX_BYTES_PER_128B_BLOCK / sizeof(uint64_t), shmemBase, copyByteCount);
    }
    helix_cp_async_bulk_commit_group();
}

__device__ __forceinline__ void startWorkspaceS2GReg(
    uint64_t* fifoEntry, uint8_t* sharedMemoryBase, int send128ByteCount, int fifo128ByteOffset, int laneId)
{
    int copyInt4Count = send128ByteCount * HELIX_BYTES_PER_128B_BLOCK / sizeof(int4);
    int4* sharedMemoryInt4 = reinterpret_cast<int4*>(sharedMemoryBase);
    uint64_t* fifoPtr = fifoEntry + fifo128ByteOffset * HELIX_UINT64_PER_128B_BLOCK;
    int4* fifoPtrInt4 = reinterpret_cast<int4*>(fifoPtr);
#pragma unroll 4
    for (int i = laneId; i < copyInt4Count; i += WARP_SIZE)
    {
        fifoPtrInt4[i] = sharedMemoryInt4[i];
    }
}

__device__ __forceinline__ uint64_t startWorkspaceG2S(uint8_t* shmemBase, uint64_t* fifoEntry, int allLoad128ByteCount,
    int fifo128ByteOffset, int loaded128ByteCount, uint64_t* smemBar, int laneId)
{
    int copyByteCount = (allLoad128ByteCount - loaded128ByteCount) * HELIX_BYTES_PER_128B_BLOCK;
    if (laneId == 0)
    {
        helix_cp_async_bulk_g2s(shmemBase + loaded128ByteCount * HELIX_BYTES_PER_128B_BLOCK,
            fifoEntry + (fifo128ByteOffset + loaded128ByteCount) * HELIX_UINT64_PER_128B_BLOCK, copyByteCount, smemBar);
    }
    return helix_mbarrier_arrive_expect_tx(smemBar, laneId == 0 ? copyByteCount : 0);
}

// ============================================================================
// Size helpers
// ============================================================================

// Compute total size needed for both fields
__host__ __device__ __forceinline__ int computeTotalUnpackedSize(HelixFieldInfo const* fields)
{
    int size = 0;
    // Field 0: note it must be aligned to 16 bytes
    size += helix_align_up(getFieldSize(fields[0]), 16);
    // Field 1: single float2
    size += helix_align_up(getFieldSize(fields[1]), 16);
    return align128(size);
}

__host__ __device__ __forceinline__ int computeTotalPackedSize(HelixFieldInfo const* fields)
{
    // because field 0 must be aligned to 16 bytes, this is the same as unpacked
    return computeTotalUnpackedSize(fields);
}

__host__ __device__ __forceinline__ int computeProtoTransferSize(HelixFieldInfo const* fields)
{
    return HelixLL128Proto::computeProtoTransfer128ByteAlignedSize(computeTotalPackedSize(fields));
}

// ============================================================================
// Main All-to-All Kernel
// ============================================================================

template <bool ALLOW_VARIABLE_FIELD1>
__global__ void helixAllToAllKernel(HelixAllToAllParams params)
{
    extern __shared__ uint8_t allWarpShmem[];
    __shared__ uint64_t allWarpSmemBar[HELIX_MAX_GROUP_COUNT_PER_BLOCK];

    bool isSender = (blockIdx.z == 0);
    // Each warp is a group handling a different peer rank
    int group = __shfl_sync(WARP_MASK, threadIdx.y, 0);
    int laneId = threadIdx.x % WARP_SIZE;
    int runChannelCount = gridDim.y;

    // Compute peer rank: blockIdx.x determines which set of peers, group
    // determines which peer in that set
    int peerRank = blockIdx.x * blockDim.y + group;

    if (peerRank >= params.cpSize)
    {
        return;
    }

    // Setup pair info for this communication
    HelixPairInfo pairInfo;
    pairInfo.channel = blockIdx.y;
    pairInfo.runChannelCount = runChannelCount;
    pairInfo.senderRank = isSender ? params.cpRank : peerRank;
    pairInfo.receiverRank = isSender ? peerRank : params.cpRank;

    // Initialize barrier for this group
    helix_initSmemBar(&allWarpSmemBar[group], laneId);
    uint32_t phaseParity = 0;

    // Get shared memory for this group
    int singlePackedSize = computeTotalPackedSize(params.sendFields);
    int singlePacked128ByteCount = singlePackedSize / HELIX_BYTES_PER_128B_BLOCK;
    int singleUnpackedSize = computeTotalUnpackedSize(params.sendFields);
    int singleProtoTransferSize = computeProtoTransferSize(params.sendFields);
    int singleProtoTransfer128ByteCount = singleProtoTransferSize / HELIX_BYTES_PER_128B_BLOCK;
    int singleShmSize = std::max(singleUnpackedSize, singleProtoTransferSize);
    uint8_t* shmem = allWarpShmem + group * singleShmSize;

    // Get FIFO pointers
    uint64_t* fifoBase = getFifoBasePtr(params, pairInfo);
    FifoInfo* senderFifo = getSenderFifoInfo(params, pairInfo);
    FifoInfo* receiverFifo = getReceiverFifoInfo(params, pairInfo);

    int fifoEntry128ByteIndexBase = HELIX_FIFO_ENTRY_128B_COUNT;
    int fifoEntryIndex = -1;

    // regardless of sender or receiver, we wait for the previous kernel here
    // receiver blocks do not need to wait at all, but they should not start
    // to stress the memory system regardless
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if (isSender)
    {
        // sender blocks should trigger next kernel immediately, s.t. they
        // do not block the next kernel from starting
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif

        // Sender logic: send data from cpRank's slice to peerRank
        int64_t head = senderFifo->head;
        int64_t tail = senderFifo->tail;

        // Each channel processes entries with stride
        // Start at channel index, increment by total channel count
        for (int entryIdx = pairInfo.channel; entryIdx < params.entryCount; entryIdx += runChannelCount)
        {

            // dataIndex points to the data for peerRank in this entry
            int dataIndex = entryIdx * params.cpSize + peerRank;

            // Load data from global to shared, then arrive on barrier
            int loadedSize = g2sAllFields<ALLOW_VARIABLE_FIELD1>(
                params.sendFields, dataIndex, shmem, &allWarpSmemBar[group], laneId);
            uint64_t arriveState = helix_mbarrier_arrive_expect_tx(&allWarpSmemBar[group], laneId == 0 ? loadedSize : 0);

            // update FIFO entry index and head if needed
            if (fifoEntry128ByteIndexBase + singleProtoTransfer128ByteCount > HELIX_FIFO_ENTRY_128B_COUNT)
            {
                if (fifoEntryIndex >= 0)
                {
                    head++;
                    __syncwarp();
                    senderFifo->head = head;
                }
                fifoEntryIndex = head % HELIX_FIFO_DEPTH;
                fifoEntry128ByteIndexBase = 0;
                while (tail + HELIX_FIFO_DEPTH <= head)
                {
                    tail = senderFifo->tail;
                }
                __syncwarp();
            }

            // wait for data to be loaded into shared memory
            waitG2sAllFields(&allWarpSmemBar[group], &phaseParity);
            // note: we don't need to pack anything, fields are already packed in
            // shared memory

            HelixLL128Proto::protoPack(shmem, head, singlePacked128ByteCount, fifoEntry128ByteIndexBase, laneId);

            uint64_t* fifoEntry = fifoBase + fifoEntryIndex * (HELIX_FIFO_ENTRY_BYTES / sizeof(uint64_t));

            // Copy from shared to workspace FIFO
            startWorkspaceS2GReg(fifoEntry, shmem, singleProtoTransfer128ByteCount, fifoEntry128ByteIndexBase, laneId);

            fifoEntry128ByteIndexBase += singleProtoTransfer128ByteCount;

            // ensure that we can over-write shmem in next iteration
            // (it must be fully read by all threads when doing S2G above)
            __syncwarp();
        }
        if (fifoEntry128ByteIndexBase > 0)
        {
            head++;
            senderFifo->head = head;
        }
    }
    else
    {
        // Receiver logic: receive data from peerRank to cpRank's slice
        int64_t tail = receiverFifo->tail;
        bool needRelease = false;

        // Each channel processes entries with stride
        // Start at channel index, increment by total channel count
        for (int entryIdx = pairInfo.channel; entryIdx < params.entryCount; entryIdx += runChannelCount)
        {
            // receiver blocks should trigger next kernel at last iteration
            // note: some blocks might not even go into this for-loop, but they
            // would exit which is equivalent to the pre-exit trigger
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            if (entryIdx + runChannelCount >= params.entryCount)
            {
                cudaTriggerProgrammaticLaunchCompletion();
            }
#endif
            // dataIndex points to where we receive data from peerRank in this entry
            int dataIndex = entryIdx * params.cpSize + peerRank;
            int loaded128ByteCount = 0;

            if (fifoEntry128ByteIndexBase + singleProtoTransfer128ByteCount > HELIX_FIFO_ENTRY_128B_COUNT)
            {
                if (fifoEntryIndex >= 0)
                {
                    tail++;
                    needRelease = true;
                }
                fifoEntryIndex = tail % HELIX_FIFO_DEPTH;
                fifoEntry128ByteIndexBase = 0;
                // receiver doesn't need to wait on FIFO entry being readable: it's
                // always readable
                __syncwarp();
            }

            uint64_t* fifoEntry = fifoBase + fifoEntryIndex * (HELIX_FIFO_ENTRY_BYTES / sizeof(uint64_t));
            while (loaded128ByteCount < singleProtoTransfer128ByteCount)
            {
                startWorkspaceG2S(shmem, fifoEntry, singleProtoTransfer128ByteCount, fifoEntry128ByteIndexBase,
                    loaded128ByteCount, &allWarpSmemBar[group], laneId);
                if (needRelease)
                {
                    receiverFifo->tail = tail;
                    senderFifo->tail = tail;
                    needRelease = false;
                }
                helix_smemBarWait(&allWarpSmemBar[group], &phaseParity);
                loaded128ByteCount += HelixLL128Proto::template checkDataReceivedInShm<false>(shmem, tail,
                    singleProtoTransfer128ByteCount, fifoEntry128ByteIndexBase, loaded128ByteCount, laneId);
            }

            HelixLL128Proto::protoUnpack(
                shmem, tail, singlePacked128ByteCount, fifoEntry128ByteIndexBase, loaded128ByteCount, laneId);

            // note: fields are already unpacked in shared memory
            s2gAllFields<ALLOW_VARIABLE_FIELD1>(params.recvFields, dataIndex, shmem, laneId);
            // wait for data to be read from shared memory
            helix_cp_async_bulk_wait_group_read<0>();

            // note: HelixLL128Proto doesn't need rearm
            // rearmFifoBuffer();
            fifoEntry128ByteIndexBase += singleProtoTransfer128ByteCount;
        }
        if (fifoEntry128ByteIndexBase > 0)
        {
            tail++;
            receiverFifo->tail = tail;
            senderFifo->tail = tail;
        }
    }
}

// ============================================================================
// Compute actual channel count
// ============================================================================

struct hash_cache_key
{
    size_t operator()(std::tuple<int, int, int> const& x) const
    {
        return std::get<0>(x) ^ std::get<1>(x) ^ std::get<2>(x);
    }
};

template <bool ALLOW_VARIABLE_FIELD1>
std::tuple<int, int, int> computeChannelAndGroupCount(int cpSize, HelixFieldInfo const* fields)
{
    static std::unordered_map<std::tuple<int, int, int>, std::tuple<int, int, int>, hash_cache_key> cache;
    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    int singleShmSize = std::max(computeTotalUnpackedSize(fields), computeProtoTransferSize(fields));
    auto key = std::make_tuple(deviceId, cpSize, singleShmSize);
    auto it = cache.find(key);
    if (it != cache.end())
    {
        return it->second;
    }

    int maxGroupCountPerCta = std::min(cpSize, HELIX_MAX_GROUP_COUNT_PER_BLOCK);
    int groupCountPerCta = maxGroupCountPerCta; // Start with max
    int totalDynamicShmemSize = singleShmSize * groupCountPerCta;
    int maxDynamicShmSize = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxDynamicShmSize, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));

    while (totalDynamicShmemSize > maxDynamicShmSize)
    {
        groupCountPerCta--;
        totalDynamicShmemSize = singleShmSize * groupCountPerCta;
    }

    TLLM_CHECK_WITH_INFO(totalDynamicShmemSize <= maxDynamicShmSize, "Single packed size %d exceeds limit %d",
        singleShmSize, maxDynamicShmSize);

    // Set shared memory attribute if needed
    if (totalDynamicShmemSize > 48 * 1024)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(helixAllToAllKernel<ALLOW_VARIABLE_FIELD1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, totalDynamicShmemSize));
    }

    int blockCountPerChannel = helix_ceil_div(cpSize, groupCountPerCta);
    blockCountPerChannel *= 2; // for send and recv

    int smCount = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));
    // TODO: we might only want to use half the SMs to overlap with other kernels.
    // note that overlap with FMHA is almost impossible because it must use
    // all SMs and probably uses >50% shmem per SM.
    // overlap with the subsequent BMM / out proj GEMMs might be possible,
    // so we need experiments to see whether it makes sense.
    int channelCount = std::max(smCount / blockCountPerChannel, 1);
    auto value = std::make_tuple(channelCount, groupCountPerCta, totalDynamicShmemSize);
    cache[key] = value;
    return value;
}

// ============================================================================
// Host Launch Function
// ============================================================================

template <bool ALLOW_VARIABLE_FIELD1>
void launchHelixAllToAllImpl(HelixAllToAllParams const& params, cudaStream_t stream)
{
    int maxChannelCount = computeHelixMaxChannelCount(params.cpSize);
    TLLM_CHECK_WITH_INFO(params.maxChannelCount == maxChannelCount,
        "maxChannelCount %d does not match computed maxChannelCount %d", params.maxChannelCount, maxChannelCount);
    auto [channelCount, groupCountPerCta, totalDynamicShmemSize]
        = computeChannelAndGroupCount<ALLOW_VARIABLE_FIELD1>(params.cpSize, params.sendFields);
    if (params.channelCount > 0)
    {
        channelCount = params.channelCount;
        TLLM_CHECK_WITH_INFO(channelCount <= maxChannelCount, "channelCount %d exceeds maxChannelCount %d",
            channelCount, maxChannelCount);
    }

    // Compute grid dimensions
    // grid.x = blocks per channel (how many blocks needed to cover all peer
    // ranks) grid.y = number of channels (parallel channels) grid.z = 2 (sender
    // and receiver)
    int ctaPerChannel = helix_ceil_div(params.cpSize, groupCountPerCta);

    auto* kernel_instance = &helixAllToAllKernel<ALLOW_VARIABLE_FIELD1>;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(ctaPerChannel, channelCount, 2);
    config.blockDim = dim3(WARP_SIZE, groupCountPerCta);
    config.dynamicSmemBytes = totalDynamicShmemSize;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_instance, params));
}

void launchHelixAllToAll(HelixAllToAllParams const& params, bool allowVariableField1, cudaStream_t stream)
{
    if (allowVariableField1)
    {
        launchHelixAllToAllImpl<true>(params, stream);
    }
    else
    {
        launchHelixAllToAllImpl<false>(params, stream);
    }
}

// ============================================================================
// Workspace Initialization
// ============================================================================

void initializeHelixWorkspace(uint64_t* local_workspace_ptr, int cpSize, cudaStream_t stream)
{
    int maxChannelCount = computeHelixMaxChannelCount(cpSize);
    // Calculate sizes with channel dimension
    size_t fifoSize = static_cast<size_t>(HELIX_FIFO_TOTAL_BYTES) * cpSize * maxChannelCount;
    size_t senderInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;
    size_t receiverInfoSize = sizeof(FifoInfo) * cpSize * maxChannelCount;

    // Initialize FIFO buffers to 0xFFFFFFFF (-1 for signed integer types)
    TLLM_CUDA_CHECK(cudaMemsetAsync(local_workspace_ptr, 0xFF, fifoSize, stream));

    // Initialize sender and receiver info to zero (single call for both)
    uint8_t* infoPtr = reinterpret_cast<uint8_t*>(local_workspace_ptr) + fifoSize;
    TLLM_CUDA_CHECK(cudaMemsetAsync(infoPtr, 0, senderInfoSize + receiverInfoSize, stream));
}

} // namespace kernels
} // namespace tensorrt_llm
