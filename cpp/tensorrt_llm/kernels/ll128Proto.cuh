/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// LL128 Protocol Implementation
//
// The LL128 protocol is a low-latency protocol for GPU-to-GPU communication
// that uses 128-byte aligned blocks with embedded flags for synchronization.
// Each 15 * 128 bytes of data is packed into 16 * 128 bytes for transfer,
// with the extra 128 bytes containing synchronization flags.
// ============================================================================

class LL128Proto
{
public:
    // Value used to initialize FIFO buffers (all 1s)
    static constexpr uint32_t INITIALIZED_VALUE = 0xFFFFFFFFU;

    /**
     * Check how many 128-byte blocks have been received in shared memory.
     *
     * @tparam USE_FINISH If true, also check for finish flag
     * @param sharedMemoryBase Base pointer to shared memory buffer
     * @param step Current step/head value for flag matching
     * @param countIn128Bytes Total number of 128-byte blocks expected
     * @param fifoEntry128ByteIndexBase Starting index within FIFO entry
     * @param loaded128ByteCount Number of blocks already confirmed loaded
     * @param laneId Thread lane ID within warp
     * @return Number of newly confirmed blocks, or -1 if finish flag received
     */
    template <bool USE_FINISH>
    static __device__ __forceinline__ int checkDataReceivedInShm(uint8_t* sharedMemoryBase, uint64_t step,
        int countIn128Bytes, int fifoEntry128ByteIndexBase, int loaded128ByteCount, int laneId)
    {
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
                    = aligned128BytesShm[idx * UINT64_PER_128B_BLOCK + indexInFifoEntry % UINT64_PER_128B_BLOCK];

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
            // Check valid in order - if previous is not true, ignore current
            int validCount = (validMask == WARP_MASK) ? WARP_SIZE : (__ffs(~validMask) - 1);

            if (USE_FINISH)
            {
                unsigned finishedMask = __ballot_sync(WARP_MASK, finish);
                // Finish should be the very first 128 bytes
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

    /**
     * Pack data with LL128 protocol flags before sending.
     *
     * For LL128, every 15 * 128 bytes is packed to 16 * 128 bytes.
     * Each half-warp (16 threads) handles one 15 * 128 byte chunk.
     *
     * @param sharedMemoryBase Base pointer to shared memory buffer
     * @param step Current step/head value for flag embedding
     * @param countIn128Bytes Number of 128-byte blocks of actual data
     * @param fifoEntry128ByteIndexBase Starting index within FIFO entry
     * @param laneId Thread lane ID within warp
     */
    static __device__ __forceinline__ void protoPack(
        uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;

        // Each 16 threads handles one 15 * 128 byte chunk
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;

            uint64_t tailValue = step;
            uint64_t tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            if (halfLaneId == 15)
            {
                tailInnerIndex = tailFlagInnerIndex;
            }

            int targetTailIndex = tailOffsetIn128Bytes * UINT64_PER_128B_BLOCK + tailInnerIndex;

            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * UINT64_PER_128B_BLOCK + idxFromFifoEntry % UINT64_PER_128B_BLOCK;
                tailValue = aligned128BytesShm[flagIndex];
                aligned128BytesShm[flagIndex] = step;
            }
            aligned128BytesShm[targetTailIndex] = tailValue;
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    /**
     * Unpack data with LL128 protocol - restore original data layout.
     *
     * @param sharedMemoryBase Base pointer to shared memory buffer
     * @param step Current step/tail value for verification
     * @param countIn128Bytes Number of 128-byte blocks of actual data
     * @param fifoEntry128ByteIndexBase Starting index within FIFO entry
     * @param loaded128ByteCount Number of blocks already loaded
     * @param laneId Thread lane ID within warp
     */
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
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;

            uint64_t tailValue = 0;
            int tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            int targetTailIndex = tailOffsetIn128Bytes * UINT64_PER_128B_BLOCK + tailInnerIndex;

            if (halfLaneId < 15)
            {
                tailValue = aligned128BytesShm[targetTailIndex];
            }
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * UINT64_PER_128B_BLOCK + idxFromFifoEntry % UINT64_PER_128B_BLOCK;
                aligned128BytesShm[flagIndex] = tailValue;
            }
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    /**
     * Rearm FIFO buffer for next use (no-op for LL128 protocol).
     */
    static __device__ __forceinline__ void rearm(
        uint32_t* u32FifoPtr, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        // LL128 protocol doesn't need rearm
    }

    /**
     * Compute the transfer size after LL128 protocol packing.
     *
     * Each 15 * 128 bytes requires one additional 128 byte tail block.
     *
     * @param compact128ByteSizeBeforeProto Original compact size in bytes (128-byte aligned)
     * @return Transfer size in bytes after protocol packing
     */
    static __device__ __host__ __forceinline__ int computeProtoTransfer128ByteAlignedSize(
        int compact128ByteSizeBeforeProto)
    {
        // Each 15 * 128 bytes needs one tail 128 byte
        int tail128ByteSize = ((compact128ByteSizeBeforeProto + 15 * 128 - 1) / (15 * 128)) * 128;
        return compact128ByteSizeBeforeProto + tail128ByteSize;
    }
};

} // namespace kernels
} // namespace tensorrt_llm

