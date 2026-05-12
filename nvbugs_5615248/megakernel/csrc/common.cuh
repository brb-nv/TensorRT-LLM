/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared device-side declarations and TinyLlama-1.1B config constants.
 *
 * Configuration macros are passed in by `setup.py`:
 *   TLLAMA_HIDDEN, TLLAMA_INTERMEDIATE, TLLAMA_NUM_HEADS, TLLAMA_NUM_KV_HEADS,
 *   TLLAMA_HEAD_DIM, TLLAMA_NUM_LAYERS, TLLAMA_VOCAB, TLLAMA_ROPE_THETA,
 *   TLLAMA_RMS_EPS, MEGAKERNEL_SEQ_LEN, NUM_BLOCKS, BLOCK_SIZE, TARGET_SM.
 */
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace lucebox {

// ---- model constants (compile-time) ----------------------------------------
constexpr int kHidden       = TLLAMA_HIDDEN;        // 2048
constexpr int kIntermediate = TLLAMA_INTERMEDIATE;  // 5632
constexpr int kNumHeads     = TLLAMA_NUM_HEADS;     // 32
constexpr int kNumKvHeads   = TLLAMA_NUM_KV_HEADS;  // 4
constexpr int kHeadDim      = TLLAMA_HEAD_DIM;      // 64
constexpr int kNumLayers    = TLLAMA_NUM_LAYERS;    // 22
constexpr int kVocab        = TLLAMA_VOCAB;         // 32000
constexpr int kKvGroups     = kNumHeads / kNumKvHeads; // 8
constexpr int kQSize        = kNumHeads * kHeadDim;    // 2048
constexpr int kKvSize       = kNumKvHeads * kHeadDim;  // 256
constexpr int kQkvSize      = kQSize + 2 * kKvSize;    // 2560
constexpr int kGateUpSize   = 2 * kIntermediate;       // 11264
constexpr int kSeqLen       = MEGAKERNEL_SEQ_LEN;      // 128 (padded ISL)
constexpr float kRmsEps     = TLLAMA_RMS_EPS;
constexpr float kRopeTheta  = TLLAMA_ROPE_THETA;

// ---- runtime constants ----------------------------------------------------
// Persistent grid: one block per SM on L40S (142). Configurable at build time.
constexpr int kNumBlocks  = NUM_BLOCKS;
constexpr int kBlockSize  = BLOCK_SIZE;
constexpr int kWarpsPerBlock = kBlockSize / 32;

// ---- per-tensor offsets in the packed weight blob -------------------------
// Filled in by `PackedWeights.layer_offsets` on the Python side; mirrored here
// so the kernel can index into the blob given the layer index.
struct LayerOffsets {
    int input_layernorm;
    int qkv_proj;
    int o_proj;
    int post_attn_norm;
    int gate_up;
    int down;
};

struct MegakernelParams {
    // Inputs.
    const int32_t* __restrict__ input_ids;     // [seq_len]
    const __nv_bfloat16* __restrict__ weights; // packed blob
    // Layer offset table (kNumLayers entries).
    const LayerOffsets* __restrict__ layer_offsets;
    int embed_offset;
    int final_norm_offset;
    int lm_head_offset;
    // Output.
    __nv_bfloat16* __restrict__ logits;        // [seq_len, vocab]
    // Workspace -- HBM-resident activation ring buffers + residual + KV cache.
    // The residual stream is BF16 to match HF Llama eager: HF's decoder layer
    // does `residual = residual + hidden_states` where both are BF16, so the
    // kernel must follow the same precision profile to give bit-comparable
    // logits.
    __nv_bfloat16* __restrict__ act_a;         // [seq_len, hidden]
    __nv_bfloat16* __restrict__ act_b;         // [seq_len, hidden]
    __nv_bfloat16* __restrict__ residual;      // [seq_len, hidden]
    __nv_bfloat16* __restrict__ qkv_buf;       // [seq_len, kQkvSize]
    __nv_bfloat16* __restrict__ attn_buf;      // [seq_len, hidden]
    __nv_bfloat16* __restrict__ gate_up_buf;   // [seq_len, kGateUpSize]
    __nv_bfloat16* __restrict__ silu_buf;      // [seq_len, intermediate]
    // Debug stage dump (nullptr if not in debug mode).
    // When non-null, the kernel writes the post-layer residual stream
    // (= input to layer L+1) into layer_residual_dump[L * seq_len * hidden]
    // for L in [0, num_layers). Layout [num_layers, seq_len, hidden] BF16.
    __nv_bfloat16* __restrict__ layer_residual_dump;
    int seq_len;
};

// ---- device helpers --------------------------------------------------------
// Gated on __CUDACC__ so host TUs (bindings.cpp) can include this header for
// the constants + struct definitions without dragging in CUDA-only intrinsics
// like `threadIdx`.
#ifdef __CUDACC__

__forceinline__ __device__ int sm_id() {
    int sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); return sm;
}

__forceinline__ __device__ int lane_id() {
    int lane; asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane)); return lane;
}

__forceinline__ __device__ int warp_id() {
    return threadIdx.x >> 5;
}

// Cast helpers ---------------------------------------------------------------
__forceinline__ __device__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__forceinline__ __device__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

#endif  // __CUDACC__

}  // namespace lucebox
