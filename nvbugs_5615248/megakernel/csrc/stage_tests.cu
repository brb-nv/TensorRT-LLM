/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone test kernels for each megakernel stage. The megakernel inlines
 * the device-side `stage_*` functions from `stages.cuh` into one persistent
 * grid. This translation unit wraps each one in its own `__global__` entry
 * point so the per-stage CUDA op can be unit-tested against HuggingFace's
 * single-module reference (RMSNorm, RoPE, eager attention, SwiGLU).
 *
 * These kernels are unused at runtime by the megakernel path -- their only
 * consumers are `tests/test_modules.py` and the `lucebox_tinyllama._C.*` ops
 * exposed from `bindings.cpp`. They share source with the megakernel via the
 * device functions in `stages.cuh`, so there is one and only one source of
 * truth for each stage's math.
 *
 * Grid sizing: every stage uses the `blockIdx.x .. gridDim.x` round-robin
 * pattern so the grid size is free. We launch `kNumBlocks` (142 on L40S),
 * matching the megakernel's resident-block count. Block size is fixed to
 * `kBlockSize` because the stages bake in `kWarpsPerBlock` for their
 * shared-memory reductions.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "stages.cuh"

namespace lucebox {

// ---- RMSNorm test kernel ---------------------------------------------------
__launch_bounds__(kBlockSize, 1)
__global__ void rms_norm_test_kernel(__nv_bfloat16* __restrict__ y,
                                     const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int seq_len, int dim) {
    stage_rms_norm(y, x, weight, seq_len, dim);
}

void launch_rms_norm_test(__nv_bfloat16* y, const __nv_bfloat16* x,
                          const __nv_bfloat16* weight,
                          int seq_len, int dim, cudaStream_t stream) {
    dim3 grid(kNumBlocks, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    rms_norm_test_kernel<<<grid, block, 0, stream>>>(y, x, weight, seq_len, dim);
}

// ---- RoPE test kernel ------------------------------------------------------
__launch_bounds__(kBlockSize, 1)
__global__ void rope_test_kernel(__nv_bfloat16* __restrict__ qkv, int seq_len) {
    stage_rope(qkv, seq_len);
}

void launch_rope_test(__nv_bfloat16* qkv, int seq_len, cudaStream_t stream) {
    dim3 grid(kNumBlocks, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    rope_test_kernel<<<grid, block, 0, stream>>>(qkv, seq_len);
}

// ---- Attention test kernel -------------------------------------------------
// The stage uses `scores_scratch` (seq_len floats per block) -- we put it in
// dynamic SMEM. The stage also declares small __shared__ reduction arrays
// internally, which the compiler allocates separately, so we only need to
// size the dynamic SMEM for the scores buffer.
__launch_bounds__(kBlockSize, 1)
__global__ void attention_test_kernel(__nv_bfloat16* __restrict__ attn_out,
                                      const __nv_bfloat16* __restrict__ qkv,
                                      int seq_len) {
    extern __shared__ unsigned char smem_raw[];
    float* scratch = reinterpret_cast<float*>(smem_raw);
    stage_attention(attn_out, qkv, scratch, seq_len);
}

static bool g_attn_attrs_set = false;

void launch_attention_test(__nv_bfloat16* attn_out, const __nv_bfloat16* qkv,
                           int seq_len, cudaStream_t stream) {
    int smem_bytes = seq_len * static_cast<int>(sizeof(float));
    if (!g_attn_attrs_set) {
        cudaFuncSetAttribute(attention_test_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        g_attn_attrs_set = true;
    }
    dim3 grid(kNumBlocks, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    attention_test_kernel<<<grid, block, smem_bytes, stream>>>(attn_out, qkv, seq_len);
}

// ---- SiLU-mul test kernel --------------------------------------------------
__launch_bounds__(kBlockSize, 1)
__global__ void silu_mul_test_kernel(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ gate_up,
                                     int seq_len) {
    stage_silu_mul(out, gate_up, seq_len);
}

void launch_silu_mul_test(__nv_bfloat16* out, const __nv_bfloat16* gate_up,
                          int seq_len, cudaStream_t stream) {
    dim3 grid(kNumBlocks, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    silu_mul_test_kernel<<<grid, block, 0, stream>>>(out, gate_up, seq_len);
}

}  // namespace lucebox
