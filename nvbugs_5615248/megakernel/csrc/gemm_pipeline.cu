/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone launcher for the per-block GEMM primitive defined in
 * `gemm_pipeline.cuh`. Used by `lucebox_tinyllama.gemm_pipeline(A, B)` for
 * M3 unit-testing the cp.async + WMMA pipeline in isolation.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "gemm_pipeline.cuh"

namespace lucebox {

__launch_bounds__(kBlockSize, 1)
__global__ void gemm_pipeline_kernel(__nv_bfloat16* __restrict__ C,
                                     const __nv_bfloat16* __restrict__ A,
                                     const __nv_bfloat16* __restrict__ B,
                                     int M, int N, int K) {
    extern __shared__ unsigned char smem_raw[];
    GemmSmem& smem = *reinterpret_cast<GemmSmem*>(smem_raw);

    int m_block_off = blockIdx.y * kBM;
    int n_block_off = blockIdx.x * kBN;

    block_gemm(C, M, N, A, K, B, m_block_off, n_block_off, smem);
}

static bool g_set_attrs = false;

void launch_gemm_pipeline(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          __nv_bfloat16* C, int M, int N, int K,
                          cudaStream_t stream) {
    int grid_x = (N + kBN - 1) / kBN;
    int grid_y = (M + kBM - 1) / kBM;
    int smem_bytes = sizeof(GemmSmem);

    if (!g_set_attrs) {
        cudaFuncSetAttribute(gemm_pipeline_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        g_set_attrs = true;
    }

    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlockSize, 1, 1);
    gemm_pipeline_kernel<<<grid, block, smem_bytes, stream>>>(C, A, B, M, N, K);
}

}  // namespace lucebox
