/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Block-resident BF16 GEMM with cp.async-pipelined operand loads + WMMA mma.
 *
 * `block_gemm` computes a (BM x BN) tile of C = A @ B (BF16, FP32-accum, BF16-out)
 * where:
 *   - A is [M, K] row-major in HBM
 *   - B is [K, N] row-major in HBM  (the packed-weight orientation, see pack_weights.py)
 *   - C is [M, N] row-major in HBM
 * One CUDA block computes one (BM x BN) output tile, reading the full K dim.
 *
 * Used in two ways:
 *   1. As a standalone kernel (`gemm_pipeline.cu`) for M3 unit testing.
 *   2. As an inlined per-stage GEMM inside the megakernel (`tinyllama_megakernel.cu`).
 *
 * Tile constants:
 *   BM=128, BN=128, BK=32   ;   8 warps per block (4 warp-rows x 2 warp-cols)
 *   each warp owns a 32x64 output tile of FP32 accumulators, issued as
 *   2x4 WMMA m16n16k16 fragments. WMMA on sm_80+ lowers to mma.sync m16n8k16.
 *
 * SMEM budget:
 *   A staging:   BM x BK x 2B * NSTAGES = 128 * 32 * 2 * 3 = 24 KB
 *   B staging:   BK x BN x 2B * NSTAGES =  32 * 128 * 2 * 3 = 24 KB
 *   C epilogue:  8 warps * 16 * 16 * 4B = 8 KB  (FP32 per-warp scratch for fragment store
 *                                                in `block_gemm`. Unused by
 *                                                `block_gemm_add` which writes the
 *                                                FP32 accumulator straight to global.)
 *   ----------------------------------------------------------------
 *                                                              56 KB  (fits 99 KB L40S cap)
 */
#pragma once

#include <mma.h>
#include <cooperative_groups.h>

#include "common.cuh"
#include "pipeline.cuh"

namespace lucebox {

namespace cg = cooperative_groups;

// Tile constants -- compile-time overridable for M6 sweep via setup.py env vars
// (MEGAKERNEL_BM / MEGAKERNEL_BN / MEGAKERNEL_BK / MEGAKERNEL_NSTAGES /
//  MEGAKERNEL_WARP_ROWS / MEGAKERNEL_WARP_COLS).
#ifndef MEGAKERNEL_BM
#define MEGAKERNEL_BM 128
#endif
#ifndef MEGAKERNEL_BN
#define MEGAKERNEL_BN 128
#endif
#ifndef MEGAKERNEL_BK
#define MEGAKERNEL_BK 32
#endif
#ifndef MEGAKERNEL_NSTAGES
#define MEGAKERNEL_NSTAGES 3
#endif
#ifndef MEGAKERNEL_WARP_ROWS
#define MEGAKERNEL_WARP_ROWS 4
#endif
#ifndef MEGAKERNEL_WARP_COLS
#define MEGAKERNEL_WARP_COLS 2
#endif

constexpr int kBM = MEGAKERNEL_BM;
constexpr int kBN = MEGAKERNEL_BN;
constexpr int kBK = MEGAKERNEL_BK;
constexpr int kNStages = MEGAKERNEL_NSTAGES;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kWarpRows = MEGAKERNEL_WARP_ROWS;
constexpr int kWarpCols = MEGAKERNEL_WARP_COLS;
constexpr int kWarpsForGemm = kWarpRows * kWarpCols;       // 8
constexpr int kWarpTileM = kBM / kWarpRows;                // 32
constexpr int kWarpTileN = kBN / kWarpCols;                // 64
constexpr int kWmmaPerWarpM = kWarpTileM / kWmmaM;         // 2
constexpr int kWmmaPerWarpN = kWarpTileN / kWmmaN;         // 4
constexpr int kKSubstepsPerStage = kBK / kWmmaK;           // 2

static_assert(kWarpsForGemm == kWarpsPerBlock,
              "kWarpRows * kWarpCols must equal BLOCK_SIZE/32");
static_assert(kBM % (kWarpRows * kWmmaM) == 0, "kBM must be divisible by kWarpRows*16");
static_assert(kBN % (kWarpCols * kWmmaN) == 0, "kBN must be divisible by kWarpCols*16");
static_assert(kBK % kWmmaK == 0, "kBK must be divisible by 16");

constexpr int kElemsPer16B = 8;                            // BF16 elements per 16-byte cp.async
constexpr int kLinesPerStage_A = (kBM * kBK) / kElemsPer16B;        // 512
constexpr int kLinesPerStage_B = (kBK * kBN) / kElemsPer16B;        // 512

// One block's full SMEM layout for `block_gemm`. Caller allocates this via
// dynamic SMEM (`extern __shared__ ...`) and passes a pointer.
struct GemmSmem {
    __nv_bfloat16 A[kNStages][kBM][kBK];
    __nv_bfloat16 B[kNStages][kBK][kBN];
    // Per-warp epilogue scratch: each warp stores its fp32 fragment into its own slot.
    float c_frag[kWarpsForGemm][kWmmaM][kWmmaN];
};

// Per-arch dynamic SMEM cap (bytes). These are the values exposed via
// `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
//   sm_80 / sm_86 / sm_89 : 99 KB (the "100 KB - 1 KB" hardware limit, see
//                                  CUDA C Programming Guide table 24).
//   sm_90 / sm_100        : 227 KB.
// We default conservatively to 99 KB (L40S target), which catches the case
// that ran in the M6 sweep where BK=64 silently overflowed to 96+ KB.
#ifndef MEGAKERNEL_SMEM_CAP_BYTES
#if TARGET_SM >= 90
#define MEGAKERNEL_SMEM_CAP_BYTES (227 * 1024)
#else
#define MEGAKERNEL_SMEM_CAP_BYTES (99 * 1024)
#endif
#endif

static_assert(sizeof(GemmSmem) <= MEGAKERNEL_SMEM_CAP_BYTES,
              "GemmSmem exceeds per-block dynamic SMEM cap for the target arch. "
              "Reduce MEGAKERNEL_BM/BN/BK or MEGAKERNEL_NSTAGES. On sm_89 the cap "
              "is 99 KB; the M6 sweep silently broke at BK=64,NSTAGES=3 (96 KB for "
              "A+B staging alone, plus 8 KB epilogue scratch = 104 KB > 99 KB).");

// Cooperative load of one stage of A and B from HBM into SMEM via cp.async.
// Caller is responsible for issuing cp_async_commit_group() after this fn returns.
__forceinline__ __device__
void issue_stage_load(GemmSmem& smem, int stage,
                      const __nv_bfloat16* __restrict__ A, int M, int K,
                      const __nv_bfloat16* __restrict__ B, int N,
                      int m_block_off, int n_block_off, int k_block_off) {
    const int tid = threadIdx.x;
    constexpr int kLinesPerThread_A = kLinesPerStage_A / kBlockSize;  // 2
    constexpr int kLinesPerThread_B = kLinesPerStage_B / kBlockSize;  // 2

    #pragma unroll
    for (int i = 0; i < kLinesPerThread_A; ++i) {
        int line = i * kBlockSize + tid;
        int row = line / (kBK / kElemsPer16B);           // line / 4
        int col = (line % (kBK / kElemsPer16B)) * kElemsPer16B;
        int gm_row = m_block_off + row;
        int gm_col = k_block_off + col;
        const __nv_bfloat16* src = A + gm_row * K + gm_col;
        uint32_t dst = cvt_to_smem(&smem.A[stage][row][col]);
        bool pred = (gm_row < M) && (gm_col + kElemsPer16B <= K);
        cp_async_16B(dst, src, pred);
    }
    #pragma unroll
    for (int i = 0; i < kLinesPerThread_B; ++i) {
        int line = i * kBlockSize + tid;
        int row = line / (kBN / kElemsPer16B);           // line / 16
        int col = (line % (kBN / kElemsPer16B)) * kElemsPer16B;
        int gm_row = k_block_off + row;
        int gm_col = n_block_off + col;
        const __nv_bfloat16* src = B + gm_row * N + gm_col;
        uint32_t dst = cvt_to_smem(&smem.B[stage][row][col]);
        bool pred = (gm_row < K) && (gm_col + kElemsPer16B <= N);
        cp_async_16B(dst, src, pred);
    }
}

// Per-warp accumulator -- FP32, register-resident.
struct WarpAcc {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
        c[kWmmaPerWarpM][kWmmaPerWarpN];

    __forceinline__ __device__ void zero() {
        #pragma unroll
        for (int i = 0; i < kWmmaPerWarpM; ++i) {
            #pragma unroll
            for (int j = 0; j < kWmmaPerWarpN; ++j) {
                nvcuda::wmma::fill_fragment(c[i][j], 0.f);
            }
        }
    }
};

// Run one BK-chunk worth of MMA into `acc` from the given SMEM stage.
__forceinline__ __device__
void warp_mma_one_stage(WarpAcc& acc, const GemmSmem& smem, int stage,
                        int warp_row, int warp_col) {
    using namespace nvcuda;

    #pragma unroll
    for (int ks = 0; ks < kKSubstepsPerStage; ++ks) {
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> b_frag;

        #pragma unroll
        for (int i = 0; i < kWmmaPerWarpM; ++i) {
            int m_off = warp_row * kWarpTileM + i * kWmmaM;
            const __nv_bfloat16* a_tile = &smem.A[stage][m_off][ks * kWmmaK];
            wmma::load_matrix_sync(a_frag, a_tile, kBK);

            #pragma unroll
            for (int j = 0; j < kWmmaPerWarpN; ++j) {
                int n_off = warp_col * kWarpTileN + j * kWmmaN;
                const __nv_bfloat16* b_tile = &smem.B[stage][ks * kWmmaK][n_off];
                wmma::load_matrix_sync(b_frag, b_tile, kBN);
                wmma::mma_sync(acc.c[i][j], a_frag, b_frag, acc.c[i][j]);
            }
        }
    }
}

// Store a warp's accumulator to HBM as BF16. C is [M, N] row-major.
__forceinline__ __device__
void warp_store_accum(__nv_bfloat16* __restrict__ C, int M, int N,
                       int m_block_off, int n_block_off,
                       int warp_row, int warp_col, int warp_id,
                       GemmSmem& smem, const WarpAcc& acc) {
    using namespace nvcuda;
    const int lane = lane_id();

    #pragma unroll
    for (int i = 0; i < kWmmaPerWarpM; ++i) {
        #pragma unroll
        for (int j = 0; j < kWmmaPerWarpN; ++j) {
            // Use the per-warp scratch slot to materialise the FP32 16x16 fragment.
            wmma::store_matrix_sync(&smem.c_frag[warp_id][0][0], acc.c[i][j],
                                    kWmmaN, wmma::mem_row_major);
            __syncwarp();
            // Cast & write back to HBM. 32 threads cover 16*16=256 BF16 elems -> 8 elems/thread.
            // Each thread writes one BF16x8 (= 16B = int4) chunk; total = 32*8 = 256 elems.
            const int tile_m_off = warp_row * kWarpTileM + i * kWmmaM;
            const int tile_n_off = warp_col * kWarpTileN + j * kWmmaN;
            // Lane mapping: lane / 2 -> row 0..15, lane % 2 -> 0 or 1 (each writes 8 cols).
            const int row = lane >> 1;
            const int col_base = (lane & 1) * 8;
            __nv_bfloat16 out[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                out[e] = float_to_bf16(smem.c_frag[warp_id][row][col_base + e]);
            }
            int gm_row = m_block_off + tile_m_off + row;
            int gm_col = n_block_off + tile_n_off + col_base;
            if (gm_row < M && gm_col + 8 <= N) {
                *reinterpret_cast<int4*>(C + gm_row * N + gm_col) =
                    *reinterpret_cast<const int4*>(out);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (gm_row < M && gm_col + e < N) {
                        C[gm_row * N + gm_col + e] = out[e];
                    }
                }
            }
            __syncwarp();
        }
    }
}

// One block computes one (BM x BN) output tile.
// `smem` is the dynamic SMEM region cast to a `GemmSmem*`.
__forceinline__ __device__
void block_gemm(__nv_bfloat16* __restrict__ C, int M, int N,
                const __nv_bfloat16* __restrict__ A, int K,
                const __nv_bfloat16* __restrict__ B,
                int m_block_off, int n_block_off,
                GemmSmem& smem) {
    const int warp = warp_id();
    const int warp_row = warp / kWarpCols;
    const int warp_col = warp % kWarpCols;

    WarpAcc acc;
    acc.zero();

    const int num_k_blocks = (K + kBK - 1) / kBK;

    // ---- Prologue: issue the first (kNStages-1) loads.
    int k_load = 0;
    #pragma unroll
    for (int s = 0; s < kNStages - 1; ++s) {
        if (k_load < num_k_blocks) {
            issue_stage_load(smem, s, A, M, K, B, N,
                             m_block_off, n_block_off, k_load * kBK);
        }
        cp_async_commit_group();
        ++k_load;
    }

    // ---- Steady state ----
    int compute_stage = 0;
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        // Wait for the stage we're about to consume.
        cp_async_wait_group<kNStages - 2>();
        __syncthreads();

        // Issue the next stage's load (if any K work remaining).
        if (k_load < num_k_blocks) {
            int load_stage = (compute_stage + (kNStages - 1)) % kNStages;
            issue_stage_load(smem, load_stage, A, M, K, B, N,
                             m_block_off, n_block_off, k_load * kBK);
        }
        cp_async_commit_group();
        ++k_load;

        // Compute one BK chunk.
        warp_mma_one_stage(acc, smem, compute_stage, warp_row, warp_col);
        compute_stage = (compute_stage + 1) % kNStages;
    }

    cp_async_wait_all();
    __syncthreads();

    // Epilogue: store BF16 outputs.
    warp_store_accum(C, M, N, m_block_off, n_block_off,
                     warp_row, warp_col, warp, smem, acc);
}

// Variant that ADDS to an existing accumulator (for residual streams).
// `C += A @ B`, where C is BF16 (matches HF Llama's BF16 residual stream).
//
// Preload: each warp reads its 8 (i, j) 16x16 BF16 tiles from C, casts to
// FP32 element-wise into a per-warp SMEM scratch slot, then loads as a
// wmma accumulator fragment. After the K-loop, `warp_store_accum` casts
// FP32 acc -> BF16 and writes back.
__forceinline__ __device__
void block_gemm_add(__nv_bfloat16* __restrict__ C, int M, int N,
                    const __nv_bfloat16* __restrict__ A, int K,
                    const __nv_bfloat16* __restrict__ B,
                    int m_block_off, int n_block_off,
                    GemmSmem& smem) {
    const int warp = warp_id();
    const int warp_row = warp / kWarpCols;
    const int warp_col = warp % kWarpCols;

    WarpAcc acc;
    {
        using namespace nvcuda;
        const int lane = lane_id();
        const int row = lane >> 1;
        const int col_base = (lane & 1) * 8;
        #pragma unroll
        for (int i = 0; i < kWmmaPerWarpM; ++i) {
            #pragma unroll
            for (int j = 0; j < kWmmaPerWarpN; ++j) {
                int gm_row = m_block_off + warp_row * kWarpTileM + i * kWmmaM;
                int gm_col = n_block_off + warp_col * kWarpTileN + j * kWmmaN;
                if (gm_row + kWmmaM <= M && gm_col + kWmmaN <= N) {
                    const __nv_bfloat16* gptr = C + (gm_row + row) * N + gm_col + col_base;
                    int4 v = *reinterpret_cast<const int4*>(gptr);
                    const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        smem.c_frag[warp][row][col_base + e] = bf16_to_float(vp[e]);
                    }
                    __syncwarp();
                    wmma::load_matrix_sync(acc.c[i][j], &smem.c_frag[warp][0][0],
                                           kWmmaN, wmma::mem_row_major);
                } else {
                    wmma::fill_fragment(acc.c[i][j], 0.f);
                }
            }
        }
    }
    __syncthreads();  // ensure preload c_frag writes are done before A/B reuse smem

    const int num_k_blocks = (K + kBK - 1) / kBK;
    int k_load = 0;
    #pragma unroll
    for (int s = 0; s < kNStages - 1; ++s) {
        if (k_load < num_k_blocks) {
            issue_stage_load(smem, s, A, M, K, B, N,
                             m_block_off, n_block_off, k_load * kBK);
        }
        cp_async_commit_group();
        ++k_load;
    }

    int compute_stage = 0;
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        cp_async_wait_group<kNStages - 2>();
        __syncthreads();

        if (k_load < num_k_blocks) {
            int load_stage = (compute_stage + (kNStages - 1)) % kNStages;
            issue_stage_load(smem, load_stage, A, M, K, B, N,
                             m_block_off, n_block_off, k_load * kBK);
        }
        cp_async_commit_group();
        ++k_load;

        warp_mma_one_stage(acc, smem, compute_stage, warp_row, warp_col);
        compute_stage = (compute_stage + 1) % kNStages;
    }
    cp_async_wait_all();
    __syncthreads();

    warp_store_accum(C, M, N, m_block_off, n_block_off,
                     warp_row, warp_col, warp, smem, acc);
}

}  // namespace lucebox
