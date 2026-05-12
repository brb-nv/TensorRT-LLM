/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyLlama-1.1B prefill MEGAKERNEL -- one persistent CUDA dispatch covers
 * the entire forward pass (embedding + 22 transformer layers + final RMSNorm
 * + LM-head) for batch=1, seq_len=128, BF16. NVBug 5615248, L40S target.
 *
 * Persistent grid: `kNumBlocks` blocks (one per SM = 142 on L40S), each
 * sticky-resident for the full forward pass. Cooperative grid-sync between
 * stages within a layer (and between layers); the host launches via
 * cudaLaunchCooperativeKernel.
 *
 * SMEM layout: ONE dynamic SMEM region per block, sized to `GemmSmem`
 * (~56 KB, fits the 99 KB L40S cap). The attention stage reuses the first
 * `seq_len * sizeof(float)` bytes of this region for its score scratch.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "common.cuh"
#include "pipeline.cuh"
#include "gemm_pipeline.cuh"
#include "stages.cuh"

namespace lucebox {

namespace cg = cooperative_groups;

// ---- helper: GEMM over the whole (M=seq_len) x N slab ---------------
// Distribute (m_tile, n_tile) work-units across the persistent grid: each
// block handles tiles whose flat index modulo gridDim.x equals blockIdx.x.
//
// IMPORTANT: must iterate the full 2D tile space. Earlier versions of this
// helper hard-coded `m_block_off = 0` and looped only over N-tiles. That
// silently corrupted output when BM < M (e.g. BM=64 with seq_len=128 left
// rows 64..127 uninitialised), which is exactly the config the M6 sweep
// reported as "best at 11.1 ms" -- the kernel was just doing half the
// GEMM math. `tests/sweep.py` doesn't validate, so this slipped through;
// `tests/test_numerics.py` catches it now.
__forceinline__ __device__
void grid_gemm(__nv_bfloat16* __restrict__ C, int M, int N,
               const __nv_bfloat16* __restrict__ A, int K,
               const __nv_bfloat16* __restrict__ B,
               GemmSmem& smem) {
    const int n_tiles = (N + kBN - 1) / kBN;
    const int m_tiles = (M + kBM - 1) / kBM;
    const int total_tiles = m_tiles * n_tiles;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        int mt = t / n_tiles;
        int nt = t - mt * n_tiles;
        block_gemm(C, M, N, A, K, B, mt * kBM, nt * kBN, smem);
    }
}

// Grid-distributed residual add: `residual[r, c] += x[r, c]` in BF16.
// Wraps `stage_residual_add` so it can be called like other grid helpers.
__forceinline__ __device__
void grid_residual_add(__nv_bfloat16* __restrict__ residual,
                       const __nv_bfloat16* __restrict__ x,
                       int seq_len) {
    stage_residual_add(residual, x, seq_len, kHidden);
}

// ---- The megakernel ----------------------------------------------------
// `__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)`. We ask the compiler to
// plan for 2 blocks/SM:
//   - SMEM per block ~44 KB (BM=64,BN=128,BK=32,NSTAGES=3,8-warp GEMM); two
//     blocks fit in the 99 KB sm_89 dynamic SMEM cap.
//   - Each block has 256 threads, so 2 blocks/SM = 512 threads/SM = 33% of
//     L40S's 1536-thread/SM ceiling.
//   - 65,536 regs/SM split across 512 threads => up to 128 regs/thread before
//     the compiler must spill. WMMA fragments cost ~8 regs each * 8 frags +
//     A/B tile regs + address arith; total is comfortably below 128.
// The host launcher then sets gridDim = min(NUM_BLOCKS, num_sms *
// cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)), so if the compiler
// fails to fit 2 blocks/SM we transparently fall back to 1 block/SM at
// gridDim = num_sms. Net: this change is safe -- worst case is no-op.
//
// Why this helps: with 1 block/SM, when that block stalls on cp.async wait
// the entire SM is idle. With 2 blocks/SM the scheduler swaps to the other
// block's warps. For the small-N GEMMs (qkv, o_proj, down_proj where
// N in {2048, 2560}), per-GEMM total tile count is 32-40 and gridDim=142
// finishes them in 1 wave -- but per-block latency was previously hiding
// cp.async stalls poorly. Multi-block-per-SM mostly attacks that.
__launch_bounds__(kBlockSize, 2)
__global__ void tinyllama_megakernel(MegakernelParams params) {
    extern __shared__ unsigned char smem_raw[];
    GemmSmem& gsmem = *reinterpret_cast<GemmSmem*>(smem_raw);
    float* attn_scratch = reinterpret_cast<float*>(smem_raw);  // alias the same region

    cg::grid_group grid = cg::this_grid();

    const int seq_len = params.seq_len;
    const __nv_bfloat16* weights = params.weights;
    const LayerOffsets* offs = params.layer_offsets;

    // ===== Stage 0: token embedding -> residual =====
    const __nv_bfloat16* embed = weights + params.embed_offset;
    stage_token_embed(params.residual, params.input_ids, embed, seq_len);
    grid.sync();

    // ===== 22 transformer layers =====
    for (int L = 0; L < kNumLayers; ++L) {
        const __nv_bfloat16* W_ln1     = weights + offs[L].input_layernorm;
        const __nv_bfloat16* W_qkv     = weights + offs[L].qkv_proj;
        const __nv_bfloat16* W_o       = weights + offs[L].o_proj;
        const __nv_bfloat16* W_ln2     = weights + offs[L].post_attn_norm;
        const __nv_bfloat16* W_gate_up = weights + offs[L].gate_up;
        const __nv_bfloat16* W_down    = weights + offs[L].down;

        // ----- Stage 1a: pre-attn RMSNorm on residual -> act_a
        stage_rms_norm(params.act_a, params.residual, W_ln1, seq_len, kHidden);
        grid.sync();

        // ----- Stage 1b: QKV proj: act_a @ W_qkv -> qkv_buf  (M, kQkvSize)
        grid_gemm(params.qkv_buf, seq_len, kQkvSize,
                  params.act_a, kHidden, W_qkv, gsmem);
        grid.sync();

        // ----- Stage 2: RoPE in-place on Q,K of qkv_buf
        stage_rope(params.qkv_buf, seq_len);
        grid.sync();

        // ----- Stage 3: prefill causal attention -> attn_buf (M, kHidden)
        stage_attention(params.attn_buf, params.qkv_buf, attn_scratch, seq_len);
        grid.sync();

        // ----- Stage 4a: O proj: act_b = attn_buf @ W_o  (plain BF16 GEMM)
        // ----- Stage 4b: residual += act_b  (BF16+BF16 -> BF16, two casts)
        //   MATCH HF: HF computes `attn_proj = self.o_proj(attn_out)` (a
        //   separate BF16 tensor) and then `residual + attn_proj` is its own
        //   BF16+BF16 cast point. Fusing the two into a single
        //   `residual += attn @ W_o` skips that intermediate cast and adds
        //   ~1 BF16 ULP/element drift per layer, which compounds to ~10% mean
        //   drift by L21. Two-stage form matches HF bit-pattern.
        grid_gemm(params.act_b, seq_len, kHidden,
                  params.attn_buf, kHidden, W_o, gsmem);
        grid.sync();
        grid_residual_add(params.residual, params.act_b, seq_len);
        grid.sync();

        // ----- Stage 5a: post-attn RMSNorm on residual -> act_a
        stage_rms_norm(params.act_a, params.residual, W_ln2, seq_len, kHidden);
        grid.sync();

        // ----- Stage 5b: gate_up GEMM: act_a @ W_gate_up -> gate_up_buf
        grid_gemm(params.gate_up_buf, seq_len, kGateUpSize,
                  params.act_a, kHidden, W_gate_up, gsmem);
        grid.sync();

        // ----- Stage 6: SiLU(gate) * up -> silu_buf
        stage_silu_mul(params.silu_buf, params.gate_up_buf, seq_len);
        grid.sync();

        // ----- Stage 7a: down proj: act_b = silu_buf @ W_down  (plain GEMM)
        // ----- Stage 7b: residual += act_b  (BF16+BF16 -> BF16)
        //   Same un-fusing as stage 4 so the kernel hits HF's intermediate
        //   BF16 cast on the MLP output before the residual add.
        grid_gemm(params.act_b, seq_len, kHidden,
                  params.silu_buf, kIntermediate, W_down, gsmem);
        grid.sync();
        grid_residual_add(params.residual, params.act_b, seq_len);
        grid.sync();

        // ----- Optional per-layer residual dump (for divide-and-conquer
        //       debugging). Copy BF16 residual to dump[L * seq_len * hidden].
        if (params.layer_residual_dump != nullptr) {
            constexpr int kElemsPer16B = 8;
            __nv_bfloat16* dst = params.layer_residual_dump +
                                 L * seq_len * kHidden;
            const int total = seq_len * kHidden / kElemsPer16B;
            const int tid = threadIdx.x;
            for (int i = blockIdx.x * blockDim.x + tid; i < total;
                 i += gridDim.x * blockDim.x) {
                int4 v = *reinterpret_cast<const int4*>(
                    params.residual + i * kElemsPer16B);
                *reinterpret_cast<int4*>(dst + i * kElemsPer16B) = v;
            }
            grid.sync();
        }
    }

    // ===== Final RMSNorm + LM head =====
    const __nv_bfloat16* W_final_norm = weights + params.final_norm_offset;
    const __nv_bfloat16* W_lm_head    = weights + params.lm_head_offset;

    stage_rms_norm(params.act_a, params.residual, W_final_norm, seq_len, kHidden);
    grid.sync();

    grid_gemm(params.logits, seq_len, kVocab,
              params.act_a, kHidden, W_lm_head, gsmem);
    // implicit grid.sync() at kernel exit is fine; no further work
}

// ---- host launcher -----------------------------------------------------
static bool g_megakernel_attrs_set = false;

void launch_megakernel(const MegakernelParams& params, cudaStream_t stream) {
    int smem_bytes = sizeof(GemmSmem);
    if (!g_megakernel_attrs_set) {
        cudaFuncSetAttribute(tinyllama_megakernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        g_megakernel_attrs_set = true;
    }

    // Clamp grid size to the resident-block ceiling so cudaLaunchCooperativeKernel
    // doesn't deadlock (every block must be co-resident).
    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &max_blocks_per_sm, tinyllama_megakernel, kBlockSize, smem_bytes,
        cudaOccupancyDefault);
    if (max_blocks_per_sm <= 0) {
        // Kernel cannot fit any blocks. Most likely: register pressure with
        // BLOCK_SIZE=256 + 8 WMMA fragments/warp. Print diagnostic; the launch
        // below will fail with cudaErrorInvalidValue.
        fprintf(stderr,
                "[lucebox_tinyllama] WARNING: cudaOccupancyMaxActiveBlocksPerMultiprocessor "
                "returned 0 for tinyllama_megakernel. Likely cause: register pressure. "
                "Try rebuilding with smaller MEGAKERNEL_BLOCK_SIZE (e.g. 128) or smaller "
                "warp tiles (MEGAKERNEL_WARP_ROWS=4 MEGAKERNEL_WARP_COLS=1, BLOCK_SIZE=128).\n");
    }
    // One-shot occupancy report: tells us whether the
    // `__launch_bounds__(BLOCK_SIZE, 2)` request actually took effect or
    // whether register pressure pinned us to 1 block/SM (the perf cliff we
    // tracked in M6 sweep).
    if (getenv("MEGAKERNEL_REPORT_OCC") != nullptr) {
        fprintf(stderr,
                "[lucebox_tinyllama] occupancy: max_blocks_per_sm=%d, smem_bytes=%d, "
                "block_size=%d, num_sms=%d, grid_size=%d (resident_cap=%d)\n",
                max_blocks_per_sm, smem_bytes, kBlockSize, num_sms,
                (kNumBlocks < num_sms * max_blocks_per_sm)
                    ? kNumBlocks : num_sms * max_blocks_per_sm,
                num_sms * max_blocks_per_sm);
    }
    int max_co_resident = num_sms * max_blocks_per_sm;
    int grid_size = (kNumBlocks < max_co_resident) ? kNumBlocks : max_co_resident;
    if (grid_size <= 0) grid_size = num_sms;  // last-resort fallback (will likely deadlock)

    dim3 grid(grid_size, 1, 1);
    dim3 block(kBlockSize, 1, 1);

    // Cooperative launch.
    void* args[] = { const_cast<MegakernelParams*>(&params) };
    cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(tinyllama_megakernel),
        grid, block, args, smem_bytes, stream);
}

}  // namespace lucebox
