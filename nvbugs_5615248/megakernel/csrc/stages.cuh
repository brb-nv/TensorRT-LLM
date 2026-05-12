/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Per-stage device functions used by `tinyllama_megakernel.cu`.
 *
 *   stage_token_embed     : input_ids -> BF16 residual[seq_len, hidden]
 *   stage_rms_norm        : per-row RMSNorm (BF16 in/out, FP32 reduction; matches HF)
 *   stage_rope_apply      : RoPE rotation on Q,K (in-place on the BF16 QKV buffer)
 *   stage_attention       : causal prefill attention with GQA, online softmax,
 *                           BF16-quantized probabilities to match HF eager
 *   stage_silu_mul        : SiLU(gate) * up -> intermediate (BF16; matches HF MLP)
 *   stage_residual_add    : x += y (BF16; unused -- megakernel uses grid_gemm_add)
 *
 * All stages are "block-cooperative round-robin": a persistent grid of
 * `kNumBlocks` blocks splits the work across `block_id, block_id+grid, ...`.
 */
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <cooperative_groups.h>
#include "common.cuh"

namespace lucebox {

namespace cg = cooperative_groups;

// ---- Stage 0: token embedding -> BF16 residual --------------------------
// Copies the (input_ids[i])-th BF16 row of embed_tokens into residual[i].
__forceinline__ __device__
void stage_token_embed(__nv_bfloat16* __restrict__ residual,
                       const int32_t* __restrict__ ids,
                       const __nv_bfloat16* __restrict__ embed,
                       int seq_len) {
    constexpr int kElemsPer16B = 8;
    constexpr int kIntsPerRow  = kHidden / kElemsPer16B;  // 256
    const int tid = threadIdx.x;

    for (int row = blockIdx.x; row < seq_len; row += gridDim.x) {
        int token_id = ids[row];
        const __nv_bfloat16* src = embed + token_id * kHidden;
        __nv_bfloat16* dst = residual + row * kHidden;
        for (int i = tid; i < kIntsPerRow; i += blockDim.x) {
            int4 v = *reinterpret_cast<const int4*>(src + i * kElemsPer16B);
            *reinterpret_cast<int4*>(dst + i * kElemsPer16B) = v;
        }
    }
}

// ---- Stage: RMSNorm (per-row) -------------------------------------------
// Matches HF Llama `LlamaRMSNorm.forward`:
//   hidden_fp32 = x.to(fp32)
//   variance    = hidden_fp32.pow(2).mean(-1)
//   hidden_fp32 = hidden_fp32 * rsqrt(variance + eps)
//   return weight * hidden_fp32.to(bf16)           <-- BF16 cast BEFORE weight multiply
//
// In particular, HF casts the normalized hidden state back to BF16 BEFORE
// multiplying by the (BF16) weight. That extra round-trip is a small precision
// loss the kernel must replicate to match HF logits.
__forceinline__ __device__
void stage_rms_norm(__nv_bfloat16* __restrict__ y,
                    const __nv_bfloat16* __restrict__ x,
                    const __nv_bfloat16* __restrict__ weight,
                    int seq_len, int dim, float eps = kRmsEps) {
    constexpr int kElemsPer16B = 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ float scratch[kWarpsPerBlock + 1];

    for (int row = blockIdx.x; row < seq_len; row += gridDim.x) {
        const __nv_bfloat16* xr = x + row * dim;
        __nv_bfloat16* yr = y + row * dim;

        // Thread-local sum of squares (FP32 reduction on BF16-loaded values).
        float sum = 0.f;
        for (int i = tid; i < dim / kElemsPer16B; i += blockDim.x) {
            int4 v = *reinterpret_cast<const int4*>(xr + i * kElemsPer16B);
            const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
            #pragma unroll
            for (int e = 0; e < kElemsPer16B; ++e) {
                float f = bf16_to_float(vp[e]);
                sum += f * f;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        }
        if (lane == 0) scratch[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float s = (lane < kWarpsPerBlock) ? scratch[lane] : 0.f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffff, s, off);
            }
            if (lane == 0) {
                float mean = s / (float)dim;
                scratch[kWarpsPerBlock] = rsqrtf(mean + eps);
            }
        }
        __syncthreads();
        float inv = scratch[kWarpsPerBlock];

        // Scale & write back -- MATCH HF: cast normalized hidden to BF16 first,
        // then multiply by BF16 weight (the multiply itself happens in FP32
        // internally on tensor cores / FP32 fma but we model it as fp32 mul
        // of two BF16-quantized operands).
        for (int i = tid; i < dim / kElemsPer16B; i += blockDim.x) {
            int4 v = *reinterpret_cast<const int4*>(xr + i * kElemsPer16B);
            int4 w = *reinterpret_cast<const int4*>(weight + i * kElemsPer16B);
            const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
            const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&w);
            __nv_bfloat16 out[kElemsPer16B];
            #pragma unroll
            for (int e = 0; e < kElemsPer16B; ++e) {
                float fx_norm = bf16_to_float(vp[e]) * inv;          // FP32 normalize
                __nv_bfloat16 hx_bf = float_to_bf16(fx_norm);        // <-- HF cast point
                float fw = bf16_to_float(wp[e]);
                out[e] = float_to_bf16(bf16_to_float(hx_bf) * fw);   // BF16 * BF16 -> BF16
            }
            *reinterpret_cast<int4*>(yr + i * kElemsPer16B) =
                *reinterpret_cast<const int4*>(out);
        }
        __syncthreads();
    }
}

// ---- Stage: RoPE on Q,K (in-place on the QKV buffer) --------------------
// QKV buffer layout per row: [Q (Q_size) | K (kv_size) | V (kv_size)].
// Q has shape (num_heads, head_dim) per row; K has shape (num_kv_heads, head_dim).
// HF-style half-rotation: pair (i, i + head_dim/2), rotate by (cos(theta), sin(theta)).
// theta_i = pos / (rope_theta ^ (i / (head_dim/2))) for i in [0..head_dim/2)
__forceinline__ __device__
void stage_rope(__nv_bfloat16* __restrict__ qkv,
                int seq_len, float rope_theta = kRopeTheta) {
    constexpr int kHalfHead = kHeadDim / 2;
    const int tid = threadIdx.x;

    for (int row = blockIdx.x; row < seq_len; row += gridDim.x) {
        __nv_bfloat16* qrow = qkv + row * kQkvSize;
        __nv_bfloat16* krow = qrow + kQSize;  // K starts after Q

        // We have Q: num_heads x head_dim = 32 x 64 = 2048 values
        //        K: num_kv_heads x head_dim = 4 x 64 = 256 values
        // Total head-pairs to rotate: (num_heads + num_kv_heads) * (head_dim/2)
        //                            = (32 + 4) * 32 = 1152 pairs
        // Spread across 256 threads -> 4-5 pairs per thread.

        constexpr int kQHeads  = kNumHeads;
        constexpr int kKHeads  = kNumKvHeads;
        constexpr int kTotalPairs = (kQHeads + kKHeads) * kHalfHead;

        for (int p = tid; p < kTotalPairs; p += blockDim.x) {
            int head_pair_idx = p;
            int head_id = head_pair_idx / kHalfHead;
            int pair_idx = head_pair_idx % kHalfHead;

            __nv_bfloat16* base;
            if (head_id < kQHeads) {
                base = qrow + head_id * kHeadDim;
            } else {
                base = krow + (head_id - kQHeads) * kHeadDim;
            }

            float pos = (float)row;
            float freq = expf(-logf(rope_theta) * (float)pair_idx /
                              (float)kHalfHead);
            float angle = pos * freq;
            // Use precise sincosf rather than __sincosf intrinsic: at
            // pair_idx=0 and row=seq_len-1 the angle is up to ~127 rad,
            // and __sincosf's range reduction is too coarse there.
            float c_fp32, s_fp32; sincosf(angle, &s_fp32, &c_fp32);
            // MATCH HF: cos/sin are stored as BF16 (cast after FP32 compute),
            // then the rotation uses BF16-quantized cos/sin values.
            float c = bf16_to_float(float_to_bf16(c_fp32));
            float s = bf16_to_float(float_to_bf16(s_fp32));

            float x0 = bf16_to_float(base[pair_idx]);
            float x1 = bf16_to_float(base[pair_idx + kHalfHead]);
            // MATCH HF apply_rotary_pos_emb: each of the two terms in
            //   q_embed = (q * cos) + (rotate_half(q) * sin)
            // is computed as a separate BF16 tensor before the addition, so
            // the kernel applies the same per-term BF16 quantization.
            float t_a = bf16_to_float(float_to_bf16(x0 * c));
            float t_b = bf16_to_float(float_to_bf16(-x1 * s));
            float t_c = bf16_to_float(float_to_bf16(x1 * c));
            float t_d = bf16_to_float(float_to_bf16(x0 * s));
            base[pair_idx]               = float_to_bf16(t_a + t_b);
            base[pair_idx + kHalfHead]   = float_to_bf16(t_c + t_d);
        }
        __syncthreads();
    }
}

// ---- Stage: causal prefill attention (one (head, query_row) per block-unit) --
//
// Input:  qkv buffer [seq_len, kQkvSize]
// Output: attn_buf  [seq_len, kHidden]
// For each (q_head=h, query_row=r):
//     k_head = h / kKvGroups
//     scores[t] = (Q[r, h, :] dot K[t, k_head, :]) * scale, for t in [0..seq_len)
//     scores[t] = -inf if t > r  (causal)
//     p[t] = softmax(scores)
//     out[r, h, :] = sum_t p[t] * V[t, k_head, :]
__forceinline__ __device__
void stage_attention(__nv_bfloat16* __restrict__ attn_out,
                     const __nv_bfloat16* __restrict__ qkv,
                     float* __restrict__ scores_scratch,  // [seq_len] floats per block
                     int seq_len) {
    constexpr int kTotalUnits = kNumHeads * kSeqLen;     // 32 * 128 = 4096
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float scale = 1.f / sqrtf((float)kHeadDim);

    float* scores = scores_scratch;  // [seq_len] floats

    for (int unit = blockIdx.x; unit < kTotalUnits; unit += gridDim.x) {
        int q_head = unit / kSeqLen;
        int q_row  = unit % kSeqLen;
        if (q_row >= seq_len) continue;
        int k_head = q_head / kKvGroups;

        const __nv_bfloat16* q_ptr = qkv + q_row * kQkvSize + q_head * kHeadDim;
        // K and V are located at qkv + t*kQkvSize + kQSize{,+kKvSize}; computed
        // inline in the per-t loops below.

        // Cache the Q row in registers (head_dim=64 = 32 fp32 vals).
        float q_reg[kHeadDim];
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            q_reg[d] = bf16_to_float(q_ptr[d]);
        }

        // Phase 1: compute scores[t] = Q . K[t] for t in [0..q_row]
        // Then mask t > q_row, softmax.
        // Spread the t's across the block's threads (256 threads, 128 t's -> 1 per thread).
        float local_max = -1e30f;
        for (int t = tid; t < seq_len; t += blockDim.x) {
            if (t > q_row) {
                scores[t] = -1e30f;
                continue;
            }
            const __nv_bfloat16* k_ptr = qkv + t * kQkvSize + kQSize + k_head * kHeadDim;
            float acc = 0.f;
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                acc += q_reg[d] * bf16_to_float(k_ptr[d]);
            }
            // MATCH HF eager: `torch.matmul(query, key.T)` returns BF16 (cublas
            // FP32 acc -> BF16 cast), then `* scaling` is BF16 * Python float ->
            // BF16. Without these two casts the kernel keeps QK^T at FP32
            // precision (more accurate than HF) and accumulates ~0.02 logit
            // drift across 22 layers.
            float qk_bf = bf16_to_float(float_to_bf16(acc));
            float s    = bf16_to_float(float_to_bf16(qk_bf * scale));
            scores[t] = s;
            local_max = fmaxf(local_max, s);
        }
        // Block-wide max reduction.
        // warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        }
        __shared__ float warp_max[kWarpsPerBlock];
        __shared__ float block_max_smem;
        __shared__ float block_sum_smem;
        if (lane == 0) warp_max[warp] = local_max;
        __syncthreads();
        if (warp == 0) {
            float m = (lane < kWarpsPerBlock) ? warp_max[lane] : -1e30f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
            }
            if (lane == 0) block_max_smem = m;
        }
        __syncthreads();
        float block_max = block_max_smem;

        // Phase 2: exp(scores - max), accumulate sum.
        float local_sum = 0.f;
        for (int t = tid; t < seq_len; t += blockDim.x) {
            float e = expf(scores[t] - block_max);
            // -inf - block_max could give 0 (fine).
            scores[t] = e;
            local_sum += e;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
        }
        if (lane == 0) warp_max[warp] = local_sum;  // reuse smem
        __syncthreads();
        if (warp == 0) {
            float s = (lane < kWarpsPerBlock) ? warp_max[lane] : 0.f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffff, s, off);
            }
            if (lane == 0) block_sum_smem = s;
        }
        __syncthreads();
        float inv_sum = 1.f / block_sum_smem;

        // Phase 2b: MATCH HF eager attention -- cast probabilities to BF16
        // before the V matmul. HF does:
        //   attn_weights = softmax(scores, dim=-1, dtype=fp32).to(bf16)
        //   attn_output  = attn_weights @ V    # BF16 x BF16 -> BF16
        // The cast happens after softmax, before V matmul. Without this cast
        // the kernel is *more* accurate than HF and the test sees ~0.01-0.02
        // logit drift purely from this quantization mismatch.
        for (int t = tid; t < seq_len; t += blockDim.x) {
            scores[t] = bf16_to_float(float_to_bf16(scores[t] * inv_sum));
        }
        __syncthreads();

        // Phase 3: out[d] = sum_t p[t] * V[t, k_head, d]
        // Each thread (tid < kHeadDim) handles one output dim.
        __nv_bfloat16* out_ptr = attn_out + q_row * kHidden + q_head * kHeadDim;
        if (tid < kHeadDim) {
            int d = tid;
            float acc = 0.f;
            for (int t = 0; t <= q_row; ++t) {
                const __nv_bfloat16* v_ptr = qkv + t * kQkvSize + kQSize + kKvSize +
                                             k_head * kHeadDim;
                acc += scores[t] * bf16_to_float(v_ptr[d]);
            }
            out_ptr[d] = float_to_bf16(acc);
        }
        __syncthreads();
    }
}

// ---- Stage: SiLU(gate) * up -> intermediate ------------------------------
// gate_up buffer: [seq_len, 2*intermediate] = [seq_len, gate||up]
//   silu_out:     [seq_len, intermediate]
__forceinline__ __device__
void stage_silu_mul(__nv_bfloat16* __restrict__ silu_out,
                    const __nv_bfloat16* __restrict__ gate_up,
                    int seq_len) {
    constexpr int kElemsPer16B = 8;
    const int tid = threadIdx.x;
    constexpr int kIntsPerRow = kIntermediate / kElemsPer16B;  // 704

    for (int row = blockIdx.x; row < seq_len; row += gridDim.x) {
        const __nv_bfloat16* gate = gate_up + row * kGateUpSize;
        const __nv_bfloat16* up   = gate + kIntermediate;
        __nv_bfloat16* out = silu_out + row * kIntermediate;
        for (int i = tid; i < kIntsPerRow; i += blockDim.x) {
            int4 g = *reinterpret_cast<const int4*>(gate + i * kElemsPer16B);
            int4 u = *reinterpret_cast<const int4*>(up   + i * kElemsPer16B);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&g);
            const __nv_bfloat16* up_p = reinterpret_cast<const __nv_bfloat16*>(&u);
            __nv_bfloat16 outb[kElemsPer16B];
            #pragma unroll
            for (int e = 0; e < kElemsPer16B; ++e) {
                float fg = bf16_to_float(gp[e]);
                float fu = bf16_to_float(up_p[e]);
                float silu_fp32 = fg / (1.f + expf(-fg));
                // MATCH HF MLP: `silu(gate) * up` is two ops in PyTorch,
                // each producing a fresh BF16 tensor. Quantize silu before
                // the multiply.
                float silu_bf16 = bf16_to_float(float_to_bf16(silu_fp32));
                outb[e] = float_to_bf16(silu_bf16 * fu);
            }
            *reinterpret_cast<int4*>(out + i * kElemsPer16B) =
                *reinterpret_cast<const int4*>(outb);
        }
        __syncthreads();
    }
}

// ---- Stage: residual add -------------------------------------------------
//   residual[r, c] += x[r, c]
__forceinline__ __device__
void stage_residual_add(__nv_bfloat16* __restrict__ residual,
                        const __nv_bfloat16* __restrict__ x,
                        int seq_len, int dim) {
    constexpr int kElemsPer16B = 8;
    const int tid = threadIdx.x;

    int n_lines_per_row = dim / kElemsPer16B;
    int total = seq_len * n_lines_per_row;
    for (int i = blockIdx.x * blockDim.x + tid; i < total; i += gridDim.x * blockDim.x) {
        int row = i / n_lines_per_row;
        int col = (i % n_lines_per_row) * kElemsPer16B;
        int4 a = *reinterpret_cast<const int4*>(residual + row * dim + col);
        int4 b = *reinterpret_cast<const int4*>(x        + row * dim + col);
        const __nv_bfloat16* ap = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&b);
        __nv_bfloat16 outb[kElemsPer16B];
        #pragma unroll
        for (int e = 0; e < kElemsPer16B; ++e) {
            outb[e] = float_to_bf16(bf16_to_float(ap[e]) + bf16_to_float(bp[e]));
        }
        *reinterpret_cast<int4*>(residual + row * dim + col) =
            *reinterpret_cast<const int4*>(outb);
    }
}

}  // namespace lucebox
