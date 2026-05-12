/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * `cp.async` + `mma.sync` primitives for sm_80+ (Ampere / Ada). We do NOT use
 * Hopper-specific paths (TMA, wgmma, mbarrier-as-tensor-async) because L40S is
 * sm_89 and lacks them. All bulk loads go through 16-byte `cp.async.cg`
 * (cache-global, bypasses L1) and are pipelined with `commit_group` /
 * `wait_group`.
 *
 * References:
 *   PTX ISA 8.4, section 8.7.6 (cp.async), section 9.7.13 (mma).
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace lucebox {

// ---- shared-memory pointer -> generic 32-bit address ----------------------
__forceinline__ __device__ uint32_t cvt_to_smem(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64a; cvta.to.shared.u64 u64a, %1; cvt.u32.u64 %0, u64a; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

// 16-byte vector cp.async, cache-global (skips L1). Fails silently if the
// source is out-of-bounds and the predicate is false; otherwise issues a load
// of 16 bytes (`int4`) from gmem into smem. Caller must ensure 16-byte alignment.
__forceinline__ __device__ void cp_async_16B(uint32_t smem_addr, const void* gmem_ptr,
                                             bool pred = true) {
    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(int(pred)));
}

__forceinline__ __device__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__forceinline__ __device__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__forceinline__ __device__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

// ---- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ------------------
//
// One warp produces a 16x8 FP32 accumulator tile from a 16x16 BF16 A fragment
// and a 16x8 BF16 B fragment. Operand layouts (PTX manual, table 130):
//   A: row-major, 16x16 BF16: each thread holds 4x BF16 in 2x .b32 regs
//   B: col-major, 16x8 BF16:  each thread holds 2x BF16 in 1x .b32 reg
//   C, D: 16x8 FP32:          each thread holds 4x FP32 in 4x .f32 regs
//
// Fragment register layout per the PTX manual (warp = 32 threads, m16n8k16):
//   A laid out as 16 rows x 16 cols. Each thread owns 8 elements; mapping is
//   row = (thread/4)*8 + (i/4)*8 ... see the PTX manual figure.
// We expose just the inline-asm wrapper; the calling code is responsible for
// arranging the fragments into the right registers via `ldmatrix` or manual loads.
__forceinline__ __device__
void mma_m16n8k16_bf16(float& d0, float& d1, float& d2, float& d3,
                       uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                       uint32_t b0, uint32_t b1,
                       float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0,%1,%2,%3},\n"
        "    {%4,%5,%6,%7},\n"
        "    {%8,%9},\n"
        "    {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// ldmatrix.sync.aligned.m8n8.x4.shared.b16: load a 16x16 BF16 matrix from smem
// into 4x BF16x2 registers per thread (one warp = 32 threads), with the rows of
// the matrix coming from 8 contiguous threads' base addresses (lane%8 supplies
// the row pointer; the warp covers 8 unique base rows, repeated 4 times for x4).
__forceinline__ __device__
void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                 uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

// Transposed variant for B (col-major load).
__forceinline__ __device__
void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                       uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

__forceinline__ __device__
void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%4];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

}  // namespace lucebox
