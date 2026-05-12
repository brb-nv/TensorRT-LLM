/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::CublasMMWrapper;

// Mirrors cublas_gemm_caller's BF16 setup in cublasScaledMM.cpp so the timing here
// reflects exactly what production traffic would see (CUBLAS_OP_T,N with swapped
// A/B and the same workspace size).
struct BestAlgo
{
    int algoId = -1;
    int tileId = 0;
    int stagesId = 0;
    int splitK = 1;
    int reduction = 0;
    int swizzle = 0;
    int customOption = 0;
    int cga = 0;
    float bestUs = std::numeric_limits<float>::infinity();
    float heuristicUs = std::numeric_limits<float>::infinity();
    int candidatesProfiled = 0;
    int candidatesReturned = 0;
};

template <typename CfgT>
CfgT readAlgoConfig(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoConfigAttributes_t cfg)
{
    CfgT v{};
    size_t written = 0;
    auto status = cublasLtMatmulAlgoConfigGetAttribute(&algo, cfg, &v, sizeof(CfgT), &written);
    // Older cuBLASLt builds may not expose every config attribute; fall back to 0
    // rather than abort the whole tuning run.
    if (status != CUBLAS_STATUS_SUCCESS || written == 0)
    {
        return CfgT{};
    }
    return v;
}

float timeAlgoOnce(CublasMMWrapper& wrapper, int m, int n, int k, void const* a_ptr, void const* b_ptr, void* out_ptr,
    cublasLtMatmulAlgo_t const& algo, int warmup, int iters, cudaStream_t stream)
{
    // The candidates we receive from cublasLtMatmulAlgoGetHeuristic have
    // already been validated for the descriptors we created; any further
    // checkTactic() call would just emit a redundant warning. The wrapper's
    // Gemm() does its own check anyway and falls back to NULL algo on
    // failure (which would just retime the heuristic-picked path).
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        wrapper.Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*A=*/b_ptr, /*lda=*/k, /*B=*/a_ptr, /*ldb=*/k, out_ptr,
            /*ldc=*/n, 1.0F, 0.0F, algo, /*hasAlgo=*/true, /*usingCublasLt=*/true);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i)
    {
        wrapper.Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*A=*/b_ptr, /*lda=*/k, /*B=*/a_ptr, /*ldb=*/k, out_ptr,
            /*ldc=*/n, 1.0F, 0.0F, algo, /*hasAlgo=*/true, /*usingCublasLt=*/true);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (ms * 1000.f) / static_cast<float>(iters); // microseconds per call
}

float timeHeuristicOnce(CublasMMWrapper& wrapper, int m, int n, int k, void const* a_ptr, void const* b_ptr,
    void* out_ptr, int warmup, int iters, cudaStream_t stream)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasLtMatmulAlgo_t dummy{};
    for (int i = 0; i < warmup; ++i)
    {
        wrapper.Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*A=*/b_ptr, /*lda=*/k, /*B=*/a_ptr, /*ldb=*/k, out_ptr,
            /*ldc=*/n, 1.0F, 0.0F, dummy, /*hasAlgo=*/false, /*usingCublasLt=*/true);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i)
    {
        wrapper.Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*A=*/b_ptr, /*lda=*/k, /*B=*/a_ptr, /*ldb=*/k, out_ptr,
            /*ldc=*/n, 1.0F, 0.0F, dummy, /*hasAlgo=*/false, /*usingCublasLt=*/true);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (ms * 1000.f) / static_cast<float>(iters);
}

} // namespace

// Tunes a BF16 row-major GEMM C[m,n] = A[m,k] @ B[k,n] using cuBLASLt heuristic
// candidates and returns the best (algo_id, tile, stages, splitK, reduction,
// swizzle, custom, cga) tuple along with a few sanity numbers.
//
// The descriptor convention matches `cublas_gemm_caller` in
// cpp/tensorrt_llm/thop/cublasScaledMM.cpp: we swap A/B and call cuBLASLt as
// (CUBLAS_OP_T, CUBLAS_OP_N, n, m, k) so the chosen attribute tuple drops
// straight into bf16_algo_list keyed on {nextPowerOfTwo(m), k, n}.
//
// Returns a 1-D int64 tensor of length 11:
//   [algo_id, tile_id, stages_id, splitK, reduction, swizzle, custom, cga,
//    candidates_returned, candidates_profiled, mp2]
// And a 1-D float64 tensor of length 2:
//   [best_us_per_call, heuristic_us_per_call]
std::tuple<Tensor, Tensor> tune_bf16_gemm(int64_t m_in, int64_t n_in, int64_t k_in, int64_t warmup_iters_in,
    int64_t timing_iters_in, int64_t max_candidates_in)
{
    int const m = static_cast<int>(m_in);
    int const n = static_cast<int>(n_in);
    int const k = static_cast<int>(k_in);
    int const warmupIters = std::max<int64_t>(1, warmup_iters_in);
    int const timingIters = std::max<int64_t>(1, timing_iters_in);
    int const maxCandidates = std::max<int64_t>(1, max_candidates_in);

    TORCH_CHECK(m > 0 && n > 0 && k > 0, "tune_bf16_gemm: m,n,k must be positive");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

    // Allocate row-major A[m,k], B[k,n], C[m,n] of dummy data. cublasLt sees these
    // through the descriptor we set up below, exactly as the production GEMM does.
    auto a = torch::randn({m, k}, opts_bf16);
    auto b = torch::randn({k, n}, opts_bf16);
    auto c = torch::empty({m, n}, opts_bf16);
    auto workspace = torch::empty({CUBLAS_WORKSPACE_SIZE}, opts_u8);

    // getCublasHandle / getCublasLtHandle are declared at tensorrt_llm
    // namespace scope in opUtils.h (after the common::op block closes), so
    // unqualified lookup from tensorrt_llm::torch_ext finds them.
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    auto wrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    wrapper->setStream(stream);
    wrapper->setWorkspace(workspace.data_ptr());
    wrapper->setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F);

    // createDescriptors uses the same (CUBLAS_OP_T, CUBLAS_OP_N, n, m, k) swap
    // as cublas_gemm_caller so the algos we enumerate here are valid for the
    // production code path.
    wrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*lda=*/k, /*ldb=*/k, /*ldc=*/n, /*fastAcc=*/0);

    auto heuristics = wrapper->getTactics(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*lda=*/k, /*ldb=*/k, /*ldc=*/n);

    BestAlgo best;
    best.candidatesReturned = static_cast<int>(heuristics.size());

    // Heuristic baseline (cuBLASLt picks for itself).
    best.heuristicUs = timeHeuristicOnce(*wrapper, m, n, k, a.data_ptr(), b.data_ptr(), c.data_ptr(), warmupIters,
        timingIters, static_cast<cudaStream_t>(stream));

    int const profileLimit = std::min<int>(static_cast<int>(heuristics.size()), maxCandidates);
    for (int idx = 0; idx < profileLimit; ++idx)
    {
        auto const& heur = heuristics[idx];
        if (heur.state != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }
        if (heur.workspaceSize > CUBLAS_WORKSPACE_SIZE)
        {
            continue;
        }

        float const us = timeAlgoOnce(*wrapper, m, n, k, a.data_ptr(), b.data_ptr(), c.data_ptr(), heur.algo,
            warmupIters, timingIters, static_cast<cudaStream_t>(stream));
        if (!std::isfinite(us))
        {
            continue;
        }
        ++best.candidatesProfiled;
        if (us < best.bestUs)
        {
            best.bestUs = us;
            best.algoId = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_ID);
            best.tileId = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            best.stagesId = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            best.splitK = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            best.reduction = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
            best.swizzle = readAlgoConfig<int>(heur.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
            best.customOption = static_cast<int>(readAlgoConfig<uint32_t>(heur.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION));
            best.cga = static_cast<int>(readAlgoConfig<uint16_t>(heur.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID));
        }
    }

    wrapper->destroyDescriptors();

    int const mp2 = std::max(nextPowerOfTwo(m), 8);

    auto int_out = torch::empty({11}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto* p = int_out.data_ptr<int64_t>();
    p[0] = best.algoId;
    p[1] = best.tileId;
    p[2] = best.stagesId;
    p[3] = best.splitK;
    p[4] = best.reduction;
    p[5] = best.swizzle;
    p[6] = best.customOption;
    p[7] = best.cga;
    p[8] = best.candidatesReturned;
    p[9] = best.candidatesProfiled;
    p[10] = mp2;

    auto float_out = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto* f = float_out.data_ptr<double>();
    f[0] = static_cast<double>(best.bestUs);
    f[1] = static_cast<double>(best.heuristicUs);

    return std::make_tuple(int_out, float_out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "tune_bf16_gemm(int m, int n, int k, int warmup_iters, int timing_iters, int max_candidates) -> (Tensor, "
        "Tensor)");
}

// Register under CompositeExplicitAutograd rather than CUDA: the schema
// has no tensor inputs (only ints), so the BackendSelect kernel cannot
// pick a backend from the args, and a CUDA-only registration produces
// "no tensor arguments to this function ... no fallback function is
// registered" at call time. The kernel allocates its own CUDA tensors
// internally, so it works correctly regardless of dispatch key.
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("tune_bf16_gemm", &tensorrt_llm::torch_ext::tune_bf16_gemm);
}
