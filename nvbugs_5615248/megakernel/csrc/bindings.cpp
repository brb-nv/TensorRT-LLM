/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PyTorch C++ bindings for the TinyLlama-1.1B prefill megakernel.
 *
 * Exposes:
 *   - `lucebox_tinyllama._C.prefill(input_ids, packed_weights, layer_offsets,
 *                                  embed_off, final_norm_off, lm_head_off,
 *                                  seq_len, layer_residual_dump=None) -> logits`
 *   - `lucebox_tinyllama._C.gemm_pipeline(A, B) -> C` (M3 standalone primitive)
 *
 * Optional `layer_residual_dump`: BF16 CUDA tensor of shape
 * (num_layers, seq_len, hidden). If provided, the kernel writes the
 * post-layer residual stream after each transformer block. Used for the
 * per-layer divide-and-conquer numerics harness.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <vector>

#include "common.cuh"

namespace lucebox {

// Forward declarations of the launchers (defined in .cu files).
void launch_megakernel(const MegakernelParams& params, cudaStream_t stream);
void launch_gemm_pipeline(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          __nv_bfloat16* C, int M, int N, int K,
                          cudaStream_t stream);

void launch_rms_norm_test(__nv_bfloat16* y, const __nv_bfloat16* x,
                          const __nv_bfloat16* weight,
                          int seq_len, int dim, cudaStream_t stream);
void launch_rope_test(__nv_bfloat16* qkv, int seq_len, cudaStream_t stream);
void launch_attention_test(__nv_bfloat16* attn_out, const __nv_bfloat16* qkv,
                           int seq_len, cudaStream_t stream);
void launch_silu_mul_test(__nv_bfloat16* out, const __nv_bfloat16* gate_up,
                          int seq_len, cudaStream_t stream);

}  // namespace lucebox

// ---- Torch op: prefill ----------------------------------------------------
torch::Tensor prefill_op(torch::Tensor input_ids,
                         torch::Tensor packed_weights,
                         torch::Tensor layer_offsets_tbl,
                         int64_t embed_off,
                         int64_t final_norm_off,
                         int64_t lm_head_off,
                         int64_t seq_len,
                         c10::optional<torch::Tensor> debug_dump) {
    TORCH_CHECK(input_ids.is_cuda(), "input_ids must be CUDA");
    TORCH_CHECK(input_ids.scalar_type() == torch::kInt32 ||
                input_ids.scalar_type() == torch::kInt64,
                "input_ids must be int32 or int64");
    TORCH_CHECK(packed_weights.is_cuda() && packed_weights.scalar_type() == torch::kBFloat16,
                "packed_weights must be CUDA BF16");
    TORCH_CHECK(layer_offsets_tbl.is_cuda() && layer_offsets_tbl.scalar_type() == torch::kInt32,
                "layer_offsets_tbl must be CUDA int32");
    TORCH_CHECK(seq_len == lucebox::kSeqLen,
                "seq_len mismatch: built with MEGAKERNEL_SEQ_LEN=", lucebox::kSeqLen,
                " but caller passed ", seq_len);

    const at::cuda::CUDAGuard guard(input_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Make sure ids are int32 for the kernel.
    torch::Tensor ids32 = input_ids.scalar_type() == torch::kInt32
        ? input_ids.contiguous()
        : input_ids.to(torch::kInt32).contiguous();

    auto opts = packed_weights.options();
    auto logits = torch::empty({seq_len, lucebox::kVocab}, opts);

    // Scratch workspaces. We allocate fresh each call to keep the binding simple;
    // a production wrapper would cache these on a per-stream pool. All buffers
    // are BF16 to match HF Llama's eager precision profile exactly (HF's
    // decoder layer keeps the residual stream in BF16).
    auto act_a       = torch::empty({seq_len, lucebox::kHidden},        opts);
    auto act_b       = torch::empty({seq_len, lucebox::kHidden},        opts);
    auto residual    = torch::empty({seq_len, lucebox::kHidden},        opts);
    auto qkv_buf     = torch::empty({seq_len, lucebox::kQkvSize},       opts);
    auto attn_buf    = torch::empty({seq_len, lucebox::kHidden},        opts);
    auto gate_up_buf = torch::empty({seq_len, lucebox::kGateUpSize},    opts);
    auto silu_buf    = torch::empty({seq_len, lucebox::kIntermediate},  opts);

    if (debug_dump.has_value()) {
        TORCH_CHECK(debug_dump->scalar_type() == torch::kBFloat16,
                    "layer_residual_dump must be bfloat16 (residual is BF16 in the kernel)");
        TORCH_CHECK(debug_dump->is_cuda(), "layer_residual_dump must be CUDA");
    }

    lucebox::MegakernelParams params{};
    params.input_ids        = ids32.data_ptr<int32_t>();
    params.weights          = reinterpret_cast<const __nv_bfloat16*>(packed_weights.data_ptr());
    params.layer_offsets    = reinterpret_cast<const lucebox::LayerOffsets*>(
                                 layer_offsets_tbl.data_ptr<int32_t>());
    params.embed_offset     = static_cast<int>(embed_off);
    params.final_norm_offset = static_cast<int>(final_norm_off);
    params.lm_head_offset   = static_cast<int>(lm_head_off);
    params.logits           = reinterpret_cast<__nv_bfloat16*>(logits.data_ptr());
    params.act_a            = reinterpret_cast<__nv_bfloat16*>(act_a.data_ptr());
    params.act_b            = reinterpret_cast<__nv_bfloat16*>(act_b.data_ptr());
    params.residual         = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
    params.qkv_buf          = reinterpret_cast<__nv_bfloat16*>(qkv_buf.data_ptr());
    params.attn_buf         = reinterpret_cast<__nv_bfloat16*>(attn_buf.data_ptr());
    params.gate_up_buf      = reinterpret_cast<__nv_bfloat16*>(gate_up_buf.data_ptr());
    params.silu_buf         = reinterpret_cast<__nv_bfloat16*>(silu_buf.data_ptr());
    params.layer_residual_dump = debug_dump.has_value()
        ? reinterpret_cast<__nv_bfloat16*>(debug_dump->data_ptr()) : nullptr;
    params.seq_len          = static_cast<int>(seq_len);

    lucebox::launch_megakernel(params, stream);
    return logits;
}

// ---- Torch op: standalone GEMM primitive (M3 deliverable) -----------------
torch::Tensor gemm_pipeline_op(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2-D");
    int M = A.size(0), K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");
    int N = B.size(1);

    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto C = torch::empty({M, N}, A.options());
    lucebox::launch_gemm_pipeline(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K, stream);
    return C;
}

// ---- Torch ops: standalone per-stage primitives (M5+ module-level tests) --
// These wrap the same device-side `stage_*` functions the megakernel inlines,
// exposed via dedicated `__global__` entry points in `stage_tests.cu`. They
// have ZERO impact on the megakernel path -- they exist so each stage can be
// unit-tested against HuggingFace's `LlamaRMSNorm` / `apply_rotary_pos_emb` /
// `eager_attention_forward` / `F.silu(gate) * up` in isolation.

torch::Tensor rms_norm_op(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x,weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (seq_len, dim)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight shape mismatch with x last-dim");
    const at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto y = torch::empty_like(x);
    lucebox::launch_rms_norm_test(
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), stream);
    return y;
}

torch::Tensor rope_op(torch::Tensor qkv) {
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16,
                "qkv must be CUDA BF16");
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == lucebox::kQkvSize,
                "qkv must be (seq_len, kQkvSize=", lucebox::kQkvSize, ")");
    const at::cuda::CUDAGuard guard(qkv.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = qkv.clone();  // RoPE is in-place; return a fresh tensor
    lucebox::launch_rope_test(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(qkv.size(0)), stream);
    return out;
}

torch::Tensor attention_op(torch::Tensor qkv) {
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16,
                "qkv must be CUDA BF16");
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == lucebox::kQkvSize,
                "qkv must be (seq_len, kQkvSize=", lucebox::kQkvSize, ")");
    const at::cuda::CUDAGuard guard(qkv.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int seq_len = static_cast<int>(qkv.size(0));
    auto attn_out = torch::empty({seq_len, lucebox::kHidden}, qkv.options());
    lucebox::launch_attention_test(
        reinterpret_cast<__nv_bfloat16*>(attn_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        seq_len, stream);
    return attn_out;
}

torch::Tensor silu_mul_op(torch::Tensor gate_up) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kBFloat16,
                "gate_up must be CUDA BF16");
    TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(1) == lucebox::kGateUpSize,
                "gate_up must be (seq_len, kGateUpSize=", lucebox::kGateUpSize, ")");
    const at::cuda::CUDAGuard guard(gate_up.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int seq_len = static_cast<int>(gate_up.size(0));
    auto out = torch::empty({seq_len, lucebox::kIntermediate}, gate_up.options());
    lucebox::launch_silu_mul_test(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
        seq_len, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &prefill_op,
          "TinyLlama-1.1B prefill megakernel (one persistent CUDA dispatch).",
          py::arg("input_ids"),
          py::arg("packed_weights"),
          py::arg("layer_offsets_tbl"),
          py::arg("embed_off"),
          py::arg("final_norm_off"),
          py::arg("lm_head_off"),
          py::arg("seq_len"),
          py::arg("debug_dump") = c10::nullopt);
    m.def("gemm_pipeline", &gemm_pipeline_op,
          "Standalone cp.async + mma.sync BF16 GEMM primitive (for unit-testing).");
    m.def("rms_norm", &rms_norm_op,
          "Standalone RMSNorm stage (HF LlamaRMSNorm BF16-cast pattern).");
    m.def("rope", &rope_op,
          "Standalone RoPE stage; takes (seq, kQkvSize) BF16 buffer and "
          "rotates Q,K (leaves V untouched). Returns a new tensor.");
    m.def("attention", &attention_op,
          "Standalone prefill attention stage (causal, GQA 32:4); input is "
          "(seq, kQkvSize) packed [Q | K | V] BF16, output is (seq, kHidden).");
    m.def("silu_mul", &silu_mul_op,
          "Standalone SwiGLU stage: silu(gate) * up where the input is "
          "(seq, 2*intermediate) [gate || up] BF16.");
}
