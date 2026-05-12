# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyLlama-1.1B prefill latency bench on L40S (NVBug 5615248).

Measures, all at batch=1, seqlen=128, BF16:
    a) HuggingFace eager forward
    b) HuggingFace `torch.compile(mode='reduce-overhead')`
    c) The lucebox megakernel (`lucebox_tinyllama.prefill`)

Comparison reference TRT engine numbers come from existing trtllm-bench runs
under `nvbugs_5615248/trtllm_bench/` -- run those separately, copy the
`enqueueV3 mean` (currently 2.31 ms) into the printed summary by hand.

Methodology:
  - locked GPU clocks via `nvidia-smi -lgc` recommended before running
  - 100 warmup iters, 1000 timed iters, CUDA-event timing per iter
  - mean / p50 / p99 reported

Model weights: the script defaults to the on-disk HF checkpoint at
    `/home/scratch.trt_llm_data_ci/llm-models/TinyLlama-1.1B-Chat-v1.0`
which is available on the standard TRT-LLM compute nodes. Override with
    --model-id <hf_id_or_local_path>
or set --use-synthetic to skip weight loading for megakernel-only timing.

Usage:
    sudo nvidia-smi -lgc 2520,2520     # L40S boost (optional)
    python tests/bench.py --backend megakernel
    python tests/bench.py --backend hf_eager
    python tests/bench.py --backend hf_compile
    python tests/bench.py --backend all
"""
import argparse
import os
import sys
import time
from statistics import mean

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama import _C_AVAILABLE, TINYLLAMA_CONFIG, pack_weights
from lucebox_tinyllama.reference import load_hf_tinyllama_weights

if _C_AVAILABLE:
    from lucebox_tinyllama import prefill


# Standard on-disk location for the TinyLlama HF checkpoint on TRT-LLM compute
# nodes. Falls back to HuggingFace Hub if --model-id points elsewhere.
DEFAULT_TINYLLAMA_PATH = (
    "/home/scratch.trt_llm_data_ci/llm-models/TinyLlama-1.1B-Chat-v1.0"
)


def time_callable(fn, warmup: int, iters: int) -> dict:
    """Time a no-arg callable using CUDA events. Returns latency stats in ms."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "mean":  mean(times),
        "p50":   times[len(times) // 2],
        "p90":   times[int(0.90 * len(times))],
        "p99":   times[int(0.99 * len(times))],
        "min":   times[0],
        "max":   times[-1],
        "n":     iters,
    }


def _layer_offsets_tensor(pw):
    flat = []
    for off in pw.layer_offsets:
        flat.extend([off.input_layernorm, off.qkv_proj, off.o_proj,
                     off.post_attn_norm, off.gate_up, off.down])
    return torch.tensor(flat, dtype=torch.int32, device=pw.blob.device)


def bench_megakernel(seq_len: int, warmup: int, iters: int,
                     model_id: str | None = None,
                     synthetic: bool = False) -> dict:
    cfg = TINYLLAMA_CONFIG
    if synthetic:
        # Random weights = same compute pattern, no HF dependency for perf-only runs.
        torch.manual_seed(0)
        sd = {}
        sd["model.embed_tokens.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size,
                                                       dtype=torch.bfloat16) * 0.02
        sd["model.norm.weight"] = torch.ones(cfg.hidden_size, dtype=torch.bfloat16)
        sd["lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size,
                                             dtype=torch.bfloat16) * 0.02
        for L in range(cfg.num_layers):
            p = f"model.layers.{L}"
            sd[f"{p}.input_layernorm.weight"] = torch.ones(cfg.hidden_size, dtype=torch.bfloat16)
            sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(cfg.q_size, cfg.hidden_size,
                                                              dtype=torch.bfloat16) * 0.02
            sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(cfg.kv_size, cfg.hidden_size,
                                                              dtype=torch.bfloat16) * 0.02
            sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(cfg.kv_size, cfg.hidden_size,
                                                              dtype=torch.bfloat16) * 0.02
            sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(cfg.hidden_size, cfg.q_size,
                                                              dtype=torch.bfloat16) * 0.02
            sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(cfg.hidden_size,
                                                                       dtype=torch.bfloat16)
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                           dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size,
                                                         dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(cfg.hidden_size, cfg.intermediate_size,
                                                           dtype=torch.bfloat16) * 0.02
    else:
        sd = load_hf_tinyllama_weights(model_id or DEFAULT_TINYLLAMA_PATH)
    pw = pack_weights(sd, device="cuda")
    offs = _layer_offsets_tensor(pw)

    ids = torch.randint(0, cfg.vocab_size, (seq_len,), dtype=torch.int32, device="cuda")

    def run():
        return prefill(ids, pw.blob, offs,
                       pw.embed_tokens_offset, pw.final_norm_offset, pw.lm_head_offset,
                       seq_len)

    return time_callable(run, warmup, iters)


def bench_hf(seq_len: int, warmup: int, iters: int, compile_mode: str | None,
             model_id: str = DEFAULT_TINYLLAMA_PATH) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa").cuda().eval()
    ids = torch.randint(0, hf.config.vocab_size, (1, seq_len),
                        dtype=torch.long, device="cuda")
    if compile_mode is not None:
        hf = torch.compile(hf, mode=compile_mode, fullgraph=False)

    @torch.no_grad()
    def run():
        return hf(ids).logits

    return time_callable(run, warmup, iters)


def fmt(label: str, stats: dict) -> str:
    return (f"{label:30s} mean={stats['mean']:7.3f}  p50={stats['p50']:7.3f}  "
            f"p90={stats['p90']:7.3f}  p99={stats['p99']:7.3f}  "
            f"min={stats['min']:7.3f}  max={stats['max']:7.3f}  n={stats['n']}  (ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["megakernel", "hf_eager", "hf_compile", "all"],
                    default="all")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--model-id", default=DEFAULT_TINYLLAMA_PATH,
                    help="HF model id or local path. Defaults to the on-disk "
                         "checkpoint at /home/scratch.trt_llm_data_ci/llm-models/...")
    ap.add_argument("--use-synthetic", action="store_true",
                    help="Use synthetic random weights for the megakernel run "
                         "(skips disk/Hub load). Default is to load real weights "
                         "from --model-id.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs CUDA")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  sm_{props.major}{props.minor}  "
          f"{props.multi_processor_count} SMs  "
          f"{props.total_memory / 1e9:.1f} GB")
    print(f"workload: seq_len={args.seq_len}  warmup={args.warmup}  iters={args.iters}")
    print("-" * 80)

    results = []
    if args.backend in ("megakernel", "all"):
        try:
            stats = bench_megakernel(args.seq_len, args.warmup, args.iters,
                                     args.model_id, synthetic=args.use_synthetic)
            label = ("lucebox megakernel (synth)" if args.use_synthetic
                     else "lucebox megakernel (HF wts)")
            print(fmt(label, stats))
            results.append(("megakernel", stats))
        except Exception as exc:
            print(f"lucebox megakernel: FAILED ({exc})")
    if args.backend in ("hf_eager", "all"):
        try:
            stats = bench_hf(args.seq_len, args.warmup, args.iters, None, args.model_id)
            print(fmt("HF eager", stats))
            results.append(("hf_eager", stats))
        except Exception as exc:
            print(f"HF eager: FAILED ({exc})")
    if args.backend in ("hf_compile", "all"):
        try:
            stats = bench_hf(args.seq_len, args.warmup, args.iters,
                             "reduce-overhead", args.model_id)
            print(fmt("HF torch.compile(red-ovh)", stats))
            results.append(("hf_compile", stats))
        except Exception as exc:
            print(f"HF compile: FAILED ({exc})")

    print("-" * 80)
    print("Reference (from POST_V5_DIVERGENCE.md, L40S, beam=10 ISL=100):")
    print("  PyT _forward_step (ctx):   5.910 ms   <-- the gap we want to close")
    print("  TRT enqueueV3:             2.310 ms   <-- the target to beat for v1")
    print("  Megakernel goal (plan):    < 2.000 ms")
    print()
    if results:
        for name, s in results:
            print(f"  -> {name:20s} {s['mean']:.3f} ms")


if __name__ == "__main__":
    main()
