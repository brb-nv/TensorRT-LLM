# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Debug helper: dump + diff first-MoE-layer tensors for MiniMax-M3 FP8 vs FP4.

Workflow
--------
1) Run one generation (1 prefill + 1 decode) per checkpoint, dumping the
   first MoE layer's tensors (input, router logits, routed/shared/final
   outputs). The model-side instrumentation lives in
   ``tensorrt_llm/_torch/models/modeling_minimaxm3.py`` and activates via the
   ``TLLM_MINIMAX_MOE_DUMP_DIR`` env var (set here automatically).

     python scripts/debug_minimax_moe_dump.py run \
         --model /path/to/MiniMax-M3-MXFP8 --dump-dir m3nvfp4/fp8

     python scripts/debug_minimax_moe_dump.py run \
         --model /path/to/Minimax-M3-NVFP4 --dump-dir m3nvfp4/fp4 \
         --moe-backend CUTLASS

2) Diff the two dumps to localize where FP4 deviates from FP8:

     python scripts/debug_minimax_moe_dump.py compare \
         --fp8-dir m3nvfp4/fp8 --fp4-dir m3nvfp4/fp4

Notes
-----
* CUDA graphs are disabled so decode runs eagerly and the dump hooks execute.
* Files are overwritten per (rank, phase, tensor) so the final post-warmup
  prefill/decode wins -- no explicit warmup detection needed.
* The PREFILL comparison is the clean control: layers 0-2 are dense MXFP8 in
  BOTH checkpoints, so the layer-3 MoE *input* should match closely; a large
  divergence in ``moe_input`` would instead point upstream (attention/norm).
"""

import argparse
import glob
import os

_DEFAULT_PROMPT = ("The capital of France is Paris. The capital of Japan is")


def _run(args):
    # Must be set before the model is imported/loaded in the MPI workers.
    os.environ["TLLM_MINIMAX_MOE_DUMP_DIR"] = os.path.abspath(args.dump_dir)
    os.environ["TLLM_MINIMAX_MOE_DUMP_LAYER"] = str(args.layer)
    os.makedirs(args.dump_dir, exist_ok=True)

    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import (KvCacheConfig,
                                     MiniMaxM3SparseAttentionConfig, MoeConfig,
                                     SamplingParams)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6,
                                    enable_block_reuse=False)
    llm_kwargs = dict(
        tensor_parallel_size=args.tp_size,
        moe_expert_parallel_size=args.ep_size,
        kv_cache_config=kv_cache_config,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(),
        max_seq_len=4096,
        trust_remote_code=True,
        # Eager decode so the per-step dump hooks run (CUDA graphs would replay
        # a captured graph and skip the Python dump).
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
    )
    if args.moe_backend:
        llm_kwargs["moe_config"] = MoeConfig(backend=args.moe_backend)

    # max_tokens=2 => exactly one prefill (context) + one decode step.
    sampling = SamplingParams(max_tokens=2, temperature=0.0)

    with LLM(args.model, **llm_kwargs) as llm:
        out = llm.generate([_DEFAULT_PROMPT], sampling_params=sampling)
    print(f"[run] generated: {out[0].outputs[0].text!r}")
    dumped = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(args.dump_dir, "*.pt")))
    print(f"[run] dumped {len(dumped)} tensors to {args.dump_dir}:")
    for name in dumped:
        print(f"        {name}")


def _compare(args):
    import torch

    fp8_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.fp8_dir, "*.pt"))
    }
    fp4_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.fp4_dir, "*.pt"))
    }
    common = sorted(set(fp8_files) & set(fp4_files))
    if not common:
        raise SystemExit(
            f"No common dump files between {args.fp8_dir} and {args.fp4_dir}.")

    # Stable, informative ordering: layer, rank, phase, then pipeline order.
    stage_order = {
        "layer_in": 0,
        "attn_out": 1,
        "block_out": 2,
        "moe_input": 3,
        "router_logits": 4,
        "shared_output": 5,
        "routed_output": 6,
        "result_pre_allreduce": 7,
        "result_final": 8,
    }

    def sort_key(fname: str):
        stem = fname[:-3]  # drop .pt
        parts = stem.split("_", 3)  # L{n}, R{r}, {phase}, {name}
        layer, rank, phase = parts[0], parts[1], parts[2]
        # Numeric layer sort (L0, L1, ... L10) rather than lexical.
        try:
            layer_num = int(layer[1:])
        except ValueError:
            layer_num = 999
        name = parts[3] if len(parts) > 3 else ""
        return (layer_num, rank, phase, stage_order.get(name, 99), name)

    hdr = f"{'file':<44}{'cosine':>10}{'rel_l2':>12}{'max_abs':>12}{'fp8_norm':>12}{'fp4_norm':>12}"
    print(hdr)
    print("-" * len(hdr))
    for fname in sorted(common, key=sort_key):
        a = fp8_files[fname]
        b = fp4_files[fname]
        ta = torch.load(a, map_location="cpu").float().flatten()
        tb = torch.load(b, map_location="cpu").float().flatten()
        n = min(ta.numel(), tb.numel())
        ta, tb = ta[:n], tb[:n]
        cos = torch.nn.functional.cosine_similarity(ta, tb, dim=0).item()
        na, nb = ta.norm().item(), tb.norm().item()
        rel = (ta - tb).norm().item() / (na + 1e-9)
        mx = (ta - tb).abs().max().item()
        print(f"{fname[:-3]:<44}{cos:>10.4f}{rel:>12.4f}{mx:>12.4g}"
              f"{na:>12.4g}{nb:>12.4g}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("run", help="Run one generation and dump MoE tensors.")
    pr.add_argument("--model", required=True, help="HF checkpoint dir.")
    pr.add_argument("--dump-dir", required=True, help="Output dir for dumps.")
    pr.add_argument("--moe-backend", default=None,
                    help="Force MoE backend (e.g. CUTLASS, TRTLLM). Default: AUTO.")
    pr.add_argument("--tp-size", type=int, default=4)
    pr.add_argument("--ep-size", type=int, default=4)
    pr.add_argument("--layer", type=int, default=3,
                    help="MoE layer index to dump (first MoE layer is 3).")

    pc = sub.add_parser("compare", help="Diff two dump dirs (FP8 vs FP4).")
    pc.add_argument("--fp8-dir", required=True)
    pc.add_argument("--fp4-dir", required=True)

    args = p.parse_args()
    if args.mode == "run":
        _run(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
