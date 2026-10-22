# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage self-test for msa-prewarm (one GPU, < 100 MB).

  python3 -m tensorrt_llm._torch.attention.backends.sparse.minimax_m3.m3ax_msa_prewarm.selftest \
      [--out FILE] [--cache-geometry tp2|tp4|adp|none] [--qmax 300]

For each geometry (TP2: 32/2 heads, 2 index heads; TP4: 16/1, 1; attention DP: 64/4, 4) the test builds real
fmha_sm100 plans the way msa_backend._build_step_plans does (dense plan; whole-row proxy plan; segmented KV-split proxy
plan via msa_indexer.segmented_proxy_plan) for every single-row q in 1..qmax on a short and a long cached prefix and for
mixed multi-row batches (short + long rows in both orders, context + 4-token generation rows), then runs fmha_sm100 with
fmha_sm100.api.get_fmha_variant replaced by a recorder that returns a no-op runner: no kernel runs and nothing
compiles (FMHAVariantManager._compile_only raises if reached). Every recorded key must be in core.enumerate_variants for
that geometry. --cache-geometry: the geometry this container prewarmed; every key recorded for it must also have its
.so in the minfer cache (what serving would dlopen). Exit 0 iff nothing is missing. Prints one JSON summary.
When o12a is on, segmented_proxy_plan above is o12a's (balanced plans where its policy takes them), and
o12a's balanced plans are also built forced for every case, on 4 CTAs (tiles cut into pieces) and on every SM, so
the o12a-split family is checked the same way.
"""
import argparse
import importlib
import json
import math
import os
import sys

GEOMS = {"tp2": (32, 2, 2), "tp4": (16, 1, 1), "adp": (64, 4, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--cache-geometry", default="tp2", choices=list(GEOMS) + ["none"])
    ap.add_argument("--qmax", type=int, default=300)
    a = ap.parse_args()

    import torch
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
    from . import core

    fmha, jit, api = core._modules()
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    page, hd = 128, 128
    fp8 = torch.float8_e4m3fn
    code = jit._dlpack_dtype_code(fp8)
    plan_so = jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so"
    plan_present = plan_so.exists()

    rec = []

    class _Noop(object):
        def run(self, *args):
            return None

    def recorder(*args, **kwargs):
        rec.append(jit._variant_key_from_runtime(*args)[0])
        return _Noop()

    def no_compile(self, name, params):
        raise RuntimeError("selftest: variant %s would compile" % name)

    orig_get, orig_compile = api.get_fmha_variant, type(jit._variant_manager)._compile_only
    api.get_fmha_variant = recorder
    type(jit._variant_manager)._compile_only = no_compile

    def rows_tensors(qs, prefix):
        q = torch.tensor(qs, dtype=torch.int32)
        p = torch.tensor([prefix] * len(qs), dtype=torch.int32)
        kv = q + p
        pages = [int(math.ceil(int(x) / page)) for x in kv.tolist()]
        kv_idx = torch.arange(sum(pages), dtype=torch.int32, device=dev)
        return q, kv, p, kv_idx, sum(pages)

    bal = forced = None
    num_sms = 0
    if core.o12a_enabled():
        bal = importlib.import_module("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_proxy_balance")
        forced = bal.config()._replace(force=True)
        o12a_api = bal.library_api(fmha)
        if o12a_api is None:
            bal = None
        else:
            num_sms = int(o12a_api._get_num_cta(dev))
    results = {}
    errors = []
    try:
        for gname, (hq, hkv, hidx) in GEOMS.items():
            enum = core.enumerate_variants(jit, api, core.families_for(hq, hkv, hidx, code, code, page))
            rec.clear()
            o12a_plans = 0
            cases = [([q], pre) for pre in (0, 5120) for q in range(1, a.qmax + 1)]
            for short in (1, 3, 6, 12, 24, 48, 100):
                for long_ in (200, 2000):
                    cases += [([short, long_], 5120), ([long_, short], 5120), ([short, short + 3, long_], 5120)]
            cases += [([q, 4, 4, 4], 5120) for q in (2, 7, 15, 30, 60, 90, 300)]
            for qs, pre in cases:
                qcpu, kvcpu, pcpu, kv_idx, npages = rows_tensors(qs, pre)
                tot = int(qcpu.sum())
                try:
                    # dense layers: plan_for(attn_rows, num_qo_heads, num_kv_heads, use_fp8_kvcache)
                    plan = fmha.fmha_sm100_plan(qcpu, kvcpu, qo_offset=pcpu, page_size=page, num_kv_splits=1,
                                                causal=True, num_qo_heads=hq, num_kv_heads=hkv, use_fp8_kvcache=True)
                    qt = torch.zeros(tot, hq, hd, dtype=fp8, device=dev)
                    kt = torch.zeros(npages, hkv, page, hd, dtype=fp8, device=dev)
                    out = torch.empty(tot, hq, hd, dtype=torch.bfloat16, device=dev)
                    fmha.fmha_sm100(qt, kt, kt, plan, kv_indices=kv_idx, kv_block_indexes=None, out=out,
                                    sm_scale=hd ** -0.5, output_maxscore=False)
                    # proxy: plan_for(proxy_rows, num_qo_heads=index heads, num_kv_heads=1, output_maxscore=True)
                    iq = torch.zeros(tot, hidx, hd, dtype=fp8, device=dev)
                    ik = torch.zeros(npages, 1, page, hd, dtype=fp8, device=dev)
                    plan = fmha.fmha_sm100_plan(qcpu, kvcpu, qo_offset=pcpu, page_size=page, num_kv_splits=1,
                                                causal=True, num_qo_heads=hidx, num_kv_heads=1, output_maxscore=True)
                    fmha.fmha_sm100(iq, ik, ik, plan, kv_indices=kv_idx, output_o=False, output_maxscore=True,
                                    sm_scale=hd ** -0.5)
                    # segmented KV-split proxy plan (None unless prefix >= 4096 and some row > 64)
                    seg = msa_indexer.segmented_proxy_plan(fmha, qcpu, pcpu, num_index_heads=hidx, page_size=page,
                                                           kv_indices=kv_idx)
                    if seg is not None:
                        splan, skv = seg
                        sik = torch.zeros(int(skv.numel()), 1, page, hd, dtype=fp8, device=dev)
                        fmha.fmha_sm100(iq, sik, sik, splan, kv_indices=skv, output_o=False, output_maxscore=True,
                                        sm_scale=hd ** -0.5)
                    # o12a's balanced proxy plans, forced, on 4 CTAs and on every SM
                    if bal is not None:
                        for nctas in (4, num_sms):
                            dec = bal.decide([int(x) for x in qcpu.tolist()], [int(x) for x in pcpu.tolist()],
                                             num_index_heads=hidx, page_size=page, num_ctas=nctas,
                                             stock_splits=0, cfg=forced)
                            built = bal.build_plan(fmha, dec, num_index_heads=hidx, kv_indices=kv_idx,
                                                   num_ctas=nctas)
                            if built is None:
                                continue
                            bplan, bkv = built
                            bik = torch.zeros(int(bkv.numel()), 1, page, hd, dtype=fp8, device=dev)
                            fmha.fmha_sm100(iq, bik, bik, bplan, kv_indices=bkv, output_o=False,
                                            output_maxscore=True, sm_scale=hd ** -0.5)
                            o12a_plans += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append("%s %s pre=%d: %s: %s" % (gname, qs, pre, type(exc).__name__, str(exc)[:200]))
            seen = sorted(set(rec))
            missing = sorted(set(seen) - set(enum))
            r = {"enumerated": sorted(enum), "recorded": seen, "missing": missing,
                 "unused": sorted(set(enum) - set(seen)), "lookups": len(rec), "cases": len(cases),
                 "o12a_forced_plans": o12a_plans}
            if a.cache_geometry == gname:
                r["cache_missing"] = sorted(n for n in seen if not core._so(jit, n).exists())
            results[gname] = r
    finally:
        api.get_fmha_variant, type(jit._variant_manager)._compile_only = orig_get, orig_compile

    ok = not errors and all(not r["missing"] and not r.get("cache_missing") for r in results.values())
    summary = {"ok": ok, "o12a": bal is not None, "plan_so_present_before": plan_present,
               "cache_base": str(jit.CACHE_BASE),
               "errors": errors[:20], "n_errors": len(errors), "geometries": results}
    txt = json.dumps(summary, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(json.dumps({"ok": ok, "n_errors": len(errors),
                      "per_geometry": {g: {"enumerated": len(r["enumerated"]), "recorded": len(r["recorded"]),
                                           "missing": r["missing"], "cache_missing": r.get("cache_missing")}
                                       for g, r in results.items()}}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
