# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msa-prewarm core: which fmha_sm100 variants a rank can select, compiling them at warmup, and a serve-time observer.

Variant key (fmha_sm100.jit): dtype_tile_singlewg_sparse_page_split_pack. The run path (fmha_sm100.api._fmha_sm100)
takes dtype from q, page from k.shape[2], sparse mode from which outputs exist, split from num_kv_splits > 1, and
tile / single-WG / pack factor from the plan: pf = _compute_pack_factor(max_q, Hq, Hkv), packed = max_q * pf,
tile = 128 if packed <= 128 else 256, single_wg = packed <= 64. Everything the key depends on besides max_q is fixed
per rank, so enumerating max_q = 1..Q_MAX with fmha_sm100's own _compute_pack_factor and _variant_key_from_runtime
gives every key a plan family can select (all q > 128 give the same tile-256 key).

Families (msa_backend._build_step_plans, msa_indexer):
  dense        main-attention plan of the 3 dense layers: q dtype = paged K/V dtype (FP8 for FP8 and NVFP4 caches,
               as _msa_main_kv_is_fp8 / run_msa_prefill_gqa), Hq/Hkv = params.sharded_head_counts, sparse Off,
               num_kv_splits 1
  proxy        indexer proxy plan: index dtype (indexer_kv_dtype), Hq = sharded_index_head_count, Hkv 1, OnlyScore,
               num_kv_splits 1 (whole-row plan and _proxy_max_score)
  proxy-split  segmented proxy plan (rows > 64 queries on a >= 4096 prefix): segments of 65..128 queries, KV split
               (the stock prewarm_split_proxy_variants covers these too; listed so the set is complete)
Not covered: the sparse GQA plan, which msa_backend builds only when the KV cache is not NVFP4 (logged if so).

o12a. When o12a's balanced proxy plans are on in this process (o12a_enabled()), families_for() adds
  o12a-split   balanced proxy plans (msa_proxy_balance.build_plan): segments of <= 64 queries are KV-split too,
               which selects the single-softmax-warpgroup split variants (and the packed one at 2+ index heads,
               <= 32 queries); TP2 gains 2 variants (14 in all), TP4 1. Segments of 65..128 queries select
               the proxy / proxy-split keys above. With M3_O12A_PROXY_SPLIT=0 (or M3AX_STACK_A=0) the set is
               msa-prewarm's exactly.

Segmented proxy. Where the backend plans the prefill indexer proxy segmented with planner-chosen split-KV
(msa_backend._proxy_segmented(), on by default on SM100/SM103), a step with any row over 64 queries splits every
segment of that step, short ones included, so the same KV-split variants as o12a-split are reachable;
families_for() adds them as proxy-autosplit (TP2: 2 variants). Without that backend function the set is unchanged.

Compile: one thread per missing variant (and the plan module), each holding that variant's own .compile.lock (the
same lock FMHAVariantManager.get_variant takes), running the library's _compile_only (ninja -j1 nvcc, the stock
command). Ranks cooperate through the locks. Nothing is loaded: the serving process later dlopens the same .so on
first use, exactly as in a seeded run, so GPU memory, dispatch, plans and kernels are stock.
"""
import importlib
import os
import threading
import time

Q_MAX = 512
TAG = "[M3AX msa-prewarm]"

_PHASE = ["init"]
_TLS = threading.local()
_REACH = {}  # variant name -> family label, union over the geometries prewarmed in this process
_OBS = {"installed": False, "seen": set(), "inline": 0}


def set_phase(phase):
    _PHASE[0] = phase
    if phase == "serve":
        _OBS["seen"] = set()


def log(msg, warn=False):
    try:
        from tensorrt_llm.logger import logger

        (logger.warning if warn else logger.info)("%s %s" % (TAG, msg))
    except Exception:  # noqa: BLE001
        print("%s %s" % (TAG, msg), flush=True)


def o12a_enabled():
    """Whether o12a's balanced proxy plans are on in this process (msa_proxy_balance is importable and its config is
    enabled: M3_O12A_PROXY_SPLIT and M3AX_STACK_A not 0)."""
    try:
        bal = importlib.import_module("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_proxy_balance")
        return bool(bal.config().enabled)
    except Exception:  # noqa: BLE001  (o12a not importable: msa-prewarm's families only)
        return False


def proxy_segmented_enabled():
    """Whether the backend plans the prefill indexer proxy segmented with split-KV on this GPU."""
    try:
        backend = importlib.import_module("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend")
        segmented = getattr(backend, "_proxy_segmented", None)
        return bool(segmented()) if segmented is not None else False
    except Exception:  # noqa: BLE001  (no segmented proxy in this backend: msa-prewarm's families only)
        return False


def stack_parts():
    """Every GB300 stack part's switch state in this process, for the log (prewarm is on, or this would not run)."""
    parts = [("master", int(os.environ.get("M3AX_STACK_A", "1").strip().lower() not in ("0", "false", "off", "no")))]
    try:
        pol = importlib.import_module(
            "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.sparse_decode_tp2_policy")
        parts.append(("o8", int(bool(pol.ENABLED))))
    except Exception:  # noqa: BLE001
        parts.append(("o8", "absent"))
    parts += [("o12a", int(o12a_enabled())), ("prewarm", 1)]
    try:
        from . import sdwarm

        parts.append(("sd_prewarm", int(sdwarm.enabled())))
    except Exception:  # noqa: BLE001
        parts.append(("sd_prewarm", "absent"))
    return " ".join("%s=%s" % kv for kv in parts)


def families_for(hq, hkv, hidx, dense_code, idx_code, page, o12a=None):
    """Plan families of one rank: msa-prewarm's three, plus o12a-split when o12a is on (o12a=None: o12a_enabled()),
    else proxy-autosplit when the backend splits segmented proxy plans (proxy_segmented_enabled())."""
    fams = _families_for(hq, hkv, hidx, dense_code, idx_code, page)
    if o12a is None:
        o12a = o12a_enabled()
    if o12a:
        fams.append(dict(family="o12a-split", dtype_code=idx_code, hq=hidx, hkv=1, sparse_mode=2, page=page,
                         split=True, q_lo=1, q_hi=64))
    elif proxy_segmented_enabled():
        fams.append(dict(family="proxy-autosplit", dtype_code=idx_code, hq=hidx, hkv=1, sparse_mode=2, page=page,
                         split=True, q_lo=1, q_hi=64))
    return fams


def _families_for(hq, hkv, hidx, dense_code, idx_code, page):
    return [
        dict(family="dense", dtype_code=dense_code, hq=hq, hkv=hkv, sparse_mode=3, page=page, split=False,
             q_lo=1, q_hi=Q_MAX),
        dict(family="proxy", dtype_code=idx_code, hq=hidx, hkv=1, sparse_mode=2, page=page, split=False,
             q_lo=1, q_hi=Q_MAX),
        dict(family="proxy-split", dtype_code=idx_code, hq=hidx, hkv=1, sparse_mode=2, page=page, split=True,
             q_lo=65, q_hi=128),
    ]


def enumerate_variants(jit, api, families):
    """{variant name: (runtime args, template params, family, first q)} with fmha_sm100's own dispatch functions."""
    out = {}
    for f in families:
        for q in range(f["q_lo"], f["q_hi"] + 1):
            pf = api._compute_pack_factor(q, f["hq"], f["hkv"])
            packed = q * pf if pf > 1 else q
            tile = 128 if packed <= 128 else 256
            args = (f["dtype_code"], tile, packed <= 64, f["sparse_mode"], f["page"], f["split"], pf)
            try:
                name, params = jit._variant_key_from_runtime(*args)
            except ValueError:
                continue
            if name not in out:
                out[name] = (args, params, f["family"], q)
    return out


def _modules():
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import require_msa_module

    fmha = require_msa_module()
    jit = importlib.import_module(fmha.__name__ + ".jit")
    api = importlib.import_module(fmha.__name__ + ".api")
    return fmha, jit, api


def _so(jit, name):
    return jit.CACHE_BASE / name / ("%s.so" % name)


def _install_observer(jit, api):
    """Serve-time log only: first use of each variant after warmup, and any inline compile not done by the prewarm."""
    if _OBS["installed"]:
        return
    orig_get = api.get_fmha_variant

    def get_fmha_variant(*args, **kwargs):
        if _PHASE[0] == "serve" and not kwargs and args not in _OBS["seen"]:
            _OBS["seen"].add(args)
            try:
                name = jit._variant_key_from_runtime(*args)[0]
                log("serve first-use variant=%s family=%s prewarmed=%d so_ready=%d"
                    % (name, _REACH.get(name, "-"), int(name in _REACH), int(_so(jit, name).exists())))
            except Exception:  # noqa: BLE001
                pass
        return orig_get(*args, **kwargs)

    mgr_cls = type(jit._variant_manager)
    orig_compile = mgr_cls._compile_only

    def _compile_only(self, variant_name, params):
        if getattr(_TLS, "prewarm", False) or _so(jit, variant_name).exists():
            return orig_compile(self, variant_name, params)
        t0 = time.time()
        log("MISS inline compile start variant=%s phase=%s family=%s"
            % (variant_name, _PHASE[0], _REACH.get(variant_name, "-")), warn=True)
        try:
            return orig_compile(self, variant_name, params)
        finally:
            _OBS["inline"] += 1
            log("MISS inline compile done variant=%s phase=%s seconds=%.1f inline_total=%d"
                % (variant_name, _PHASE[0], time.time() - t0, _OBS["inline"]), warn=True)

    api.get_fmha_variant = get_fmha_variant
    mgr_cls._compile_only = _compile_only
    _OBS["installed"] = True


def _build_variant(jit, name, params):
    so = _so(jit, name)
    if so.exists():
        return "present", 0.0
    t0 = time.time()
    _TLS.prewarm = True
    fd = jit._acquire_file_lock(jit.CACHE_BASE / name / ".compile.lock")
    try:
        if so.exists():
            return "peer", time.time() - t0
        jit._variant_manager._compile_only(name, params)
        return "built", time.time() - t0
    finally:
        jit._release_file_lock(fd)
        _TLS.prewarm = False


def _build_plan(jit):
    so = jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so"
    if so.exists():
        return "present", 0.0
    t0 = time.time()
    fd = jit._acquire_file_lock(jit.CACHE_BASE / "plan.lock")
    try:
        if so.exists():
            return "peer", time.time() - t0
        jit._do_compile_plan()
        return "built", time.time() - t0
    finally:
        jit._release_file_lock(fd)


def compile_variants(jit, reach, with_plan=True):
    """Compile every missing variant of `reach` (and the plan module) in parallel; returns {name: (status, s)}."""
    from concurrent.futures import ThreadPoolExecutor

    jit._variant_manager._load_templates()
    # longest first: sparse Off (dense) builds take ~90 s, OnlyScore ~72 s
    names = sorted(reach, key=lambda n: (reach[n][2] != "dense", n))
    todo = [n for n in names if not _so(jit, n).exists()]
    res = {n: ("present", 0.0) for n in names if n not in todo}
    plan_todo = with_plan and not (jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so").exists()
    if not todo and not plan_todo:
        res["plan"] = ("present", 0.0)
        return res, 0
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        ncpu = os.cpu_count() or 8
    workers = max(1, min(len(todo) + int(plan_todo), 16, max(2, ncpu // 4)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="m3ax-prewarm") as ex:
        futs = {n: ex.submit(_build_variant, jit, n, reach[n][1]) for n in todo}
        pf = ex.submit(_build_plan, jit) if plan_todo else None
        for n, f in futs.items():
            try:
                res[n] = f.result()
            except Exception as exc:  # noqa: BLE001
                res[n] = ("failed:%s:%s" % (type(exc).__name__, str(exc).splitlines()[0][:160] if str(exc) else ""), 0.0)
        if pf is not None:
            try:
                res["plan"] = pf.result()
            except Exception as exc:  # noqa: BLE001
                res["plan"] = ("failed:%s" % type(exc).__name__, 0.0)
    res.setdefault("plan", ("present", 0.0))
    return res, workers


def _geometry(runner, kvm, runner_mod):
    """Same gating as DecoderRunner._prewarm_minimax_m3_msa_indexer; None when this runner has no MSA attention."""
    import torch
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import MiniMaxM3KVCacheManagerV2

    if not isinstance(kvm, MiniMaxM3KVCacheManagerV2):
        return None
    sc = getattr(runner, "sparse_attention_config", None)
    cfg_cls = getattr(runner_mod, "MiniMaxM3SparseAttentionConfig", None)
    if cfg_cls is None or not isinstance(sc, cfg_cls) or getattr(sc, "implementation", None) != "msa":
        return None
    params = sc.to_sparse_metadata_params(pretrained_config=runner.model.model_config.pretrained_config)
    mapping = runner.mapping
    hq, hkv = params.sharded_head_counts(mapping)
    hidx = params.sharded_index_head_count(mapping)
    try:
        from tensorrt_llm.bindings import DataType

        dense_fp8 = kvm.dtype in (DataType.FP8, DataType.NVFP4)
        nvfp4 = kvm.dtype == DataType.NVFP4
    except Exception:  # noqa: BLE001
        n = str(getattr(kvm.dtype, "name", kvm.dtype)).upper()
        dense_fp8, nvfp4 = n.endswith("FP8") or n.endswith("NVFP4"), n.endswith("NVFP4")
    return dict(
        hq=int(hq), hkv=int(hkv), hidx=int(hidx), page=int(kvm.tokens_per_block),
        dense_dtype=torch.float8_e4m3fn if dense_fp8 else torch.bfloat16,
        idx_dtype=torch.float8_e4m3fn if kvm.indexer_kv_dtype == "fp8" else torch.bfloat16,
        nvfp4=bool(nvfp4), rank=int(getattr(mapping, "rank", 0)), tp=int(getattr(mapping, "tp_size", 1)),
    )


def prewarm_for_runner(runner, kvm, runner_mod):
    """Compile, without loading, every fmha_sm100 variant this rank's MSA plans can select. Best effort: on any error
    the stock lazy JIT applies unchanged."""
    t0 = time.time()
    try:
        geo = _geometry(runner, kvm, runner_mod)
        if geo is None:
            return
        _, jit, api = _modules()
        _install_observer(jit, api)
        dense_code, idx_code = jit._dlpack_dtype_code(geo["dense_dtype"]), jit._dlpack_dtype_code(geo["idx_dtype"])
        reach = enumerate_variants(jit, api, families_for(geo["hq"], geo["hkv"], geo["hidx"], dense_code,
                                                         idx_code, geo["page"]))
        for name, v in reach.items():
            _REACH.setdefault(name, v[2])
        res, workers = compile_variants(jit, reach)
        st = [res[n][0] for n in reach]
        failed = [n for n in reach if res[n][0].startswith("failed")]
        missing = [n for n in reach if not _so(jit, n).exists()]
        built_s = max([res[n][1] for n in reach if res[n][0] == "built"] or [0.0])
        fams = {}
        for n, v in reach.items():
            fams.setdefault(v[2], []).append(n)
        log("prewarm done rank=%d tp=%d dense=%s:%d/%d proxy=%s:%d/1 page=%d reachable=%d present=%d built=%d "
            "peer=%d failed=%d plan=%s missing_after=%d wall_s=%.1f max_build_s=%.1f threads=%d"
            % (geo["rank"], geo["tp"], str(geo["dense_dtype"]).replace("torch.", ""), geo["hq"], geo["hkv"],
               str(geo["idx_dtype"]).replace("torch.", ""), geo["hidx"], geo["page"], len(reach),
               st.count("present"), st.count("built"), st.count("peer"), len(failed), res["plan"][0],
               len(missing), time.time() - t0, built_s, workers))
        log("prewarm variants rank=%d %s cache=%s" % (
            geo["rank"], " ".join("%s=%s" % (k, ",".join(sorted(v))) for k, v in sorted(fams.items())),
            jit.CACHE_BASE))
        log("stack-a parts rank=%d %s" % (geo["rank"], stack_parts()))
        for n in failed:
            log("prewarm build failed variant=%s %s (compiles lazily on first use, as stock)" % (n, res[n][0]),
                warn=True)
        if not geo["nvfp4"]:
            log("KV cache is not NVFP4: the sparse GQA plan family is not prewarmed", warn=True)
    except Exception as exc:  # noqa: BLE001
        log("prewarm skipped after %.1f s (stock lazy JIT applies): %s: %s"
            % (time.time() - t0, type(exc).__name__, exc), warn=True)
