#!/usr/bin/env python3
"""Compute effective HBM bandwidth from nsys-exported sqlite traces.

Bounds analysis by the cudaProfilerApi NVTX ranges (overlap_<tag>_measurement_N),
each of which wraps exactly ONE TTFT request (1 prefill + 20 decode iterations
for ISL=107 / OSL=20 / max_beam_width=10 on TinyLlama-1.1B).

Per measurement:
  - bytes_weights = 21 forwards x sizeof(model in bf16)        # 1 prefill + 20 decode
  - bytes_kvcache_decode  ≈ sum_t(2 * num_layers * num_kv_heads * head_dim
                              * (ISL + t) * num_beams * 2 bytes)
  - kernel_active_us = total GPU-busy time inside the window

Bandwidth is computed three ways:
  1) wall-clock effective bandwidth = weights / wall
  2) kernel-active effective bandwidth = weights / kernel_active
  3) "model-forward only" bandwidth = weights / time_in_model_forward_kernels
     using a regex classification of kernel names.
"""

import sqlite3
import argparse
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Model spec for TinyLlama-1.1B-Chat-v1.0 (Llama-2 family)
# ---------------------------------------------------------------------------
MODEL = dict(
    name="TinyLlama-1.1B-Chat-v1.0",
    hidden=2048,
    intermediate=5632,
    num_layers=22,
    num_q_heads=32,
    num_kv_heads=4,
    head_dim=64,
    vocab=32000,
    dtype_bytes=2,  # bf16
    tie_word_embeddings=False,
)


def model_bytes(m=MODEL) -> int:
    """Total bytes a single forward must read for weights + embedding + lm_head."""
    h = m["hidden"]
    inter = m["intermediate"]
    n_layers = m["num_layers"]
    n_q = m["num_q_heads"]
    n_kv = m["num_kv_heads"]
    d = m["head_dim"]
    v = m["vocab"]

    per_layer = (
        # QKV: Q is [h, n_q*d] = [h, h] for grouped-q; K/V are [h, n_kv*d]
        h * (n_q * d) + 2 * h * (n_kv * d)
        # O proj
        + h * h
        # Up + gate + down
        + 2 * h * inter + inter * h
        # 2x RMSNorm vectors
        + 2 * h
    )
    embed = v * h          # token embedding
    final_norm = h
    lm_head = 0 if m["tie_word_embeddings"] else v * h
    total_params = n_layers * per_layer + embed + final_norm + lm_head
    return total_params * m["dtype_bytes"], total_params


# ---------------------------------------------------------------------------
# Kernel classification
#   Class -> regex (matched against the demangled kernel name)
# ---------------------------------------------------------------------------
KERNEL_CLASSES = [
    # The actual model-forward arithmetic.
    ("model_matmul",   re.compile(r"gemvx::kernel|sm.._.*gemm|cublas.*gemm|cutlass.*gemm|"
                                  r"sm.._xmma|cublasLt|sm.._dgrad|gemv2N|gemv2T|"
                                  r"trtllm.*gemm|finegrained.*gemm")),
    ("model_attention", re.compile(r"fmha|flash_attention|trt_llm.*attention|"
                                   r"masked_multihead|mmha_kernel|paged_kv")),
    ("model_norm_rope", re.compile(r"applyBiasRope|RMSNorm|rms_norm|rotary|"
                                   r"computeSeqAndPaddingOffsets|invokeAddBiasResidual|"
                                   r"invoke.*norm|silu|swiglu|invoke.*activation")),
    # Sampling pipeline (beam search top-k, sort, softmax, scatter).
    ("sampling",       re.compile(r"mbtopk|gatherTopK|bitonicSort|cunn_SoftMax|"
                                  r"_scatter_gather|DeviceScanByKey|InclusiveSumByKey|"
                                  r"DeviceScanKernel|DeviceScanInitKernel|cub::detail::scan|"
                                  r"masked_fill|where_kernel|compare_scalar|where_scalar")),
    # Pure index/copy/elementwise plumbing — almost certainly not model forward.
    ("plumbing",       re.compile(r"index_elementwise_kernel|unrolled_elementwise_kernel|"
                                  r"vectorized_elementwise_kernel|elementwise_kernel|"
                                  r"reduce_kernel|FillFunctor|arange|index_copy|"
                                  r"index_fill|index_put|direct_copy|elementwise_kernel_with_index")),
]


def classify_kernel(name: str) -> str:
    if not name:
        return "unknown"
    for cls, pat in KERNEL_CLASSES:
        if pat.search(name):
            return cls
    return "other"


# ---------------------------------------------------------------------------
# Time-overlap helpers
# ---------------------------------------------------------------------------
def overlap_dur(a_start, a_end, w_start, w_end):
    """Duration that interval [a_start,a_end] spends inside [w_start,w_end]."""
    s = max(a_start, w_start)
    e = min(a_end, w_end)
    return max(0, e - s)


def union_busy_time(intervals):
    """Length of the union of (start,end) pairs, in same units as input."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    busy = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    return busy


# ---------------------------------------------------------------------------
# Main analysis per sqlite file
# ---------------------------------------------------------------------------
def analyze(db_path, tag, peak_bw_gbs=864.0,
            isl=107, osl=20, num_beams=10):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # GPU spec (sanity)
    gpu_name, hbm_bw = cur.execute(
        "SELECT name, memoryBandwidth FROM TARGET_INFO_GPU LIMIT 1"
    ).fetchone()
    hbm_bw_gbs = hbm_bw / 1e9

    # Bounding NVTX ranges (5 per file).
    measurements = list(cur.execute("""
        SELECT COALESCE(e.text, s.value) AS t, e.start, e.end
        FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId
        WHERE COALESCE(e.text, s.value) LIKE ?
        ORDER BY e.start
    """, (f"%{tag}_measurement_%",)))

    # Pre-fetch all kernels and memcpy/memset with names.
    kernels = list(cur.execute("""
        SELECT k.start, k.end, COALESCE(sd.value, sm.value) AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds sd ON sd.id = k.demangledName
        LEFT JOIN StringIds sm ON sm.id = k.shortName
    """))
    memcpy = list(cur.execute("""
        SELECT m.start, m.end FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    """))
    memset = list(cur.execute("""
        SELECT m.start, m.end FROM CUPTI_ACTIVITY_KIND_MEMSET m
    """))
    # CUDA graph replays — the model forward (matmuls, attention, RMSNorm,
    # RoPE, etc.) runs *inside* these replays and does NOT show up as
    # individual KERNEL rows in the sqlite. Each row is the wall time of
    # one entire graph replay on the GPU.
    graphs = list(cur.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
    """))

    # Model byte budget per forward
    bytes_per_forward, n_params = model_bytes()

    # Per-measurement KV cache reads (decode steps only)
    # Each decode step at position p reads K and V for positions [0..p-1] for ALL beams.
    # KV cache layout: [num_layers, 2 (K,V), num_kv_heads, head_dim, dtype_bytes]
    kv_per_position_per_beam = (
        2 * MODEL["num_layers"] * MODEL["num_kv_heads"]
        * MODEL["head_dim"] * MODEL["dtype_bytes"]
    )

    def kv_bytes_for_decode_step(decode_idx):
        # decode_idx: 0..osl-1 (0th decode reads ISL positions, ...)
        seq_len_at_step = isl + decode_idx
        return seq_len_at_step * num_beams * kv_per_position_per_beam

    print("=" * 96)
    print(f"FILE : {db_path}")
    print(f"GPU  : {gpu_name}  | peak HBM = {hbm_bw_gbs:.0f} GB/s "
          f"(spec from sqlite; using {peak_bw_gbs} GB/s as peak)")
    print(f"MODEL: {MODEL['name']}  | params = {n_params/1e9:.3f} B  | "
          f"bf16 size = {bytes_per_forward/1e9:.3f} GB")
    print(f"WORKLOAD: ISL={isl}, OSL={osl}, beams={num_beams}, "
          f"forwards/measurement = {1+osl}")
    print()

    rows = []
    for label, ws, we in measurements:
        wall_ns = we - ws

        # Sum kernel time within the window: "kernel activity" (overlap-aware union).
        kintervals = []
        per_class = defaultdict(list)
        per_class_count = defaultdict(int)
        for ks, ke, kname in kernels:
            if ke <= ws or ks >= we:
                continue
            s = max(ks, ws); e = min(ke, we)
            kintervals.append((s, e))
            cls = classify_kernel(kname)
            per_class[cls].append((s, e))
            per_class_count[cls] += 1

        # Memcpy / memset overlap as well
        mintervals = []
        for ms, me in memcpy:
            if me <= ws or ms >= we: continue
            mintervals.append((max(ms, ws), min(me, we)))
        for ms, me in memset:
            if me <= ws or ms >= we: continue
            mintervals.append((max(ms, ws), min(me, we)))

        # CUDA graph replays in this window. Bucket by duration: the trace
        # is bimodal (small ≤500us → decode/postproc, large ≥500us → prefill
        # or full-forward graph), and we use that to attribute graph time to
        # decode vs prefill below.
        gintervals = []
        gdurs_small = []
        gdurs_large = []
        for gs, ge in graphs:
            if ge <= ws or gs >= we: continue
            sg = max(gs, ws); eg = min(ge, we)
            gintervals.append((sg, eg))
            d_us = (eg - sg) / 1e3
            if d_us < 500:
                gdurs_small.append(d_us)
            else:
                gdurs_large.append(d_us)

        kernel_busy_ns = union_busy_time(kintervals)
        memcpy_busy_ns = union_busy_time(mintervals)
        graph_busy_ns  = union_busy_time(gintervals)
        # Total GPU-busy = union of (kernels + memcpy + memset + graph replays)
        gpu_busy_ns    = union_busy_time(kintervals + mintervals + gintervals)
        gpu_idle_pct   = 100.0 * (wall_ns - gpu_busy_ns) / wall_ns

        per_class_busy = {cls: union_busy_time(ivs) for cls, ivs in per_class.items()}

        # Bytes for this measurement (weights only, plus KV-cache reads on decode)
        bytes_weights = (1 + osl) * bytes_per_forward
        bytes_kv = sum(kv_bytes_for_decode_step(i) for i in range(osl))
        bytes_total = bytes_weights + bytes_kv

        # Idealized "model forward" GPU time: graph replays carry the matmul
        # / attention / RMSNorm / RoPE work. Add the few model-class kernels
        # that DO appear outside graphs (e.g. prefill RoPE, FMHA when not in
        # a graph) for completeness.
        model_classes = ("model_matmul", "model_attention", "model_norm_rope")
        model_kernel_intervals = sum((per_class.get(c, []) for c in model_classes), [])
        model_busy_ns = union_busy_time(gintervals + model_kernel_intervals)

        # Effective bandwidth metrics
        def gbs(bytes_, ns):
            return bytes_ / ns if ns > 0 else 0.0  # bytes/ns == GB/s (with 1e9 cancel)

        bw_wall_weights         = gbs(bytes_weights, wall_ns)
        bw_gpubusy_weights      = gbs(bytes_weights, gpu_busy_ns)
        bw_modelactive_weights  = gbs(bytes_weights, model_busy_ns)
        bw_modelactive_total    = gbs(bytes_total,    model_busy_ns)

        rows.append(dict(
            label=label,
            wall_us=wall_ns/1e3,
            kernel_busy_us=kernel_busy_ns/1e3,
            memcpy_busy_us=memcpy_busy_ns/1e3,
            graph_busy_us=graph_busy_ns/1e3,
            gdurs_small_us=gdurs_small,
            gdurs_large_us=gdurs_large,
            gpu_busy_us=gpu_busy_ns/1e3,
            gpu_idle_pct=gpu_idle_pct,
            model_busy_us=model_busy_ns/1e3,
            per_class_busy_us={k: v/1e3 for k, v in per_class_busy.items()},
            per_class_count=dict(per_class_count),
            bytes_weights_gb=bytes_weights/1e9,
            bytes_kv_gb=bytes_kv/1e9,
            bw_wall=bw_wall_weights,
            bw_gpubusy=bw_gpubusy_weights,
            bw_modelactive_w=bw_modelactive_weights,
            bw_modelactive_t=bw_modelactive_total,
        ))

    # Pretty print
    hdr = (f"{'measurement':<32} {'wall(ms)':>9} {'gpubusy(ms)':>12} "
           f"{'idle%':>6} {'model(ms)':>10} "
           f"{'BWwall':>8} {'BWbusy':>8} {'BWmodel':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<32} "
              f"{r['wall_us']/1e3:>9.2f} "
              f"{r['gpu_busy_us']/1e3:>12.2f} "
              f"{r['gpu_idle_pct']:>5.1f}% "
              f"{r['model_busy_us']/1e3:>10.2f} "
              f"{r['bw_wall']:>6.0f}GB "
              f"{r['bw_gpubusy']:>6.0f}GB "
              f"{r['bw_modelactive_w']:>6.0f}GB ")

    # Aggregate
    n = len(rows)
    avg_wall   = sum(r['wall_us'] for r in rows)/n
    avg_busy   = sum(r['gpu_busy_us'] for r in rows)/n
    avg_kbusy  = sum(r['kernel_busy_us'] for r in rows)/n
    avg_gbusy  = sum(r['graph_busy_us'] for r in rows)/n
    avg_idle   = sum(r['gpu_idle_pct'] for r in rows)/n
    avg_model  = sum(r['model_busy_us'] for r in rows)/n
    avg_w      = sum(r['bytes_weights_gb'] for r in rows)/n
    avg_kv     = sum(r['bytes_kv_gb'] for r in rows)/n
    # GB / us == 1e3 * GB/s, so (avg_w / (avg_wall/1e6)) = GB/s. Equivalently
    # avg_w * 1e6 / avg_wall_us when wall is in us.
    avg_bw_wall  = avg_w * 1e6 / avg_wall  if avg_wall  > 0 else 0
    avg_bw_busy  = avg_w * 1e6 / avg_busy  if avg_busy  > 0 else 0
    avg_bw_mdl_w = avg_w * 1e6 / avg_model if avg_model > 0 else 0
    avg_bw_mdl_t = (avg_w + avg_kv) * 1e6 / avg_model if avg_model > 0 else 0

    # Per-forward (decode-dominated) bandwidth:
    #   1 prefill + osl decode steps ≈ osl forwards (prefill ≈ same time)
    #   Each "forward" loads ~ bytes_per_forward of weights from HBM.
    avg_per_forward_us = avg_model / (1 + osl)
    bw_per_forward = bytes_per_forward * 1e6 / avg_per_forward_us / 1e9   # GB/s
    # Above: bytes/us * 1e6 = bytes/s, /1e9 = GB/s

    print()
    print("=== Per-measurement averages ===")
    print(f"  Wall-clock per request                  : {avg_wall/1e3:7.2f} ms")
    print(f"  GPU busy time (kernels + memcpy + graphs): {avg_busy/1e3:7.2f} ms"
          f"  [graphs={avg_gbusy/1e3:.2f} ms, kernels={avg_kbusy/1e3:.2f} ms]")
    print(f"  GPU idle fraction                       : {avg_idle:7.1f} %")
    print(f"  Time in model-forward (graphs+model_kern): {avg_model/1e3:7.2f} ms")
    print(f"  Bytes/request (weights, {1+osl} forwards)     : {avg_w:7.2f} GB")
    print(f"  Bytes/request (KV reads, {osl} decode steps) : {avg_kv:7.3f} GB")
    print()
    print("=== Effective HBM bandwidth (per measurement) ===")
    print(f"  weights / wall clock              : "
          f"{avg_bw_wall:6.1f} GB/s = {100*avg_bw_wall/peak_bw_gbs:5.1f}% of peak ({peak_bw_gbs:.0f} GB/s)")
    print(f"  weights / GPU-busy                : "
          f"{avg_bw_busy:6.1f} GB/s = {100*avg_bw_busy/peak_bw_gbs:5.1f}% of peak")
    print(f"  weights / model-forward time      : "
          f"{avg_bw_mdl_w:6.1f} GB/s = {100*avg_bw_mdl_w/peak_bw_gbs:5.1f}% of peak  "
          f"<-- this is what the megakernel actually optimizes")
    print(f"  (weights+KV) / model-forward time : "
          f"{avg_bw_mdl_t:6.1f} GB/s = {100*avg_bw_mdl_t/peak_bw_gbs:5.1f}% of peak")
    print()
    print(f"  Per-forward time  (avg over {1+osl} fwds): "
          f"{avg_per_forward_us:7.0f} us → {bw_per_forward:.0f} GB/s "
          f"= {100*bw_per_forward/peak_bw_gbs:.1f}% of peak per forward")

    # Class breakdown (avg)
    print()
    print("=== GPU time breakdown (avg per measurement) ===")
    avg_class_us = defaultdict(float)
    avg_class_n  = defaultdict(int)
    for r in rows:
        for c, us in r['per_class_busy_us'].items():
            avg_class_us[c] += us / n
        for c, k in r['per_class_count'].items():
            avg_class_n[c] += k / n
    # Graph time is reported separately — it represents model forward fused
    # into CUDA-graph replays.
    avg_graph_us = avg_gbusy
    total_us = sum(avg_class_us.values()) + avg_graph_us or 1
    print(f"  {'class':<28} {'busy (ms)':>10} {'% of all':>9} {'avg count':>11}")
    print(f"  {'CUDA graph replays':<28} {avg_graph_us/1e3:>10.2f} "
          f"{100*avg_graph_us/total_us:>8.1f}% {n*42/n:>11.0f}    [model fwd lives here]")
    for cls in ["model_matmul", "model_attention", "model_norm_rope",
                "sampling", "plumbing", "other"]:
        us = avg_class_us.get(cls, 0)
        cnt = avg_class_n.get(cls, 0)
        print(f"  {cls:<28} {us/1e3:>10.2f} "
              f"{100*us/total_us:>8.1f}% {cnt:>11.0f}")

    # Show graph duration buckets so we can see prefill vs decode split.
    print()
    print(f"=== CUDA graph durations (avg per measurement) ===")
    avg_n_small = sum(len(r['gdurs_small_us']) for r in rows)/n
    avg_n_large = sum(len(r['gdurs_large_us']) for r in rows)/n
    avg_t_small = sum(sum(r['gdurs_small_us']) for r in rows)/n
    avg_t_large = sum(sum(r['gdurs_large_us']) for r in rows)/n
    if avg_n_small:
        avg_d_small = avg_t_small / avg_n_small
        bw_small = bytes_per_forward / (avg_d_small * 1e3) / 1e9 * 1e9   # GB/s
        # bytes / us = bytes/(us)*(1e6 us/s) = bytes/s; /1e9 = GB/s
        bw_small = bytes_per_forward / (avg_d_small * 1e-6) / 1e9
        print(f"  small  (<500us, decode/aux) : n={avg_n_small:5.1f}  "
              f"avg dur={avg_d_small:7.1f} us  total={avg_t_small/1e3:6.2f} ms  "
              f"-> implied BW if 1 fwd: {bw_small:6.0f} GB/s = {100*bw_small/peak_bw_gbs:.1f}% peak")
    if avg_n_large:
        avg_d_large = avg_t_large / avg_n_large
        bw_large = bytes_per_forward / (avg_d_large * 1e-6) / 1e9
        print(f"  large  (>=500us, full fwd)  : n={avg_n_large:5.1f}  "
              f"avg dur={avg_d_large:7.1f} us  total={avg_t_large/1e3:6.2f} ms  "
              f"-> implied BW if 1 fwd: {bw_large:6.0f} GB/s = {100*bw_large/peak_bw_gbs:.1f}% peak")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--tag", required=True,
                   help="Measurement NVTX tag prefix, e.g. 'overlap_on'.")
    p.add_argument("--isl", type=int, default=107)
    p.add_argument("--osl", type=int, default=20)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--peak-bw-gbs", type=float, default=864.0,
                   help="L40S peak HBM bandwidth, GB/s.")
    args = p.parse_args()
    analyze(args.db, args.tag,
            peak_bw_gbs=args.peak_bw_gbs,
            isl=args.isl, osl=args.osl, num_beams=args.beams)


if __name__ == "__main__":
    main()
