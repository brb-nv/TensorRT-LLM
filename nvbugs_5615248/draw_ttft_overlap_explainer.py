"""Visual explainer: why TTFT is *higher* with the overlap scheduler.

Generates a stacked-timeline figure that compares the first three executor
iterations of `_executor_loop` (overlap OFF) vs `_executor_loop_overlap`
(overlap ON), as observed in the NVBug 5615248 nsys profile.

Headline numbers (5 measurements, TinyLlama, ISL=107, beam=10, gen=20):

    overlap OFF :  TTFT mean = 10.94 ms,  Total = 110.64 ms
    overlap ON  :  TTFT mean = 14.94 ms,  Total =  91.57 ms

The +4 ms TTFT delta with overlap ON is *almost exactly* one decode-iter:
the first token isn't enqueued at the end of the prefill iter (iter 0),
because `_update_requests` / `_handle_responses` for the prefill batch
are deferred to iter 1 (where they overlap that iter's GPU forward).

Important nuance (corrected from a previous version of this script):
    `sample_async` itself is NON-BLOCKING in both modes. It only enqueues
    decoder kernels (`forward_async`) and async D2H copies (`_copy_to_host`),
    then records a `torch.cuda.Event` and returns. The actual host->GPU
    barrier is `state.sampler_event.synchronize()`, which is the FIRST line
    of every `update_requests` implementation (TorchSampler line 3298,
    TRTLLMSampler line 4717). The wait is drawn explicitly as the leading
    sub-segment of the `update_requests` box.

Iteration phase durations are taken from
nvbugs_5615248/nsys_analysis/overlap_{on,off}/nvtx_pushpop_sum.column.txt
(prefill _forward_step ~5.5 ms, gen _forward_step ~0.78 ms,
_sample_async ~2.6 ms CPU, _update_requests ~1.1 ms in non-overlap mode
~0.07 ms in overlap mode).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Phase durations in ms, taken directly from nsys NVTX wall-clock medians
# in nvbugs_5615248/nsys_analysis/overlap_*/nvtx_pushpop_sum.column.txt
# so the chart's TTFT/total numbers reproduce the measured ones (~10.9 ms
# off, ~14.9 ms on) instead of an under-counted theoretical model.
#
# (`_forward_step` host wall is ~6.1 ms for prefill, 0.78 ms for gen, even
# though the actual GPU compute is shorter. The host stays inside the call
# for the full wall-clock duration — partly launch loop, partly perf-event
# recording / KV-cache bookkeeping. We draw it that way for visual fidelity.)
PREP_MS = 0.5
PREFILL_FWD_MS = 6.1            # host wall ≈ GPU wall for forward
GEN_FWD_MS = 0.8
SAMPLE_CPU_MS = 2.6             # host-only python work in sample_async
SAMPLE_GPU_MS = 0.5             # decoder kernels triggered inside sample_async
# `update_requests` decomposes into a leading sampler-event sync (wait for
# GPU decoder + async D2H copies) and the actual host-side update work.
UPDATE_SYNC_NONOVERLAP_MS = 0.7   # GPU still has decoder/D2H in flight
UPDATE_WORK_NONOVERLAP_MS = 0.4
UPDATE_SYNC_OVERLAP_MS = 0.02     # event already signaled long ago
UPDATE_WORK_OVERLAP_MS = 0.07
RESP_MS = 0.4

LANE_HOST = 1
LANE_GPU = 0
BAR_HEIGHT = 0.7

PHASE_COLORS = {
    "prep":         "#cccccc",
    "fwd_pre":      "#ff7f0e",
    "fwd_gen":      "#1f77b4",
    "sample":       "#9467bd",
    "sample_gpu":   "#b39dd6",
    "sync_wait":    "#fdd0a2",   # the actual GPU barrier
    "update":       "#2ca02c",
    "resp":         "#d62728",
    "wait":         "#eeeeee",
}


def _draw_box(ax, lane: int, t0: float, dur: float, color: str, label: str,
              text_color: str = "black", fontsize: int = 8,
              hatch: str | None = None):
    """Draw a single timeline box with a centered text label."""
    rect = mpatches.Rectangle(
        (t0, lane - BAR_HEIGHT / 2), dur, BAR_HEIGHT,
        facecolor=color, edgecolor="black", linewidth=0.6,
        hatch=hatch,
    )
    ax.add_patch(rect)
    if dur > 0.4:  # only label boxes wide enough to read
        ax.text(t0 + dur / 2, lane, label,
                ha="center", va="center", fontsize=fontsize, color=text_color)


def _draw_forward(ax, t0: float, dur: float, label: str, color_key: str) -> float:
    """Forward step: host wall ≈ GPU wall (NVTX shows ~6.1ms host for prefill,
    ~0.78ms for gen). We draw a host box and a GPU box of the same width."""
    _draw_box(ax, LANE_HOST, t0, dur, PHASE_COLORS[color_key], label,
              fontsize=7)
    _draw_box(ax, LANE_GPU, t0, dur, PHASE_COLORS[color_key], label,
              fontsize=7)
    return t0 + dur


def _draw_sample_async_host(ax, t0: float) -> float:
    """Sample_async on host: pure CPU prep + kernel launches. No GPU sync."""
    _draw_box(ax, LANE_HOST, t0, SAMPLE_CPU_MS, PHASE_COLORS["sample"],
              "sample_async\n(CPU only;\nno GPU sync)", fontsize=6)
    return t0 + SAMPLE_CPU_MS


def _draw_sample_kernels_gpu(ax, t0: float) -> float:
    """Decoder/sample kernels on GPU. Triggered by sample_async; finish later."""
    _draw_box(ax, LANE_GPU, t0, SAMPLE_GPU_MS, PHASE_COLORS["sample_gpu"],
              "sample\nkernels", fontsize=6)
    return t0 + SAMPLE_GPU_MS


def _draw_update_requests(ax, t0: float, sync_ms: float, work_ms: float,
                          prev_batch_label: str | None = None,
                          show_sync_label: bool = True) -> float:
    """update_requests = [sampler_event.synchronize() wait] + [host update work].

    The leading sync block is drawn with a hatched fill to make the wait
    visible. `prev_batch_label`, if set, identifies which batch's sample_state
    is being processed (used in overlap-ON to make the +1-iter slip explicit).
    """
    if sync_ms > 0.3 and show_sync_label:
        wait_label = "sync\nwait"
    else:
        wait_label = ""
    _draw_box(ax, LANE_HOST, t0, sync_ms, PHASE_COLORS["sync_wait"],
              wait_label, fontsize=6, hatch="//")

    if prev_batch_label is not None:
        work_label = f"upd\n({prev_batch_label})"
    elif work_ms > 0.3:
        work_label = "upd"
    else:
        work_label = ""
    _draw_box(ax, LANE_HOST, t0 + sync_ms, work_ms, PHASE_COLORS["update"],
              work_label, fontsize=6)
    return t0 + sync_ms + work_ms


def _draw_overlap_off(ax) -> tuple[float, float]:
    """One executor iteration is fully serial: prep → forward → sample_async
    → update_requests (sync + work) → handle_responses.

    First token is enqueued at the *end* of iter 0.
    Returns (ttft_t, total_t).
    """
    t = 0.0

    # ---- Iter 0 = prefill ----
    _draw_box(ax, LANE_HOST, t, PREP_MS, PHASE_COLORS["prep"], "prep")
    t += PREP_MS

    fwd_start = t
    t = _draw_forward(ax, t, PREFILL_FWD_MS, "forward (prefill)\n~6 ms",
                      "fwd_pre")

    sample_host_end = _draw_sample_async_host(ax, t)
    _draw_sample_kernels_gpu(ax, fwd_start + PREFILL_FWD_MS)
    t = sample_host_end

    t = _draw_update_requests(ax, t,
                              sync_ms=UPDATE_SYNC_NONOVERLAP_MS,
                              work_ms=UPDATE_WORK_NONOVERLAP_MS)

    _draw_box(ax, LANE_HOST, t, RESP_MS, PHASE_COLORS["resp"],
              "resp", fontsize=6)
    ttft_t = t + RESP_MS
    t = ttft_t

    # ---- Iters 1, 2 = gen ----
    for _ in range(1, 3):
        _draw_box(ax, LANE_HOST, t, PREP_MS, PHASE_COLORS["prep"], "")
        t += PREP_MS

        fwd_start = t
        t = _draw_forward(ax, t, GEN_FWD_MS, "fwd\ngen", "fwd_gen")

        sample_host_end = _draw_sample_async_host(ax, t)
        _draw_sample_kernels_gpu(ax, fwd_start + GEN_FWD_MS)
        t = sample_host_end

        t = _draw_update_requests(ax, t,
                                  sync_ms=UPDATE_SYNC_NONOVERLAP_MS,
                                  work_ms=UPDATE_WORK_NONOVERLAP_MS,
                                  show_sync_label=False)

        _draw_box(ax, LANE_HOST, t, RESP_MS, PHASE_COLORS["resp"],
                  "resp", fontsize=6)
        t += RESP_MS

    return ttft_t, t


def _draw_overlap_on(ax) -> tuple[float, float, float]:
    """Overlap loop. Per iter N host order:

        prep → forward(N) → update_requests(N-1) → sample_async(N)
        → _process_previous_batch(N-1)  [handle_responses for N-1]

    On iter 0 the previous_batch is None so update/handle_responses are
    skipped — TTFT slips by one iter.

    Returns (ttft_t, total_t, iter0_end_t).
    """
    t = 0.0

    # ---- Iter 0 = prefill. previous_batch is None. ----
    _draw_box(ax, LANE_HOST, t, PREP_MS, PHASE_COLORS["prep"], "prep")
    t += PREP_MS

    fwd0_start = t
    t = _draw_forward(ax, t, PREFILL_FWD_MS, "forward (prefill)\n~6 ms",
                      "fwd_pre")

    # No previous_batch ⇒ no update_requests in this iter.
    sample_host_end = _draw_sample_async_host(ax, t)
    _draw_sample_kernels_gpu(ax, fwd0_start + PREFILL_FWD_MS)
    t = sample_host_end

    # No _process_previous_batch either: iter 0 ends here, prefill's first
    # token is on the GPU but NOT yet enqueued to the client.
    iter0_end = t

    # ---- Iter 1 = first gen iter. previous_batch = iter 0. ----
    _draw_box(ax, LANE_HOST, t, PREP_MS, PHASE_COLORS["prep"], "")
    t += PREP_MS

    fwd1_start = t
    t = _draw_forward(ax, t, GEN_FWD_MS, "fwd\ngen", "fwd_gen")

    # update_requests for the PREVIOUS batch (iter 0).
    t = _draw_update_requests(
        ax, t,
        sync_ms=UPDATE_SYNC_OVERLAP_MS,
        work_ms=UPDATE_WORK_OVERLAP_MS,
        prev_batch_label="iter 0",
    )

    # sample_async for the CURRENT batch (iter 1).
    sample_host_end = _draw_sample_async_host(ax, t)
    _draw_sample_kernels_gpu(ax, fwd1_start + GEN_FWD_MS)
    t = sample_host_end

    # _process_previous_batch → _handle_responses for iter 0.
    _draw_box(ax, LANE_HOST, t, RESP_MS, PHASE_COLORS["resp"],
              "resp\n(iter 0\ntoken!)", fontsize=6)
    ttft_t = t + RESP_MS
    t = ttft_t

    # ---- Iter 2 = steady-state gen iter. previous_batch = iter 1. ----
    _draw_box(ax, LANE_HOST, t, PREP_MS, PHASE_COLORS["prep"], "")
    t += PREP_MS

    fwd2_start = t
    t = _draw_forward(ax, t, GEN_FWD_MS, "fwd\ngen", "fwd_gen")

    t = _draw_update_requests(
        ax, t,
        sync_ms=UPDATE_SYNC_OVERLAP_MS,
        work_ms=UPDATE_WORK_OVERLAP_MS,
        prev_batch_label="iter 1",
    )

    sample_host_end = _draw_sample_async_host(ax, t)
    _draw_sample_kernels_gpu(ax, fwd2_start + GEN_FWD_MS)
    t = sample_host_end

    _draw_box(ax, LANE_HOST, t, RESP_MS, PHASE_COLORS["resp"],
              "resp", fontsize=6)
    t += RESP_MS

    return ttft_t, t, iter0_end


def main() -> None:
    fig, (ax_off, ax_on) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"hspace": 1.1},
    )

    ttft_off, end_off = _draw_overlap_off(ax_off)
    ttft_on, end_on, iter0_end_on = _draw_overlap_on(ax_on)

    for ax in (ax_off, ax_on):
        ax.set_ylim(-2.0, 2.2)
        ax.set_yticks([LANE_GPU, LANE_HOST])
        ax.set_yticklabels(["GPU", "Host (Python)"])
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    xmax = max(end_off, end_on) + 4.5  # extra room on the right for callouts
    ax_off.set_xlim(0, xmax)

    # ----- TTFT marker (overlap OFF) -----
    ax_off.axvline(ttft_off, color="red", linestyle="--", linewidth=1.5)
    ax_off.annotate(
        f"TTFT = {ttft_off:.1f} ms\nfirst token enqueued",
        xy=(ttft_off, 1.4), xytext=(ttft_off - 0.2, 2.05),
        color="red", fontsize=9, va="top", ha="right",
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
    )

    # Callout: where the actual GPU sync lives. Placed in the right margin.
    sync_x = PREP_MS + PREFILL_FWD_MS + SAMPLE_CPU_MS + UPDATE_SYNC_NONOVERLAP_MS / 2
    ax_off.annotate(
        "the hatched bar is\n"
        "state.sampler_event.synchronize()\n"
        "→ the only place the host actually\n"
        "    waits for the GPU sample kernels",
        xy=(sync_x, LANE_HOST + 0.35),
        xytext=(end_off + 0.5, LANE_HOST + 1.0),
        color="darkorange", fontsize=9, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff5e6", ec="darkorange",
                  lw=1),
    )

    # ----- TTFT markers (overlap ON) -----
    ax_on.axvline(iter0_end_on, color="gray", linestyle=":", linewidth=1.2)
    ax_on.axvline(ttft_on, color="red", linestyle="--", linewidth=1.5)

    # Gray label: where iter 0 actually ends.
    ax_on.annotate(
        "iter 0 ends here:\nprefill's first token is\nalready on the GPU,\n"
        "but NOT yet enqueued —\nprevious_batch was None",
        xy=(iter0_end_on, -0.4), xytext=(iter0_end_on - 0.3, -1.95),
        color="gray", fontsize=8.5, va="bottom", ha="right",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    # Red label: TTFT in overlap mode.
    ax_on.annotate(
        f"TTFT = {ttft_on:.1f} ms\nprefill's token finally\n"
        "enqueued during iter 1",
        xy=(ttft_on, 1.4), xytext=(ttft_on + 0.3, 2.05),
        color="red", fontsize=9, va="top", ha="left",
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
    )

    # Purple "+1-iter slip" arrow: from iter 0's sample_state on the GPU lane
    # to iter 1's resp box on the host lane.
    sample0_gpu_x = PREP_MS + PREFILL_FWD_MS + SAMPLE_GPU_MS / 2
    resp_x = ttft_on - RESP_MS / 2
    ax_on.annotate(
        "", xy=(resp_x, LANE_HOST - 0.1),
        xytext=(sample0_gpu_x, LANE_GPU + 0.45),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.6,
                        connectionstyle="arc3,rad=-0.4"),
    )
    ax_on.text((sample0_gpu_x + resp_x) / 2, 1.85,
               "iter 0's sample_state is delivered to the client during iter 1\n"
               "→ this is the +1-iter slip that increases TTFT",
               color="purple", fontsize=9, ha="center", va="bottom")

    # ----- Iter labels (below each panel) -----
    iter_y = -0.85
    ax_off.text(ttft_off / 2, iter_y, "iter 0  (prefill)",
                fontsize=9.5, color="#444", style="italic", ha="center")
    ax_off.text(ttft_off + (end_off - ttft_off) / 4, iter_y,
                "iter 1  (gen)", fontsize=9.5, color="#444",
                style="italic", ha="center")
    ax_on.text(iter0_end_on / 2, iter_y, "iter 0  (prefill)",
               fontsize=9.5, color="#444", style="italic", ha="center")
    ax_on.text((iter0_end_on + ttft_on) / 2, iter_y, "iter 1  (gen)",
               fontsize=9.5, color="#444", style="italic", ha="center")
    ax_on.text(ttft_on + (end_on - ttft_on) / 2, iter_y, "iter 2  (gen)",
               fontsize=9.5, color="#444", style="italic", ha="center")

    # ----- Δ TTFT annotation in the right margin of the overlap-ON panel -----
    ax_on.text(
        end_on + 0.5, LANE_HOST + 0.5,
        f"Δ TTFT = +{ttft_on - ttft_off:.1f} ms\n≈ one decode iteration\n\n"
        "(measured: 10.94 → 14.94 ms,\nΔ = +4.0 ms; chart slightly\n"
        "underestimates because we\n"
        "model only the first few iters)",
        fontsize=9.5, color="darkred", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff5f5", ec="darkred", lw=1),
    )

    # Titles.
    ax_off.set_title(
        "overlap scheduler OFF  (`_executor_loop`)  —  "
        "host work runs strictly after each forward; "
        "TTFT ≈ 1 prefill iteration",
        fontsize=11, loc="left",
    )
    ax_on.set_title(
        "overlap scheduler ON  (`_executor_loop_overlap`)  —  "
        "host work for iter N processes batch N−1; "
        "first token slips by 1 iter",
        fontsize=11, loc="left",
    )
    ax_on.set_xlabel("time (ms)")

    # Legend.
    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS["prep"],       label="prep / schedule"),
        mpatches.Patch(color=PHASE_COLORS["fwd_pre"],    label="forward (prefill)"),
        mpatches.Patch(color=PHASE_COLORS["fwd_gen"],    label="forward (generation)"),
        mpatches.Patch(color=PHASE_COLORS["sample"],     label="sample_async (host CPU prep)"),
        mpatches.Patch(color=PHASE_COLORS["sample_gpu"], label="decoder/sample kernels + D2H (GPU)"),
        mpatches.Patch(facecolor=PHASE_COLORS["sync_wait"], hatch="//", edgecolor="black",
                       label="sampler_event.synchronize() (GPU barrier)"),
        mpatches.Patch(color=PHASE_COLORS["update"],     label="update_requests (host work)"),
        mpatches.Patch(color=PHASE_COLORS["resp"],       label="handle_responses (enqueue token)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Why TTFT is higher with overlap scheduler ON\n"
        "(NVBug 5615248: TinyLlama, ISL=107, beam=10, max_batch_size=1)\n"
        "measured: TTFT  off=10.94 ms   on=14.94 ms   (Δ ≈ +4 ms ≈ 1 decode iter)\n"
        "          total off=110.6 ms   on=91.6 ms   (overlap pays back across all gen iters)\n"
        "note: sample_async is non-blocking; the actual GPU sync lives in update_requests "
        "(state.sampler_event.synchronize())",
        fontsize=12, y=1.0,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = "/home/bbuddharaju/scratch/TensorRT-LLM/nvbugs_5615248/ttft_overlap_explainer.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
