# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the DSR1 FP4 Pareto plot used in the Helix Parallelism blog post.

Renders ``docs/source/blogs/media/tech_blog21_dsr1_fp4_pareto.png`` with each
Pareto point annotated by its parallelism configuration, and the connecting
curve smoothed via monotone-preserving cubic Hermite (PCHIP) interpolation.
Run from the repo root::

    python docs/source/blogs/tech_blog/blog21_helix_pareto_plot.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class ParetoPoint:
    name: str
    concurrency: int
    attn_dp: bool
    kvp: int
    tp: int
    pp: int
    ep: int
    tokens_per_s_per_user: float
    tokens_per_s_per_gpu: float

    def label(self) -> str:
        # Use DP=N when attention is data-parallel; TP=N otherwise. Show all
        # dims so the configuration along the curve is unambiguous, but omit
        # PP=1 to keep labels short (PP>1 is rare and highlighted when present).
        attn_field = f"DP={self.tp}" if self.attn_dp else f"TP={self.tp}"
        fields = [
            f"C={self.concurrency}",
            f"KVP={self.kvp}",
            attn_field,
            f"EP={self.ep}",
        ]
        if self.pp != 1:
            fields.append(f"PP={self.pp}")
        return ", ".join(fields)


# Only rows with non-empty Tokens/s/user / Tokens/s/GPU are included.
HELIX_POINTS: list[ParetoPoint] = [
    ParetoPoint("ck-1_ctp32tp2",     1,   False, 32, 2,  1, 8,   131.75, 2.06),
    ParetoPoint("ck-2_ctp8cep4tep2", 2,   False, 32, 2,  1, 8,   117.10, 3.66),
    ParetoPoint("ck-2_ctp4cep8tep2", 2,   False, 32, 2,  1, 16,  116.82, 3.65),
    ParetoPoint("ck-4_cep64",        4,   False, 64, 1,  1, 64,  111.11, 6.94),
    ParetoPoint("ck-8_ctp64",        8,   False, 64, 1,  1, 8,   103.41, 12.93),
    ParetoPoint("ck-16_ctp64",       16,  False, 64, 1,  1, 8,   88.97,  22.24),
    ParetoPoint("ck-32_ctp64",       32,  False, 64, 1,  1, 8,   70.82,  35.41),
    ParetoPoint("ck-64_ctp64",       64,  False, 64, 1,  1, 8,   53.19,  53.19),
    ParetoPoint("ck-2_ep64",         128, True,  1,  64, 1, 64,  39.14,  78.28),
    ParetoPoint("ck-4_ep64",         256, True,  1,  64, 1, 64,  25.65,  102.59),
    ParetoPoint("ck-224_cep32ep2",   448, True,  32, 2,  1, 64,  17.66,  123.63),
]

BASELINE_POINTS: list[ParetoPoint] = [
    ParetoPoint("ck-1_tp64",         1,   False, 1,  64, 1, 8,   74.68,  1.17),
    ParetoPoint("ck-1_ep64",         64,  True,  1,  64, 1, 64,  52.38,  52.38),
    ParetoPoint("ck-2_ep64",         128, True,  1,  64, 1, 64,  39.11,  78.22),
    ParetoPoint("ck-4_ep64",         256, True,  1,  64, 1, 64,  25.44,  101.76),
]


# Layout strategy: ALL Helix (blue) labels are placed ABOVE the blue Pareto
# curve, in the upper-right whitespace. ALL baseline (orange) labels are
# placed BELOW the orange Pareto curve, in the lower-left whitespace. This
# spatial separation lets the reader map each label to its curve at a
# glance, even where the two curves overlap (C=128 / C=256 share the same
# config in both runs - labeled on the baseline side only to avoid clutter).
# Format: ABSOLUTE label position (lx, ly, ha, va, draw_leader). None
# suppresses the label entirely.
HELIX_LABEL_POSITIONS = {
    # Top-left Helix-only point - label snug above-right of the marker.
    "ck-224_cep32ep2":   ( 22, 130, "left",   "center", True),  # C=448
    # Top-left points shared with baseline - suppressed (labeled on baseline).
    "ck-4_ep64":         None,                                   # C=256
    "ck-2_ep64":         None,                                   # C=128
    # Knee of the Helix curve - label snug above-right of the marker.
    "ck-64_ctp64":       ( 55,  70, "left",   "center", True),  # C=64
    # Bottom-right cluster: stack labels in a tight column above the curve,
    # with text starting just to the right of x=140 (close to the markers,
    # which span x=70..132).
    "ck-32_ctp64":       (140,  60, "left",   "center", True),  # C=32
    "ck-16_ctp64":       (140,  47, "left",   "center", True),  # C=16
    "ck-8_ctp64":        (140,  35, "left",   "center", True),  # C=8
    "ck-4_cep64":        (140,  24, "left",   "center", True),  # C=4
    # The two C=2 points are tied (~0.3% apart) - keep only one label.
    "ck-2_ctp8cep4tep2": (140,  14, "left",   "center", True),  # C=2 (EP=8)
    "ck-2_ctp4cep8tep2": None,                                   # tied with above
    "ck-1_ctp32tp2":     (140,   5, "left",   "center", True),  # C=1
}

BASELINE_LABEL_POSITIONS = {
    # Top-left baseline points: labels are placed in the small margin to
    # the left of x=0 so the text never crosses the Helix curve. The C=256
    # and C=128 markers sit on the part of the Helix curve that runs
    # diagonally through this region; without the margin shift, the right
    # end of each label would intersect the blue curve.
    "ck-4_ep64":         (-22,  92, "left",   "center", True),  # C=256
    "ck-2_ep64":         (-22,  68, "left",   "center", True),  # C=128
    # Top of the long diagonal segment - label kept in the margin for
    # visual alignment with the labels above.
    "ck-1_ep64":         (-22,  42, "left",   "center", True),  # C=64
    # Bottom-right baseline point - label just below the marker.
    "ck-1_tp64":         ( 78,  -7, "left",   "center", True),  # C=1
}

OUT_PATH = Path(__file__).resolve().parents[2] / "blogs" / "media" / \
    "tech_blog21_dsr1_fp4_pareto.png"


def _smooth_pareto_xy(points):
    """Return densely-sampled (x, y) along a monotone Pareto curve.

    Points are sorted by tokens/s/user; PchipInterpolator preserves
    monotonicity, so the smoothed curve never overshoots its data points.
    """
    xs = np.array([p.tokens_per_s_per_user for p in points])
    ys = np.array([p.tokens_per_s_per_gpu for p in points])
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    # PchipInterpolator requires strictly-increasing x.
    keep = np.concatenate([[True], np.diff(xs_sorted) > 0])
    xs_unique = xs_sorted[keep]
    ys_unique = ys_sorted[keep]
    interp = PchipInterpolator(xs_unique, ys_unique)
    x_dense = np.linspace(xs_unique.min(), xs_unique.max(), 400)
    return x_dense, interp(x_dense), xs, ys


def _plot_curve(ax, points, color, marker, positions):
    x_dense, y_dense, xs, ys = _smooth_pareto_xy(points)
    # Smooth Pareto curve.
    ax.plot(x_dense, y_dense, color=color, linewidth=2.0, zorder=2.5)
    # Markers at actual measured points.
    ax.plot(xs, ys, color=color, marker=marker, markersize=7,
            linestyle="none", zorder=3)
    for p in points:
        spec = positions[p.name]
        if spec is None:  # label suppressed (tied / identical config elsewhere)
            continue
        lx, ly, ha, va, draw_leader = spec
        x, y = p.tokens_per_s_per_user, p.tokens_per_s_per_gpu
        arrowprops = (
            dict(arrowstyle="-", color=color, alpha=0.45, linewidth=0.6,
                 shrinkA=0, shrinkB=4)
            if draw_leader else None
        )
        ax.annotate(
            p.label(),
            xy=(x, y),
            xytext=(lx, ly),
            ha=ha, va=va,
            fontsize=7,
            color=color,
            zorder=4,
            arrowprops=arrowprops,
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 8), dpi=150)

    helix_color = "#1F4E79"
    baseline_color = "#ED7D31"

    _plot_curve(ax, BASELINE_POINTS, baseline_color, "o",
                BASELINE_LABEL_POSITIONS)
    _plot_curve(ax, HELIX_POINTS, helix_color, "o",
                HELIX_LABEL_POSITIONS)

    # xlim is extended slightly to the left so the C=256 / C=128 / C=64
    # baseline labels sit in the left margin without crossing the Helix
    # curve, which runs diagonally through the (10..50, 50..125) region.
    ax.set_xlim(-25, 195)
    ax.set_ylim(-10, 145)
    ax.set_xlabel("Tokens/s/user", fontsize=11)
    ax.set_ylabel("Tokens/s/GPU", fontsize=11)
    ax.set_title("DSR1 FP4 1M/16k Pareto on GB300 NVL72, "
                 "#Gen GPUs Per Instance=64", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Custom legend handles so each entry shows both line and marker (the
    # smooth curve and the markers are plotted separately above).
    legend_handles = [
        Line2D([0], [0], color=helix_color, linewidth=2, marker="o",
               markersize=7, label="Helix"),
        Line2D([0], [0], color=baseline_color, linewidth=2, marker="o",
               markersize=7, label="Baseline"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
