# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Kimi K2 Pareto plot used in the Helix Parallelism blog post.

Renders ``docs/source/blogs/media/tech_blog21_kimi_k2_pareto.png`` with each
Pareto point annotated by its parallelism configuration, and the connecting
curve smoothed via monotone-preserving cubic Hermite (PCHIP) interpolation.
Run from the repo root::

    python docs/source/blogs/tech_blog/blog21_kimi_k2_pareto_plot.py
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


# Helix runs (KVP>1). Sorted by tokens/s/user (ascending) so the curve walks
# from the top-left (max tokens/s/GPU) to the bottom-right (max tokens/s/user).
HELIX_POINTS: list[ParetoPoint] = [
    ParetoPoint("k2-c512_kvp8",       512, True,   8, 8, 1, 64,  36.36, 290.86),
    ParetoPoint("k2-c32_kvp8",         32, False,  8, 1, 1,  8,  53.19, 212.77),
    ParetoPoint("k2-c16_kvp8",         16, False,  8, 1, 1,  8,  74.80, 149.59),
    ParetoPoint("k2-c8_kvp8",           8, False,  8, 1, 1,  8,  92.75,  92.75),
    ParetoPoint("k2-c8_kvp16",          8, False, 16, 1, 1, 16, 106.31,  53.16),
    ParetoPoint("k2-c1_kvp4_tp1",       1, False,  4, 1, 1,  4, 121.06,  30.26),
    ParetoPoint("k2-c1_kvp4_tp2",       1, False,  4, 2, 1,  8, 151.83,  18.98),
    ParetoPoint("k2-c1_kvp8_tp2",       1, False,  8, 2, 1, 16, 158.69,   9.92),
]

# Baseline runs (KVP=1, no Helix).
BASELINE_POINTS: list[ParetoPoint] = [
    ParetoPoint("k2base-c256_dp16",   256, True, 1, 16, 1, 16, 27.51, 440.22),
    ParetoPoint("k2base-c64_dp8",      64, True, 1,  8, 1,  8, 41.47, 331.78),
    ParetoPoint("k2base-c128_dp32",   128, True, 1, 32, 1, 32, 55.81, 223.24),
    ParetoPoint("k2base-c8_dp4",        8, True, 1,  4, 1,  4, 66.31, 132.61),
    ParetoPoint("k2base-c16_dp16",     16, True, 1, 16, 1, 16, 73.27,  73.27),
]


# Layout strategy: ALL Helix (blue) labels are placed ABOVE/RIGHT of the blue
# Pareto curve, and most baseline (orange) labels are placed in the left
# margin. The two curves CROSS near (x=58, y=200): for tokens/s/user < ~58
# the baseline curve is the upper envelope, and the helix curve is the upper
# envelope after that. Because of this, a few leader lines for left-margin
# baseline labels brush against the helix curve - this is unavoidable when
# the marker sits within a few units of the other curve.
# Format: ABSOLUTE label position (lx, ly, ha, va, draw_leader). None
# suppresses the label entirely.
HELIX_LABEL_POSITIONS = {
    # Top-left points: stack the labels diagonally above the curve so each
    # text sits in the upper-right whitespace of its marker.
    "k2-c512_kvp8":     ( 60, 320, "left", "center", True),
    "k2-c32_kvp8":      ( 78, 240, "left", "center", True),
    "k2-c16_kvp8":      ( 95, 175, "left", "center", True),
    "k2-c8_kvp8":       (113, 115, "left", "center", True),
    "k2-c8_kvp16":      (128,  78, "left", "center", True),
    # Bottom-right cluster (all C=1): stack labels in a tight column to the
    # right of the markers, which sit between x=121 and x=159.
    "k2-c1_kvp4_tp1":   (170,  60, "left", "center", True),
    "k2-c1_kvp4_tp2":   (170,  45, "left", "center", True),
    "k2-c1_kvp8_tp2":   (170,  30, "left", "center", True),
}

BASELINE_LABEL_POSITIONS = {
    # Top-left point sits above all helix data, so its label can ride along
    # in the same upper-right whitespace.
    "k2base-c256_dp16": ( 48, 458, "left", "center", True),
    # Remaining baseline points: labels in the left margin, stacked
    # vertically. Leader lines stay below the helix curve except for
    # k2base-c128_dp32, whose marker sits ~3 units above the helix curve and
    # therefore unavoidably brushes it.
    "k2base-c64_dp8":   (-55, 360, "left", "center", True),
    "k2base-c128_dp32": (-55, 245, "left", "center", True),
    "k2base-c8_dp4":    (-55, 130, "left", "center", True),
    "k2base-c16_dp16":  (-55,  75, "left", "center", True),
}

OUT_PATH = Path(__file__).resolve().parents[2] / "blogs" / "media" / \
    "tech_blog21_kimi_k2_pareto.png"


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

    # xlim is extended to the left so the stacked baseline labels sit in the
    # margin without crossing the Helix curve, and to the right to host the
    # bottom-right Helix label cluster. ylim is extended downward to fit the
    # bottom-right baseline labels and upward to host the highest-y baseline
    # label (C=256 baseline, ~440 tokens/s/GPU).
    ax.set_xlim(-60, 200)
    ax.set_ylim(-32, 478)
    ax.set_xlabel("Tokens/s/user", fontsize=11)
    ax.set_ylabel("Tokens/s/GPU", fontsize=11)
    ax.set_title("Kimi K2 250k/8k Pareto on GB300 NVL72, "
                 "#Gen GPUs Per Instance \u2264 64", fontsize=12)
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
