# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Kimi K2.5 Pareto plot used in the Helix Parallelism blog post.

Renders ``docs/source/blogs/media/tech_blog21_kimi_k2_5_pareto.png`` with each
Pareto point annotated by its parallelism configuration, and the connecting
curve smoothed via monotone-preserving cubic Hermite (PCHIP) interpolation.
Run from the repo root::

    python docs/source/blogs/tech_blog/blog21_kimi_2_5_pareto_plot.py
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
    ParetoPoint("kim-c64_kvp8",        64, False,  8, 1, 1,  8,  35.04, 280.35),
    ParetoPoint("kim-c128_kvp32",     128, False, 32, 1, 1, 32,  51.22, 204.89),
    ParetoPoint("kim-c16_kvp8",        16, False,  8, 1, 1,  8,  72.27, 144.54),
    ParetoPoint("kim-c8_kvp8",          8, False,  8, 1, 1,  8,  90.36,  90.36),
    ParetoPoint("kim-c8_kvp16",         8, False, 16, 1, 1, 16, 102.59,  51.29),
    ParetoPoint("kim-c1_kvp4_tp1",      1, False,  4, 1, 1,  4, 114.75,  28.69),
    ParetoPoint("kim-c4_kvp16_tp2",     4, False, 16, 2, 1, 32, 116.75,  14.59),
    ParetoPoint("kim-c2_kvp16_tp2",     2, False, 16, 2, 1, 32, 120.04,   7.50),
    ParetoPoint("kim-c2_kvp32_tp2",     2, False, 32, 2, 1, 64, 123.63,   3.86),
]

# Baseline runs (KVP=1, no Helix).
BASELINE_POINTS: list[ParetoPoint] = [
    ParetoPoint("base-c8_tp4",    8, False, 1, 4, 1, 4, 45.46, 90.91),
    ParetoPoint("base-c4_tp4",    4, False, 1, 4, 1, 4, 63.27, 63.27),
    ParetoPoint("base-c2_tp4",    2, False, 1, 4, 1, 4, 77.68, 38.84),
    ParetoPoint("base-c2_tp8",    2, False, 1, 8, 1, 8, 94.15, 23.54),
]


# Layout strategy: ALL Helix (blue) labels are placed ABOVE/RIGHT of the blue
# Pareto curve. ALL baseline (orange) labels are placed BELOW/LEFT of the
# orange Pareto curve. This spatial separation lets the reader map each label
# to its curve at a glance, even where the two curves come close.
# Format: ABSOLUTE label position (lx, ly, ha, va, draw_leader). None
# suppresses the label entirely.
HELIX_LABEL_POSITIONS = {
    # Top-left points: stack the labels diagonally above the curve so each
    # text sits in the upper-right whitespace of its marker.
    "kim-c64_kvp8":      ( 52, 295, "left", "center", True),
    "kim-c128_kvp32":    ( 70, 222, "left", "center", True),
    "kim-c16_kvp8":      ( 90, 162, "left", "center", True),
    "kim-c8_kvp8":       (108, 110, "left", "center", True),
    "kim-c8_kvp16":      (125,  74, "left", "center", True),
    # Bottom-right cluster: the four lowest-y points sit in a tight column
    # at x=115..124, so we stack their labels in a column to the right of
    # the markers, with leader lines connecting each label to its marker.
    "kim-c1_kvp4_tp1":   (140,  52, "left", "center", True),
    "kim-c4_kvp16_tp2":  (140,  37, "left", "center", True),
    "kim-c2_kvp16_tp2":  (140,  22, "left", "center", True),
    "kim-c2_kvp32_tp2":  (140,   8, "left", "center", True),
}

BASELINE_LABEL_POSITIONS = {
    # Top-left baseline points: labels are placed in the small margin to the
    # left of x=0 so the text never crosses the Helix curve. These markers
    # sit on the part of the Helix curve that runs diagonally through this
    # region; without the margin shift, the right end of each label would
    # intersect the blue curve.
    "base-c8_tp4":     (-32, 100, "left", "center", True),
    "base-c4_tp4":     (-32,  65, "left", "center", True),
    "base-c2_tp4":     (-32,  32, "left", "center", True),
    # Bottom-right baseline point - label below the marker / below the
    # bottom-right cluster of helix labels.
    "base-c2_tp8":     ( 65, -15, "left", "center", True),
}

OUT_PATH = Path(__file__).resolve().parents[2] / "blogs" / "media" / \
    "tech_blog21_kimi_k2_5_pareto.png"


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

    # xlim is extended slightly to the left so the top-three baseline labels
    # sit in the left margin without crossing the Helix curve, which runs
    # diagonally through the (35..80, 50..200) region. ylim is extended
    # downward to fit the bottom-right baseline label and upward to host the
    # Helix label for the highest-throughput-per-GPU point.
    ax.set_xlim(-35, 175)
    ax.set_ylim(-25, 310)
    ax.set_xlabel("Tokens/s/user", fontsize=11)
    ax.set_ylabel("Tokens/s/GPU", fontsize=11)
    ax.set_title("Kimi K2.5 250k/8k Pareto on GB300 NVL72, "
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
