# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined DSR1 FP4 + Kimi K2 Pareto plot for the Helix Parallelism blog post.

Renders ``docs/source/blogs/media/tech_blog21_dsr1_kimi_k2_side_by_side.png``
as a single figure with two panels: DSR1 FP4 on the left, Kimi K2 on the
right. ``ParetoPoint`` instances, label-position dictionaries, and the
``_plot_curve`` helper are imported from ``blog21_helix_pareto_plot.py`` and
``blog21_kimi_k2_pareto_plot.py``, so any update to those datasets flows
through automatically.

Run from the repo root::

    python docs/source/blogs/tech_blog/blog21_dsr1_kimi_k2_side_by_side.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_module(stem: str):
    """Load a sibling-module file by stem (e.g. ``blog21_helix_pareto_plot``).

    The blog-post plotting scripts are individual files rather than a
    package, so ``importlib`` is used to pull them in by path.
    """
    path = Path(__file__).parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:  # defensive; should not happen
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


dsr1 = _load_module("blog21_helix_pareto_plot")
k2 = _load_module("blog21_kimi_k2_pareto_plot")

OUT_PATH = Path(__file__).resolve().parents[2] / "blogs" / "media" / \
    "tech_blog21_dsr1_kimi_k2_side_by_side.png"

HELIX_COLOR = "#1F4E79"
BASELINE_COLOR = "#ED7D31"

LEGEND_HANDLES = [
    Line2D([0], [0], color=HELIX_COLOR, linewidth=2, marker="o",
           markersize=7, label="Helix"),
    Line2D([0], [0], color=BASELINE_COLOR, linewidth=2, marker="o",
           markersize=7, label="Baseline"),
]


def _draw_panel(ax, mod, *, xlim, ylim, title) -> None:
    mod._plot_curve(ax, mod.BASELINE_POINTS, BASELINE_COLOR, "o",
                    mod.BASELINE_LABEL_POSITIONS)
    mod._plot_curve(ax, mod.HELIX_POINTS, HELIX_COLOR, "o",
                    mod.HELIX_LABEL_POSITIONS)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Tokens/s/user", fontsize=11)
    ax.set_ylabel("Tokens/s/GPU", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles=LEGEND_HANDLES, loc="upper right", fontsize=10)


def main() -> None:
    # constrained_layout packs the two panels with minimal padding while
    # still leaving room for tick labels, axis titles, and the panel titles.
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(18, 6.5), dpi=150, constrained_layout=True
    )

    # Tightened DSR1 xlim: bottom-right Helix labels now sit at x=132 (was
    # 140 in earlier revisions); xlim right = 165 hosts the ~30 data-unit
    # wide label text with a small visual buffer.
    _draw_panel(
        ax_l, dsr1,
        xlim=(-25, 165),
        ylim=(-10, 145),
        title="DSR1 FP4 1M/16k Pareto on GB300 NVL72, "
              "#Gen GPUs Per Instance=64",
    )

    # K2 has wider data range than DSR1, so each character takes more data
    # units; the right-side label text width is ~40 data units. Bottom-right
    # Helix labels at x=160 keep the column close to the rightmost marker
    # (x=158.69) and fit comfortably inside xlim right = 200.
    _draw_panel(
        ax_r, k2,
        xlim=(-58, 200),
        ylim=(-32, 478),
        title="Kimi K2 250k/8k Pareto on GB300 NVL72, "
              "#Gen GPUs Per Instance \u2264 64",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
