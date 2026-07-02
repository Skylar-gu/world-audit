"""Acceptance/QA plots from the site manifest (reads manifest.json only).

Small multiples of response curves: truth reference vs. model mean with the
K-sample CRN band. Colors follow the validated reference palette
(dataviz skill): series blue for the model, primary ink for truth, status
colors only for the traffic-light dot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"
STATUS = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}

PLOTS_DIR = Path("experiments/artifacts/plots")


def response_small_multiples(scene: str,
                             manifest_path: Path = Path("site/public/manifest.json"),
                             subtest: str = "response") -> Path:
    m = json.loads(manifest_path.read_text())
    scene_row = next(s for s in m["scenes"] if s["name"] == scene)
    models = scene_row["models"]

    ncols = 3
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows),
                             sharex=True, sharey=True, layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for ax, row in zip(np.ravel(axes), models):
        st = next(s for s in row["subtests"] if s["name"] == subtest)
        c = st["entries"][0]["curves"]
        x = np.array(c["x"], dtype=float)
        y_t = np.array([np.nan if v is None else v for v in c["y_truth"]], dtype=float)
        y_m = np.array([np.nan if v is None else v for v in c["y_model"]], dtype=float)
        lo = np.array([np.nan if v is None else v for v in c["ci"]["lo"]], dtype=float)
        hi = np.array([np.nan if v is None else v for v in c["ci"]["hi"]], dtype=float)

        floor = 1e-4  # log-y floor so zero/None values stay plottable
        y_t, y_m = np.maximum(y_t, floor), np.maximum(y_m, floor)
        lo, hi = np.maximum(lo, floor), np.maximum(hi, floor)
        ax.set_facecolor(SURFACE)
        ax.fill_between(x, lo, hi, color=SERIES_BLUE, alpha=0.18, linewidth=0)
        ax.plot(x, y_m, color=SERIES_BLUE, linewidth=2, label="model mean")
        ax.plot(x, y_t, color=INK, linewidth=2, linestyle=(0, (4, 2)),
                label="simulated ground truth")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{row['capacity']} × {row['budget']}   score {st['score']:.2f}",
                     fontsize=10, color=INK, loc="left")
        ax.plot([0.965], [0.92], transform=ax.transAxes, marker="o", markersize=8,
                color=STATUS[st["light"]], markeredgecolor=SURFACE, clip_on=False)
        if st["diverged"]:
            ax.text(0.96, 0.80, "diverged", transform=ax.transAxes, fontsize=8,
                    color=INK_MUTED, ha="right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=8)

    for ax in np.ravel(axes)[len(models):]:
        ax.set_visible(False)

    fig.supxlabel(c["x_label"], fontsize=10, color=INK_MUTED)
    fig.supylabel("‖Δx(H)‖ of poked body (m, log)", fontsize=10, color=INK_MUTED)
    np.ravel(axes)[0].legend(loc="upper left", fontsize=9, frameon=False,
                             labelcolor=INK)
    fig.suptitle(f"{scene}: impulse-response curves vs simulated ground truth  "
                 f"(band = K-sample 10–90%)",
                 fontsize=12, color=INK)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{scene}-{subtest}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="billiards")
    args = ap.parse_args()
    print(response_small_multiples(args.scene))
