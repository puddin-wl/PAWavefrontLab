#!/usr/bin/env python3
"""Mark representative noise/signal points in re and plot their PA A-lines."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from denoising.noise_review.scripts.review_noise_alines import _read_aline


SOURCE = Path("/mnt/c/neuws_data/raw/2026-09-19/re_20260919-123731_1_600_600_512_PA1.bin")
PROJECTION = PROJECT_ROOT / "data/2026-09-19_manual_noise_review/denoise_re/01_original_full_depth_mip_float32.tif"
OUTPUT = PROJECT_ROOT / "data/2026-09-19_manual_noise_review/re_marked_comparison"

NOISE_POINTS = [
    ("N1", 17, 325),
    ("N2", 494, 466),
    ("N3", 479, 329),
    ("N4", 466, 379),
    ("N5", 585, 18),
    ("N6", 291, 370),
]
SIGNAL_POINTS = [
    ("S1", 80, 77),
    ("S2", 102, 24),
    ("S3", 150, 70),
    ("S4", 300, 54),
    ("S5", 400, 79),
    ("S6", 200, 288),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    projection = tifffile.imread(PROJECTION).astype(np.float32)
    positive = projection[projection > 0]
    low, high = np.percentile(positive, (1, 99.7))

    fig, axis = plt.subplots(figsize=(9.2, 8.2), constrained_layout=True)
    shown = axis.imshow(projection, cmap="gray", origin="lower", vmin=low, vmax=high)
    for label, y, x in NOISE_POINTS:
        axis.scatter(x, y, s=90, facecolors="none", edgecolors="#ff3030", linewidths=1.7)
        axis.text(x + 7, y + 7, label, color="#ff3030", fontsize=10, weight="bold")
    for label, y, x in SIGNAL_POINTS:
        axis.scatter(x, y, s=90, facecolors="none", edgecolors="#00d4ff", linewidths=1.7)
        axis.text(x + 7, y + 7, label, color="#00d4ff", fontsize=10, weight="bold")
    axis.set_title("re: manually marked PA noise points (red) and vessel signals (cyan)")
    axis.set_xlabel("X pixel")
    axis.set_ylabel("Y pixel")
    fig.colorbar(shown, ax=axis, label="full-depth positive MIP (ADC)")
    fig.savefig(OUTPUT / "re_marked_noise_and_signal_points.png", dpi=190)
    plt.close(fig)

    samples = np.arange(512)
    fig, axes = plt.subplots(6, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    records = []
    trace_rows = []
    for column, (category, points, color) in enumerate(
        (("noise", NOISE_POINTS, "#d62728"), ("signal", SIGNAL_POINTS, "#0077bb"))
    ):
        for row, (label, y, x) in enumerate(points):
            trace = _read_aline(SOURCE, y=y, x=x, width=600, depth=512)
            peak = int(np.argmax(trace))
            amplitude = float(trace[peak])
            axis = axes[row, column]
            axis.plot(samples, trace, color=color, linewidth=1.05)
            axis.axvspan(222, 313, color="#6abf69", alpha=0.16)
            axis.axvline(peak, color=color, linewidth=0.8, linestyle="--")
            axis.axhline(0, color="#777777", linewidth=0.6)
            axis.grid(True, alpha=0.35, linewidth=0.5)
            axis.set_xlim(0, 511)
            axis.set_ylabel("relative ADC")
            axis.set_title(f"{label} {category}: (y={y}, x={x}), peak={peak}, +{amplitude:.0f} ADC")
            records.append(
                {
                    "label": label,
                    "category": category,
                    "y": y,
                    "x": x,
                    "peak_index": peak,
                    "peak_relative_adc": amplitude,
                    "peak_inside_signal_window_222_313": 222 <= peak < 313,
                }
            )
            trace_rows.append((label, category, y, x, trace))
    axes[-1, 0].set_xlabel("A-line sample index")
    axes[-1, 1].set_xlabel("A-line sample index")
    fig.suptitle(
        "re PA A-lines: isolated noise versus spatially continuous vessel signal\n"
        "green band = signal depth 222:313",
        fontsize=15,
    )
    fig.savefig(OUTPUT / "re_noise_vs_signal_palines.png", dpi=190)
    plt.close(fig)

    with (OUTPUT / "re_selected_palines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "category", "y", "x", "sample_index", "relative_adc"])
        for label, category, y, x, trace in trace_rows:
            for sample, value in enumerate(trace):
                writer.writerow([label, category, y, x, sample, f"{float(value):.6f}"])
    (OUTPUT / "re_selected_points.json").write_text(
        json.dumps({"source": str(SOURCE), "points": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
