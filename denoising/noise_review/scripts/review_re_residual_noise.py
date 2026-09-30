#!/usr/bin/env python3
"""Review isolated residuals after the first re-specific denoising pass."""

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
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from denoising.pa_denoising import load_template_csv
from denoising.noise_review.scripts.denoise_re_marked_noise import SOURCE, clean_gated
from denoising.noise_review.scripts.review_noise_alines import _read_aline


INPUT_DIR = PROJECT_ROOT / "data/2026-09-19_manual_noise_review/re_marked_denoised"
OUTPUT = PROJECT_ROOT / "data/2026-09-19_manual_noise_review/re_residual_review"


def first_pass_trace(template: np.ndarray, y: int, x: int) -> np.ndarray:
    raw = _read_aline(SOURCE, y=y, x=x, width=600, depth=512) + 2048.0
    cleaned, _, _, _, _ = clean_gated(raw[None, :], template)
    return cleaned[0] - 2048.0


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = tifffile.imread(INPUT_DIR / "02_re_denoised_mip_float32.tif").astype(np.float32)
    template = load_template_csv(INPUT_DIR / "re_broad_noise_template.csv")
    local = median_filter(image, size=5, mode="reflect")
    residual = np.maximum(image - local, 0)
    order = np.argsort(residual.ravel())[::-1]
    selected = []
    for flat in order:
        y, x = np.unravel_index(int(flat), residual.shape)
        if x < 20 or y < 20 or x >= 580 or y >= 580:
            continue
        if any((y - old_y) ** 2 + (x - old_x) ** 2 < 12**2 for old_y, old_x in selected):
            continue
        selected.append((int(y), int(x)))
        if len(selected) == 48:
            break

    positive = image[image > 0]
    low, high = np.percentile(positive, (1, 99.7))
    records = []
    traces = []
    page_size = 8
    for page_start in range(0, len(selected), page_size):
        points = selected[page_start : page_start + page_size]
        fig, axes = plt.subplots(len(points), 2, figsize=(14, 2.5 * len(points)), constrained_layout=True)
        for row, (y, x) in enumerate(points):
            label = f"R{page_start + row + 1:03d}"
            crop = image[y - 18 : y + 19, x - 18 : x + 19]
            axes[row, 0].imshow(crop, cmap="gray", origin="lower", vmin=low, vmax=high)
            axes[row, 0].scatter([18], [18], s=48, facecolors="none", edgecolors="#ff3030")
            axes[row, 0].set_title(f"{label}: (y={y}, x={x}), residual={residual[y, x]:.0f} ADC")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            trace = first_pass_trace(template, y, x)
            neighbor_traces = [
                first_pass_trace(template, yy, xx)
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            ]
            neighbor = np.median(np.asarray(neighbor_traces), axis=0)
            peak = int(np.argmax(trace))
            axes[row, 1].plot(trace, color="#d62728", linewidth=1.0, label="candidate after pass 1")
            axes[row, 1].plot(neighbor, color="#1f77b4", linewidth=0.85, label="4-neighbor median")
            axes[row, 1].axvspan(222, 313, color="#6abf69", alpha=0.16)
            axes[row, 1].axvline(peak, color="#d62728", linestyle="--", linewidth=0.7)
            axes[row, 1].axhline(0, color="#777777", linewidth=0.6)
            axes[row, 1].grid(True, alpha=0.35, linewidth=0.5)
            axes[row, 1].set_xlim(0, 511)
            axes[row, 1].set_ylabel("relative ADC")
            axes[row, 1].set_title(f"peak={peak}, +{trace[peak]:.0f} ADC")
            axes[row, 1].legend(frameon=False, fontsize=8)
            records.append(
                {
                    "label": label, "y": y, "x": x,
                    "projection_adc_after_pass1": float(image[y, x]),
                    "local_residual_adc_after_pass1": float(residual[y, x]),
                    "peak_index_after_pass1": peak,
                    "peak_relative_adc_after_pass1": float(trace[peak]),
                    "peak_inside_signal_window": 222 <= peak < 313,
                }
            )
            traces.append((label, y, x, trace))
        axes[-1, 1].set_xlabel("A-line sample")
        fig.suptitle("re residual candidates after first denoising pass")
        fig.savefig(OUTPUT / f"residual_review_page_{page_start // page_size + 1:02d}.png", dpi=180)
        plt.close(fig)

    with (OUTPUT / "residual_candidate_palines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "y", "x", "sample_index", "relative_adc_after_pass1"])
        for label, y, x, trace in traces:
            for sample, value in enumerate(trace):
                writer.writerow([label, y, x, sample, f"{float(value):.6f}"])
    (OUTPUT / "residual_candidates.json").write_text(
        json.dumps({"source": str(SOURCE), "candidates": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
