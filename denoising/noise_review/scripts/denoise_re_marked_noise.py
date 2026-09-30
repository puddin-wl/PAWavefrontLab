#!/usr/bin/env python3
"""Denoise re using the six manually marked broad-transient PA A-lines."""

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

from dataio.packed12 import decode_packed12
from denoising.pa_denoising import correlation_terms
from denoising.noise_review.scripts.build_manual_noise_template import fit_template
from denoising.noise_review.scripts.review_noise_alines import _read_aline


SOURCE = Path("/mnt/c/neuws_data/raw/2026-09-19/re_20260919-123731_1_600_600_512_PA1.bin")
OUTPUT = PROJECT_ROOT / "data/2026-09-19_manual_noise_review/re_marked_denoised"
NOISE_POINTS = [
    ("N1", 17, 325), ("N2", 494, 466), ("N3", 479, 329),
    ("N4", 466, 379), ("N5", 585, 18), ("N6", 291, 370),
]
SIGNAL_POINTS = [
    ("S1", 80, 77), ("S2", 102, 24), ("S3", 150, 70),
    ("S4", 300, 54), ("S5", 400, 79), ("S6", 200, 288),
]
SIGNAL_WINDOW = (222, 313)
NCC_THRESHOLD = 0.90
MINIMUM_FITTED_PEAK_ADC = 80.0


def clean_gated(raw: np.ndarray, template: np.ndarray):
    values = raw.astype(np.float32, copy=False)
    centered = values - np.median(values, axis=1, keepdims=True)
    correlation, template_energy, dot = correlation_terms(centered, template, depth=512)
    coefficients_by_center = dot / np.maximum(template_energy[None, :], np.float32(1e-12))
    valid = np.abs(coefficients_by_center) >= MINIMUM_FITTED_PEAK_ADC
    valid[:, SIGNAL_WINDOW[0] : SIGNAL_WINDOW[1]] = False
    score = np.where(valid, np.abs(correlation), -1.0)
    centers = np.argmax(score, axis=1).astype(np.int16)
    rows = np.arange(values.shape[0])
    best_ncc = correlation[rows, centers]
    coefficients = coefficients_by_center[rows, centers]
    matched = np.any(valid, axis=1) & (np.abs(best_ncc) >= NCC_THRESHOLD)

    cleaned = values.copy()
    selected = np.flatnonzero(matched)
    if selected.size:
        half_width = template.size // 2
        selected_centers = centers[selected].astype(np.int32)
        indices = np.arange(512, dtype=np.int32)[None, :] - selected_centers[:, None] + half_width
        supported = (indices >= 0) & (indices < template.size)
        shifted = np.zeros((selected.size, 512), dtype=np.float32)
        clipped = np.clip(indices, 0, template.size - 1)
        shifted[supported] = template[clipped[supported]]
        cleaned[selected] -= coefficients[selected, None] * shifted
    return cleaned, best_ncc, centers, coefficients, matched


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    noise_traces = [
        _read_aline(SOURCE, y=y, x=x, width=600, depth=512)
        for _, y, x in NOISE_POINTS
    ]
    template, aligned = fit_template(noise_traces, half_width=180)
    with (OUTPUT / "re_broad_noise_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_sample", "template_relative_adc"])
        for offset, value in zip(range(-180, 181), template):
            writer.writerow([offset, f"{float(value):.9f}"])

    original_mip = np.empty((600, 600), dtype=np.float32)
    cleaned_mip = np.empty_like(original_mip)
    ncc_map = np.empty_like(original_mip)
    center_map = np.empty((600, 600), dtype=np.uint16)
    matched_map = np.empty((600, 600), dtype=bool)
    with SOURCE.open("rb") as stream:
        for row_start in range(0, 600, 10):
            row_count = min(10, 600 - row_start)
            byte_count = row_count * 600 * 512 * 3 // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=byte_count)
            raw = decode_packed12(packed).reshape(row_count * 600, 512)
            cleaned, ncc, centers, _, matched = clean_gated(raw, template)
            original_mip[row_start : row_start + row_count] = np.maximum(
                np.max(raw, axis=1).reshape(row_count, 600).astype(np.float32) - 2048.0, 0
            )
            cleaned_mip[row_start : row_start + row_count] = np.maximum(
                np.max(cleaned, axis=1).reshape(row_count, 600) - 2048.0, 0
            )
            ncc_map[row_start : row_start + row_count] = np.abs(ncc).reshape(row_count, 600)
            center_map[row_start : row_start + row_count] = centers.reshape(row_count, 600)
            matched_map[row_start : row_start + row_count] = matched.reshape(row_count, 600)

    tifffile.imwrite(OUTPUT / "01_re_original_mip_float32.tif", original_mip)
    tifffile.imwrite(OUTPUT / "02_re_denoised_mip_float32.tif", cleaned_mip)
    tifffile.imwrite(OUTPUT / "03_re_max_abs_ncc_float32.tif", ncc_map)
    tifffile.imwrite(OUTPUT / "04_re_matched_center_uint16.tif", center_map)
    tifffile.imwrite(OUTPUT / "05_re_subtraction_mask_uint8.tif", matched_map.astype(np.uint8))

    removed = np.maximum(original_mip - cleaned_mip, 0)
    positive = original_mip[original_mip > 0]
    low, high = np.percentile(positive, (1, 99.7))
    removed_high = max(1.0, float(np.percentile(removed, 99.7)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    panels = (
        (original_mip, "A. re original MIP", "gray", low, high),
        (cleaned_mip, "B. re broad-noise denoised MIP", "gray", low, high),
        (removed, "C. removed broad transient", "magma", 0, removed_high),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        fig.colorbar(shown, ax=axis, shrink=0.82, label="ADC")
    fig.suptitle("re-specific denoising: |NCC|>=0.90, fitted peak>=80 ADC, center outside 222:313")
    fig.savefig(OUTPUT / "06_re_before_after.png", dpi=190)
    plt.close(fig)

    point_metrics = []
    fig, axes = plt.subplots(6, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    for column, (category, points) in enumerate((("noise", NOISE_POINTS), ("signal", SIGNAL_POINTS))):
        for row, (label, y, x) in enumerate(points):
            raw = _read_aline(SOURCE, y=y, x=x, width=600, depth=512) + 2048.0
            cleaned, ncc, centers, _, matched = clean_gated(raw[None, :], template)
            before = raw - 2048.0
            after = cleaned[0] - 2048.0
            axis = axes[row, column]
            axis.plot(before, color="#999999", linewidth=0.9, label="before")
            axis.plot(after, color="#d62728" if category == "noise" else "#0077bb", linewidth=1.05, label="after")
            axis.axvspan(*SIGNAL_WINDOW, color="#6abf69", alpha=0.16)
            axis.axhline(0, color="#777777", linewidth=0.6)
            axis.grid(True, alpha=0.35, linewidth=0.5)
            axis.set_xlim(0, 511)
            axis.set_ylabel("relative ADC")
            axis.set_title(
                f"{label} {category}: matched={bool(matched[0])}, center={int(centers[0])}, |NCC|={abs(float(ncc[0])):.3f}"
            )
            axis.legend(frameon=False, fontsize=8)
            point_metrics.append(
                {
                    "label": label, "category": category, "y": y, "x": x,
                    "matched": bool(matched[0]), "matched_center": int(centers[0]),
                    "absolute_ncc": abs(float(ncc[0])),
                    "maximum_absolute_change_adc": float(np.max(np.abs(after - before))),
                }
            )
    axes[-1, 0].set_xlabel("A-line sample")
    axes[-1, 1].set_xlabel("A-line sample")
    fig.suptitle("Marked re PA A-lines before and after re-specific subtraction")
    fig.savefig(OUTPUT / "07_re_marked_palines_before_after.png", dpi=190)
    plt.close(fig)

    local_before = median_filter(original_mip, size=5, mode="reflect")
    local_after = median_filter(cleaned_mip, size=5, mode="reflect")
    residual_before = np.maximum(original_mip - local_before, 0)
    residual_after = np.maximum(cleaned_mip - local_after, 0)
    threshold = float(np.percentile(residual_before, 99.7))
    candidates = residual_before >= threshold
    survivors = candidates & (residual_after >= threshold)
    metrics = {
        "source": str(SOURCE),
        "method": "manual re broad-noise template + gated NCC/LS subtraction",
        "correlation_threshold_absolute": NCC_THRESHOLD,
        "minimum_fitted_peak_adc": MINIMUM_FITTED_PEAK_ADC,
        "excluded_match_center_window_start_stop": list(SIGNAL_WINDOW),
        "matched_pixel_count": int(matched_map.sum()),
        "matched_pixel_fraction": float(matched_map.mean()),
        "isolated_candidate_threshold_adc": threshold,
        "isolated_candidates_before": int(candidates.sum()),
        "isolated_candidates_surviving": int(survivors.sum()),
        "isolated_candidates_removed": int(candidates.sum() - survivors.sum()),
        "marked_points": point_metrics,
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
