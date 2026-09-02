#!/usr/bin/env python3
"""Fit a shifted PA-noise template and test normalized-xcorr subtraction.

The template is built from the saved representative noise A-lines after their
positive peaks are aligned.  The full volume is processed by rows.  For every
A-line, normalized cross-correlation estimates the template center; a least-
squares coefficient estimates its amplitude.  Only strong, high-correlation
matches are subtracted.  The raw acquisition file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter
from scipy.signal.windows import tukey


WORK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pa_denoising import clean_traces, decode_packed12  # noqa: E402


DEFAULT_ANALYSIS = WORK_DIR / "aline_analysis" / "analysis.json"
DEFAULT_OUTPUT = WORK_DIR / "results" / "template_xcorr_v1"
DEFAULT_COHERENT_REFERENCE = (
    WORK_DIR / "time_gate_test" / "01_excess_rms_without_pd_float32.tif"
)
BITS_PER_SAMPLE = 12


def read_aline(source: Path, *, y: int, x: int, width: int, depth: int) -> np.ndarray:
    if depth % 2:
        raise ValueError("depth must be even")
    offset = (y * width + x) * depth * 3 // 2
    byte_count = depth * 3 // 2
    with source.open("rb") as stream:
        stream.seek(offset)
        packed = np.fromfile(stream, dtype=np.uint8, count=byte_count)
    if packed.size != byte_count:
        raise EOFError(f"Could not read A-line ({y}, {x}) from {source}")
    return decode_packed12(packed)


def fit_template(
    traces: list[np.ndarray], *, half_width: int
) -> tuple[np.ndarray, list[int], list[float], np.ndarray]:
    """Align positive peaks and return a robust median template."""
    if len(traces) < 3:
        raise ValueError("At least three noise traces are required")
    length = 2 * half_width + 1
    centered = [trace.astype(np.float32) - np.median(trace) for trace in traces]
    peaks = [int(np.argmax(trace)) for trace in centered]
    peak_amplitudes = np.asarray(
        [trace[peak] for trace, peak in zip(centered, peaks)], dtype=np.float32
    )
    if np.any(peak_amplitudes <= 0):
        raise ValueError("Every template trace must have a positive peak")
    target_peak = float(np.median(peak_amplitudes))

    aligned = np.full((len(centered), length), np.nan, dtype=np.float32)
    relative = np.arange(-half_width, half_width + 1)
    for row, (trace, peak, amplitude) in enumerate(
        zip(centered, peaks, peak_amplitudes)
    ):
        source_indices = relative + peak
        valid = (source_indices >= 0) & (source_indices < trace.size)
        aligned[row, valid] = trace[source_indices[valid]] * target_peak / amplitude

    with np.errstate(invalid="ignore"):
        template = np.nanmedian(aligned, axis=0).astype(np.float32)
    if not np.isfinite(template).all():
        raise RuntimeError("Template contains unsupported relative positions")

    edge_count = min(15, max(3, length // 20))
    left = float(np.median(template[:edge_count]))
    right = float(np.median(template[-edge_count:]))
    template -= np.linspace(left, right, length, dtype=np.float32)
    template *= tukey(length, alpha=0.15).astype(np.float32)
    if int(np.argmax(template)) != half_width:
        raise RuntimeError("Fitted template peak moved away from its aligned center")
    return template, peaks, peak_amplitudes.astype(float).tolist(), aligned


def isolated_metrics(before: np.ndarray, after: np.ndarray) -> dict:
    local_before = median_filter(before, size=5, mode="reflect")
    local_after = median_filter(after, size=5, mode="reflect")
    residual_before = np.maximum(before - local_before, 0.0)
    residual_after = np.maximum(after - local_after, 0.0)
    threshold = float(np.percentile(residual_before, 99.7))
    before_mask = residual_before >= threshold
    after_at_same_locations = before_mask & (residual_after >= threshold)
    before_count = int(before_mask.sum())
    survivor_count = int(after_at_same_locations.sum())
    return {
        "definition": "positive 5x5 local-median residual above original P99.7",
        "threshold_adc": threshold,
        "original_candidate_count": before_count,
        "surviving_at_same_locations_and_threshold": survivor_count,
        "removed_count": before_count - survivor_count,
        "removed_fraction": (
            (before_count - survivor_count) / before_count if before_count else 0.0
        ),
    }


def save_template_figure(
    destination: Path,
    *,
    aligned: np.ndarray,
    template: np.ndarray,
    labels: list[str],
    half_width: int,
) -> None:
    relative = np.arange(-half_width, half_width + 1)
    figure, axes = plt.subplots(
        len(labels) + 1,
        1,
        figsize=(11.5, 2.0 * (len(labels) + 1)),
        sharex=True,
        constrained_layout=True,
    )
    finite_values = aligned[np.isfinite(aligned)]
    low, high = np.percentile(finite_values, [0.2, 99.8])
    padding = max(10.0, float(high - low) * 0.08)
    for axis, label, trace in zip(axes[:-1], labels, aligned):
        axis.plot(relative, trace, color="#d62728", linewidth=1.0)
        axis.axvline(0, color="#555555", linewidth=0.7)
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set_ylim(float(low - padding), float(high + padding))
        axis.set_ylabel(label)
        axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
    axes[-1].plot(relative, template, color="#1f77b4", linewidth=1.6)
    axes[-1].axvline(0, color="#555555", linewidth=0.7)
    axes[-1].axhline(0, color="#777777", linewidth=0.6)
    axes[-1].set_ylim(float(low - padding), float(high + padding))
    axes[-1].set_ylabel("Template")
    axes[-1].set_xlabel("Sample offset from aligned noise peak")
    axes[-1].grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
    figure.suptitle("Aligned noise A-lines and robust fitted template")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_projection_report(
    destination: Path,
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    removed = np.maximum(before - after, 0.0)
    positive = before[before > 0]
    vmin, vmax = np.percentile(positive, [1.0, 99.7]) if positive.size else (0.0, 1.0)
    removed_max = max(1.0, float(np.percentile(removed, 99.7)))
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), constrained_layout=True)
    panels = (
        (before, "A. Original full-depth MIP", "gray", float(vmin), float(vmax)),
        (after, "B. Template-xcorr cleaned MIP", "gray", float(vmin), float(vmax)),
        (removed, "C. Removed projection amplitude", "magma", 0.0, removed_max),
    )
    for axis, (image, title, cmap, low, high) in zip(axes, panels):
        view = axis.imshow(image, cmap=cmap, origin="lower", vmin=low, vmax=high)
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        figure.colorbar(view, ax=axis, shrink=0.82, label="ADC")
    figure.suptitle("Shifted-template noise subtraction, no PD correction")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_diagnostic_maps(
    destination: Path,
    *,
    correlation: np.ndarray,
    centers: np.ndarray,
    matched: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), constrained_layout=True)
    panels = (
        (np.abs(correlation), "A. Maximum absolute NCC", "viridis", 0.0, 1.0),
        (centers, "B. Matched template center", "turbo", 0.0, 511.0),
        (matched.astype(float), "C. Subtraction mask", "gray", 0.0, 1.0),
    )
    for axis, (image, title, cmap, low, high) in zip(axes, panels):
        view = axis.imshow(image, cmap=cmap, origin="lower", vmin=low, vmax=high)
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        figure.colorbar(view, ax=axis, shrink=0.82)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_selected_noise_cleaning(
    destination: Path,
    *,
    labels: list[str],
    originals: list[np.ndarray],
    fitted_components: list[np.ndarray],
    residuals: list[np.ndarray],
) -> None:
    samples = np.arange(originals[0].size)
    combined = np.concatenate(originals + fitted_components + residuals)
    low, high = np.percentile(combined, [0.2, 99.8])
    padding = max(10.0, float(high - low) * 0.08)
    figure, axes = plt.subplots(
        len(labels),
        3,
        figsize=(16.0, 2.6 * len(labels)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, (label, original, fitted, residual) in enumerate(
        zip(labels, originals, fitted_components, residuals)
    ):
        panels = (
            (original, "Original", "#d62728"),
            (fitted, "Fitted template component", "#1f77b4"),
            (residual, "After subtraction", "#333333"),
        )
        for column, (trace, title, color) in enumerate(panels):
            axis = axes[row, column]
            axis.plot(samples, trace, color=color, linewidth=1.0)
            axis.axvspan(250, 320, color="#b8b8b8", alpha=0.20)
            axis.axhline(0, color="#777777", linewidth=0.6)
            axis.set_xlim(0, originals[0].size - 1)
            axis.set_ylim(float(low - padding), float(high + padding))
            axis.set_title(f"{label} — {title}")
            axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
            if column == 0:
                axis.set_ylabel("Relative ADC")
    for axis in axes[-1]:
        axis.set_xlabel("A-line sample index")
    figure.suptitle("Selected noise A-lines: original, fitted component, and residual")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coherent-reference", type=Path, default=DEFAULT_COHERENT_REFERENCE
    )
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--baseline-adc", type=float, default=2048.0)
    parser.add_argument("--template-half-width", type=int, default=180)
    parser.add_argument("--correlation-threshold", type=float, default=0.70)
    parser.add_argument("--minimum-fitted-peak-adc", type=float, default=80.0)
    parser.add_argument("--chunk-rows", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.correlation_threshold <= 1:
        raise ValueError("correlation threshold must be in (0, 1]")
    if args.minimum_fitted_peak_adc < 0:
        raise ValueError("minimum fitted peak must be nonnegative")
    if args.chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    analysis_path = args.analysis.expanduser().resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    pa_source = Path(analysis["source_pa"])
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    noise_points = sorted(
        (point for point in analysis["selected_points"] if point["category"] == "noise"),
        key=lambda point: point["label"],
    )
    signal_points = sorted(
        (point for point in analysis["selected_points"] if point["category"] == "signal"),
        key=lambda point: point["label"],
    )
    template_traces = [
        read_aline(
            pa_source,
            y=int(point["y"]),
            x=int(point["x"]),
            width=args.width,
            depth=args.depth,
        )
        for point in noise_points
    ]
    template, template_peaks, peak_amplitudes, aligned = fit_template(
        template_traces, half_width=args.template_half_width
    )
    labels = [point["label"] for point in noise_points]
    save_template_figure(
        output / "01_fitted_noise_template.png",
        aligned=aligned,
        template=template,
        labels=labels,
        half_width=args.template_half_width,
    )
    with (output / "fitted_noise_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_sample", "template_relative_adc"])
        for relative, value in zip(
            range(-args.template_half_width, args.template_half_width + 1), template
        ):
            writer.writerow([relative, f"{float(value):.6f}"])

    sample_count = args.height * args.width * args.depth
    expected_bytes = (sample_count * BITS_PER_SAMPLE + 7) // 8
    if pa_source.stat().st_size != expected_bytes:
        raise ValueError("PA source size does not match the requested dimensions")
    samples_per_row = args.width * args.depth
    if samples_per_row % 2:
        raise ValueError("width * depth must be even")

    before = np.empty((args.height, args.width), dtype=np.float32)
    after = np.empty_like(before)
    correlation_map = np.empty_like(before)
    center_map = np.empty((args.height, args.width), dtype=np.uint16)
    coefficient_map = np.empty_like(before)
    matched_map = np.zeros((args.height, args.width), dtype=bool)

    with pa_source.open("rb") as stream:
        for row_start in range(0, args.height, args.chunk_rows):
            row_stop = min(row_start + args.chunk_rows, args.height)
            row_count = row_stop - row_start
            pair_count = row_count * samples_per_row // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
            if packed.size != pair_count * 3:
                raise EOFError(f"Unexpected EOF at rows {row_start}:{row_stop}")
            raw = decode_packed12(packed).reshape(row_count * args.width, args.depth)
            cleaned, best_corr, centers, coefficients, matched = clean_traces(
                raw,
                template,
                correlation_threshold=args.correlation_threshold,
                minimum_fitted_peak_adc=args.minimum_fitted_peak_adc,
            )
            before[row_start:row_stop] = np.maximum(
                np.max(raw, axis=1).reshape(row_count, args.width).astype(np.float32)
                - args.baseline_adc,
                0.0,
            )
            after[row_start:row_stop] = np.maximum(
                np.max(cleaned, axis=1).reshape(row_count, args.width)
                - args.baseline_adc,
                0.0,
            )
            correlation_map[row_start:row_stop] = best_corr.reshape(row_count, args.width)
            center_map[row_start:row_stop] = centers.reshape(row_count, args.width).astype(np.uint16)
            coefficient_map[row_start:row_stop] = coefficients.reshape(row_count, args.width)
            matched_map[row_start:row_stop] = matched.reshape(row_count, args.width)

    tifffile.imwrite(output / "02_original_full_depth_mip_float32.tif", before)
    tifffile.imwrite(output / "03_template_cleaned_mip_float32.tif", after)
    tifffile.imwrite(output / "04_max_correlation_float32.tif", correlation_map)
    tifffile.imwrite(output / "05_matched_center_uint16.tif", center_map)
    tifffile.imwrite(output / "06_fitted_coefficient_float32.tif", coefficient_map)
    save_projection_report(
        output / "07_before_after_projection.png", before=before, after=after
    )
    save_diagnostic_maps(
        output / "08_match_diagnostics.png",
        correlation=correlation_map,
        centers=center_map,
        matched=matched_map,
    )

    point_diagnostics = []
    noise_originals: list[np.ndarray] = []
    noise_fitted_components: list[np.ndarray] = []
    noise_residuals: list[np.ndarray] = []
    for category, points in (("noise", noise_points), ("signal", signal_points)):
        for point in points:
            raw = read_aline(
                pa_source,
                y=int(point["y"]),
                x=int(point["x"]),
                width=args.width,
                depth=args.depth,
            )[None, :]
            cleaned, corr, centers, coefficients, matched = clean_traces(
                raw,
                template,
                correlation_threshold=args.correlation_threshold,
                minimum_fitted_peak_adc=args.minimum_fitted_peak_adc,
            )
            if category == "noise":
                baseline = float(np.median(raw))
                noise_originals.append(raw[0].astype(np.float32) - baseline)
                noise_fitted_components.append(
                    raw[0].astype(np.float32) - cleaned[0]
                )
                noise_residuals.append(cleaned[0] - baseline)
            point_diagnostics.append(
                {
                    "category": category,
                    "label": point["label"],
                    "y": int(point["y"]),
                    "x": int(point["x"]),
                    "best_signed_correlation": float(corr[0]),
                    "matched_center_sample": int(centers[0]),
                    "least_squares_coefficient": float(coefficients[0]),
                    "subtracted": bool(matched[0]),
                    "positive_peak_before_adc": float(
                        max(float(np.max(raw)) - args.baseline_adc, 0.0)
                    ),
                    "positive_peak_after_adc": float(
                        max(float(np.max(cleaned)) - args.baseline_adc, 0.0)
                    ),
                }
            )

    save_selected_noise_cleaning(
        output / "09_selected_noise_cleaning.png",
        labels=labels,
        originals=noise_originals,
        fitted_components=noise_fitted_components,
        residuals=noise_residuals,
    )

    matched_centers = center_map[matched_map]
    edge_center_fraction = (
        float(np.mean((matched_centers < 10) | (matched_centers > args.depth - 11)))
        if matched_centers.size
        else 0.0
    )
    coherent_validation: dict = {"available": False}
    coherent_path = args.coherent_reference.expanduser().resolve()
    if coherent_path.is_file():
        coherent = np.asarray(tifffile.imread(coherent_path), dtype=np.float32)
        if coherent.shape != before.shape:
            raise ValueError(
                f"Coherent reference shape {coherent.shape} does not match {before.shape}"
            )
        coherent_threshold = float(np.percentile(coherent, 99.3))
        coherent_mask = coherent >= coherent_threshold
        absolute_change = np.abs(after[coherent_mask] - before[coherent_mask])
        coherent_validation = {
            "available": True,
            "source": str(coherent_path),
            "mask_definition": "top 0.7% coherent excess-RMS pixels",
            "pixel_count": int(coherent_mask.sum()),
            "matched_pixel_count": int(np.sum(matched_map & coherent_mask)),
            "matched_pixel_fraction": float(np.mean(matched_map[coherent_mask])),
            "absolute_projection_change_median_p95_max_adc": [
                float(np.median(absolute_change)),
                float(np.percentile(absolute_change, 95)),
                float(np.max(absolute_change)),
            ],
        }

    metrics = {
        "method": "robust shifted template + normalized cross-correlation + least-squares subtraction",
        "source_pa": str(pa_source),
        "source_modified": False,
        "shape_height_width_depth": [args.height, args.width, args.depth],
        "template_source_labels": labels,
        "template_alignment_positive_peak_samples": template_peaks,
        "template_source_peak_amplitudes_relative_to_trace_median_adc": peak_amplitudes,
        "template_half_width_samples": args.template_half_width,
        "template_length_samples": int(template.size),
        "correlation_threshold_absolute": args.correlation_threshold,
        "minimum_fitted_peak_adc": args.minimum_fitted_peak_adc,
        "matched_pixel_count": int(matched_map.sum()),
        "matched_pixel_fraction": float(matched_map.mean()),
        "matched_center_p1_p50_p99": [
            float(value)
            for value in np.percentile(center_map[matched_map], [1, 50, 99])
        ] if matched_map.any() else [],
        "matched_center_within_10_samples_of_record_edge_fraction": edge_center_fraction,
        "isolated_bright_point_check": isolated_metrics(before, after),
        "coherent_structure_validation": coherent_validation,
        "median_projection_before_after_adc": [
            float(np.median(before)),
            float(np.median(after)),
        ],
        "p99_7_projection_before_after_adc": [
            float(np.percentile(before, 99.7)),
            float(np.percentile(after, 99.7)),
        ],
        "selected_point_diagnostics": point_diagnostics,
        "caveat": (
            "The template was fitted from six selected noise examples. This first trial "
            "saves projection-level outputs only and does not rewrite the raw PA volume."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
