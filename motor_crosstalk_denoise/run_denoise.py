#!/usr/bin/env python3
"""Standalone motor-crosstalk denoising for a packed-12 PA volume.

Edit the settings block and click "Run Python File" in VS Code, or pass the
equivalent command-line arguments. The raw acquisition is opened read-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pa_denoising import (  # noqa: E402
    TemplateXcorrProjectionResult,
    load_packed12_template_xcorr_mip_projection,
    load_template_csv,
)


# ------------------------- Direct-run settings -------------------------
# Edit these two paths, then click "Run Python File" in VS Code.
INPUT_BIN = Path(
    "/mnt/c/neuws_data/raw/2026_0821/"
    "LaserData_20260821-032156_2_600_600_512_PA1.bin"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "2026_08_21_example"

HEIGHT = 600
WIDTH = 600
DEPTH = 512
BASELINE_ADC = 2048.0
CORRELATION_THRESHOLD = 0.70
MINIMUM_FITTED_PEAK_ADC = 80.0
CHUNK_ROWS = 10
TEMPLATE_CSV = (
    Path(__file__).resolve().parent
    / "template"
    / "motor_crosstalk_template_v1.csv"
)
# ----------------------------------------------------------------------


def _isolated_metrics(before: np.ndarray, after: np.ndarray) -> dict:
    local_before = median_filter(before, size=5, mode="reflect")
    local_after = median_filter(after, size=5, mode="reflect")
    residual_before = np.maximum(before - local_before, 0.0)
    residual_after = np.maximum(after - local_after, 0.0)
    threshold = float(np.percentile(residual_before, 99.7))
    candidates = residual_before >= threshold
    survivors = candidates & (residual_after >= threshold)
    original_count = int(candidates.sum())
    survivor_count = int(survivors.sum())
    return {
        "definition": "positive 5x5 local-median residual above original P99.7",
        "threshold_adc": threshold,
        "original_candidate_count": original_count,
        "surviving_candidate_count": survivor_count,
        "removed_candidate_count": original_count - survivor_count,
        "removed_fraction": (
            (original_count - survivor_count) / original_count
            if original_count
            else 0.0
        ),
    }


def _save_comparison(
    destination: Path, result: TemplateXcorrProjectionResult
) -> None:
    before = result.original_projection
    after = result.projection
    removed = np.maximum(before - after, 0.0)
    positive = before[before > 0]
    low, high = (
        np.percentile(positive, (1.0, 99.7)) if positive.size else (0.0, 1.0)
    )
    removed_high = max(1.0, float(np.percentile(removed, 99.7)))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), constrained_layout=True)
    panels = (
        (before, "A. Original full-depth MIP", "gray", low, high),
        (after, "B. Template-xcorr cleaned MIP", "gray", low, high),
        (removed, "C. Removed crosstalk amplitude", "magma", 0.0, removed_high),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        shown = axis.imshow(
            image, cmap=cmap, origin="lower", vmin=float(vmin), vmax=float(vmax)
        )
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        fig.colorbar(shown, ax=axis, shrink=0.82, label="ADC")
    fig.suptitle("Motor-crosstalk shifted-template subtraction (no PD)")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _save_diagnostics(
    destination: Path, result: TemplateXcorrProjectionResult
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), constrained_layout=True)
    panels = (
        (np.abs(result.correlation_map), "A. Maximum absolute NCC", "viridis", 0, 1),
        (result.center_map, "B. Matched template center", "turbo", 0, 511),
        (result.matched_map, "C. Subtraction mask", "gray", 0, 1),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        fig.colorbar(shown, ax=axis, shrink=0.82)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _metrics(
    result: TemplateXcorrProjectionResult,
    *,
    source: Path,
    template_path: Path,
    args: argparse.Namespace,
) -> dict:
    matched = result.matched_map
    matched_count = int(matched.sum())
    correlations = np.abs(result.correlation_map[matched])
    centers = result.center_map[matched]
    return {
        "method": "shifted template normalized-xcorr + least-squares subtraction",
        "source_pa": str(source),
        "source_modified": False,
        "shape_height_width_depth": [args.height, args.width, args.depth],
        "template": str(template_path),
        "template_length_samples": int(load_template_csv(template_path).size),
        "correlation_threshold_absolute": args.correlation_threshold,
        "minimum_fitted_peak_adc": args.minimum_fitted_peak_adc,
        "baseline_adc": args.baseline_adc,
        "pd_used": False,
        "time_gate_used": False,
        "matched_pixel_count": matched_count,
        "matched_pixel_fraction": float(matched.mean()),
        "matched_abs_ncc_p10_p50_p90": (
            [float(value) for value in np.percentile(correlations, (10, 50, 90))]
            if matched_count
            else []
        ),
        "matched_center_p1_p50_p99": (
            [float(value) for value in np.percentile(centers, (1, 50, 99))]
            if matched_count
            else []
        ),
        "median_projection_before_after_adc": [
            float(np.median(result.original_projection)),
            float(np.median(result.projection)),
        ],
        "p99_7_projection_before_after_adc": [
            float(np.percentile(result.original_projection, 99.7)),
            float(np.percentile(result.projection, 99.7)),
        ],
        "isolated_bright_point_check": _isolated_metrics(
            result.original_projection, result.projection
        ),
    }


def denoise(args: argparse.Namespace) -> Path:
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PA BIN 不存在：{source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{output}")
    if output.exists():
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    source_state = (source.stat().st_size, source.stat().st_mtime_ns)
    try:
        template = load_template_csv(template_path)
        result = load_packed12_template_xcorr_mip_projection(
            source,
            template=template,
            height=args.height,
            width=args.width,
            depth=args.depth,
            baseline_adc=args.baseline_adc,
            correlation_threshold=args.correlation_threshold,
            minimum_fitted_peak_adc=args.minimum_fitted_peak_adc,
            chunk_rows=args.chunk_rows,
        )
        tifffile.imwrite(
            temporary / "01_original_full_depth_mip_float32.tif",
            result.original_projection,
        )
        tifffile.imwrite(
            temporary / "02_denoised_mip_float32.tif", result.projection
        )
        tifffile.imwrite(
            temporary / "03_max_abs_ncc_float32.tif",
            np.abs(result.correlation_map).astype(np.float32),
        )
        tifffile.imwrite(
            temporary / "04_matched_center_uint16.tif", result.center_map
        )
        tifffile.imwrite(
            temporary / "05_fitted_coefficient_float32.tif", result.coefficient_map
        )
        tifffile.imwrite(
            temporary / "06_subtraction_mask_uint8.tif",
            result.matched_map.astype(np.uint8),
        )
        _save_comparison(temporary / "07_before_after.png", result)
        _save_diagnostics(temporary / "08_match_diagnostics.png", result)
        metrics = _metrics(
            result,
            source=source,
            template_path=template_path,
            args=args,
        )
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if (source.stat().st_size, source.stat().st_mtime_ns) != source_state:
            raise RuntimeError("原始 PA BIN 在处理期间发生变化。")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"去噪完成：{output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_BIN))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--template", default=str(TEMPLATE_CSV))
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--baseline-adc", type=float, default=BASELINE_ADC)
    parser.add_argument(
        "--correlation-threshold", type=float, default=CORRELATION_THRESHOLD
    )
    parser.add_argument(
        "--minimum-fitted-peak-adc", type=float, default=MINIMUM_FITTED_PEAK_ADC
    )
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return parser


if __name__ == "__main__":
    denoise(build_parser().parse_args())
