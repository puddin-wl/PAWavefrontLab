#!/usr/bin/env python3
"""First-pass PA denoising by a fixed A-line response window.

The packed unsigned-12 PA volume has shape (height, width, depth).  For each
spatial point, this script compares the legacy maximum over all depth samples
with a maximum restricted to the known PA response window.  No PD correction
or spatial smoothing is applied in this version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import median_filter


DEFAULT_SOURCE = Path(
    "/mnt/c/neuws_data/raw/2026_0821/"
    "LaserData_20260821-032156_2_600_600_512_PA1.bin"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "fixed_window_v1"
BITS_PER_SAMPLE = 12


def decode_pairwise_ubit12_little(packed: np.ndarray) -> np.ndarray:
    """Decode three bytes into two little-endian unsigned 12-bit samples."""
    triples = np.asarray(packed, dtype=np.uint8).reshape(-1, 3)
    decoded = np.empty(triples.shape[0] * 2, dtype=np.uint16)
    decoded[0::2] = triples[:, 0].astype(np.uint16) | (
        (triples[:, 1] & 0x0F).astype(np.uint16) << 8
    )
    decoded[1::2] = (triples[:, 1] >> 4).astype(np.uint16) | (
        triples[:, 2].astype(np.uint16) << 4
    )
    return decoded


def make_projections(
    source: Path,
    *,
    height: int,
    width: int,
    depth: int,
    gate_start: int,
    gate_stop: int,
    baseline: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-depth MIP, gated MIP, and full-depth peak indices."""
    if not 0 <= gate_start < gate_stop <= depth:
        raise ValueError("The gate must satisfy 0 <= start < stop <= depth.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    sample_count = height * width * depth
    expected_bytes = (sample_count * BITS_PER_SAMPLE + 7) // 8
    actual_bytes = source.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch: expected {expected_bytes} bytes for "
            f"{height}x{width}x{depth} packed-12 data, got {actual_bytes}."
        )

    samples_per_row = width * depth
    if samples_per_row % 2:
        raise ValueError("width * depth must be even for pairwise packed-12 decoding.")

    full_mip = np.empty((height, width), dtype=np.float32)
    gated_mip = np.empty_like(full_mip)
    full_peak_index = np.empty((height, width), dtype=np.uint16)

    with source.open("rb") as stream:
        for row_start in range(0, height, chunk_rows):
            row_stop = min(row_start + chunk_rows, height)
            row_count = row_stop - row_start
            pair_count = row_count * samples_per_row // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
            if packed.size != pair_count * 3:
                raise EOFError(f"Unexpected end of file while reading rows {row_start}:{row_stop}.")

            volume = decode_pairwise_ubit12_little(packed).reshape(
                row_count, width, depth
            )
            peak_indices = np.argmax(volume, axis=2)
            full_peak_index[row_start:row_stop] = peak_indices.astype(np.uint16)
            full_mip[row_start:row_stop] = np.maximum(
                np.max(volume, axis=2).astype(np.float32) - baseline, 0.0
            )
            gated_mip[row_start:row_stop] = np.maximum(
                np.max(volume[:, :, gate_start:gate_stop], axis=2).astype(np.float32)
                - baseline,
                0.0,
            )

    return full_mip, gated_mip, full_peak_index


def robust_display_limits(reference: np.ndarray) -> tuple[float, float]:
    positive = reference[reference > 0]
    if positive.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(positive, [1.0, 99.7])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def isolated_bright_point_metrics(
    before: np.ndarray,
    after: np.ndarray,
    *,
    neighborhood: int = 5,
    candidate_percentile: float = 99.7,
) -> dict[str, float | int]:
    """Compare high positive local residuals using one pre-denoising threshold."""
    local_before = median_filter(before, size=neighborhood, mode="reflect")
    local_after = median_filter(after, size=neighborhood, mode="reflect")
    residual_before = np.maximum(before - local_before, 0.0)
    residual_after = np.maximum(after - local_after, 0.0)
    threshold = float(np.percentile(residual_before, candidate_percentile))
    if threshold <= 0:
        threshold = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    before_mask = residual_before >= threshold
    surviving_mask = before_mask & (residual_after >= threshold)
    before_count = int(before_mask.sum())
    surviving_count = int(surviving_mask.sum())
    removed_count = before_count - surviving_count
    removed_fraction = removed_count / before_count if before_count else 0.0
    return {
        "definition": (
            f"original positive residual above P{candidate_percentile:g} after "
            f"{neighborhood}x{neighborhood} median background subtraction"
        ),
        "threshold_adc": threshold,
        "original_candidate_count": before_count,
        "surviving_candidate_count_at_same_locations_and_threshold": surviving_count,
        "removed_candidate_count": removed_count,
        "removed_fraction": removed_fraction,
    }


def save_display(
    image: np.ndarray,
    destination: Path,
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    view = ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    fig.colorbar(view, ax=ax, label="Positive peak amplitude (ADC)")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_comparison(
    before: np.ndarray,
    after: np.ndarray,
    destination: Path,
    *,
    gate_start: int,
    gate_stop: int,
    vmin: float,
    vmax: float,
) -> None:
    removed = before - after
    removed_vmax = float(np.percentile(removed, 99.7))
    if removed_vmax <= 0:
        removed_vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), constrained_layout=True)
    panels = (
        (before, "A. Full-depth maximum (512 samples)", "gray", vmin, vmax),
        (
            after,
            f"B. Fixed-window maximum ({gate_start}:{gate_stop})",
            "gray",
            vmin,
            vmax,
        ),
        (removed, "C. Amplitude excluded by the gate", "magma", 0.0, removed_vmax),
    )
    for ax, (image, title, cmap, panel_min, panel_max) in zip(axes, panels):
        view = ax.imshow(
            image,
            cmap=cmap,
            origin="lower",
            vmin=panel_min,
            vmax=panel_max,
        )
        ax.set_title(title)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        fig.colorbar(view, ax=ax, shrink=0.82, label="ADC")
    fig.suptitle("PA fixed response-window denoising, no PD and no spatial smoothing")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--gate-start", type=int, default=250)
    parser.add_argument("--gate-stop", type=int, default=320)
    parser.add_argument("--baseline", type=int, default=2048)
    parser.add_argument("--sampling-rate-msps", type=float, default=500.0)
    parser.add_argument("--trigger-delay-us", type=float, default=12.3)
    parser.add_argument("--chunk-rows", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    before, after, peak_index = make_projections(
        source,
        height=args.height,
        width=args.width,
        depth=args.depth,
        gate_start=args.gate_start,
        gate_stop=args.gate_stop,
        baseline=args.baseline,
        chunk_rows=args.chunk_rows,
    )
    removed = before - after
    vmin, vmax = robust_display_limits(before)

    tifffile.imwrite(output / "01_full_depth_mip_float32.tif", before)
    tifffile.imwrite(output / "02_fixed_window_mip_float32.tif", after)
    tifffile.imwrite(output / "03_excluded_amplitude_float32.tif", removed)
    save_display(
        before,
        output / "01_full_depth_mip_display.png",
        title="Full-depth maximum projection",
        vmin=vmin,
        vmax=vmax,
    )
    save_display(
        after,
        output / "02_fixed_window_mip_display.png",
        title=f"Fixed-window maximum projection ({args.gate_start}:{args.gate_stop})",
        vmin=vmin,
        vmax=vmax,
    )
    save_comparison(
        before,
        after,
        output / "04_fixed_window_comparison.png",
        gate_start=args.gate_start,
        gate_stop=args.gate_stop,
        vmin=vmin,
        vmax=vmax,
    )

    isolated = isolated_bright_point_metrics(before, after)
    outside_gate = (peak_index < args.gate_start) | (peak_index >= args.gate_stop)
    positive = before > 0
    gate_start_us = args.trigger_delay_us + args.gate_start / args.sampling_rate_msps
    gate_stop_us = args.trigger_delay_us + args.gate_stop / args.sampling_rate_msps
    metrics = {
        "method": "positive maximum amplitude restricted to a fixed A-line window",
        "source_pa": str(source),
        "shape_height_width_depth": [args.height, args.width, args.depth],
        "packed_format": "unsigned 12-bit little-endian, two samples per three bytes",
        "baseline_adc": args.baseline,
        "gate_samples_start_stop": [args.gate_start, args.gate_stop],
        "gate_stop_is_exclusive": True,
        "sampling_rate_msps": args.sampling_rate_msps,
        "trigger_delay_us": args.trigger_delay_us,
        "gate_time_us_start_stop": [gate_start_us, gate_stop_us],
        "pd_used": False,
        "spatial_smoothing_used": False,
        "fraction_of_positive_pixels_whose_full_depth_peak_is_outside_gate": float(
            outside_gate[positive].mean()
        ),
        "median_amplitude_before_adc": float(np.median(before)),
        "median_amplitude_after_adc": float(np.median(after)),
        "p99_7_amplitude_before_adc": float(np.percentile(before, 99.7)),
        "p99_7_amplitude_after_adc": float(np.percentile(after, 99.7)),
        "isolated_bright_point_check": isolated,
        "display_limits_shared_adc": [vmin, vmax],
        "caveat": (
            "This is a fixed-depth gate for the current acquisition. Verify or adapt the "
            "window before applying it to samples at a different depth or trigger timing."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
