#!/usr/bin/env python3
"""Finalize a real NeuWS reconstruction when no reference image was acquired."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_unit_png, write_wrapped_phase_png  # noqa: E402


def _finite_2d(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite 2-D array, got {value.shape}.")
    return value


def _minmax(value: np.ndarray) -> np.ndarray:
    minimum, maximum = float(value.min()), float(value.max())
    if maximum <= minimum:
        return np.zeros_like(value, dtype=np.float32)
    return np.asarray((value - minimum) / (maximum - minimum), dtype=np.float32)


def _percentile_display(value: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lower, upper = np.percentile(value, [low, high])
    if upper <= lower:
        return np.zeros_like(value, dtype=np.float32)
    return np.asarray(np.clip((value - lower) / (upper - lower), 0, 1), dtype=np.float32)


def finalize(data_dir: Path, final_dir: Path, output_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    final_dir = final_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_raw = _finite_2d(
        sio.loadmat(final_dir / "final_I_est_network_units.mat")["image"],
        "reconstructed image",
    )
    phase = _finite_2d(
        sio.loadmat(final_dir / "final_aberration.mat")["phase"],
        "reconstructed phase",
    )
    field = np.exp(1j * phase).astype(np.complex64)
    measurements = np.load(data_dir / "measurements.npy", mmap_mode="r")
    training = json.loads((final_dir / "training_summary.json").read_text(encoding="utf-8"))

    image_normalized = _minmax(image_raw)
    image_display = _percentile_display(image_raw)
    measurement_mean = np.asarray(np.mean(measurements, axis=0), dtype=np.float32)

    # Repair the legacy complex-field export and add explicit, stable products.
    sio.savemat(
        final_dir / "final_aberration.mat",
        {"field": field, "phase": phase},
        do_compression=True,
    )
    np.save(final_dir / "reconstructed_object_normalized.npy", image_normalized)
    sio.savemat(
        final_dir / "reconstructed_object_normalized.mat",
        {"reconstructed_object": image_normalized},
        do_compression=True,
    )
    write_unit_png(final_dir / "reconstructed_object_normalized.png", image_normalized)
    write_unit_png(final_dir / "reconstructed_object_percentile_display.png", image_display)
    np.save(final_dir / "reconstructed_aberration_phase.npy", phase)
    np.save(final_dir / "reconstructed_aberration_field.npy", field)
    write_wrapped_phase_png(final_dir / "reconstructed_aberration_phase.png", phase)

    wrapped_unit = np.asarray((np.angle(field) + np.pi) / (2 * np.pi), dtype=np.float32)
    wrapped_u8 = np.uint8(np.clip(wrapped_unit, 0, 1) * 255)
    imageio.imwrite(final_dir / "final_aberrations_angle.png", wrapped_u8)
    imageio.mimsave(final_dir / "final_aberrations_angle.gif", [wrapped_u8], duration=1.0)

    losses = np.asarray(training["loss_history"], dtype=np.float64)
    report = {
        "reference_available": False,
        "metrics_available": False,
        "reason": "Only S1-S50 were used; no reference or no-SLM acquisition was processed.",
        "frames": int(measurements.shape[0]),
        "measurement_shape": list(measurements.shape[1:]),
        "image_network_units": {
            "min": float(image_raw.min()),
            "max": float(image_raw.max()),
            "mean": float(image_raw.mean()),
            "p01": float(np.percentile(image_raw, 1)),
            "p99": float(np.percentile(image_raw, 99)),
        },
        "phase_rad": {
            "min": float(phase.min()),
            "max": float(phase.max()),
            "mean": float(phase.mean()),
            "std": float(phase.std()),
        },
        "field_phase_consistency_max_error": float(
            np.max(np.abs(field - np.exp(1j * phase)))
        ),
        "training": {
            "epochs_completed": training["num_epochs"],
            "best_epoch": training["early_stopping"]["best_epoch"],
            "best_smoothed_loss": training["early_stopping"]["best_smoothed_loss"],
            "last_observed_loss": float(losses[-1]),
            "minimum_epoch_loss": float(losses.min()),
            "elapsed_seconds": training["elapsed_seconds"],
            "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
        },
    }
    report_path = output_dir / "samples_only_reconstruction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reconstruction"] = {
        "completed": True,
        "reference_metrics_available": False,
        "final_directory": str(final_dir),
        "evaluation_directory": str(output_dir),
        "epochs_completed": training["num_epochs"],
        "best_epoch": training["early_stopping"]["best_epoch"],
        "best_smoothed_loss": training["early_stopping"]["best_smoothed_loss"],
        "network_zernike_features": training["network_zernike_features"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    panels = (
        (_percentile_display(measurement_mean), "Mean of S1-S50", "gray"),
        (image_display, "NeuWS reconstruction\nP1-P99 display", "gray"),
        (_minmax(phase), "Estimated phase\nlinear display", "rainbow"),
        (np.angle(field), "Estimated phase\nwrapped to [-pi, pi]", "twilight"),
    )
    for axis, (panel, title, cmap) in zip(axes[:4], panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    axes[4].plot(np.arange(1, losses.size + 1), losses)
    axes[4].set_title("Training loss")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("Mean MSE")
    axes[4].grid(alpha=0.25)
    fig.suptitle("2026-08-19 radial-15 real-data reconstruction", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "samples_only_reconstruction_overview.png", dpi=150)
    plt.close(fig)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reconstruction-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = finalize(args.data_dir, args.reconstruction_dir, args.output_dir)
    print(f"Finalized samples-only reconstruction: {result}")
