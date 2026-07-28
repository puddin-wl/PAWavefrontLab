#!/usr/bin/env python3
"""Generate NeuWS paper-style SLM patterns and synthetic measurements."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics import (  # noqa: E402
    PAPER_PHASE_SIGN,
    crop_to_aperture,
    make_static_aberration,
    sample_slm_coefficients,
    simulate_measurements,
    slm_complex_field,
    validate_geometry,
    zernike_basis_numpy,
)


SCHEMA_VERSION = 1
SLM_NUM_MODES = 15
ABERRATION_NUM_MODES = 28


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _read_object(path: Path, size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype(np.float32)
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif array.ndim > 3:
        array = np.asarray(array[0])
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {array.shape}.")
    height, width = array.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    array = array[top : top + side, left : left + side].astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Input image contains NaN or infinite values.")
    minimum, maximum = float(array.min()), float(array.max())
    if maximum <= minimum:
        raise ValueError("Input image must have a non-zero intensity range.")
    array = (array - minimum) / (maximum - minimum)
    image = Image.fromarray(array).resize((size, size), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def _write_phase_png(path: Path, phase: np.ndarray) -> None:
    wrapped = np.mod(phase, 2.0 * math.pi)
    encoded = np.rint(wrapped * (65535.0 / (2.0 * math.pi))).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def _common_manifest(args, geometry, coefficients: np.ndarray) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "NeuWS AOtools Python generator",
        "command": args.command,
        "size": geometry.size,
        "measurement_shape": [geometry.size, geometry.size],
        "aperture_height": geometry.aperture_height,
        "aperture_shape": [geometry.aperture_height, geometry.size],
        "aperture_ratio": geometry.aperture_height / geometry.size,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "slm_num_modes": SLM_NUM_MODES,
        "slm_noll_indices": list(range(1, SLM_NUM_MODES + 1)),
        "slm_sigma_rad": args.slm_sigma,
        "phase_sign": PAPER_PHASE_SIGN,
        "phase_convention": "aperture * exp(-1j * proj_sim)",
        "slm_coefficients_file": "slm_coefficients.npy",
        "slm_patterns_file": "slm_patterns.npy",
        "slm_coefficients_shape": list(coefficients.shape),
        "png_phase_encoding": {
            "wrap_interval_rad": [0.0, 2.0 * math.pi],
            "integer_range": [0, 65535],
            "calibrated_for_hardware": False,
        },
    }


def _prepare_patterns(args, output_dir: Path):
    geometry = validate_geometry(args.size, args.aperture_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / "slm_png"
    png_dir.mkdir(exist_ok=True)
    coefficients = sample_slm_coefficients(
        args.num_frames, SLM_NUM_MODES, args.slm_sigma, args.seed
    )
    basis = zernike_basis_numpy(SLM_NUM_MODES, geometry.size)
    patterns = np.lib.format.open_memmap(
        output_dir / "slm_patterns.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.num_frames, geometry.aperture_height, geometry.size),
    )
    for index, coefficient in enumerate(coefficients, start=1):
        full_phase = np.einsum("m,mhw->hw", coefficient, basis, optimize=True)
        active_phase = np.asarray(
            crop_to_aperture(full_phase, geometry.aperture_height), dtype=np.float32
        )
        patterns[index - 1] = active_phase
        sio.savemat(
            output_dir / f"SLM_sim{index}.mat",
            {"proj_sim": active_phase},
            do_compression=True,
        )
        _write_phase_png(png_dir / f"SLM_sim{index}.png", active_phase)
    patterns.flush()
    np.save(output_dir / "slm_coefficients.npy", coefficients)
    return geometry, coefficients, patterns


def generate_patterns(args) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    geometry, coefficients, _ = _prepare_patterns(args, output_dir)
    manifest = _common_manifest(args, geometry, coefficients)
    manifest["output_files"] = {
        "mat_phase_pattern": "SLM_simN.mat:proj_sim",
        "phase_array": "slm_patterns.npy",
        "phase_png_directory": "slm_png",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Generated {args.num_frames} patterns with shape "
        f"{geometry.aperture_height}x{geometry.size} in {output_dir}"
    )


def generate_simulation(args) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    geometry, coefficients, patterns = _prepare_patterns(args, output_dir)
    device = _device(args.device)
    object_image = _read_object(Path(args.input_image).expanduser(), geometry.size)
    aberration, aberration_coefficients = make_static_aberration(
        args.aberration_mode,
        geometry.size,
        geometry.aperture_height,
        sigma=args.aberration_sigma,
        seed=args.seed,
        num_modes=ABERRATION_NUM_MODES,
        device=device,
    )
    object_tensor = torch.as_tensor(object_image, dtype=torch.float32, device=device)
    noise_rng = np.random.default_rng(args.seed + 1)
    measurement_max = 0.0
    for start in range(0, args.num_frames, args.batch_size):
        stop = min(start + args.batch_size, args.num_frames)
        active_phase = torch.as_tensor(
            np.asarray(patterns[start:stop]), dtype=torch.float32, device=device
        )
        fields = slm_complex_field(active_phase, geometry.size, PAPER_PHASE_SIGN)
        measurements, _ = simulate_measurements(object_tensor, aberration, fields)
        batch = measurements[:, 0].detach().cpu().numpy().astype(np.float32)
        if args.noise_std:
            batch += noise_rng.normal(0.0, args.noise_std, batch.shape).astype(np.float32)
            batch = np.clip(batch, 0.0, None)
        if not np.isfinite(batch).all():
            raise RuntimeError("Simulation produced NaN or infinite measurements.")
        measurement_max = max(measurement_max, float(batch.max()))
        for offset, measurement in enumerate(batch, start=start + 1):
            sio.savemat(
                output_dir / f"SLM_raw{offset}.mat",
                {"imsdata": measurement},
                do_compression=True,
            )

    aberration_np = aberration.detach().cpu().numpy().astype(np.complex64)
    sio.savemat(
        output_dir / "ground_truth.mat",
        {
            "object_image": object_image,
            "aberration_field": aberration_np,
            "aberration_amplitude": np.abs(aberration_np).astype(np.float32),
            "aberration_phase": np.angle(aberration_np).astype(np.float32),
            "aberration_coefficients": aberration_coefficients,
            "slm_coefficients": coefficients,
        },
        do_compression=True,
    )
    manifest = _common_manifest(args, geometry, coefficients)
    manifest.update(
        {
            "input_image": str(Path(args.input_image).expanduser().resolve()),
            "image_preprocessing": "grayscale, center-square crop, bicubic resize, min-max [0,1]",
            "aberration_mode": args.aberration_mode,
            "aberration_num_modes": ABERRATION_NUM_MODES if args.aberration_mode == "zernike" else None,
            "aberration_noll_indices": list(range(1, ABERRATION_NUM_MODES + 1)) if args.aberration_mode == "zernike" else None,
            "aberration_sigma_rad": args.aberration_sigma,
            "noise_std": args.noise_std,
            "measurement_max": measurement_max,
            "simulation_device": str(device),
            "static_object": True,
            "static_aberration": True,
            "output_files": {
                "mat_phase_pattern": "SLM_simN.mat:proj_sim",
                "mat_measurement": "SLM_rawN.mat:imsdata",
                "phase_array": "slm_patterns.npy",
                "phase_png_directory": "slm_png",
                "ground_truth": "ground_truth.mat",
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Generated {args.num_frames} measurements with shape "
        f"{geometry.size}x{geometry.size} in {output_dir} using {device}"
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--aperture-height", type=int)
    parser.add_argument("--slm-sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    patterns = subparsers.add_parser("patterns", help="Generate paper-style SLM patterns.")
    _add_common_arguments(patterns)
    patterns.set_defaults(func=generate_patterns)
    simulate = subparsers.add_parser(
        "simulate", help="Generate SLM patterns and closed-loop synthetic camera frames."
    )
    _add_common_arguments(simulate)
    simulate.add_argument("--input-image", required=True)
    simulate.add_argument(
        "--aberration-mode", choices=("zernike", "complex-gaussian"), default="zernike"
    )
    simulate.add_argument("--aberration-sigma", type=float, default=1.0)
    simulate.add_argument("--noise-std", type=float, default=0.0)
    simulate.add_argument("--device", default="auto")
    simulate.add_argument("--batch-size", type=int, default=8)
    simulate.set_defaults(func=generate_simulation)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.slm_sigma < 0:
        parser.error("--slm-sigma must be non-negative")
    if getattr(args, "aberration_sigma", 0.0) < 0:
        parser.error("--aberration-sigma must be non-negative")
    if getattr(args, "noise_std", 0.0) < 0:
        parser.error("--noise-std must be non-negative")
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("--batch-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
