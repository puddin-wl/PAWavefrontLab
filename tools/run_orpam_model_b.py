#!/usr/bin/env python3
"""Run the independent physical OR-PAM forward model B on real SLM phases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.orpam_slm import HardwareSLMPhaseLibrary  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    kernel_energy_diagnostics,
)


def _parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise argparse.ArgumentTypeError("Frame ranges must be ascending.")
            indices.extend(range(start, stop + 1))
        else:
            indices.append(int(part))
    if not indices or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("Frames must be a non-empty list without duplicates.")
    return indices


def _load_array(path: Path, mat_variable: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".mat":
        values = sio.loadmat(path)
        if mat_variable:
            if mat_variable not in values:
                raise KeyError(f"{path} does not contain {mat_variable!r}.")
            value = values[mat_variable]
        else:
            candidates = [
                item for key, item in values.items() if not key.startswith("__") and np.asarray(item).ndim >= 2
            ]
            if len(candidates) != 1:
                raise ValueError("MAT input needs --mat-variable when it has multiple arrays.")
            value = candidates[0]
    elif suffix in {".tif", ".tiff"}:
        value = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported array format: {path.suffix}")
    value = np.asarray(value).squeeze()
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{path} must contain one finite 2-D array.")
    return np.asarray(value, dtype=np.float32)


def make_fibre_phantom(size: int = 600) -> np.ndarray:
    """Deterministic thin-fibre phantom used when no object is supplied."""

    image = np.zeros((size, size), dtype=np.float32)
    for column in range(70, size - 70):
        rows = (
            int(0.34 * size + 0.22 * column),
            int(0.76 * size - 0.31 * column),
            int(0.18 * size + 0.52 * column),
        )
        for row, amplitude in zip(rows, (1.0, 0.75, 0.55)):
            if 1 <= row < size - 1:
                image[row - 1 : row + 2, column] = amplitude
    return image


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    parser.add_argument("--frames", type=_parse_indices, default=[1])
    parser.add_argument("--measurement-dir", type=Path)
    parser.add_argument("--object", type=Path)
    parser.add_argument("--object-mat-variable")
    parser.add_argument("--aberration", type=Path)
    parser.add_argument("--aberration-mat-variable")
    parser.add_argument("--fft-size", type=int, default=4320)
    parser.add_argument("--kernel-radius", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validate-png", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/orpam_model_b"
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library = HardwareSLMPhaseLibrary(args.phase_root)
    mapping = None
    if args.measurement_dir:
        mapping = library.validate_measurement_mapping(args.measurement_dir)
    device = _device(args.device)
    config = ORPAMModelBConfig(
        slm_size=library.shape[0],
        fft_size=args.fft_size,
        phase_sign=library.phase_sign,
        kernel_radius=args.kernel_radius,
        quadrature_order=args.quadrature_order,
    )
    simulator = ORPAMForwardModelB(config).to(device)

    object_array = (
        make_fibre_phantom()
        if args.object is None
        else _load_array(args.object, args.object_mat_variable)
    )
    if object_array.shape != (600, 600):
        raise ValueError(f"The model-B object must be 600x600, got {object_array.shape}.")
    object_tensor = torch.as_tensor(object_array, device=device)

    if args.aberration is None:
        aberration = torch.zeros(library.shape, device=device)
        aberration_source = "zero phase"
    else:
        aberration_array = _load_array(args.aberration, args.aberration_mat_variable)
        if aberration_array.shape != library.shape:
            raise ValueError(
                f"Aberration must use the hardware grid {library.shape}, got {aberration_array.shape}."
            )
        aberration = torch.as_tensor(aberration_array, device=device)
        aberration_source = str(args.aberration.expanduser().resolve())

    frame_records = []
    for index in args.frames:
        start = time.perf_counter()
        phase = library.load(
            index,
            device=device,
            validate_png=args.validate_png,
        )
        psf_result = simulator.optical_psf(phase, aberration)
        kernel = simulator.equivalent_kernel(psf_result.intensity)
        image = simulator.scan_forward(object_tensor, kernel)
        diagnostics = kernel_energy_diagnostics(kernel)
        elapsed = time.perf_counter() - start

        psf_np = psf_result.intensity[0, 0].detach().cpu().numpy().astype(np.float32)
        kernel_np = kernel[0, 0].detach().cpu().numpy().astype(np.float32)
        image_np = image[0, 0].detach().cpu().numpy().astype(np.float32)
        coordinate_np = psf_result.x_m.detach().cpu().numpy().astype(np.float64)
        np.savez_compressed(
            output_dir / f"model_b_frame_{index:02d}.npz",
            psf=psf_np,
            focus_x_m=coordinate_np,
            focus_y_m=coordinate_np,
            equivalent_kernel=kernel_np,
            simulated_image=image_np,
            object_image=object_array,
        )
        tifffile.imwrite(output_dir / f"model_b_frame_{index:02d}_image.tif", image_np)
        frame_records.append(
            {
                "frame_index": index,
                "hardware_phase_mat": str(library.mat_paths[index]),
                "raw_psf_energy": float(psf_result.raw_energy[0].detach().cpu()),
                "normalized_psf_integral": float(
                    psf_result.intensity.sum().detach().cpu()
                    * config.focus_pixel_pitch_m**2
                ),
                "kernel_retained_energy": float(
                    diagnostics["retained_energy"][0].detach().cpu()
                ),
                "kernel_truncation_error": float(
                    diagnostics["truncation_error"][0].detach().cpu()
                ),
                "psf_peak_inverse_m2": float(psf_result.intensity.max().detach().cpu()),
                "simulated_image_shape": list(image_np.shape),
                "elapsed_seconds": elapsed,
            }
        )
        del phase, psf_result, kernel, image
        if device.type == "cuda":
            torch.cuda.empty_cache()

    record = {
        "model": "OR-PAM physical forward model B",
        "physical_and_numerical_parameters": config.to_record(),
        "phase_library": str(library.root),
        "phase_manifest": library.manifest,
        "measurement_mapping_validated": args.measurement_dir is not None,
        "measurement_mapping": mapping,
        "object_source": "built-in deterministic fibre phantom" if args.object is None else str(args.object),
        "aberration_source": aberration_source,
        "device": str(device),
        "frames": frame_records,
    }
    (output_dir / "model_b_run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run(args)
    print(f"Model-B outputs written to {output_dir}")


if __name__ == "__main__":
    main()
