#!/usr/bin/env python3
"""Compare legacy NeuWS model A with physical OR-PAM forward model B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.orpam_slm import HardwareSLMPhaseLibrary  # noqa: E402
from optics.legacy import fft_2xPad_Conv2D  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    kernel_energy_diagnostics,
)
from optics.propagation import pupil_psf  # noqa: E402
from tools.run_orpam_model_b import make_fibre_phantom  # noqa: E402


def _model_phase(library: HardwareSLMPhaseLibrary, index: int) -> np.ndarray:
    specification = str(library.manifest["model_output"]["mat"])
    path_text, variable = specification.rsplit(":", 1)
    path = library.root / path_text.replace("N", str(index))
    values = sio.loadmat(path)
    if variable not in values:
        raise KeyError(f"{path} does not contain {variable!r}.")
    phase = np.asarray(values[variable], dtype=np.float32).squeeze()
    expected = tuple(int(value) for value in library.manifest["model_shape"])
    if phase.shape != expected:
        raise ValueError(f"Legacy model phase shape {phase.shape} != {expected}.")
    return phase


def _centroid_and_rms(array: np.ndarray, pixel_pitch: float) -> dict[str, float]:
    values = np.maximum(np.asarray(array, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return {"centroid_row_m": 0.0, "centroid_column_m": 0.0, "rms_radius_m": 0.0}
    rows = (np.arange(values.shape[0]) - values.shape[0] // 2) * pixel_pitch
    columns = (np.arange(values.shape[1]) - values.shape[1] // 2) * pixel_pitch
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    centroid_row = float((values * row_grid).sum() / total)
    centroid_column = float((values * column_grid).sum() / total)
    radius2 = (row_grid - centroid_row) ** 2 + (column_grid - centroid_column) ** 2
    return {
        "centroid_row_m": centroid_row,
        "centroid_column_m": centroid_column,
        "rms_radius_m": float(np.sqrt((values * radius2).sum() / total)),
    }


def _normalized_difference(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / max(float(np.linalg.norm(first)), np.finfo(float).eps)
    second = second / max(float(np.linalg.norm(second)), np.finfo(float).eps)
    return float(np.linalg.norm(first - second))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--fft-size", type=int, default=4320)
    parser.add_argument("--kernel-radius", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/orpam_model_a_b"
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    library = HardwareSLMPhaseLibrary(args.phase_root)
    hardware_phase = library.load(args.frame, device=device)
    model_phase_np = _model_phase(library, args.frame)
    model_phase = torch.as_tensor(model_phase_np, device=device)
    object_np = make_fibre_phantom(600)
    object_tensor = torch.as_tensor(object_np, device=device)[None, None]

    # Model A baseline exactly preserves its array-level assumptions: a 600 x
    # 600 phase/pupil, unit unknown complex amplitude for this controlled
    # comparison, sum-normalized PSF, explicit PSF flip, then legacy FFT
    # convolution.  A trained StaticDiffuseNet may instead learn free signed
    # amplitude and phase; that distinct modelling freedom is not attributed to
    # physical sampling here.
    slm_field_a = torch.exp(1j * library.phase_sign * model_phase)[None, None]
    unit_aberration_a = torch.ones_like(slm_field_a)
    random_psf_a = pupil_psf(unit_aberration_a, slm_field_a)
    ideal_psf_a = pupil_psf(unit_aberration_a, torch.ones_like(slm_field_a))
    legacy_kernel_a = random_psf_a.flip(-2, -1)
    image_a = fft_2xPad_Conv2D(object_tensor, legacy_kernel_a)

    config = ORPAMModelBConfig(
        slm_size=library.shape[0],
        fft_size=args.fft_size,
        phase_sign=library.phase_sign,
        kernel_radius=args.kernel_radius,
        quadrature_order=args.quadrature_order,
    )
    simulator = ORPAMForwardModelB(config).to(device)
    zero_aberration = torch.zeros(library.shape, device=device)
    random_psf_b = simulator.optical_psf(hardware_phase, zero_aberration)
    ideal_psf_b = simulator.optical_psf(torch.zeros_like(hardware_phase), zero_aberration)
    kernel_b = simulator.equivalent_kernel(random_psf_b.intensity)
    image_b = simulator.scan_forward(object_tensor, kernel_b)
    support = kernel_energy_diagnostics(kernel_b)

    arrays = {
        "object_image": object_np,
        "model_a_ideal_psf": ideal_psf_a[0, 0].detach().cpu().numpy(),
        "model_a_random_psf": random_psf_a[0, 0].detach().cpu().numpy(),
        "model_a_kernel_after_legacy_flip": legacy_kernel_a[0, 0].detach().cpu().numpy(),
        "model_a_simulated_image": image_a[0, 0].detach().cpu().numpy(),
        "model_b_ideal_psf": ideal_psf_b.intensity[0, 0].detach().cpu().numpy(),
        "model_b_random_psf": random_psf_b.intensity[0, 0].detach().cpu().numpy(),
        "model_b_equivalent_kernel": kernel_b[0, 0].detach().cpu().numpy(),
        "model_b_simulated_image": image_b[0, 0].detach().cpu().numpy(),
        "model_b_focus_coordinate_m": random_psf_b.x_m.detach().cpu().numpy(),
    }
    np.savez_compressed(output_dir / f"model_a_b_frame_{args.frame:02d}.npz", **arrays)

    ideal_first_zero = config.wavelength_m * config.focal_length_m / config.slm_width_m
    model_a_spread = _centroid_and_rms(arrays["model_a_random_psf"], 1.0)
    report = {
        "frame_index": args.frame,
        "controlled_comparison": {
            "same_random_coefficients": bool(
                library.manifest.get("same_coefficients_on_both_grids")
            ),
            "model_a_phase": "independently evaluated model-grid phase",
            "model_b_phase": "real 1080x1080 hardware-grid MAT phase",
            "common_aberration": "zero phase",
            "common_pupil_amplitude_for_comparison": "uniform square",
            "important_model_a_difference": (
                "StaticDiffuseNet normally predicts a free signed pupil amplitude and phase; "
                "this comparison fixes amplitude to one to isolate physical sampling."
            ),
        },
        "model_a": {
            "array_shape": list(random_psf_a.shape[-2:]),
            "physical_pixel_pitch": None,
            "normalization": "discrete pixel sum equals one",
            "random_psf_discrete_sum": float(random_psf_a.sum().detach().cpu()),
            "random_psf_peak": float(random_psf_a.max().detach().cpu()),
            "ideal_psf_peak": float(ideal_psf_a.max().detach().cpu()),
            "random_vs_ideal_normalized_l2": _normalized_difference(
                arrays["model_a_random_psf"], arrays["model_a_ideal_psf"]
            ),
            "random_psf_rms_radius_pixels": model_a_spread["rms_radius_m"],
            "simulated_image_peak": float(image_a.max().detach().cpu()),
            "note": "Direct 600-grid pupil/PSF coupling has no wavelength/focal-length/SLM-pitch map.",
        },
        "model_b": {
            **config.to_record(),
            "ideal_square_pupil_first_zero_m": ideal_first_zero,
            "ideal_square_pupil_sinc2_fwhm_m": 0.8859 * ideal_first_zero,
            "random_psf_peak_inverse_m2": float(
                random_psf_b.intensity.max().detach().cpu()
            ),
            "random_psf_area_integral": float(
                random_psf_b.intensity.sum().detach().cpu()
                * config.focus_pixel_pitch_m**2
            ),
            "ideal_psf_peak_inverse_m2": float(ideal_psf_b.intensity.max().detach().cpu()),
            "random_vs_ideal_normalized_l2": _normalized_difference(
                arrays["model_b_random_psf"], arrays["model_b_ideal_psf"]
            ),
            "kernel_peak": float(kernel_b.max().detach().cpu()),
            "kernel_retained_energy": float(support["retained_energy"][0].detach().cpu()),
            "kernel_truncation_error": float(support["truncation_error"][0].detach().cpu()),
            "simulated_image_peak": float(image_b.max().detach().cpu()),
            "random_psf_spread": _centroid_and_rms(
                arrays["model_b_random_psf"], config.focus_pixel_pitch_m
            ),
        },
        "final_image_normalized_l2_difference": _normalized_difference(
            arrays["model_a_simulated_image"], arrays["model_b_simulated_image"]
        ),
    }
    (output_dir / f"model_a_b_frame_{args.frame:02d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run(args)
    print(f"Model A/B comparison written to {output_dir}")


if __name__ == "__main__":
    main()
