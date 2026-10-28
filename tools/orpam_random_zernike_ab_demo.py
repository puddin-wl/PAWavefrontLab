#!/usr/bin/env python3
"""Create a self-contained random-Zernike model-A/model-B comparison folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pawavefrontlab-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optics import zernike_basis_numpy  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    kernel_energy_diagnostics,
    pupil_amplitude_mask,
)
from optics.propagation import pupil_psf  # noqa: E402
from tools.orpam_pupil_mask_loss_sweep import (  # noqa: E402
    SLM_WIDTH_M,
    circular_pupil_mask,
    model_a_diagnostic_forward,
)


def _phase_from_coefficients(coefficients: np.ndarray, size: int) -> np.ndarray:
    basis = zernike_basis_numpy(int(coefficients.size), size)
    phase = np.einsum("m,mhw->hw", coefficients, basis, optimize=True)
    return np.asarray(phase, dtype=np.float32)


def random_zernike_phase_pair(
    *,
    seed: int,
    num_modes: int,
    target_rms_rad: float,
    model_size: int = 600,
    hardware_size: int = 1080,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use one Noll coefficient vector on the model-A and model-B grids.

    Piston and tip/tilt are constrained so the visible change is an aberration
    rather than an arbitrary global phase or image translation.  Noll 4 onward
    are random with a mild order-dependent decay, then jointly scaled to the
    requested piston-removed RMS on the 600 grid.
    """

    if num_modes < 4:
        raise ValueError("num_modes must be at least four.")
    if not math.isfinite(target_rms_rad) or target_rms_rad <= 0:
        raise ValueError("target_rms_rad must be finite and positive.")
    rng = np.random.default_rng(seed)
    noll = np.arange(1, num_modes + 1, dtype=np.float64)
    unscaled = rng.standard_normal(num_modes) / np.sqrt(noll)
    unscaled[:3] = 0.0
    initial_phase = _phase_from_coefficients(unscaled.astype(np.float32), model_size)
    initial_centered = initial_phase - float(initial_phase.mean(dtype=np.float64))
    initial_rms = float(np.sqrt(np.mean(initial_centered.astype(np.float64) ** 2)))
    if initial_rms <= 0:
        raise RuntimeError("Random Zernike draw unexpectedly has zero RMS.")
    coefficients = unscaled * (target_rms_rad / initial_rms)

    # Noll 1 is constant.  Adjust it so that the saved 600-grid phase is
    # piston-free while retaining an exact coefficient-to-phase relationship.
    model_basis = zernike_basis_numpy(num_modes, model_size)
    nonpiston = np.einsum(
        "m,mhw->hw", coefficients, model_basis, optimize=True
    ).astype(np.float32)
    piston_basis_mean = float(model_basis[0].mean(dtype=np.float64))
    coefficients[0] -= float(nonpiston.mean(dtype=np.float64)) / piston_basis_mean
    model_phase = np.einsum(
        "m,mhw->hw", coefficients, model_basis, optimize=True
    ).astype(np.float32)
    hardware_phase = _phase_from_coefficients(
        coefficients.astype(np.float32), hardware_size
    )
    return (
        coefficients.astype(np.float32),
        model_phase,
        hardware_phase,
        unscaled.astype(np.float32),
    )


def _image_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    data_range = float(reference64.max() - reference64.min())
    if data_range <= 0:
        raise ValueError("Reference object must have nonzero dynamic range.")
    return {
        "ssim": float(
            structural_similarity(reference64, estimate64, data_range=data_range)
        ),
        "psnr_db": float(
            peak_signal_noise_ratio(reference64, estimate64, data_range=data_range)
        ),
        "mse": float(np.mean((estimate64 - reference64) ** 2)),
        "relative_l2": float(
            np.linalg.norm(estimate64 - reference64)
            / np.linalg.norm(reference64)
        ),
    }


def _kernel_spread(kernel: np.ndarray) -> dict[str, float]:
    """Return energy centroid and RMS radius in the kernel's array pixels."""

    values = np.maximum(np.asarray(kernel, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Kernel must have positive energy.")
    row, column = np.meshgrid(
        np.arange(values.shape[0], dtype=np.float64),
        np.arange(values.shape[1], dtype=np.float64),
        indexing="ij",
    )
    centroid_row = float(np.sum(values * row) / total)
    centroid_column = float(np.sum(values * column) / total)
    radius_squared = (row - centroid_row) ** 2 + (column - centroid_column) ** 2
    radius = np.sqrt(radius_squared).ravel()
    energy = values.ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(energy[order]) / total
    containment = {
        f"radius_{int(fraction * 100)}pct_energy_pixel": float(
            radius[order][np.searchsorted(cumulative, fraction)]
        )
        for fraction in (0.5, 0.8, 0.9, 0.95, 0.99)
    }
    return {
        "energy_sum": total,
        "centroid_row_pixel": centroid_row,
        "centroid_column_pixel": centroid_column,
        "rms_radius_pixel": float(np.sqrt(np.sum(values * radius_squared) / total)),
        **containment,
    }


def _write_grayscale_png(path: Path, image: np.ndarray, maximum: float) -> None:
    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=maximum)


def _write_phase_figure(path: Path, phase: np.ndarray, coefficients: np.ndarray) -> None:
    limit = float(np.max(np.abs(phase)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    shown = axes[0].imshow(
        phase, cmap="twilight", vmin=-limit, vmax=limit, origin="lower"
    )
    axes[0].set(title="Random Zernike phase (600 x 600)", xlabel="x pixel", ylabel="y pixel")
    fig.colorbar(shown, ax=axes[0], label="phase (rad)", shrink=0.85)
    indices = np.arange(1, coefficients.size + 1)
    axes[1].bar(indices, coefficients, width=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        title="Noll coefficients",
        xlabel="Noll index",
        ylabel="coefficient (rad)",
        xlim=(0.3, coefficients.size + 0.7),
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_image_overview(
    path: Path,
    phase: np.ndarray,
    reference: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    b_pupil_diameter_mm: float,
    a_pupil: str,
) -> None:
    display_maximum = float(max(reference.max(), image_a.max(), image_b.max()))
    errors = (np.abs(image_a - reference), np.abs(image_b - reference))
    error_maximum = float(max(value.max() for value in errors))
    phase_limit = float(np.max(np.abs(phase)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    phase_view = axes[0, 0].imshow(
        phase, cmap="twilight", vmin=-phase_limit, vmax=phase_limit, origin="lower"
    )
    axes[0, 0].set_title("Random Zernike phase (rad)")
    fig.colorbar(phase_view, ax=axes[0, 0], shrink=0.75)
    for axis, value, title in (
        (axes[0, 1], reference, "Original object"),
        (
            axes[0, 2],
            image_a,
            f"Model A-{a_pupil} aberrated\n"
            f"SSIM={metrics_a['ssim']:.4f}, PSNR={metrics_a['psnr_db']:.2f} dB",
        ),
        (
            axes[1, 0],
            image_b,
            f"Model B-circle {b_pupil_diameter_mm:g} mm aberrated\n"
            f"SSIM={metrics_b['ssim']:.4f}, PSNR={metrics_b['psnr_db']:.2f} dB",
        ),
    ):
        view = axis.imshow(
            value, cmap="gray", vmin=0.0, vmax=display_maximum, origin="lower"
        )
        axis.set_title(title)
    fig.colorbar(view, ax=(axes[0, 1], axes[0, 2], axes[1, 0]), shrink=0.75)
    for axis, value, title in (
        (axes[1, 1], errors[0], "|Model A - original|"),
        (axes[1, 2], errors[1], "|Model B - original|"),
    ):
        error_view = axis.imshow(
            value, cmap="magma", vmin=0.0, vmax=error_maximum, origin="lower"
        )
        axis.set_title(title)
    fig.colorbar(error_view, ax=(axes[1, 1], axes[1, 2]), shrink=0.75)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Same object and random aberration; zero known SLM phase")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_psf_figure(
    path: Path,
    psf_a: np.ndarray,
    psf_b: np.ndarray,
    focus_coordinate_m: np.ndarray,
    pupil_diameter_mm: float,
    a_pupil: str,
) -> None:
    half_a = 16
    centre_a = psf_a.shape[0] // 2
    crop_a = psf_a[
        centre_a - half_a : centre_a + half_a + 1,
        centre_a - half_a : centre_a + half_a + 1,
    ]
    half_b = 40
    centre_b = psf_b.shape[0] // 2
    crop_b = psf_b[
        centre_b - half_b : centre_b + half_b + 1,
        centre_b - half_b : centre_b + half_b + 1,
    ]
    coordinate_b_um = focus_coordinate_m[
        centre_b - half_b : centre_b + half_b + 1
    ] * 1e6
    norm_a = crop_a / float(crop_a.max())
    norm_b = crop_b / float(crop_b.max())
    log_a = np.log10(np.maximum(norm_a, 1e-8))
    log_b = np.log10(np.maximum(norm_b, 1e-8))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    a_view = axes[0, 0].imshow(
        log_a,
        cmap="magma",
        vmin=-8,
        vmax=0,
        origin="lower",
        extent=[-half_a, half_a, -half_a, half_a],
    )
    axes[0, 0].set(
        title=f"Model A-{a_pupil} zero-phase PSF",
        xlabel="x (array pixel)",
        ylabel="y (array pixel)",
    )
    axes[0, 1].plot(np.arange(-half_a, half_a + 1), norm_a[half_a])
    axes[0, 1].set(
        title="Model A centre profile", xlabel="array pixel", ylabel="peak-normalized intensity"
    )
    b_view = axes[1, 0].imshow(
        log_b,
        cmap="magma",
        vmin=-8,
        vmax=0,
        origin="lower",
        extent=[coordinate_b_um[0], coordinate_b_um[-1]] * 2,
    )
    axes[1, 0].set(
        title=f"Model B-circle {pupil_diameter_mm:g} mm zero-phase PSF",
        xlabel="x (um)",
        ylabel="y (um)",
    )
    axes[1, 1].plot(coordinate_b_um, norm_b[half_b])
    axes[1, 1].set(
        title="Model B centre profile", xlabel="x (um)", ylabel="peak-normalized intensity"
    )
    fig.colorbar(a_view, ax=axes[0, 0], label="log10 normalized intensity")
    fig.colorbar(b_view, ax=axes[1, 0], label="log10 normalized intensity")
    for axis in (axes[0, 1], axes[1, 1]):
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.05)
    fig.suptitle("Zero known phase and zero system aberration; actual A/B pupil assumptions")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/orpam_b_to_a/dataset_nfft2160/object_true_normalized.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/orpam_random_zernike_ab_demo_seed20261028",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--num-modes", type=int, default=28)
    parser.add_argument("--phase-rms-rad", type=float, default=1.0)
    parser.add_argument("--fft-size", type=int, default=2160)
    parser.add_argument("--kernel-radius", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--a-pupil", choices=("square", "circle"), default="square")
    parser.add_argument("--b-pupil-diameter-mm", type=float, default=6.9)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    object_path = args.object.expanduser().resolve()
    object_image = np.load(object_path, allow_pickle=False).astype(np.float32)
    if object_image.shape != (600, 600) or not np.isfinite(object_image).all():
        raise ValueError("The object must be one finite 600x600 array.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    coefficients, phase_a, phase_b, unscaled = random_zernike_phase_pair(
        seed=args.seed,
        num_modes=args.num_modes,
        target_rms_rad=args.phase_rms_rad,
    )
    object_tensor = torch.from_numpy(object_image).to(device)[None, None]
    phase_a_tensor = torch.from_numpy(phase_a).to(device)
    zero_a = torch.zeros_like(phase_a_tensor)
    mask_a = None
    if args.a_pupil == "circle":
        mask_a = circular_pupil_mask(
            600,
            SLM_WIDTH_M,
            args.b_pupil_diameter_mm * 1e-3,
            device=device,
        )

    with torch.no_grad():
        image_a_tensor, kernel_a_aberrated = model_a_diagnostic_forward(
            object_tensor,
            zero_a,
            phase_a_tensor,
            pupil_mask=mask_a,
        )
        amplitude_a = (
            torch.ones_like(phase_a_tensor)
            if mask_a is None
            else mask_a.to(device=device, dtype=phase_a_tensor.dtype)
        )
        optical_psf_a_aberrated = pupil_psf(
            (amplitude_a * torch.exp(1j * phase_a_tensor))[None, None],
            torch.ones((1, 1, 600, 600), dtype=torch.complex64, device=device),
        )
        optical_psf_a_zero = pupil_psf(
            amplitude_a.to(torch.complex64)[None, None],
            torch.ones((1, 1, 600, 600), dtype=torch.complex64, device=device),
        )

        config = ORPAMModelBConfig(
            fft_size=args.fft_size,
            kernel_radius=args.kernel_radius,
            quadrature_order=args.quadrature_order,
            pupil_shape="circle",
            pupil_diameter_m=args.b_pupil_diameter_mm * 1e-3,
        )
        model_b = ORPAMForwardModelB(config).to(device)
        phase_b_tensor = torch.from_numpy(phase_b).to(device)
        zero_b = torch.zeros_like(phase_b_tensor)
        optical_psf_b_aberrated = model_b.optical_psf(zero_b, phase_b_tensor)
        kernel_b_aberrated = model_b.equivalent_kernel(
            optical_psf_b_aberrated.intensity
        )
        image_b_tensor = model_b.scan_forward(object_tensor, kernel_b_aberrated)
        optical_psf_b_zero = model_b.optical_psf(zero_b, zero_b)

    image_a = image_a_tensor[0].cpu().numpy().astype(np.float32)
    image_b = image_b_tensor[0, 0].cpu().numpy().astype(np.float32)
    psf_a_zero = optical_psf_a_zero[0, 0].cpu().numpy().astype(np.float32)
    psf_a_aberrated = optical_psf_a_aberrated[0, 0].cpu().numpy().astype(np.float32)
    psf_b_zero = optical_psf_b_zero.intensity[0, 0].cpu().numpy().astype(np.float32)
    psf_b_aberrated = (
        optical_psf_b_aberrated.intensity[0, 0].cpu().numpy().astype(np.float32)
    )
    focus_coordinate_m = optical_psf_b_zero.x_m.cpu().numpy().astype(np.float32)
    kernel_a = kernel_a_aberrated[0, 0].cpu().numpy().astype(np.float32)
    kernel_b = kernel_b_aberrated[0, 0].cpu().numpy().astype(np.float32)
    metrics_a = _image_metrics(object_image, image_a)
    metrics_b = _image_metrics(object_image, image_b)
    energy_b = kernel_energy_diagnostics(kernel_b_aberrated)
    spread_a = _kernel_spread(kernel_a)
    spread_b = _kernel_spread(kernel_b)
    active_a = np.ones_like(phase_a, dtype=bool)
    if mask_a is not None:
        active_a = mask_a.cpu().numpy().astype(bool)
    effective_phase_a = phase_a[active_a].astype(np.float64)
    effective_phase_a -= float(effective_phase_a.mean())
    effective_phase_a_rms = float(np.sqrt(np.mean(effective_phase_a**2)))
    circle_mask = pupil_amplitude_mask(config).cpu().numpy().astype(bool)
    effective_phase_b = phase_b[circle_mask].astype(np.float64)
    effective_phase_b -= float(effective_phase_b.mean())
    effective_phase_b_rms = float(np.sqrt(np.mean(effective_phase_b**2)))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        np.save(temporary / "zernike_phase_model_a_600.npy", phase_a)
        np.save(temporary / "zernike_phase_model_b_1080.npy", phase_b)
        np.save(temporary / "object_original.npy", object_image)
        np.save(temporary / "model_a_aberrated_image.npy", image_a)
        np.save(temporary / "model_b_aberrated_image.npy", image_b)
        tifffile.imwrite(temporary / "zernike_phase_model_a_600.tif", phase_a)
        tifffile.imwrite(temporary / "zernike_phase_model_b_1080.tif", phase_b)
        tifffile.imwrite(temporary / "object_original.tif", object_image)
        tifffile.imwrite(temporary / "model_a_aberrated_image.tif", image_a)
        tifffile.imwrite(temporary / "model_b_aberrated_image.tif", image_b)
        np.savez_compressed(
            temporary / "model_a_numerical_results.npz",
            zero_phase_psf=psf_a_zero,
            aberrated_psf=psf_a_aberrated,
            aberrated_kernel_after_legacy_flip=kernel_a,
        )
        np.savez_compressed(
            temporary / "model_b_numerical_results.npz",
            zero_phase_psf=psf_b_zero,
            aberrated_psf=psf_b_aberrated,
            focus_coordinate_m=focus_coordinate_m,
            aberrated_equivalent_kernel=kernel_b,
        )

        with (temporary / "zernike_coefficients.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("noll_index", "coefficient_rad", "unscaled_random_draw"),
                lineterminator="\n",
            )
            writer.writeheader()
            for index, (coefficient, draw) in enumerate(
                zip(coefficients, unscaled), start=1
            ):
                writer.writerow(
                    {
                        "noll_index": index,
                        "coefficient_rad": float(coefficient),
                        "unscaled_random_draw": float(draw),
                    }
                )
        metric_rows = [
            {"model": "A", **metrics_a},
            {"model": "B", **metrics_b},
        ]
        with (temporary / "image_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=tuple(metric_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(metric_rows)

        display_maximum = float(max(object_image.max(), image_a.max(), image_b.max()))
        _write_grayscale_png(
            temporary / "object_original.png", object_image, display_maximum
        )
        _write_grayscale_png(
            temporary / "model_a_aberrated_image.png", image_a, display_maximum
        )
        _write_grayscale_png(
            temporary / "model_b_aberrated_image.png", image_b, display_maximum
        )
        _write_phase_figure(
            temporary / "zernike_phase_and_coefficients.png", phase_a, coefficients
        )
        _write_image_overview(
            temporary / "aberrated_image_comparison.png",
            phase_a,
            object_image,
            image_a,
            image_b,
            metrics_a,
            metrics_b,
            args.b_pupil_diameter_mm,
            args.a_pupil,
        )
        _write_psf_figure(
            temporary / "zero_phase_psf_comparison.png",
            psf_a_zero,
            psf_b_zero,
            focus_coordinate_m,
            args.b_pupil_diameter_mm,
            args.a_pupil,
        )

        phase_a_centered = phase_a.astype(np.float64) - float(
            phase_a.mean(dtype=np.float64)
        )
        phase_b_centered = phase_b.astype(np.float64) - float(
            phase_b.mean(dtype=np.float64)
        )
        report = {
            "experiment": "random Zernike aberration through model A and model B",
            "object_source": str(object_path),
            "known_slm_phase": "zero for both aberrated-image simulations",
            "pupil": {
                "model_a": (
                    "original NeuWS uniform square pupil"
                    if args.a_pupil == "square"
                    else f"diagnostic uniform hard circular pupil, diameter {args.b_pupil_diameter_mm:g} mm"
                ),
                "model_b": f"uniform hard circular pupil, diameter {args.b_pupil_diameter_mm:g} mm",
            },
            "zernike": {
                "basis": "project AOtools Noll-normalized basis",
                "seed": args.seed,
                "num_modes": args.num_modes,
                "random_modes": "Noll 4 through num_modes",
                "constraints": "piston adjusted to zero mean; Noll 2 and 3 tip/tilt set to zero",
                "coefficients_rad": coefficients.tolist(),
                "phase_600_rms_rad": float(
                    np.sqrt(np.mean(phase_a_centered**2))
                ),
                "phase_1080_rms_rad": float(
                    np.sqrt(np.mean(phase_b_centered**2))
                ),
                "phase_600_peak_to_valley_rad": float(phase_a.max() - phase_a.min()),
                "phase_1080_peak_to_valley_rad": float(phase_b.max() - phase_b.min()),
            },
            "model_a": {
                "grid": [600, 600],
                "forward": "original NeuWS 600 FFT, sum-normalized PSF, PSF flip, padded FFT convolution",
                "zero_phase_psf_sum": float(psf_a_zero.sum(dtype=np.float64)),
                "effective_phase_rms_rad": effective_phase_a_rms,
                "aberrated_kernel_spread_scan_pixels": spread_a,
                "metrics_against_original_object": metrics_a,
            },
            "model_b": {
                **config.to_record(),
                "zero_phase_psf_area_integral": float(
                    psf_b_zero.sum(dtype=np.float64) * config.focus_pixel_pitch_m**2
                ),
                "effective_phase_rms_inside_circle_rad": effective_phase_b_rms,
                "aberrated_kernel_spread_scan_pixels": spread_b,
                "aberrated_kernel_retained_energy": float(
                    energy_b["retained_energy"][0].cpu()
                ),
                "aberrated_kernel_truncation_error": float(
                    energy_b["truncation_error"][0].cpu()
                ),
                "metrics_against_original_object": metrics_b,
            },
            "metric_definition": {
                "comparison": "raw simulated image versus object_original; no gain fit or image normalization",
                "data_range": float(object_image.max() - object_image.min()),
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        readme = f"""# Random Zernike A/B forward demonstration

- Seed: `{args.seed}`
- Zernike modes: Noll 1–{args.num_modes}; random modes 4–{args.num_modes}
- Phase RMS: A `{report['zernike']['phase_600_rms_rad']:.6f} rad`, B `{report['zernike']['phase_1080_rms_rad']:.6f} rad`
- Effective pupil phase RMS: A-{args.a_pupil} `{report['model_a']['effective_phase_rms_rad']:.6f} rad`, B-circle `{effective_phase_b_rms:.6f} rad`
- Aberrated-kernel RMS radius: A `{spread_a['rms_radius_pixel']:.6f}` scan pixels, B `{spread_b['rms_radius_pixel']:.6f}` scan pixels
- Aberrated-kernel 90% energy radius: A `{spread_a['radius_90pct_energy_pixel']:.6f}` scan pixels, B `{spread_b['radius_90pct_energy_pixel']:.6f}` scan pixels
- Known SLM phase: zero
- Pupil: model A `{args.a_pupil}`; model B hard circle `{args.b_pupil_diameter_mm:g} mm`
- Model B: NFFT={args.fft_size}, kernel radius={args.kernel_radius}, quadrature={args.quadrature_order}
- Metric scale: raw fixed scale; no gain fit and no output normalization

| Model | SSIM vs original | PSNR vs original | MSE | Relative L2 |
|---|---:|---:|---:|---:|
| A | {metrics_a['ssim']:.8f} | {metrics_a['psnr_db']:.6f} dB | {metrics_a['mse']:.8e} | {metrics_a['relative_l2']:.8f} |
| B | {metrics_b['ssim']:.8f} | {metrics_b['psnr_db']:.6f} dB | {metrics_b['mse']:.8e} | {metrics_b['relative_l2']:.8f} |

Start with `aberrated_image_comparison.png`, `zero_phase_psf_comparison.png`, and
`zernike_phase_and_coefficients.png`. TIFF/NPY/NPZ files preserve float data.
"""
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output_dir), "A": metrics_a, "B": metrics_b}, indent=2))
    return output_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
