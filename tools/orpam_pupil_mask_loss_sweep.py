#!/usr/bin/env python3
"""Generate circular-pupil model-B data and scan fixed model-A phase scale.

This is a diagnostic workflow only.  It never constructs an optimizer or a
trainable network: the object, pupil amplitude, known SLM phases, and tested
aberration are explicit tensors.  The model-A helper below reproduces the
legacy NeuWS FFT, PSF normalization, PSF flip, and padded FFT convolution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from torch.fft import fft2, fftshift


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pawavefrontlab-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataio.orpam_slm import HardwareSLMPhaseLibrary  # noqa: E402
from optics.legacy import fft_2xPad_Conv2D  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    kernel_energy_diagnostics,
    pupil_amplitude_mask,
)


DEFAULT_ALPHAS = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SLM_WIDTH_M = 1080 * 6.4e-6
CURVE_SPECS = (
    ("B-square / A-square", "B-square", "A-square", None),
    ("B-circle-6.9 / A-square", "B-circle-6.9mm", "A-square", 6.9),
    ("B-circle-6.9 / A-circle-6.9", "B-circle-6.9mm", "A-circle-6.9mm", 6.9),
    ("B-circle-6.5 / A-square", "B-circle-6.5mm", "A-square", 6.5),
    ("B-circle-6.5 / A-circle-6.5", "B-circle-6.5mm", "A-circle-6.5mm", 6.5),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must not exist or must be empty: {path}")


def circular_pupil_mask(
    size: int,
    physical_width_m: float,
    diameter_m: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a hard circular pupil at pixel centres on a square physical grid."""

    if type(size) is not int or size <= 0 or size % 2:
        raise ValueError("size must be a positive even integer.")
    if not math.isfinite(physical_width_m) or physical_width_m <= 0:
        raise ValueError("physical_width_m must be finite and positive.")
    if not math.isfinite(diameter_m) or not 0 < diameter_m <= physical_width_m:
        raise ValueError("diameter_m must lie in (0, physical_width_m].")
    pitch = physical_width_m / size
    coordinate = (
        torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    ) * pitch
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return (x.square() + y.square() <= (0.5 * diameter_m) ** 2).to(dtype)


def model_a_diagnostic_forward(
    object_image: torch.Tensor,
    known_phase: torch.Tensor,
    aberration_phase: torch.Tensor,
    *,
    pupil_mask: torch.Tensor | None = None,
    phase_sign: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the original NeuWS model-A optical and convolution formula.

    Returns predictions with shape ``[B,H,W]`` and the already-flipped PSF
    kernels with shape ``[B,1,H,W]``.  A circle changes only the SLM-plane
    amplitude before the same 600 x 600 FFT; no model-B physical resampling or
    5 um cell integration appears here.
    """

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1.")
    if object_image.ndim == 2:
        object_image = object_image[None, None]
    if known_phase.ndim == 2:
        known_phase = known_phase[None, None]
    elif known_phase.ndim == 3:
        known_phase = known_phase[:, None]
    if aberration_phase.ndim == 2:
        aberration_phase = aberration_phase[None, None]
    elif aberration_phase.ndim == 3:
        aberration_phase = aberration_phase[:, None]
    if object_image.ndim != 4 or object_image.shape[:2] != (1, 1):
        raise ValueError("object_image must be one single-channel image.")
    if known_phase.ndim != 4 or known_phase.shape[1] != 1:
        raise ValueError("known_phase must have shape [B,1,H,W] or an equivalent form.")
    if aberration_phase.ndim != 4 or aberration_phase.shape[1] != 1:
        raise ValueError("aberration_phase must have shape [B,1,H,W] or equivalent.")
    spatial_shape = object_image.shape[-2:]
    if spatial_shape[0] != spatial_shape[1] or spatial_shape[0] % 2:
        raise ValueError("model A requires a square even-sized object.")
    if known_phase.shape[-2:] != spatial_shape or aberration_phase.shape[-2:] != spatial_shape:
        raise ValueError("object and phase spatial shapes must match.")
    batch = known_phase.shape[0]
    if aberration_phase.shape[0] not in (1, batch):
        raise ValueError("aberration batch must be one or match known_phase.")
    if not all(torch.isfinite(value).all() for value in (object_image, known_phase, aberration_phase)):
        raise ValueError("model-A diagnostic inputs must be finite.")
    object_image = object_image.to(device=known_phase.device, dtype=known_phase.dtype)
    aberration_phase = aberration_phase.to(device=known_phase.device, dtype=known_phase.dtype)
    # Preserve the original NeuWS floating-point operation order as well as
    # its mathematics: the unknown and known complex phase factors are built
    # separately and then multiplied.
    field = torch.exp(1j * aberration_phase) * torch.exp(
        1j * phase_sign * known_phase
    )
    if pupil_mask is not None:
        if pupil_mask.ndim == 2:
            pupil_mask = pupil_mask[None, None]
        if pupil_mask.shape != (1, 1, *spatial_shape):
            raise ValueError("pupil_mask must have the same spatial shape as the phases.")
        field = field * pupil_mask.to(device=field.device, dtype=known_phase.dtype)
    kernel = fftshift(fft2(field, norm="forward"), dim=(-2, -1)).abs().square()
    kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)
    kernel = kernel.flip(-2).flip(-1)
    prediction = fft_2xPad_Conv2D(object_image, kernel)
    if prediction.shape != (1, batch, *spatial_shape):
        raise RuntimeError(f"Unexpected legacy convolution shape: {prediction.shape}.")
    return prediction[0], kernel


def _load_mat(path: Path, variable: str) -> np.ndarray:
    values = sio.loadmat(path)
    if variable not in values:
        raise KeyError(f"{path} does not contain {variable!r}.")
    value = np.asarray(values[variable]).squeeze()
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{path}:{variable} must be a finite 2-D array.")
    return value


def _copy_unified_sources(base_dir: Path, destination: Path) -> None:
    for name in ("object_true_normalized.npy", "phi_true_600.npy", "phi_true_1080.npy"):
        source = base_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)


def generate_circle_dataset(args: argparse.Namespace) -> Path:
    base_dir = Path(args.base_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase_root = Path(args.phase_root).expanduser().resolve()
    if output_dir == base_dir:
        raise ValueError("Circular data must not overwrite the square data directory.")
    _empty_destination(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        object_true = np.load(base_dir / "object_true_normalized.npy", allow_pickle=False)
        phi_true_hw = np.load(base_dir / "phi_true_1080.npy", allow_pickle=False)
        phi_true_model = np.load(base_dir / "phi_true_600.npy", allow_pickle=False)
        if object_true.shape != (600, 600) or phi_true_model.shape != (600, 600):
            raise ValueError("Base object and model-grid truth must both be 600x600.")
        if phi_true_hw.shape != (1080, 1080):
            raise ValueError("Base hardware-grid truth must be 1080x1080.")
        library = HardwareSLMPhaseLibrary(phase_root)
        if (
            library.shape != (1080, 1080)
            or args.num_frames <= 0
            or args.num_frames > library.num_frames
        ):
            raise ValueError(
                "Requested frames must be a positive prefix of the 50-frame "
                "1080 phase library."
            )
        config = ORPAMModelBConfig(
            fft_size=args.fft_size,
            kernel_radius=args.kernel_radius,
            quadrature_order=args.quadrature_order,
            phase_sign=library.phase_sign,
            pupil_shape="circle",
            pupil_diameter_m=args.pupil_diameter_mm * 1e-3,
        )
        simulator = ORPAMForwardModelB(config)
        mask = pupil_amplitude_mask(config).cpu().numpy().astype(np.uint8)
        np.save(temporary / "pupil_mask.npy", mask)
        _copy_unified_sources(base_dir, temporary)
        object_tensor = torch.from_numpy(np.asarray(object_true, dtype=np.float32))[None, None]
        truth_tensor = torch.from_numpy(np.asarray(phi_true_hw, dtype=np.float32))
        frame_records: list[dict[str, object]] = []
        selected_crops: dict[str, np.ndarray] = {}
        crop_half_width_m = 200e-6
        crop_half_pixels = int(round(crop_half_width_m / config.focus_pixel_pitch_m))
        centre = config.fft_size // 2
        crop_slice = slice(centre - crop_half_pixels, centre + crop_half_pixels + 1)
        global_maximum = 0.0
        start = time.perf_counter()
        with torch.no_grad():
            for frame in range(1, args.num_frames + 1):
                frame_start = time.perf_counter()
                phase = library.load(frame, validate_png=True)
                optical = simulator.optical_psf(phase, truth_tensor)
                kernel = simulator.equivalent_kernel(optical.intensity)
                observation = simulator.scan_forward(object_tensor, kernel)
                image = observation[0, 0].cpu().numpy().astype(np.float32)
                if image.shape != (600, 600) or not np.isfinite(image).all():
                    raise RuntimeError(f"Invalid generated frame {frame}.")
                sio.savemat(
                    temporary / f"SLM_raw{frame}.mat",
                    {"imsdata": image},
                    do_compression=True,
                )
                phase_source = base_dir / f"SLM_sim{frame}.mat"
                if not phase_source.is_file():
                    raise FileNotFoundError(phase_source)
                shutil.copy2(phase_source, temporary / phase_source.name)
                psf = optical.intensity[0, 0]
                dx = config.focus_pixel_pitch_m
                x = optical.x_m
                marginal_x = psf.sum(dim=0) * dx**2
                marginal_y = psf.sum(dim=1) * dx**2
                centroid_x = float((marginal_x * x).sum())
                centroid_y = float((marginal_y * x).sum())
                energy = kernel_energy_diagnostics(kernel)
                global_maximum = max(global_maximum, float(image.max()))
                frame_records.append(
                    {
                        "frame": frame,
                        "minimum": float(image.min()),
                        "maximum": float(image.max()),
                        "mean": float(image.mean()),
                        "psf_peak_per_m2": float(psf.max()),
                        "psf_centroid_x_m": centroid_x,
                        "psf_centroid_y_m": centroid_y,
                        "kernel_retained_energy": float(energy["retained_energy"][0]),
                        "kernel_truncation_error": float(energy["truncation_error"][0]),
                        "elapsed_seconds": time.perf_counter() - frame_start,
                    }
                )
                if frame in {1, 25, 50}:
                    selected_crops[f"psf_frame_{frame}"] = (
                        psf[crop_slice, crop_slice].cpu().numpy().astype(np.float32)
                    )
                del phase, optical, kernel, observation
                if frame % 10 == 0:
                    print(f"Generated circle frame {frame}/{args.num_frames}", flush=True)
        coordinate_um = (
            np.arange(2 * crop_half_pixels + 1) - crop_half_pixels
        ) * config.focus_pixel_pitch_m * 1e6
        np.savez_compressed(
            temporary / "psf_diagnostics.npz",
            coordinate_um=coordinate_um.astype(np.float32),
            **selected_crops,
        )
        manifest = {
            "schema_version": 1,
            "experiment": "OR-PAM model B circular-pupil fixed-aberration data",
            "size": 600,
            "aperture_height": 600,
            "num_frames": args.num_frames,
            "measurement_variable": "imsdata",
            "phase_variable": "proj_sim",
            "phase_sign": library.phase_sign,
            "measurement_max": global_maximum,
            "per_frame_minmax_applied": False,
            "normalization_note": "raw frames saved; one shared maximum recorded only",
            "pupil": {
                "shape": "circle",
                "diameter_m": config.pupil_diameter_m,
                "diameter_mm": args.pupil_diameter_mm,
                "binary_hard_edge": True,
                "active_pixel_count": int(mask.sum()),
                "illuminated_area_m2_pixel_sum": float(mask.sum() * config.slm_pixel_pitch_m**2),
                "absolute_power_note": (
                    "PSF is normalized to unit physical-area integral; aperture clipping "
                    "does not reduce total simulated optical power."
                ),
            },
            "model_b": config.to_record(),
            "source": {
                "base_square_data_dir": str(base_dir),
                "phase_library": str(phase_root),
                "object_sha256": _sha256(base_dir / "object_true_normalized.npy"),
                "phi_true_600_sha256": _sha256(base_dir / "phi_true_600.npy"),
                "phi_true_1080_sha256": _sha256(base_dir / "phi_true_1080.npy"),
                "same_object_and_truth_files_copied": True,
                "known_phase_files_copied_from_base_by_frame": True,
            },
            "generation": {
                "device": "cpu",
                "torch_no_grad": True,
                "sequential_frames": True,
                "elapsed_seconds": time.perf_counter() - start,
            },
            "frames": frame_records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "parameters.json").write_text(
            json.dumps(config.to_record(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Circular model-B data written to {output_dir}")
    return output_dir


def validate_dataset_alignment(
    data_dirs: Iterable[str | Path], *, num_frames: int = 50
) -> dict[str, object]:
    """Require byte-identical object/truth arrays and phase values by frame."""

    directories = [Path(path).expanduser().resolve() for path in data_dirs]
    if len(directories) < 2:
        raise ValueError("At least two data directories are required for alignment.")
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    reference = directories[0]
    names = ("object_true_normalized.npy", "phi_true_600.npy")
    hashes = {name: _sha256(reference / name) for name in names}
    reference_phases = [
        _load_mat(reference / f"SLM_sim{frame}.mat", "proj_sim")
        for frame in range(1, num_frames + 1)
    ]
    phase_digest = hashlib.sha256()
    for value in reference_phases:
        phase_digest.update(np.ascontiguousarray(value).view(np.uint8))
    for directory in directories[1:]:
        for name in names:
            if _sha256(directory / name) != hashes[name]:
                raise ValueError(f"{directory}/{name} differs from {reference}/{name}.")
        for frame, expected in enumerate(reference_phases, start=1):
            observed = _load_mat(directory / f"SLM_sim{frame}.mat", "proj_sim")
            if not np.array_equal(observed, expected):
                raise ValueError(f"Phase frame {frame} differs in {directory}.")
            if not (directory / f"SLM_raw{frame}.mat").is_file():
                raise FileNotFoundError(directory / f"SLM_raw{frame}.mat")
    return {
        "num_frames": num_frames,
        "object_sha256": hashes["object_true_normalized.npy"],
        "phi_true_600_sha256": hashes["phi_true_600.npy"],
        "phase_value_sha256": phase_digest.hexdigest(),
        "directories": [str(path) for path in directories],
    }


def _load_observations(directory: Path, num_frames: int) -> np.ndarray:
    values = [
        _load_mat(directory / f"SLM_raw{frame}.mat", "imsdata").astype(np.float32)
        for frame in range(1, num_frames + 1)
    ]
    result = np.stack(values)
    if result.shape != (num_frames, 600, 600) or float(result.min()) < 0:
        raise ValueError(f"Invalid observations in {directory}: {result.shape}.")
    return result


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction64 = prediction.astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    denominator = float(np.sum(prediction64 * prediction64))
    if denominator <= 0:
        raise ValueError("Prediction has no energy.")
    gain = float(np.sum(prediction64 * target64) / denominator)
    residual = prediction64 - target64
    gained_residual = gain * prediction64 - target64
    target_norm = np.linalg.norm(target64.reshape(target.shape[0], -1), axis=1)
    if np.any(target_norm <= 0):
        raise ValueError("A target frame has no energy.")
    raw_relative = np.linalg.norm(
        residual.reshape(target.shape[0], -1), axis=1
    ) / target_norm
    gained_relative = np.linalg.norm(
        gained_residual.reshape(target.shape[0], -1), axis=1
    ) / target_norm
    return {
        "raw_mse": float(np.mean(residual**2)),
        "global_gain": gain,
        "gain_mse": float(np.mean(gained_residual**2)),
        "mean_relative_l2": float(np.mean(raw_relative)),
        "gain_mean_relative_l2": float(np.mean(gained_relative)),
    }


def _model_a_predictions(
    object_true: np.ndarray,
    known_phases: np.ndarray,
    phi_true: np.ndarray,
    alpha: float,
    pupil_mask: torch.Tensor | None,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    object_tensor = torch.from_numpy(object_true).to(device=device, dtype=torch.float32)[None, None]
    aberration = torch.from_numpy(phi_true).to(device=device, dtype=torch.float32) * alpha
    mask = None if pupil_mask is None else pupil_mask.to(device=device, dtype=torch.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, known_phases.shape[0], batch_size):
            phase = torch.from_numpy(known_phases[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            prediction, _ = model_a_diagnostic_forward(
                object_tensor,
                phase,
                aberration,
                pupil_mask=mask,
                phase_sign=-1,
            )
            outputs.append(prediction.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_curve(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: float(row["alpha"]))
    alpha_one = next((row for row in ordered if abs(float(row["alpha"]) - 1.0) < 1e-9), None)
    if alpha_one is None:
        raise ValueError("The alpha grid must contain alpha=1.")
    result: dict[str, object] = {
        "observation": ordered[0]["observation"],
        "predictor": ordered[0]["predictor"],
        "pupil_diameter_mm": ordered[0]["pupil_diameter_mm"],
        "alpha_count": len(ordered),
    }
    for metric in ("raw_mse", "gain_mse", "mean_relative_l2"):
        best = min(ordered, key=lambda row: float(row[metric]))
        at_one = float(alpha_one[metric])
        minimum = float(best[metric])
        result[metric] = {
            "alpha_best": float(best["alpha"]),
            "minimum": minimum,
            "alpha_1": at_one,
            "alpha_1_minus_minimum": at_one - minimum,
        }
    gain_best = min(ordered, key=lambda row: float(row["gain_mse"]))
    result["global_gain_at_gain_mse_best"] = float(gain_best["global_gain"])
    result["sampled_minimum_at_boundary"] = any(
        float(result[metric]["alpha_best"]) in {float(ordered[0]["alpha"]), float(ordered[-1]["alpha"])}
        for metric in ("raw_mse", "gain_mse", "mean_relative_l2")
    )
    return result


def _plot_loss(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    styles = ["o-", "s-", "^-", "D-", "v-"]
    for style, (name, group) in zip(
        styles,
        ((name, [row for row in rows if row["curve"] == name]) for name in dict.fromkeys(row["curve"] for row in rows)),
    ):
        group = sorted(group, key=lambda row: float(row["alpha"]))
        alpha = [float(row["alpha"]) for row in group]
        for axis, metric in zip(axes, ("raw_mse", "gain_mse", "mean_relative_l2")):
            axis.plot(alpha, [float(row[metric]) for row in group], style, ms=3, label=name)
    titles = ("Fixed-scale MSE", "One global-gain MSE", "Mean frame relative L2")
    for axis, title in zip(axes, titles):
        axis.axvline(1.0, color="black", ls="--", lw=1, alpha=0.7)
        axis.set(xlabel="alpha", title=title)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("loss")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Fixed-object, unit-amplitude model-A phase-scale sweep (50 frames)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_pupils(output: Path) -> None:
    masks = [
        torch.ones(600, 600),
        circular_pupil_mask(600, SLM_WIDTH_M, 6.9e-3),
        circular_pupil_mask(600, SLM_WIDTH_M, 6.5e-3),
    ]
    titles = ("A-square", "A-circle 6.9 mm", "A-circle 6.5 mm")
    extent = [-SLM_WIDTH_M * 0.5e3, SLM_WIDTH_M * 0.5e3] * 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for axis, mask, title in zip(axes, masks, titles):
        axis.imshow(mask, cmap="gray", vmin=0, vmax=1, extent=extent, origin="lower")
        axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle("Physical pupil masks mapped to the 600 x 600 model-A grid")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_psf_comparison(
    phase_root: Path, phi_true_hw: np.ndarray, output: Path, *, fft_size: int
) -> None:
    library = HardwareSLMPhaseLibrary(phase_root)
    known = library.load(1)
    configurations = [
        ("B-square", ORPAMModelBConfig(fft_size=fft_size)),
        (
            "B-circle 6.9 mm",
            ORPAMModelBConfig(fft_size=fft_size, pupil_shape="circle", pupil_diameter_m=6.9e-3),
        ),
        (
            "B-circle 6.5 mm",
            ORPAMModelBConfig(fft_size=fft_size, pupil_shape="circle", pupil_diameter_m=6.5e-3),
        ),
    ]
    crops = []
    extent = None
    with torch.no_grad():
        for title, config in configurations:
            optical = ORPAMForwardModelB(config).optical_psf(
                known, torch.from_numpy(phi_true_hw.astype(np.float32))
            )
            half = int(round(160e-6 / config.focus_pixel_pitch_m))
            centre = fft_size // 2
            crop = optical.intensity[0, 0, centre - half : centre + half + 1, centre - half : centre + half + 1]
            crops.append((title, crop.cpu().numpy()))
            extent = [-half * config.focus_pixel_pitch_m * 1e6, half * config.focus_pixel_pitch_m * 1e6] * 2
    global_peak = max(float(value.max()) for _, value in crops)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for axis, (title, value) in zip(axes, crops):
        shown = axis.imshow(
            np.log10(np.maximum(value / global_peak, 1e-8)),
            cmap="magma",
            vmin=-6,
            vmax=0,
            extent=extent,
            origin="lower",
        )
        axis.set(title=title, xlabel="x (um)", ylabel="y (um)")
    fig.colorbar(shown, ax=axes, label="log10(global-peak normalized PSF)", shrink=0.85)
    fig.suptitle("Model-B frame 1 PSF comparison; common physical/log scale")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> Path:
    square_dir = Path(args.square_data_dir).expanduser().resolve()
    circle69_dir = Path(args.circle_69_data_dir).expanduser().resolve()
    circle65_dir = Path(args.circle_65_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _empty_destination(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = validate_dataset_alignment(
        (square_dir, circle69_dir, circle65_dir), num_frames=args.num_frames
    )
    object_true = np.load(square_dir / "object_true_normalized.npy", allow_pickle=False).astype(np.float32)
    phi_true = np.load(square_dir / "phi_true_600.npy", allow_pickle=False).astype(np.float32)
    phi_true_hw = np.load(square_dir / "phi_true_1080.npy", allow_pickle=False).astype(np.float32)
    known_phases = np.stack(
        [
            _load_mat(square_dir / f"SLM_sim{frame}.mat", "proj_sim").astype(np.float32)
            for frame in range(1, args.num_frames + 1)
        ]
    )
    targets = {
        "B-square": _load_observations(square_dir, args.num_frames),
        "B-circle-6.9mm": _load_observations(circle69_dir, args.num_frames),
        "B-circle-6.5mm": _load_observations(circle65_dir, args.num_frames),
    }
    alphas = sorted(set(float(value) for value in args.alphas))
    if not alphas or not any(abs(value - 1.0) < 1e-9 for value in alphas):
        raise ValueError("The alpha list must be non-empty and contain 1.0.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the diagnostic sweep but unavailable.")
    operators = {
        "A-square": None,
        "A-circle-6.9mm": circular_pupil_mask(600, SLM_WIDTH_M, 6.9e-3),
        "A-circle-6.5mm": circular_pupil_mask(600, SLM_WIDTH_M, 6.5e-3),
    }
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for operator_name, mask in operators.items():
        matching_curves = [curve for curve in CURVE_SPECS if curve[2] == operator_name]
        for alpha in alphas:
            prediction = _model_a_predictions(
                object_true,
                known_phases,
                phi_true,
                alpha,
                mask,
                device=device,
                batch_size=args.batch_size,
            )
            for curve_name, observation_name, _, diameter in matching_curves:
                rows.append(
                    {
                        "curve": curve_name,
                        "observation": observation_name,
                        "predictor": operator_name,
                        "pupil_diameter_mm": "" if diameter is None else diameter,
                        "alpha": alpha,
                        **_metrics(prediction, targets[observation_name]),
                    }
                )
            print(f"{operator_name}: alpha={alpha:g}", flush=True)
            del prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(output_dir / "loss_sweep.csv", rows)
    curve_names = list(dict.fromkeys(str(row["curve"]) for row in rows))
    summaries = {
        name: _summarize_curve([row for row in rows if row["curve"] == name])
        for name in curve_names
    }
    summary = {
        "schema_version": 1,
        "experiment": "fixed_aberration_alpha_sweep_no_training",
        "network_training_performed": False,
        "object_fixed": True,
        "pupil_amplitude_fixed_uniform_inside_mask": True,
        "phase_formula": "Phi_alpha = alpha * Phi_true",
        "phase_sign": "exp(+1j*Phi_alpha) exp(-1j*Gamma_i)",
        "alpha_values": alphas,
        "num_frames": args.num_frames,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "loss_definitions": {
            "raw_mse": "mean over all 50*600*600 pixels with fixed physical scale",
            "gain_mse": "same MSE after one scalar gain fitted jointly to all 50 frames",
            "mean_relative_l2": "mean_i ||prediction_i-target_i||_2 / ||target_i||_2",
            "gain": "dot(prediction,target)/dot(prediction,prediction), no bias",
        },
        "alignment": alignment,
        "curves": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _plot_loss(rows, output_dir / "loss_sweep.png")
    _plot_pupils(output_dir / "pupil_mask_comparison.png")
    _plot_psf_comparison(
        Path(args.phase_root).expanduser().resolve(),
        phi_true_hw,
        output_dir / "psf_comparison.png",
        fft_size=args.fft_size,
    )
    print(f"Loss sweep written to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-circle")
    generate.add_argument("--base-data-dir", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--pupil-diameter-mm", required=True, type=float)
    generate.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    generate.add_argument("--num-frames", type=int, default=50)
    generate.add_argument("--fft-size", type=int, default=2160)
    generate.add_argument("--kernel-radius", type=int, default=32)
    generate.add_argument("--quadrature-order", type=int, default=12)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--square-data-dir", required=True, type=Path)
    sweep.add_argument("--circle-69-data-dir", required=True, type=Path)
    sweep.add_argument("--circle-65-data-dir", required=True, type=Path)
    sweep.add_argument("--output-dir", required=True, type=Path)
    sweep.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    sweep.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    sweep.add_argument("--num-frames", type=int, default=50)
    sweep.add_argument("--batch-size", type=int, default=10)
    sweep.add_argument("--device", default="cuda")
    sweep.add_argument("--fft-size", type=int, default=2160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-circle":
        generate_circle_dataset(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
