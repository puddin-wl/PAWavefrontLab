#!/usr/bin/env python3
"""Prepare and evaluate the one-way OR-PAM model-B to NeuWS model-A experiment.

The ``prepare`` command invokes model B once, without gradients, to make a
NeuWS-compatible 50-frame data set.  Training is deliberately delegated to
the existing ``workflows/reconstruction/reconstruct_neuws.py`` entry point so
that the inverse forward operator remains the unmodified model A.  The
``analyze`` command evaluates that checkpoint against the known object and
fixed aberration.
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

import numpy as np
import scipy.io as sio
import tifffile
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pawavefrontlab-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataio import BatchDataset  # noqa: E402
from dataio.orpam_slm import HardwareSLMPhaseLibrary  # noqa: E402
from networks.mmes import model_from_checkpoint  # noqa: E402
from optics import zernike_basis_numpy  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    kernel_energy_diagnostics,
)


ABERRATION_TERMS = {
    "radial_quadratic_rad": 0.45,
    "square_astigmatism_rad": 0.35,
    "x_cubic_coma_like_rad": 0.30,
}


def fixed_aberration(size: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the recorded grid-independent model-B truth phase in radians."""

    coordinate = (
        torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    ) / (size / 2.0)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    radius2 = x.square() + y.square()
    return (
        ABERRATION_TERMS["radial_quadratic_rad"] * radius2
        + ABERRATION_TERMS["square_astigmatism_rad"] * (x.square() - y.square())
        + ABERRATION_TERMS["x_cubic_coma_like_rad"] * x * radius2
    )


def load_vessel_crop(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Read the native TIFF and apply exactly image[200:800, 200:800]."""

    path = Path(path).expanduser().resolve()
    image = np.asarray(tifffile.imread(path))
    source_shape = list(image.shape)
    image = np.squeeze(image)
    if image.ndim != 2 or image.shape != (1000, 1000):
        raise ValueError(
            f"Expected a singleton-channel 1000x1000 TIFF, got {source_shape}."
        )
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise ValueError(f"Unsupported vessel dtype: {image.dtype}.")
    if not np.isfinite(image).all() or float(image.min()) < 0:
        raise ValueError("Vessel TIFF must contain finite non-negative values.")
    crop = np.ascontiguousarray(image[200:800, 200:800])
    if crop.shape != (600, 600):
        raise RuntimeError(f"Unexpected crop shape: {crop.shape}.")
    maximum = float(crop.max())
    if maximum <= 0:
        raise ValueError("Vessel crop has no positive intensity.")
    record = {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stored_shape": source_shape,
        "squeezed_shape": list(image.shape),
        "dtype": str(image.dtype),
        "source_minimum": float(image.min()),
        "source_maximum": float(image.max()),
        "crop_slice": "image[200:800, 200:800]",
        "crop_shape": list(crop.shape),
        "crop_minimum": float(crop.min()),
        "crop_maximum": maximum,
        "crop_mean": float(crop.mean()),
        "channels": 1,
        "processing": "no denoising/enhancement; float32 conversion divided by crop maximum",
        "linear_scale_divisor": maximum,
    }
    return crop, record


def _empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must not exist or must be empty: {path}")


def _model_phase_source(library: HardwareSLMPhaseLibrary, frame: int) -> Path:
    specification = str(library.manifest["model_output"]["mat"])
    path_text, _ = specification.rsplit(":", 1)
    return library.root / path_text.replace("N", str(frame))


def prepare_dataset(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    _empty_destination(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        crop_native, image_record = load_vessel_crop(args.vessel_image)
        object_true = crop_native.astype(np.float32) / float(crop_native.max())
        tifffile.imwrite(temporary / "object_crop_native.tif", crop_native)
        np.save(temporary / "object_true_normalized.npy", object_true)

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for generation but is unavailable.")
        library = HardwareSLMPhaseLibrary(args.phase_root)
        if args.num_frames != library.num_frames:
            raise ValueError(
                f"This experiment requires all {library.num_frames} real phases; "
                f"got --num-frames {args.num_frames}."
            )
        phi_true_hw = fixed_aberration(library.shape[0], device=device)
        model_size = int(library.manifest["model_shape"][0])
        phi_true_model = fixed_aberration(model_size).cpu().numpy()
        phi_true_hw_np = phi_true_hw.detach().cpu().numpy()
        np.save(temporary / "phi_true_1080.npy", phi_true_hw_np)
        np.save(temporary / "phi_true_600.npy", phi_true_model)
        sio.savemat(
            temporary / "ground_truth.mat",
            {
                "object_true_normalized": object_true,
                "phi_true_600": phi_true_model,
                "phi_true_1080": phi_true_hw_np,
            },
            do_compression=True,
        )

        config = ORPAMModelBConfig(
            slm_size=library.shape[0],
            fft_size=args.fft_size,
            phase_sign=library.phase_sign,
            kernel_radius=args.kernel_radius,
            quadrature_order=args.quadrature_order,
        )
        simulator = ORPAMForwardModelB(config).to(device)
        object_tensor = torch.as_tensor(object_true, device=device)[None, None]
        frame_records: list[dict[str, object]] = []
        global_maximum = 0.0
        start = time.perf_counter()
        with torch.no_grad():
            for frame in range(1, args.num_frames + 1):
                frame_start = time.perf_counter()
                phase = library.load(frame, device=device, validate_png=True)
                optical = simulator.optical_psf(phase, phi_true_hw)
                kernel = simulator.equivalent_kernel(optical.intensity)
                observation = simulator.scan_forward(object_tensor, kernel)
                observation_np = (
                    observation[0, 0].detach().cpu().numpy().astype(np.float32)
                )
                if observation_np.shape != (600, 600):
                    raise RuntimeError(
                        f"Frame {frame} output shape is {observation_np.shape}."
                    )
                if not np.isfinite(observation_np).all() or observation_np.min() < 0:
                    raise RuntimeError(f"Frame {frame} observation is invalid.")
                sio.savemat(
                    temporary / f"SLM_raw{frame}.mat",
                    {"imsdata": observation_np},
                    do_compression=True,
                )
                shutil.copy2(
                    _model_phase_source(library, frame),
                    temporary / f"SLM_sim{frame}.mat",
                )
                energy = kernel_energy_diagnostics(kernel)
                maximum = float(observation_np.max())
                global_maximum = max(global_maximum, maximum)
                frame_records.append(
                    {
                        "frame": frame,
                        "minimum": float(observation_np.min()),
                        "maximum": maximum,
                        "mean": float(observation_np.mean()),
                        "kernel_retained_energy": float(
                            energy["retained_energy"][0].cpu()
                        ),
                        "kernel_truncation_error": float(
                            energy["truncation_error"][0].cpu()
                        ),
                        "elapsed_seconds": time.perf_counter() - frame_start,
                    }
                )
                del phase, optical, kernel, observation
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if frame % 10 == 0:
                    print(f"Generated frame {frame}/{args.num_frames}", flush=True)

        if not math.isfinite(global_maximum) or global_maximum <= 0:
            raise RuntimeError("Generated observations have no positive intensity.")
        centered_truth = phi_true_model - float(phi_true_model.mean())
        manifest = {
            "schema_version": 1,
            "experiment": "OR-PAM model B observations inverted by NeuWS model A",
            "size": 600,
            "aperture_height": 600,
            "num_frames": args.num_frames,
            "phase_sign": library.phase_sign,
            "measurement_variable": "imsdata",
            "phase_variable": "proj_sim",
            "measurement_normalization": "shared-max only during model-A loading",
            "measurement_max": global_maximum,
            "per_frame_minmax_applied": False,
            "object": image_record,
            "object_assumption": "known 600x600 proxy object; not claimed as continuous absorption truth",
            "phi_true": {
                "formula_rad": "0.45*(x^2+y^2)+0.35*(x^2-y^2)+0.30*x*(x^2+y^2)",
                "coordinate": "x,y=(index-(N-1)/2)/(N/2)",
                "terms": ABERRATION_TERMS,
                "phase_sign_in_pupil": "exp(+1j*Phi_true) exp(-1j*Gamma_i)",
                "piston_removed_rms_rad_600": float(
                    np.sqrt(np.mean(centered_truth.astype(np.float64) ** 2))
                ),
                "minimum_rad_600": float(phi_true_model.min()),
                "maximum_rad_600": float(phi_true_model.max()),
            },
            "model_b": config.to_record(),
            "generation": {
                "device": str(device),
                "torch_no_grad": True,
                "sequential_frames": True,
                "elapsed_seconds": time.perf_counter() - start,
                "numerical_limit": (
                    "NFFT=2160 is the validated first-pass setting; representative "
                    "2160-to-4320 image differences were approximately 0.0033-0.0097."
                ),
            },
            "phase_source": {
                "library": str(library.root),
                "hardware_grid": [1080, 1080],
                "model_a_grid": [model_size, model_size],
                "same_random_coefficients": bool(
                    library.manifest["same_coefficients_on_both_grids"]
                ),
                "known_input_excludes_phi_true": True,
            },
            "frames": frame_records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"B→A data set written to {output_dir}")
    return output_dir


def _time_coordinate(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 1:
        return torch.zeros_like(indices, dtype=torch.float32) - 0.5
    return indices.float() / (count - 1) - 0.5


def _center_piston(phase: np.ndarray) -> np.ndarray:
    return np.asarray(phase - np.mean(phase, dtype=np.float64), dtype=np.float32)


def _phase_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = _center_piston(estimate)
    truth = _center_piston(truth)
    residual = estimate - truth
    wrapped = np.angle(np.exp(1j * residual)).astype(np.float32)
    truth_rms = float(np.sqrt(np.mean(truth.astype(np.float64) ** 2)))
    rmse = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
    return {
        "rmse_rad": rmse,
        "mae_rad": float(np.mean(np.abs(residual), dtype=np.float64)),
        "correlation": float(np.corrcoef(estimate.ravel(), truth.ravel())[0, 1]),
        "wrapped_rmse_rad": float(
            np.sqrt(np.mean(wrapped.astype(np.float64) ** 2))
        ),
        "wrapped_mae_rad": float(np.mean(np.abs(wrapped), dtype=np.float64)),
        "estimate_rms_rad": float(
            np.sqrt(np.mean(estimate.astype(np.float64) ** 2))
        ),
        "truth_rms_rad": truth_rms,
        "relative_rmse_to_truth_rms": rmse / truth_rms,
    }


def _array_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate64 = np.asarray(estimate, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    residual = estimate64 - truth64
    estimate_norm = np.linalg.norm(estimate64)
    truth_norm = np.linalg.norm(truth64)
    denominator = float(np.dot(estimate64.ravel(), estimate64.ravel()))
    fitted_gain = (
        float(np.dot(estimate64.ravel(), truth64.ravel()) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": float(np.corrcoef(estimate64.ravel(), truth64.ravel())[0, 1]),
        "normalized_l2": float(
            np.linalg.norm(estimate64 / estimate_norm - truth64 / truth_norm)
        ),
        "negative_fraction": float(np.mean(estimate64 < 0)),
        "shape_only_fitted_gain_truth_per_estimate": fitted_gain,
        "shape_only_gain_fitted_rmse": float(
            np.sqrt(np.mean((fitted_gain * estimate64 - truth64) ** 2))
        ),
    }


def _fit_zernike(phase: np.ndarray, num_modes: int) -> np.ndarray:
    basis = zernike_basis_numpy(num_modes, phase.shape[0])
    design = basis.reshape(num_modes, -1).T.astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(
        design, phase.reshape(-1).astype(np.float64), rcond=None
    )
    return coefficients.astype(np.float32)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(
    output_dir: Path,
    truth_phase: np.ndarray,
    estimate_phase: np.ndarray,
    signed_amplitude: np.ndarray,
    truth_object: np.ndarray,
    estimate_object: np.ndarray,
    zernike_true: np.ndarray,
    zernike_estimate: np.ndarray,
    loss: np.ndarray,
    selected: list[dict[str, object]],
) -> None:
    truth_phase = _center_piston(truth_phase)
    estimate_phase = _center_piston(estimate_phase)
    residual_phase = estimate_phase - truth_phase
    limit = max(float(np.max(np.abs(truth_phase))), float(np.max(np.abs(estimate_phase))))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, value, title in (
        (axes[0, 0], truth_phase, "Phi_true (piston removed)"),
        (axes[0, 1], estimate_phase, "Phi_est from model A (piston removed)"),
    ):
        shown = axis.imshow(value, cmap="twilight", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(shown, ax=axes[0], label="rad", shrink=0.85)
    residual_limit = float(np.percentile(np.abs(residual_phase), 99.5))
    shown = axes[1, 0].imshow(
        residual_phase, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit
    )
    axes[1, 0].set_title("Phi_est - Phi_true")
    axes[1, 0].axis("off")
    fig.colorbar(shown, ax=axes[1, 0], label="rad")
    shown = axes[1, 1].imshow(signed_amplitude, cmap="viridis")
    axes[1, 1].set_title("Recovered signed pupil amplitude")
    axes[1, 1].axis("off")
    fig.colorbar(shown, ax=axes[1, 1])
    fig.savefig(output_dir / "phase_and_amplitude.png", dpi=180)
    plt.close(fig)

    common_max = float(
        np.percentile(np.concatenate((truth_object.ravel(), estimate_object.ravel())), 99.8)
    )
    difference = estimate_object - truth_object
    difference_limit = float(np.percentile(np.abs(difference), 99.8))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    shown = axes[0].imshow(truth_object, cmap="gray", vmin=0, vmax=common_max)
    axes[0].set_title("Known vessel proxy")
    axes[1].imshow(estimate_object, cmap="gray", vmin=0, vmax=common_max)
    axes[1].set_title("Recovered object (same scale)")
    fig.colorbar(shown, ax=axes[:2], label="source-normalized intensity")
    shown = axes[2].imshow(
        difference, cmap="coolwarm", vmin=-difference_limit, vmax=difference_limit
    )
    axes[2].set_title("Estimate - truth")
    fig.colorbar(shown, ax=axes[2])
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_dir / "object_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].semilogy(np.arange(1, loss.size + 1), loss)
    axes[0].set(xlabel="epoch", ylabel="measurement MSE", title="Training loss")
    axes[0].grid(alpha=0.25)
    indices = np.arange(1, len(zernike_true) + 1)
    axes[1].plot(indices, zernike_true, "o-", ms=3, label="truth")
    axes[1].plot(indices, zernike_estimate, "o-", ms=3, label="estimate")
    axes[1].set(xlabel="Noll index", ylabel="coefficient (rad)", title="Zernike comparison")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output_dir / "loss_and_zernike.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)), constrained_layout=True)
    if len(selected) == 1:
        axes = axes[None]
    common_max = max(
        float(np.percentile(np.concatenate((row["target"].ravel(), row["prediction"].ravel())), 99.8))
        for row in selected
    )
    difference_limit = max(
        float(np.percentile(np.abs(row["prediction"] - row["target"]), 99.8))
        for row in selected
    )
    for axes_row, row in zip(axes, selected):
        axes_row[0].imshow(row["target"], cmap="magma", vmin=0, vmax=common_max)
        axes_row[0].set_title(f"B observation, frame {row['frame']}")
        axes_row[1].imshow(row["prediction"], cmap="magma", vmin=0, vmax=common_max)
        axes_row[1].set_title("A prediction")
        axes_row[2].imshow(
            row["prediction"] - row["target"],
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        axes_row[2].set_title(f"difference; MSE={row['mse']:.3e}")
        for axis in axes_row:
            axis.axis("off")
    fig.savefig(output_dir / "observation_fits.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    reconstruction_dir = Path(args.reconstruction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _empty_destination(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for analysis but is unavailable.")

    checkpoint_path = reconstruction_dir / "final/model_final.pt"
    summary_path = reconstruction_dir / "final/training_summary.json"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = json.loads(summary_path.read_text(encoding="utf-8"))
    if saved.get("model_mode") != "object_mmes":
        raise ValueError("B→A analysis requires the object_mmes model-A checkpoint.")
    network = model_from_checkpoint(saved, device=device)
    dataset = BatchDataset(data_dir, num=50, normalization="shared-max")
    loader = DataLoader(dataset, batch_size=int(saved["model_config"]["batch_size"]), shuffle=False)

    with torch.no_grad():
        basis = network.basis[:1]
        parameters = network.g_g(basis).permute(0, 3, 1, 2)
        signed_amplitude = parameters[:, 0:1]
        estimated_phase = parameters[:, 1:2]
        object_estimate = network.g_im()

        frame_rows: list[dict[str, object]] = []
        prediction_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for x_batch, y_batch, indices in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            indices = indices.to(device)
            current_time = _time_coordinate(indices, len(dataset))
            prediction = network(x_batch, current_time)[0].reshape_as(y_batch)
            for local, frame_index in enumerate(indices.tolist()):
                target = y_batch[local].detach().cpu().numpy().astype(np.float32)
                estimate = prediction[local].detach().cpu().numpy().astype(np.float32)
                residual = estimate.astype(np.float64) - target.astype(np.float64)
                target_norm = np.linalg.norm(target.astype(np.float64))
                frame_rows.append(
                    {
                        "frame": int(frame_index) + 1,
                        "mse": float(np.mean(residual**2)),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "mae": float(np.mean(np.abs(residual))),
                        "relative_l2": float(np.linalg.norm(residual) / target_norm),
                        "target_maximum": float(target.max()),
                        "prediction_maximum": float(estimate.max()),
                    }
                )
                prediction_cache[int(frame_index) + 1] = (target, estimate)

    phi_true = np.load(data_dir / "phi_true_600.npy", allow_pickle=False)
    object_true = np.load(data_dir / "object_true_normalized.npy", allow_pickle=False)
    phase_np = estimated_phase[0, 0].cpu().numpy().astype(np.float32)
    amplitude_np = signed_amplitude[0, 0].cpu().numpy().astype(np.float32)
    field_amplitude_np = np.abs(amplitude_np).astype(np.float32)
    object_network_np = object_estimate[0, 0].cpu().numpy().astype(np.float32)
    # BatchDataset divides every B observation by one shared maximum.  Undo
    # that same scalar on the recovered model-A object for comparison with the
    # source-normalized [0,1] proxy object; no fitted gain is used here.
    object_source_units = object_network_np * float(dataset.max_intensity)

    phase_metrics = _phase_metrics(phase_np, phi_true)
    object_metrics = _array_metrics(object_source_units, object_true)
    truth_centered = _center_piston(phi_true)
    estimate_centered = _center_piston(phase_np)
    zernike_true = _fit_zernike(truth_centered, args.num_zernike_modes)
    zernike_estimate = _fit_zernike(estimate_centered, args.num_zernike_modes)
    zernike_residual = zernike_estimate - zernike_true
    coefficient_rows = [
        {
            "noll_index": index,
            "truth_rad": float(truth),
            "estimate_rad": float(estimate),
            "residual_rad": float(estimate - truth),
        }
        for index, (truth, estimate) in enumerate(
            zip(zernike_true, zernike_estimate), start=1
        )
    ]
    _write_csv(output_dir / "zernike_coefficients.csv", coefficient_rows)
    _write_csv(output_dir / "frame_residuals.csv", frame_rows)

    sorted_frames = sorted(frame_rows, key=lambda row: float(row["mse"]))
    representative_rows = [
        sorted_frames[0],
        sorted_frames[len(sorted_frames) // 2],
        sorted_frames[-1],
    ]
    selected = []
    for row in representative_rows:
        frame = int(row["frame"])
        target, prediction = prediction_cache[frame]
        selected.append({**row, "target": target, "prediction": prediction})

    losses = np.asarray(training["measurement_mse_history"], dtype=np.float64)
    amplitude_mean = float(amplitude_np.mean())
    amplitude_std = float(amplitude_np.std())
    phase_absolute_error = np.abs(estimate_centered - truth_centered)
    amplitude_deviation = np.abs(amplitude_np - amplitude_mean)
    result = {
        "schema_version": 1,
        "experiment": "B_to_A_only",
        "data_dir": str(data_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "checkpoint": str(checkpoint_path),
        "model_a_forward_unchanged": True,
        "model_mode": saved["model_mode"],
        "device": str(device),
        "training": {
            "epochs_completed": int(training["num_epochs"]),
            "requested_epochs": int(training["requested_num_epochs"]),
            "elapsed_seconds": float(training["elapsed_seconds"]),
            "early_stopping": training["early_stopping"],
            "initial_measurement_mse": float(losses[0]),
            "final_epoch_measurement_mse": float(losses[-1]),
            "minimum_epoch_measurement_mse": float(losses.min()),
            "final_evaluation_measurement_mse": float(training["final_measurement_mse"]),
            "final_total_loss": float(training["final_total_loss"]),
        },
        "phase": {
            "piston_handling": "subtract each full-square 600x600 mean; no tilt removal or registration",
            "sign": "direct Phi_est versus Phi_true in exp(+1j*Phi)",
            **phase_metrics,
        },
        "zernike": {
            "num_modes": args.num_zernike_modes,
            "basis": "AOtools Noll-normalized basis used by this repository",
            "coefficient_rmse_rad": float(
                np.sqrt(np.mean(zernike_residual.astype(np.float64) ** 2))
            ),
            "coefficient_mae_rad": float(
                np.mean(np.abs(zernike_residual), dtype=np.float64)
            ),
            "coefficient_correlation": float(
                np.corrcoef(zernike_true[1:], zernike_estimate[1:])[0, 1]
            ),
        },
        "amplitude": {
            "semantics": "signed amplitude predicted by original StaticDiffuseNet g_g branch",
            "signed_minimum": float(amplitude_np.min()),
            "signed_maximum": float(amplitude_np.max()),
            "signed_mean": amplitude_mean,
            "signed_std": amplitude_std,
            "coefficient_of_variation_abs": float(
                field_amplitude_np.std() / max(field_amplitude_np.mean(), 1e-12)
            ),
            "negative_fraction": float(np.mean(amplitude_np < 0)),
            "amplitude_deviation_vs_absolute_phase_error_correlation": float(
                np.corrcoef(amplitude_deviation.ravel(), phase_absolute_error.ravel())[0, 1]
            ),
        },
        "object": {
            "comparison_scale": (
                "recovered network object multiplied by BatchDataset shared maximum; "
                "no fitted gain in primary metrics"
            ),
            **object_metrics,
        },
        "observations": {
            "frame_mse_mean": float(np.mean([row["mse"] for row in frame_rows])),
            "frame_mse_median": float(np.median([row["mse"] for row in frame_rows])),
            "frame_mse_minimum": float(sorted_frames[0]["mse"]),
            "frame_mse_minimum_frame": int(sorted_frames[0]["frame"]),
            "frame_mse_maximum": float(sorted_frames[-1]["mse"]),
            "frame_mse_maximum_frame": int(sorted_frames[-1]["frame"]),
            "relative_l2_mean": float(
                np.mean([row["relative_l2"] for row in frame_rows])
            ),
            "representative_frames_minimum_median_maximum": [
                int(row["frame"]) for row in representative_rows
            ],
        },
        "interpretation": {
            "low_loss_threshold": 1e-4,
            "phase_inaccuracy_criteria": (
                "relative RMSE > 0.5 or direct phase correlation < 0.9; "
                "reported as a transparent diagnostic, not a universal acceptance rule"
            ),
            "low_loss_but_inaccurate_phase": bool(
                float(training["final_measurement_mse"]) < 1e-4
                and (
                    phase_metrics["relative_rmse_to_truth_rms"] > 0.5
                    or phase_metrics["correlation"] < 0.9
                )
            ),
        },
    }
    np.savez_compressed(
        output_dir / "recovered_arrays.npz",
        phi_true=phi_true,
        phi_estimate=phase_np,
        phi_residual=estimate_centered - truth_centered,
        signed_amplitude=amplitude_np,
        field_amplitude=field_amplitude_np,
        object_true=object_true,
        object_estimate_source_units=object_source_units,
    )
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _make_figures(
        output_dir,
        phi_true,
        phase_np,
        amplitude_np,
        object_true,
        object_source_units,
        zernike_true,
        zernike_estimate,
        losses,
        selected,
    )
    print(f"B→A analysis written to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Generate the model-B data set once.")
    prepare.add_argument("--vessel-image", required=True, type=Path)
    prepare.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--num-frames", type=int, default=50)
    prepare.add_argument("--fft-size", type=int, default=2160)
    prepare.add_argument("--kernel-radius", type=int, default=32)
    prepare.add_argument("--quadrature-order", type=int, default=12)
    prepare.add_argument("--device", default="cpu")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Evaluate an existing object_mmes model-A checkpoint."
    )
    analyze_parser.add_argument("--data-dir", required=True, type=Path)
    analyze_parser.add_argument("--reconstruction-dir", required=True, type=Path)
    analyze_parser.add_argument("--output-dir", required=True, type=Path)
    analyze_parser.add_argument("--device", default="cuda")
    analyze_parser.add_argument("--num-zernike-modes", type=int, default=28)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_dataset(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
