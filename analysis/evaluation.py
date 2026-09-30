"""Image and wrapped-phase evaluation utilities for NeuWS reconstructions."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation
from skimage.restoration import unwrap_phase

from optics import zernike_basis_numpy


def _unit_image(image: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image).squeeze()
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D image, got {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float64) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float64)
    if image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
        raise ValueError(
            f"{name} must be normalized to [0,1], got range [{image.min()}, {image.max()}]."
        )
    return np.clip(image, 0.0, 1.0)


def _ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    smallest = min(reference.shape)
    window = min(7, smallest if smallest % 2 else smallest - 1)
    if window < 3:
        raise ValueError("SSIM requires images at least 3x3.")
    return float(structural_similarity(reference, estimate, data_range=1.0, win_size=window))


def _metric_pair(reference: np.ndarray, estimate: np.ndarray) -> dict:
    return {
        "psnr_db": float(peak_signal_noise_ratio(reference, estimate, data_range=1.0)),
        "ssim": _ssim(reference, estimate),
    }


def overlapping_views(
    reference: np.ndarray, estimate: np.ndarray, shift_yx
) -> tuple[np.ndarray, np.ndarray, dict]:
    dy, dx = (int(round(float(value))) for value in shift_yx)
    height, width = reference.shape
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(f"Shift {(dy, dx)} leaves no overlapping pixels.")
    if dy >= 0:
        reference_rows, estimate_rows = slice(dy, height), slice(0, height - dy)
    else:
        reference_rows, estimate_rows = slice(0, height + dy), slice(-dy, height)
    if dx >= 0:
        reference_cols, estimate_cols = slice(dx, width), slice(0, width - dx)
    else:
        reference_cols, estimate_cols = slice(0, width + dx), slice(-dx, width)
    reference_view = reference[reference_rows, reference_cols]
    estimate_view = estimate[estimate_rows, estimate_cols]
    overlap = {
        "height": int(reference_view.shape[0]),
        "width": int(reference_view.shape[1]),
        "reference_rows": [reference_rows.start, reference_rows.stop],
        "reference_cols": [reference_cols.start, reference_cols.stop],
        "estimate_rows": [estimate_rows.start, estimate_rows.stop],
        "estimate_cols": [estimate_cols.start, estimate_cols.stop],
    }
    return reference_view, estimate_view, overlap


def evaluate_images(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    register: bool = False,
) -> tuple[dict, tuple[np.ndarray, np.ndarray] | None]:
    reference = _unit_image(reference, "reference")
    estimate = _unit_image(estimate, "estimate")
    if reference.shape != estimate.shape:
        raise ValueError(f"Image shapes must match, got {reference.shape} and {estimate.shape}.")
    result = {"shape": list(reference.shape), "raw": _metric_pair(reference, estimate)}
    registered_views = None
    if register:
        shift, registration_error, phase_difference = phase_cross_correlation(
            reference, estimate, upsample_factor=1, normalization=None
        )
        reference_view, estimate_view, overlap = overlapping_views(reference, estimate, shift)
        result["registered"] = {
            **_metric_pair(reference_view, estimate_view),
            "shift_yx_pixels": [int(round(float(value))) for value in shift],
            "phase_correlation_error": float(registration_error),
            "phase_difference_rad": float(phase_difference),
            "overlap": overlap,
        }
        registered_views = (reference_view, estimate_view)
    return result, registered_views


def wrapped_phase_error(reference_phase: np.ndarray, estimate_phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(estimate_phase) - np.asarray(reference_phase))))


def _phase_statistics(error: np.ndarray, mask: np.ndarray) -> dict:
    values = np.asarray(error)[mask]
    rmse = float(np.sqrt(np.mean(values**2)))
    mae = float(np.mean(np.abs(values)))
    return {
        "rmse_rad": rmse,
        "mae_rad": mae,
        "rmse_deg": math.degrees(rmse),
        "mae_deg": math.degrees(mae),
        "max_abs_rad": float(np.max(np.abs(values))),
    }


def remove_zernike_modes(
    wrapped_error: np.ndarray,
    mask: np.ndarray,
    noll_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    size = wrapped_error.shape[0]
    maximum_index = max(noll_indices)
    basis = zernike_basis_numpy(maximum_index, size)[np.asarray(noll_indices) - 1]
    unwrapped = np.asarray(
        unwrap_phase(np.ma.array(wrapped_error, mask=~mask)).filled(0.0), dtype=np.float64
    )
    design = basis[:, mask].T.astype(np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, unwrapped[mask], rcond=None)
    fitted = np.einsum("m,mhw->hw", coefficients, basis, optimize=True)
    residual = np.angle(np.exp(1j * (wrapped_error - fitted)))
    return residual, coefficients


def evaluate_phases(
    reference_phase: np.ndarray,
    estimate_phase: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    reference_phase = np.asarray(reference_phase).squeeze()
    estimate_phase = np.asarray(estimate_phase).squeeze()
    mask = np.asarray(mask, dtype=bool).squeeze()
    if reference_phase.ndim != 2 or reference_phase.shape != estimate_phase.shape:
        raise ValueError("Reference and estimate phases must be matching 2-D arrays.")
    if reference_phase.shape[0] != reference_phase.shape[1] or reference_phase.shape[0] % 2:
        raise ValueError("Phase evaluation supports positive even square arrays only.")
    if mask.shape != reference_phase.shape or not mask.any():
        raise ValueError("The aperture mask must match the phase shape and contain active pixels.")
    if not np.isfinite(reference_phase[mask]).all() or not np.isfinite(estimate_phase[mask]).all():
        raise ValueError("Phase arrays contain NaN or infinite values inside the aperture.")
    raw = wrapped_phase_error(reference_phase, estimate_phase)
    primary, primary_coefficients = remove_zernike_modes(raw, mask, (1, 2, 3))
    diagnostic, diagnostic_coefficients = remove_zernike_modes(raw, mask, (1, 2, 3, 4, 7, 8))
    result = {
        "shape": list(reference_phase.shape),
        "active_pixels": int(mask.sum()),
        "raw": _phase_statistics(raw, mask),
        "primary_piston_tip_tilt_removed": {
            **_phase_statistics(primary, mask),
            "removed_noll_indices": [1, 2, 3],
            "fitted_coefficients_rad": primary_coefficients.tolist(),
        },
        "diagnostic_defocus_coma_removed": {
            **_phase_statistics(diagnostic, mask),
            "removed_noll_indices": [1, 2, 3, 4, 7, 8],
            "fitted_coefficients_rad": diagnostic_coefficients.tolist(),
        },
    }
    return result, primary, diagnostic
