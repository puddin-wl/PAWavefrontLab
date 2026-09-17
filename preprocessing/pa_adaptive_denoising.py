"""Experiment-calibrated A-line denoising for packed-12 PA volumes.

The calibration is deliberately one-dimensional: it learns depth windows and
repeatable transient waveforms from representative A-lines, then freezes that
calibration for every acquisition in the experiment.  No operation in this
module mixes neighbouring image pixels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d, label, uniform_filter1d

from preprocessing.pa_denoising import correlation_terms
from preprocessing.packed12 import decode_packed12, packed12_byte_count


ALGORITHM_VERSION = "adaptive-aline-v1"


class CalibrationError(RuntimeError):
    """Raised when automatic calibration cannot satisfy its safety checks."""

    def __init__(self, message: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class AdaptiveTemplate:
    """A repeatable transient and its empirically calibrated match threshold."""

    values: np.ndarray = field(repr=False)
    correlation_threshold: float
    amplitude_snr_threshold: float = 6.0
    source_event_count: int = 0
    source_frame_count: int = 0
    median_member_ncc: float = 0.0
    signal_false_match_rate: float = 0.0
    signal_power_retention: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 1 or values.size < 3 or values.size % 2 != 1:
            raise ValueError("自适应模板必须是一维、奇数长度且至少包含 3 点。")
        if values.size > 361 or not np.isfinite(values).all():
            raise ValueError("自适应模板必须有限且长度不超过 361 点。")
        peak = float(np.max(np.abs(values)))
        if peak <= 0:
            raise ValueError("自适应模板不能全为零。")
        object.__setattr__(self, "values", values / np.float32(peak))
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("模板相关阈值必须位于 (0, 1]。")
        if self.amplitude_snr_threshold <= 0:
            raise ValueError("模板幅度信噪比阈值必须为正数。")

    def metadata(self) -> dict:
        values = np.asarray(self.values, dtype="<f4")
        return {
            "length_samples": int(values.size),
            "relative_samples_inclusive": [
                -int(values.size // 2),
                int(values.size // 2),
            ],
            "correlation_threshold_absolute": float(self.correlation_threshold),
            "amplitude_snr_threshold": float(self.amplitude_snr_threshold),
            "source_event_count": int(self.source_event_count),
            "source_frame_count": int(self.source_frame_count),
            "median_member_ncc": float(self.median_member_ncc),
            "signal_false_match_rate": float(self.signal_false_match_rate),
            "signal_power_retention": float(self.signal_power_retention),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }


@dataclass(frozen=True)
class AdaptiveCalibration:
    """Frozen experiment-level parameters used for every source volume."""

    height: int
    width: int
    depth: int
    signal_window: tuple[int, int]
    noise_windows: tuple[tuple[int, int], tuple[int, int]]
    templates: tuple[AdaptiveTemplate, ...] = ()
    seed: int = 0
    grid_size: int = 20
    qc: dict = field(default_factory=dict)
    source_records: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        dimensions = (self.height, self.width, self.depth)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError(f"采集尺寸必须为正数，实际为 {dimensions}。")
        _validate_windows(self.depth, self.signal_window, self.noise_windows)
        if len(self.templates) > 3:
            raise ValueError("一次实验最多允许 3 个自适应模板。")

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "source_shape": [self.height, self.width, self.depth],
            "signal_window_start_inclusive_stop_exclusive": list(self.signal_window),
            "noise_windows_start_inclusive_stop_exclusive": [
                list(window) for window in self.noise_windows
            ],
            "seed": int(self.seed),
            "grid_size": int(self.grid_size),
            "spatial_filter": None,
            "projection": "depth-adaptive noise-power-subtracted RMS",
            "templates": [template.metadata() for template in self.templates],
            "quality_control": self.qc,
            "sources": list(self.source_records),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.metadata(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(payload)
        for template in self.templates:
            digest.update(np.asarray(template.values, dtype="<f4").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class AdaptiveProjectionResult:
    projection: np.ndarray
    template_match_counts: tuple[int, ...]
    template_match_fractions: tuple[float, ...]
    signal_power_before: float
    signal_power_after: float


def _validate_windows(
    depth: int,
    signal_window: tuple[int, int],
    noise_windows: Sequence[tuple[int, int]],
) -> None:
    windows = (signal_window, *noise_windows)
    for position, window in enumerate(windows):
        if len(window) != 2:
            raise ValueError(f"深度窗口必须是 (start, stop)，实际为 {window!r}。")
        start, stop = window
        if not 0 <= int(start) < int(stop) <= int(depth):
            raise ValueError(f"深度窗口越界：{window!r}，depth={depth}。")
        for other in windows[:position]:
            if max(start, other[0]) < min(stop, other[1]):
                raise ValueError(f"深度窗口不能重叠：{window!r} 与 {other!r}。")


def robust_linear_detrend(
    traces: np.ndarray,
    *,
    iterations: int = 3,
    clip_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a robust per-A-line linear baseline and return residual + scale."""
    values = np.asarray(traces, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("traces 必须是有限的二维 A-line 数组。")
    depth = values.shape[1]
    positions = np.arange(depth, dtype=np.float32)
    mask = np.ones(values.shape, dtype=bool)
    residual = np.empty_like(values)
    for _ in range(iterations):
        weights = mask.astype(np.float32)
        sum_w = np.maximum(weights.sum(axis=1), np.float32(2.0))
        sum_z = np.sum(weights * positions[None, :], axis=1, dtype=np.float32)
        sum_x = np.sum(weights * values, axis=1, dtype=np.float32)
        sum_zz = np.sum(
            weights * positions[None, :] * positions[None, :],
            axis=1,
            dtype=np.float32,
        )
        sum_zx = np.sum(
            weights * positions[None, :] * values, axis=1, dtype=np.float32
        )
        denominator = np.maximum(
            sum_w * sum_zz - sum_z * sum_z, np.float32(1e-6)
        )
        slope = (sum_w * sum_zx - sum_z * sum_x) / denominator
        intercept = (sum_x - slope * sum_z) / sum_w
        residual = values - (
            intercept[:, None] + slope[:, None] * positions[None, :]
        )
        center = np.median(residual, axis=1)
        scale = np.float32(1.4826) * np.median(
            np.abs(residual - center[:, None]), axis=1
        )
        scale = np.maximum(scale, np.float32(1e-3))
        mask = np.abs(residual - center[:, None]) <= (
            np.float32(clip_sigma) * scale[:, None]
        )

    differences = np.diff(residual, axis=1)
    difference_center = np.median(differences, axis=1)
    # MAD(diff)/sqrt(2) is a robust estimate of the original white-noise sigma.
    noise_scale = (
        np.float32(1.4826 / np.sqrt(2.0))
        * np.median(np.abs(differences - difference_center[:, None]), axis=1)
    )
    noise_scale = np.maximum(noise_scale, np.float32(1e-3))
    return residual.astype(np.float32, copy=False), noise_scale.astype(
        np.float32, copy=False
    )


def _signal_profile_from_residual(
    residual: np.ndarray, noise_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    local_power = uniform_filter1d(
        residual * residual, size=9, axis=1, mode="nearest"
    )
    standardized = local_power / np.maximum(
        noise_scale[:, None] * noise_scale[:, None], np.float32(1e-6)
    )
    sparse_excess = np.maximum(
        np.percentile(standardized, 99.0, axis=0)
        - np.percentile(standardized, 90.0, axis=0),
        0.0,
    )
    profile = gaussian_filter1d(sparse_excess.astype(np.float32), sigma=2.0)
    baseline = float(np.median(profile))
    spread = float(1.4826 * np.median(np.abs(profile - baseline)))
    threshold = baseline + 2.0 * max(spread, 1e-6)
    evidence = np.maximum(profile - np.float32(threshold), 0.0)
    return profile.astype(np.float32), evidence.astype(np.float32), threshold


def detect_signal_window(
    traces: np.ndarray,
    *,
    minimum_width: int = 32,
    maximum_width: int = 160,
    padding: int = 8,
) -> tuple[tuple[int, int], dict]:
    """Detect the dominant sparse depth-signal component in sampled A-lines."""
    residual, noise_scale = robust_linear_detrend(traces)
    profile, evidence, threshold = _signal_profile_from_residual(
        residual, noise_scale
    )
    components, component_count = label(evidence > 0)
    candidates: list[tuple[float, int, int, int]] = []
    for component in range(1, component_count + 1):
        indices = np.flatnonzero(components == component)
        if indices.size:
            candidates.append(
                (
                    float(np.sum(evidence[indices])),
                    int(indices[0]),
                    int(indices[-1]),
                    int(indices.size),
                )
            )
    if not candidates:
        raise CalibrationError(
            "没有检测到稳定的深度信号分量。",
            diagnostics={"profile_threshold": threshold},
        )
    candidates.sort(reverse=True)
    weight, component_start, component_stop, _ = candidates[0]
    if weight <= 0:
        raise CalibrationError("深度信号分量没有正的显著能量。")

    component_indices = np.arange(component_start, component_stop + 1)
    component_weights = evidence[component_indices]
    cumulative = np.cumsum(component_weights, dtype=np.float64)
    lower_target = cumulative[-1] * 0.0025
    upper_target = cumulative[-1] * 0.9975
    lower = int(component_indices[np.searchsorted(cumulative, lower_target)])
    upper = int(component_indices[np.searchsorted(cumulative, upper_target)]) + 1
    start = max(0, lower - padding)
    stop = min(traces.shape[1], upper + padding)
    width = stop - start
    if width < minimum_width:
        missing = minimum_width - width
        start = max(0, start - missing // 2)
        stop = min(traces.shape[1], start + minimum_width)
        start = max(0, stop - minimum_width)
        width = stop - start
    if width > maximum_width:
        raise CalibrationError(
            f"自动信号窗口过宽（{width} 点），无法安全区分信号与噪声。",
            diagnostics={
                "candidate_window": [start, stop],
                "maximum_width": maximum_width,
            },
        )
    # Coverage is defined against the selected signal component, not against
    # unrelated transient components elsewhere in depth (those are handled by
    # template discovery).
    coverage = float(
        np.sum(evidence[max(start, component_start) : min(stop, component_stop + 1)], dtype=np.float64)
        / max(weight, 1e-12)
    )
    diagnostics = {
        "profile": profile,
        "evidence": evidence,
        "profile_threshold": float(threshold),
        "dominant_component": [component_start, component_stop + 1],
        "window": [start, stop],
        "significant_energy_coverage": coverage,
        "candidate_components": [
            {"weight": item[0], "start": item[1], "stop": item[2] + 1}
            for item in candidates[:10]
        ],
    }
    return (start, stop), diagnostics


def _window_score(
    residual: np.ndarray,
    noise_scale: np.ndarray,
    profile: np.ndarray,
    start: int,
    stop: int,
) -> float:
    window = residual[:, start:stop]
    normalized = window / noise_scale[:, None]
    depth_power = np.median(normalized * normalized, axis=0)
    x = np.arange(stop - start, dtype=np.float32)
    slope = float(np.polyfit(x, depth_power, 1)[0]) if x.size >= 2 else 0.0
    spike_fraction = float(np.mean(np.abs(normalized) > 6.0))
    return (
        float(np.mean(np.log1p(np.maximum(profile[start:stop], 0.0))))
        + 3.0 * float(np.mean(np.log1p(np.maximum(depth_power, 0.0))))
        + 4.0 * spike_fraction
        + 10.0 * abs(slope)
    )


def choose_noise_windows(
    traces: np.ndarray,
    signal_window: tuple[int, int],
    *,
    guard: int = 16,
    preferred_width: int = 48,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], dict]:
    """Choose low-energy stationary windows on both sides of the signal gate."""
    residual, noise_scale = robust_linear_detrend(traces)
    profile, evidence, _ = _signal_profile_from_residual(residual, noise_scale)
    depth = traces.shape[1]
    signal_start, signal_stop = signal_window

    def candidates(side_start: int, side_stop: int) -> list[tuple[float, int, int]]:
        available = side_stop - side_start
        if available < 32:
            return []
        width = min(64, preferred_width, available)
        width = max(32, width)
        result = []
        for start in range(side_start, side_stop - width + 1):
            stop = start + width
            score = _window_score(
                residual, noise_scale, profile, start, stop
            )
            result.append((score, start, stop))
        return result

    left_candidates = candidates(0, max(0, signal_start - guard))
    right_candidates = candidates(min(depth, signal_stop + guard), depth)
    if not left_candidates or not right_candidates:
        raise CalibrationError(
            "信号窗口两侧没有足够空间建立噪声窗口。",
            diagnostics={"signal_window": list(signal_window), "guard": guard},
        )
    left_score, left_start, left_stop = min(left_candidates)
    right_score, right_start, right_stop = min(right_candidates)
    windows = ((left_start, left_stop), (right_start, right_stop))

    # Validate the fitted variance trend against the experiment-level depth
    # power profile.  Aggregating held-out A-lines avoids treating the sampling
    # uncertainty of one 48-point variance estimate as model error.
    positions = np.arange(depth, dtype=np.float32)
    power_profile = np.mean(residual * residual, axis=0, dtype=np.float64)
    left_power = float(np.mean(power_profile[left_start:left_stop]))
    right_power = float(np.mean(power_profile[right_start:right_stop]))
    left_center = (left_start + left_stop - 1) / 2.0
    right_center = (right_start + right_stop - 1) / 2.0
    selected_indices = np.r_[left_start:left_stop, right_start:right_stop]
    interpolation = left_power + (right_power - left_power) * (
        (positions[selected_indices] - left_center)
        / max(right_center - left_center, 1.0)
    )
    relative_error = np.abs(power_profile[selected_indices] - interpolation) / np.maximum(
        power_profile[selected_indices], 1e-6
    )
    prediction_error = float(np.median(relative_error))
    peak_evidence = float(np.max(evidence))
    window_evidence = float(
        max(np.max(evidence[left_start:left_stop]), np.max(evidence[right_start:right_stop]))
    )
    evidence_ratio = window_evidence / max(peak_evidence, 1e-12)
    if prediction_error > 0.20:
        raise CalibrationError(
            f"噪声功率预测误差过高（{prediction_error:.3f} > 0.20）。",
            diagnostics={
                "noise_windows": [list(window) for window in windows],
                "noise_power_prediction_median_relative_error": prediction_error,
            },
        )
    if evidence_ratio > 0.10:
        raise CalibrationError(
            "自动噪声窗口仍包含显著深度信号。",
            diagnostics={
                "noise_windows": [list(window) for window in windows],
                "noise_window_to_signal_evidence_ratio": evidence_ratio,
            },
        )
    return windows, {
        "left_score": float(left_score),
        "right_score": float(right_score),
        "noise_power_prediction_median_relative_error": prediction_error,
        "noise_window_to_signal_evidence_ratio": evidence_ratio,
    }


def _extract_template_candidates(
    residual: np.ndarray,
    noise_scale: np.ndarray,
    frame_ids: np.ndarray,
    signal_window: tuple[int, int],
    *,
    maximum_candidates: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = residual.shape[1]
    masked = np.abs(residual).copy()
    guard_start = max(0, signal_window[0] - 8)
    guard_stop = min(depth, signal_window[1] + 8)
    masked[:, guard_start:guard_stop] = 0.0
    centers = np.argmax(masked, axis=1)
    rows = np.arange(residual.shape[0])
    amplitudes = masked[rows, centers]
    snr = amplitudes / noise_scale
    selected = np.flatnonzero(snr >= 6.0)
    if selected.size > maximum_candidates:
        strongest = np.argpartition(snr[selected], -maximum_candidates)[
            -maximum_candidates:
        ]
        selected = selected[strongest]
    order = selected[np.argsort(snr[selected])[::-1]]
    half_width = 180
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int32)
    indices = centers[order, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < depth)
    clipped = np.clip(indices, 0, depth - 1)
    candidates = np.where(valid, residual[order[:, None], clipped], 0.0)
    signs = np.sign(candidates[:, half_width])
    signs[signs == 0] = 1.0
    candidates *= signs[:, None]
    candidates /= np.maximum(
        np.max(np.abs(candidates), axis=1, keepdims=True), np.float32(1e-6)
    )
    return (
        candidates.astype(np.float32, copy=False),
        frame_ids[order].astype(np.int32, copy=False),
        order.astype(np.int32, copy=False),
        snr[order].astype(np.float32, copy=False),
    )


def _trim_template(values: np.ndarray) -> np.ndarray:
    template = np.asarray(values, dtype=np.float32)
    center = template.size // 2
    energy = template * template
    total = float(np.sum(energy))
    if total <= 0:
        raise ValueError("候选模板没有正能量。")
    order = np.argsort(np.abs(np.arange(template.size) - center))
    cumulative = np.cumsum(energy[order], dtype=np.float64)
    kept = order[cumulative <= total * 0.995]
    if kept.size < template.size:
        kept = np.r_[kept, order[min(kept.size, template.size - 1)]]
    radius = int(np.max(np.abs(kept - center))) if kept.size else 0
    radius = min(180, max(16, radius))
    trimmed = template[center - radius : center + radius + 1].copy()
    trimmed -= np.float32(np.median(np.r_[trimmed[:4], trimmed[-4:]]))
    peak = float(np.max(np.abs(trimmed)))
    if peak <= 0:
        raise ValueError("截取后的候选模板没有正峰值。")
    return (trimmed / np.float32(peak)).astype(np.float32)


def _template_match_summary(
    residual: np.ndarray,
    noise_scale: np.ndarray,
    template: np.ndarray,
    threshold: float,
    signal_window: tuple[int, int],
) -> tuple[float, float]:
    correlation, template_energy, dot = correlation_terms(
        residual, template, depth=residual.shape[1]
    )
    half = template.size // 2
    valid_centers = np.zeros(residual.shape[1], dtype=bool)
    valid_centers[half : residual.shape[1] - half] = True
    correlation[:, ~valid_centers] = 0.0
    coefficients = dot / np.maximum(template_energy[None, :], np.float32(1e-12))
    fitted_peak = np.abs(coefficients) * float(np.max(np.abs(template)))
    signal_slice = slice(signal_window[0], signal_window[1])
    signal_corr = np.abs(correlation[:, signal_slice])
    signal_amp = fitted_peak[:, signal_slice]
    accepted = (signal_corr >= threshold) & (
        signal_amp >= 6.0 * noise_scale[:, None]
    )
    high_signal = (
        np.max(np.abs(residual[:, signal_slice]), axis=1) >= 6.0 * noise_scale
    )
    denominator = max(int(np.sum(high_signal)), 1)
    false_rate = float(np.sum(np.any(accepted[high_signal], axis=1)) / denominator)
    if not np.any(accepted):
        return false_rate, 1.0

    cleaned = residual.copy()
    rows, relative_centers = np.nonzero(accepted)
    # One strongest accepted match for this template per A-line.
    chosen_rows = []
    chosen_centers = []
    for row in np.unique(rows):
        possible = relative_centers[rows == row]
        best = possible[np.argmax(signal_corr[row, possible])]
        chosen_rows.append(int(row))
        chosen_centers.append(int(best + signal_window[0]))
    offsets = np.arange(residual.shape[1])[None, :] - np.asarray(
        chosen_centers
    )[:, None] + half
    valid = (offsets >= 0) & (offsets < template.size)
    shifted = np.zeros((len(chosen_rows), residual.shape[1]), dtype=np.float32)
    clipped = np.clip(offsets, 0, template.size - 1)
    shifted[valid] = template[clipped[valid]]
    row_array = np.asarray(chosen_rows)
    center_array = np.asarray(chosen_centers)
    coeff = coefficients[row_array, center_array]
    cleaned[row_array] -= coeff[:, None] * shifted
    before = float(np.sum(residual[high_signal, signal_slice] ** 2, dtype=np.float64))
    after = float(np.sum(cleaned[high_signal, signal_slice] ** 2, dtype=np.float64))
    return false_rate, after / max(before, 1e-12)


def discover_templates(
    traces: np.ndarray,
    frame_ids: np.ndarray,
    signal_window: tuple[int, int],
    *,
    maximum_templates: int = 3,
    minimum_events: int = 100,
    minimum_frames: int = 5,
) -> tuple[tuple[AdaptiveTemplate, ...], dict]:
    """Discover zero to three repeatable transient templates."""
    residual, noise_scale = robust_linear_detrend(traces)
    candidates, candidate_frames, candidate_rows, candidate_snr = (
        _extract_template_candidates(
            residual, noise_scale, frame_ids, signal_window
        )
    )
    if candidates.shape[0] < minimum_events:
        return (), {
            "candidate_event_count": int(candidates.shape[0]),
            "accepted_template_count": 0,
            "reason": "insufficient_repeated_transient_candidates",
        }

    core = candidates[:, 180 - 32 : 180 + 33]
    core_norm = core / np.maximum(
        np.linalg.norm(core, axis=1, keepdims=True), np.float32(1e-6)
    )
    available = np.ones(candidates.shape[0], dtype=bool)
    templates: list[AdaptiveTemplate] = []
    rejected_stable_clusters: list[dict] = []

    for _ in range(maximum_templates):
        remaining = np.flatnonzero(available)
        if remaining.size < minimum_events:
            break
        # Try high-SNR seeds until a stable cluster is found or 64 seeds fail.
        seed_order = remaining[np.argsort(candidate_snr[remaining])[::-1]][:64]
        best_members = np.empty(0, dtype=np.int32)
        for seed in seed_order:
            similarity = np.abs(core_norm[remaining] @ core_norm[seed])
            members = remaining[similarity >= 0.85]
            if members.size > best_members.size:
                best_members = members.astype(np.int32)
        if best_members.size < minimum_events:
            break
        frame_count = int(np.unique(candidate_frames[best_members]).size)
        if frame_count < minimum_frames:
            available[best_members] = False
            continue

        full_template = np.median(candidates[best_members], axis=0).astype(np.float32)
        template = _trim_template(full_template)
        half = template.size // 2
        member_segments = candidates[
            best_members, 180 - half : 180 + half + 1
        ]
        normalized_members = member_segments / np.maximum(
            np.linalg.norm(member_segments, axis=1, keepdims=True),
            np.float32(1e-6),
        )
        normalized_template = template / max(float(np.linalg.norm(template)), 1e-6)
        member_ncc = np.abs(normalized_members @ normalized_template)
        median_ncc = float(np.median(member_ncc))
        available[best_members] = False
        if median_ncc < 0.90:
            continue

        null_rows = np.setdiff1d(
            np.arange(residual.shape[0]), candidate_rows[best_members], assume_unique=False
        )
        if null_rows.size > 10000:
            null_rows = null_rows[:10000]
        if null_rows.size:
            null_corr, _, _ = correlation_terms(
                residual[null_rows], template, depth=residual.shape[1]
            )
            edge = template.size // 2
            if edge:
                null_corr[:, :edge] = 0.0
                null_corr[:, -edge:] = 0.0
            null_max = np.max(np.abs(null_corr), axis=1)
            threshold = max(0.75, float(np.percentile(null_max, 99.99)))
        else:
            threshold = 0.75
        threshold = min(threshold, 0.9999)
        false_rate, retention = _template_match_summary(
            residual,
            noise_scale,
            template,
            threshold,
            signal_window,
        )
        if false_rate > 1e-4 or retention < 0.99:
            rejected_stable_clusters.append(
                {
                    "event_count": int(best_members.size),
                    "frame_count": frame_count,
                    "median_member_ncc": median_ncc,
                    "signal_false_match_rate": false_rate,
                    "signal_power_retention": retention,
                }
            )
            continue
        templates.append(
            AdaptiveTemplate(
                values=template,
                correlation_threshold=threshold,
                amplitude_snr_threshold=6.0,
                source_event_count=int(best_members.size),
                source_frame_count=frame_count,
                median_member_ncc=median_ncc,
                signal_false_match_rate=false_rate,
                signal_power_retention=retention,
            )
        )

    if rejected_stable_clusters and not templates:
        raise CalibrationError(
            "检测到重复瞬态，但候选模板无法通过信号安全验证。",
            diagnostics={"rejected_template_clusters": rejected_stable_clusters},
        )
    return tuple(templates), {
        "candidate_event_count": int(candidates.shape[0]),
        "accepted_template_count": len(templates),
        "accepted_templates": [template.metadata() for template in templates],
        "rejected_stable_clusters": rejected_stable_clusters,
    }


def _window_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return float(intersection / max(union, 1))


def calibrate_from_samples(
    traces_by_frame: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
    depth: int,
    seed: int = 0,
    grid_size: int = 20,
    source_records: Sequence[dict] = (),
) -> AdaptiveCalibration:
    """Calibrate one experiment from per-frame representative A-lines."""
    if len(traces_by_frame) < 5:
        raise CalibrationError("五折自动标定至少需要 5 个测量帧。")
    frames = [np.asarray(frame, dtype=np.float32) for frame in traces_by_frame]
    if any(
        frame.ndim != 2
        or frame.shape[1] != depth
        or not np.isfinite(frame).all()
        for frame in frames
    ):
        raise ValueError("每个标定帧必须是有限的 (A-line 数, depth) 数组。")

    fold_windows: list[tuple[int, int]] = []
    for fold in range(5):
        training = np.concatenate(
            [frame for index, frame in enumerate(frames) if index % 5 != fold],
            axis=0,
        )
        window, _ = detect_signal_window(training)
        fold_windows.append(window)
    minimum_iou = min(
        _window_iou(first, second)
        for index, first in enumerate(fold_windows)
        for second in fold_windows[index + 1 :]
    )
    if minimum_iou < 0.85:
        raise CalibrationError(
            f"五折信号窗口不稳定（最小 IoU={minimum_iou:.3f}）。",
            diagnostics={
                "fold_signal_windows": [list(window) for window in fold_windows],
                "minimum_pairwise_iou": minimum_iou,
            },
        )

    all_traces = np.concatenate(frames, axis=0)
    frame_ids = np.concatenate(
        [np.full(frame.shape[0], index, dtype=np.int32) for index, frame in enumerate(frames)]
    )
    signal_window, signal_diagnostics = detect_signal_window(all_traces)
    coverage = float(signal_diagnostics["significant_energy_coverage"])
    if coverage < 0.995:
        raise CalibrationError(
            f"信号窗口显著能量覆盖率不足（{coverage:.5f} < 0.995）。",
            diagnostics={"signal_window": list(signal_window), "coverage": coverage},
        )
    noise_windows, noise_diagnostics = choose_noise_windows(
        all_traces, signal_window
    )
    templates, template_diagnostics = discover_templates(
        all_traces, frame_ids, signal_window
    )

    per_frame_windows = []
    per_frame_ious = []
    for frame in frames:
        try:
            frame_window, _ = detect_signal_window(frame)
        except CalibrationError:
            # Weak individual frames are allowed; experiment-level consistency
            # is already established by folds that contain many frames.
            per_frame_windows.append(None)
            continue
        per_frame_windows.append(list(frame_window))
        per_frame_ious.append(_window_iou(frame_window, signal_window))
    overlapping_frame_count = int(sum(value >= 0.50 for value in per_frame_ious))
    required_overlapping_frames = int(np.ceil(len(frames) * 0.60))
    if overlapping_frame_count < required_overlapping_frames:
        raise CalibrationError(
            "多数测量帧的信号深度无法复现实验级窗口。",
            diagnostics={
                "experiment_signal_window": list(signal_window),
                "per_frame_signal_windows": per_frame_windows,
                "overlapping_frame_count": overlapping_frame_count,
                "required_overlapping_frames": required_overlapping_frames,
            },
        )

    serializable_signal = {
        key: value
        for key, value in signal_diagnostics.items()
        if key not in {"profile", "evidence"}
    }
    qc = {
        "passed": True,
        "fold_signal_windows": [list(window) for window in fold_windows],
        "minimum_fold_window_iou": minimum_iou,
        "per_frame_signal_windows": per_frame_windows,
        "frames_overlapping_experiment_gate": overlapping_frame_count,
        "required_overlapping_frames": required_overlapping_frames,
        "minimum_detectable_frame_iou": (
            float(min(per_frame_ious)) if per_frame_ious else None
        ),
        "signal": serializable_signal,
        "noise": noise_diagnostics,
        "template_discovery": template_diagnostics,
        "no_spatial_filtering": True,
    }
    calibration = AdaptiveCalibration(
        height=height,
        width=width,
        depth=depth,
        signal_window=signal_window,
        noise_windows=noise_windows,
        templates=templates,
        seed=seed,
        grid_size=grid_size,
        qc=qc,
        source_records=tuple(source_records),
    )
    # Retain arrays for diagnostic figures without putting them in JSON.
    object.__setattr__(calibration, "_signal_profile", signal_diagnostics["profile"])
    object.__setattr__(calibration, "_signal_evidence", signal_diagnostics["evidence"])
    return calibration


def _grid_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or count > length:
        raise ValueError(f"grid_size 必须位于 1–{length}，实际为 {count}。")
    edges = np.linspace(0, length, count + 1)
    centers = np.floor((edges[:-1] + edges[1:]) / 2.0).astype(np.int32)
    return np.clip(centers, 0, length - 1)


def sample_packed12_alines(
    source: str | Path,
    *,
    height: int,
    width: int,
    depth: int,
    grid_size: int = 20,
) -> np.ndarray:
    """Read a deterministic spatial grid of complete A-lines without full decode."""
    path = Path(source).expanduser().resolve()
    expected = packed12_byte_count(height * width * depth)
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path} 大小错误：应为 {expected} 字节，实际为 {path.stat().st_size}。"
        )
    if depth % 2:
        raise ValueError("A-line 深度必须为偶数，才能进行按 A-line 随机读取。")
    rows = _grid_indices(height, grid_size)
    columns = _grid_indices(width, grid_size)
    flat_indices = (rows[:, None] * width + columns[None, :]).reshape(-1)
    bytes_per_aline = packed12_byte_count(depth)
    mapped = np.memmap(
        path, dtype=np.uint8, mode="r", shape=(height * width, bytes_per_aline)
    )
    packed = np.asarray(mapped[flat_indices]).reshape(-1)
    return decode_packed12(packed).reshape(-1, depth)


def calibrate_experiment(
    sources: Sequence[str | Path],
    *,
    height: int,
    width: int,
    depth: int,
    grid_size: int = 20,
    seed: int = 0,
) -> AdaptiveCalibration:
    """Sample and calibrate all measurement sources in one experiment."""
    if len(sources) < 5:
        raise CalibrationError("实验至少需要 5 个测量帧才能自动标定。")
    traces_by_frame = []
    records = []
    for source_value in sources:
        path = Path(source_value).expanduser().resolve()
        stat = path.stat()
        traces_by_frame.append(
            sample_packed12_alines(
                path,
                height=height,
                width=width,
                depth=depth,
                grid_size=grid_size,
            )
        )
        records.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return calibrate_from_samples(
        traces_by_frame,
        height=height,
        width=width,
        depth=depth,
        seed=seed,
        grid_size=grid_size,
        source_records=records,
    )


def _clean_one_template_cpu(
    residual: np.ndarray,
    noise_scale: np.ndarray,
    model: AdaptiveTemplate,
) -> tuple[np.ndarray, int]:
    template = model.values
    depth = residual.shape[1]
    correlation, template_energy, dot = correlation_terms(
        residual, template, depth=depth
    )
    half = template.size // 2
    if half:
        correlation[:, :half] = 0.0
        correlation[:, depth - half :] = 0.0
    coefficient_by_center = dot / np.maximum(
        template_energy[None, :], np.float32(1e-12)
    )
    fitted_peak = np.abs(coefficient_by_center) * float(np.max(np.abs(template)))
    accepted = (
        np.abs(correlation) >= model.correlation_threshold
    ) & (
        fitted_peak >= model.amplitude_snr_threshold * noise_scale[:, None]
    )
    has_match = np.any(accepted, axis=1)
    if not np.any(has_match):
        return residual, 0
    score = np.where(accepted, np.abs(correlation), -1.0)
    centers = np.argmax(score, axis=1)
    rows = np.flatnonzero(has_match)
    chosen = centers[rows]
    offsets = np.arange(depth, dtype=np.int32)[None, :] - chosen[:, None] + half
    valid = (offsets >= 0) & (offsets < template.size)
    shifted = np.zeros((rows.size, depth), dtype=np.float32)
    clipped = np.clip(offsets, 0, template.size - 1)
    shifted[valid] = template[clipped[valid]]
    coefficients = coefficient_by_center[rows, chosen]
    cleaned = residual.copy()
    cleaned[rows] -= coefficients[:, None] * shifted
    return cleaned, int(rows.size)


def clean_adaptive_traces(
    traces: np.ndarray,
    calibration: AdaptiveCalibration,
    *,
    backend: str = "cpu",
) -> tuple[np.ndarray, tuple[int, ...], float, float]:
    """Detrend and sequentially subtract every accepted experiment template."""
    residual, noise_scale = robust_linear_detrend(traces)
    before = float(
        np.sum(
            residual[:, calibration.signal_window[0] : calibration.signal_window[1]]
            ** 2,
            dtype=np.float64,
        )
    )
    counts = []
    cleaned = residual
    for model in calibration.templates:
        if backend == "cuda":
            from preprocessing.pa_denoising_gpu import clean_traces_gpu_adaptive

            cleaned, matched = clean_traces_gpu_adaptive(
                cleaned,
                model.values,
                correlation_threshold=model.correlation_threshold,
                minimum_fitted_peak_adc=(
                    model.amplitude_snr_threshold * noise_scale
                ),
            )
            count = int(np.sum(matched))
        elif backend == "cpu":
            cleaned, count = _clean_one_template_cpu(cleaned, noise_scale, model)
        else:
            raise ValueError(f"未知后端：{backend}")
        counts.append(count)
    after = float(
        np.sum(
            cleaned[:, calibration.signal_window[0] : calibration.signal_window[1]]
            ** 2,
            dtype=np.float64,
        )
    )
    return cleaned, tuple(counts), before, after


def adaptive_excess_rms_projection(
    residual: np.ndarray,
    calibration: AdaptiveCalibration,
) -> np.ndarray:
    """Return per-A-line excess RMS using interpolated left/right noise power."""
    values = np.asarray(residual, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != calibration.depth:
        raise ValueError(
            f"residual 必须为 (A-line 数, {calibration.depth})，实际为 {values.shape}。"
        )
    left, right = calibration.noise_windows
    signal = calibration.signal_window
    noise_indices = np.r_[left[0] : left[1], right[0] : right[1]].astype(np.int32)
    positions = noise_indices.astype(np.float32)
    position_mean = np.float32(np.mean(positions))
    centered_positions = positions - position_mean
    denominator = np.sum(centered_positions * centered_positions, dtype=np.float32)
    noise_values = values[:, noise_indices]

    # Two clipped least-squares iterations make the local baseline insensitive
    # to occasional residual transients while remaining vectorized by A-line.
    weights = np.ones_like(noise_values, dtype=np.float32)
    slope = np.zeros(values.shape[0], dtype=np.float32)
    intercept = np.zeros_like(slope)
    for _ in range(2):
        sum_w = np.maximum(weights.sum(axis=1), np.float32(2.0))
        weighted_z = weights * positions[None, :]
        sum_z = weighted_z.sum(axis=1, dtype=np.float32)
        sum_x = (weights * noise_values).sum(axis=1, dtype=np.float32)
        sum_zz = (weighted_z * positions[None, :]).sum(axis=1, dtype=np.float32)
        sum_zx = (weighted_z * noise_values).sum(axis=1, dtype=np.float32)
        denom = np.maximum(sum_w * sum_zz - sum_z * sum_z, np.float32(1e-6))
        slope = (sum_w * sum_zx - sum_z * sum_x) / denom
        intercept = (sum_x - slope * sum_z) / sum_w
        noise_residual = noise_values - (
            intercept[:, None] + slope[:, None] * positions[None, :]
        )
        scale = np.float32(1.4826) * np.median(
            np.abs(noise_residual - np.median(noise_residual, axis=1)[:, None]),
            axis=1,
        )
        scale = np.maximum(scale, np.float32(1e-3))
        weights = (np.abs(noise_residual) <= 4.0 * scale[:, None]).astype(np.float32)

    full_positions = np.arange(calibration.depth, dtype=np.float32)
    detrended = values - (
        intercept[:, None] + slope[:, None] * full_positions[None, :]
    )
    scale_limit = 4.0 * np.maximum(scale, np.float32(1e-3))
    left_residual = np.clip(
        detrended[:, left[0] : left[1]],
        -scale_limit[:, None],
        scale_limit[:, None],
    )
    right_residual = np.clip(
        detrended[:, right[0] : right[1]],
        -scale_limit[:, None],
        scale_limit[:, None],
    )
    left_power = np.mean(left_residual * left_residual, axis=1, dtype=np.float32)
    right_power = np.mean(right_residual * right_residual, axis=1, dtype=np.float32)
    left_center = np.float32((left[0] + left[1] - 1) / 2.0)
    right_center = np.float32((right[0] + right[1] - 1) / 2.0)
    signal_positions = np.arange(signal[0], signal[1], dtype=np.float32)
    interpolation = (signal_positions - left_center) / max(
        float(right_center - left_center), 1.0
    )
    predicted_power = left_power[:, None] + (
        right_power - left_power
    )[:, None] * interpolation[None, :]
    predicted_power = np.maximum(predicted_power, 0.0)
    expected_noise_power = np.mean(predicted_power, axis=1, dtype=np.float32)
    signal_values = detrended[:, signal[0] : signal[1]]
    signal_power = np.mean(signal_values * signal_values, axis=1, dtype=np.float32)
    excess = np.maximum(signal_power - expected_noise_power, np.float32(0.0))
    return np.sqrt(excess, out=excess).astype(np.float32, copy=False)


def load_packed12_adaptive_projection(
    source: str | Path,
    *,
    calibration: AdaptiveCalibration,
    backend: str = "cpu",
    chunk_rows: int = 10,
) -> AdaptiveProjectionResult:
    """Stream one packed-12 volume through the frozen adaptive calibration."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows 必须是正整数。")
    path = Path(source).expanduser().resolve()
    expected = packed12_byte_count(
        calibration.height * calibration.width * calibration.depth
    )
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path} 大小错误：应为 {expected} 字节，实际为 {path.stat().st_size}。"
        )
    samples_per_row = calibration.width * calibration.depth
    if samples_per_row % 2:
        raise ValueError("流式 packed-12 解码要求 width×depth 为偶数。")

    projection = np.empty((calibration.height, calibration.width), dtype=np.float32)
    total_counts = np.zeros(len(calibration.templates), dtype=np.int64)
    total_before = 0.0
    total_after = 0.0
    with path.open("rb") as stream:
        for row_start in range(0, calibration.height, chunk_rows):
            row_stop = min(calibration.height, row_start + chunk_rows)
            row_count = row_stop - row_start
            pair_count = row_count * samples_per_row // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
            if packed.size != pair_count * 3:
                raise EOFError(f"读取 {path} 时意外到达文件末尾。")
            raw = decode_packed12(packed).reshape(
                row_count * calibration.width, calibration.depth
            )
            cleaned, counts, before, after = clean_adaptive_traces(
                raw, calibration, backend=backend
            )
            projected = adaptive_excess_rms_projection(cleaned, calibration)
            projection[row_start:row_stop] = projected.reshape(
                row_count, calibration.width
            )
            if counts:
                total_counts += np.asarray(counts, dtype=np.int64)
            total_before += before
            total_after += after

    if not np.isfinite(projection).all() or float(np.min(projection)) < 0:
        raise RuntimeError("自适应 A-line 投影产生了无效值。")
    aline_count = calibration.height * calibration.width
    return AdaptiveProjectionResult(
        projection=projection,
        template_match_counts=tuple(int(value) for value in total_counts),
        template_match_fractions=tuple(
            float(value / aline_count) for value in total_counts
        ),
        signal_power_before=float(total_before),
        signal_power_after=float(total_after),
    )


def verify_source_records(records: Iterable[dict]) -> None:
    """Fail if any calibrated measurement changed before processing finished."""
    for record in records:
        path = Path(record["path"])
        stat = path.stat()
        current = (int(stat.st_size), int(stat.st_mtime_ns))
        expected = (int(record["size"]), int(record["mtime_ns"]))
        if current != expected:
            raise RuntimeError(f"标定后原始 BIN 发生变化：{path}")
