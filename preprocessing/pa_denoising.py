"""Noise-aware projection for packed-12 photoacoustic A-line scans.

The acquisition stores one depth trace for every image pixel.  A full-depth
maximum projection preferentially selects isolated noise peaks.  This module
instead estimates a linear baseline from two noise-only depth windows and
returns the square root of the signal-window power in excess of the measured
noise power.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


BITS_PER_SAMPLE = 12


@dataclass(frozen=True)
class TemplateXcorrProjectionResult:
    """A cleaned MIP and the per-A-line template-matching diagnostics."""

    projection: np.ndarray
    original_projection: np.ndarray
    correlation_map: np.ndarray
    center_map: np.ndarray
    coefficient_map: np.ndarray
    matched_map: np.ndarray


def decode_packed12(packed: np.ndarray) -> np.ndarray:
    """Decode little-endian packed unsigned-12 samples in pairs."""
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 1 or values.size % 3:
        raise ValueError("packed-12 输入必须是一维数组，且字节数是 3 的整数倍。")
    triples = values.reshape(-1, 3)
    decoded = np.empty(triples.shape[0] * 2, dtype=np.uint16)
    decoded[0::2] = triples[:, 0].astype(np.uint16) | (
        (triples[:, 1] & 0x0F).astype(np.uint16) << 8
    )
    decoded[1::2] = (triples[:, 1] >> 4).astype(np.uint16) | (
        triples[:, 2].astype(np.uint16) << 4
    )
    return decoded


def load_template_csv(path_value: str | Path) -> np.ndarray:
    """Load and validate a centered, odd-length motor-crosstalk template."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"串扰模板不存在：{path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"relative_sample", "template_relative_adc"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"串扰模板缺少列：{sorted(required)}")
    relative = np.asarray([int(row["relative_sample"]) for row in rows])
    template = np.asarray(
        [float(row["template_relative_adc"]) for row in rows], dtype=np.float32
    )
    if template.size % 2 != 1 or template.size < 3:
        raise ValueError("串扰模板长度必须是大于等于 3 的奇数。")
    half_width = template.size // 2
    expected_relative = np.arange(-half_width, half_width + 1)
    if not np.array_equal(relative, expected_relative):
        raise ValueError("串扰模板 relative_sample 必须连续并以 0 为中心。")
    if not np.isfinite(template).all() or float(np.max(np.abs(template))) <= 0:
        raise ValueError("串扰模板必须包含有限的非零数值。")
    return template


def correlation_terms(
    centered: np.ndarray,
    template: np.ndarray,
    *,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return NCC, overlap template energy, and dot product at each center."""
    traces = np.asarray(centered, dtype=np.float32)
    kernel = np.asarray(template, dtype=np.float32)
    if traces.ndim != 2 or traces.shape[1] != depth:
        raise ValueError(f"centered 必须为 (A-line 数, {depth})，实际为 {traces.shape}。")
    if kernel.ndim != 1 or kernel.size % 2 != 1:
        raise ValueError("template 必须是一维奇数长度数组。")
    half_width = kernel.size // 2
    dot_full = fftconvolve(
        traces,
        kernel[::-1][None, :],
        mode="full",
        axes=1,
    )
    dot = dot_full[:, half_width : half_width + depth]
    signal_energy_full = fftconvolve(
        traces * traces,
        np.ones((1, kernel.size), dtype=np.float32),
        mode="full",
        axes=1,
    )
    signal_energy = signal_energy_full[:, half_width : half_width + depth]
    template_energy_full = fftconvolve(
        np.ones(depth, dtype=np.float32),
        (kernel[::-1] * kernel[::-1]).astype(np.float32),
        mode="full",
    )
    template_energy = template_energy_full[half_width : half_width + depth]
    denominator = np.sqrt(
        np.maximum(signal_energy * template_energy[None, :], np.float32(1e-12))
    )
    correlation = np.divide(
        dot, denominator, out=np.zeros_like(dot), where=denominator > 0
    )
    np.clip(correlation, -1.0, 1.0, out=correlation)
    return (
        correlation.astype(np.float32, copy=False),
        template_energy.astype(np.float32, copy=False),
        dot.astype(np.float32, copy=False),
    )


def clean_traces(
    raw: np.ndarray,
    template: np.ndarray,
    *,
    correlation_threshold: float,
    minimum_fitted_peak_adc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the strongest accepted shifted template from each A-line."""
    values = np.asarray(raw, dtype=np.float32)
    kernel = np.asarray(template, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("raw 必须是有限的二维 A-line 数组。")
    if kernel.ndim != 1 or kernel.size % 2 != 1 or not np.isfinite(kernel).all():
        raise ValueError("template 必须是有限的一维奇数长度数组。")
    if not np.isfinite(correlation_threshold) or not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold 必须位于 (0, 1]。")
    if not np.isfinite(minimum_fitted_peak_adc) or minimum_fitted_peak_adc < 0:
        raise ValueError("minimum_fitted_peak_adc 必须是非负有限数值。")

    baselines = np.median(values, axis=1, keepdims=True)
    centered = values - baselines
    depth = values.shape[1]
    correlation, template_energy, dot = correlation_terms(
        centered, kernel, depth=depth
    )
    coefficient_by_center = dot / np.maximum(
        template_energy[None, :], np.float32(1e-12)
    )
    fitted_peak_by_center = (
        np.abs(coefficient_by_center) * float(np.max(np.abs(kernel)))
    )
    valid_amplitude = fitted_peak_by_center >= minimum_fitted_peak_adc
    selection_score = np.where(valid_amplitude, np.abs(correlation), -1.0)
    has_valid_amplitude = np.any(valid_amplitude, axis=1)
    best_valid_centers = np.argmax(selection_score, axis=1)
    best_global_centers = np.argmax(np.abs(correlation), axis=1)
    best_centers = np.where(
        has_valid_amplitude, best_valid_centers, best_global_centers
    ).astype(np.int16)
    rows = np.arange(values.shape[0])
    best_correlation = correlation[rows, best_centers]
    coefficients = coefficient_by_center[rows, best_centers]
    matched = has_valid_amplitude & (
        np.abs(best_correlation) >= correlation_threshold
    )

    half_width = kernel.size // 2
    cleaned = values.copy()
    candidate_rows = np.flatnonzero(matched)
    if candidate_rows.size:
        centers = best_centers[candidate_rows].astype(np.int32)
        template_indices = (
            np.arange(depth, dtype=np.int32)[None, :]
            - centers[:, None]
            + half_width
        )
        valid = (template_indices >= 0) & (template_indices < kernel.size)
        shifted = np.zeros((candidate_rows.size, depth), dtype=np.float32)
        clipped = np.clip(template_indices, 0, kernel.size - 1)
        shifted[valid] = kernel[clipped[valid]]
        cleaned[candidate_rows] -= coefficients[candidate_rows, None] * shifted

    return cleaned, best_correlation, best_centers, coefficients, matched


def load_packed12_template_xcorr_mip_projection(
    source: str | Path,
    *,
    template: np.ndarray,
    height: int,
    width: int,
    depth: int,
    baseline_adc: float = 2048.0,
    correlation_threshold: float = 0.70,
    minimum_fitted_peak_adc: float = 80.0,
    chunk_rows: int = 10,
) -> TemplateXcorrProjectionResult:
    """Stream a packed-12 volume, subtract crosstalk, and return full-depth MIPs."""
    dimensions = (height, width, depth)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError(f"height、width、depth 必须是正整数，实际为 {dimensions}。")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows 必须是正整数。")
    if not np.isfinite(baseline_adc):
        raise ValueError("baseline_adc 必须是有限数值。")

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")
    sample_count = height * width * depth
    expected_bytes = (sample_count * BITS_PER_SAMPLE + 7) // 8
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"文件大小与 {height}×{width}×{depth} 个 12 位值不匹配："
            f"应为 {expected_bytes} 字节，实际为 {path.stat().st_size} 字节。"
        )
    samples_per_row = width * depth
    if samples_per_row % 2:
        raise ValueError("流式解码要求 width×depth 为偶数。")

    projection = np.empty((height, width), dtype=np.float32)
    original_projection = np.empty_like(projection)
    correlation_map = np.empty_like(projection)
    center_map = np.empty((height, width), dtype=np.uint16)
    coefficient_map = np.empty_like(projection)
    matched_map = np.empty((height, width), dtype=bool)
    with path.open("rb") as stream:
        for row_start in range(0, height, chunk_rows):
            row_stop = min(row_start + chunk_rows, height)
            row_count = row_stop - row_start
            pair_count = row_count * samples_per_row // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
            if packed.size != pair_count * 3:
                raise EOFError(f"读取 {path} 时意外到达文件末尾。")
            raw = decode_packed12(packed).reshape(row_count * width, depth)
            cleaned, correlation, centers, coefficients, matched = clean_traces(
                raw,
                template,
                correlation_threshold=correlation_threshold,
                minimum_fitted_peak_adc=minimum_fitted_peak_adc,
            )
            output_shape = (row_count, width)
            original_projection[row_start:row_stop] = np.maximum(
                np.max(raw, axis=1).reshape(output_shape).astype(np.float32)
                - np.float32(baseline_adc),
                0.0,
            )
            projection[row_start:row_stop] = np.maximum(
                np.max(cleaned, axis=1).reshape(output_shape)
                - np.float32(baseline_adc),
                0.0,
            )
            correlation_map[row_start:row_stop] = correlation.reshape(output_shape)
            center_map[row_start:row_stop] = centers.reshape(output_shape).astype(
                np.uint16
            )
            coefficient_map[row_start:row_stop] = coefficients.reshape(output_shape)
            matched_map[row_start:row_stop] = matched.reshape(output_shape)

    arrays = (
        projection,
        original_projection,
        correlation_map,
        coefficient_map,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("去相关投影产生了 NaN 或无穷值。")
    if float(projection.min()) < 0:
        raise RuntimeError("去相关投影产生了负值。")
    return TemplateXcorrProjectionResult(
        projection=projection,
        original_projection=original_projection,
        correlation_map=correlation_map,
        center_map=center_map,
        coefficient_map=coefficient_map,
        matched_map=matched_map,
    )


def _validate_windows(
    depth: int,
    signal_window: tuple[int, int],
    noise_windows: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError(f"depth 必须是正整数，实际为 {depth!r}。")
    if not noise_windows:
        raise ValueError("至少需要一个噪声窗口。")

    def indices(window: tuple[int, int], label: str) -> np.ndarray:
        if len(window) != 2:
            raise ValueError(f"{label} 必须是 (start, stop)，实际为 {window!r}。")
        start, stop = window
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in window):
            raise ValueError(f"{label} 边界必须是整数，实际为 {window!r}。")
        if not 0 <= start < stop <= depth:
            raise ValueError(f"{label} 必须满足 0 <= start < stop <= {depth}，实际为 {window!r}。")
        return np.arange(start, stop, dtype=np.int32)

    signal_indices = indices(signal_window, "信号窗口")
    noise_indices = np.concatenate(
        [indices(window, f"噪声窗口 {position + 1}") for position, window in enumerate(noise_windows)]
    )
    if np.unique(noise_indices).size != noise_indices.size:
        raise ValueError("噪声窗口彼此重叠。")
    if np.intersect1d(signal_indices, noise_indices).size:
        raise ValueError("信号窗口不能与噪声窗口重叠。")
    if np.unique(noise_indices).size < 2:
        raise ValueError("线性基线拟合至少需要两个不同深度点。")
    return signal_indices, noise_indices


def excess_rms_projection(
    volume: np.ndarray,
    *,
    signal_window: tuple[int, int],
    noise_windows: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Project ``(..., depth)`` traces to noise-subtracted RMS amplitudes."""
    array = np.asarray(volume)
    if array.ndim != 3 or any(length <= 0 for length in array.shape):
        raise ValueError(f"输入必须是非空三维数组，实际形状为 {array.shape}。")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"输入必须是数值数组，实际类型为 {array.dtype}。")
    if not np.isfinite(array).all():
        raise ValueError("输入包含 NaN 或无穷值。")

    signal_indices, noise_indices = _validate_windows(
        int(array.shape[-1]), signal_window, noise_windows
    )
    values = np.asarray(array, dtype=np.float32)
    noise_positions = noise_indices.astype(np.float32)
    position_mean = np.float32(noise_positions.mean())
    centered_positions = noise_positions - position_mean
    denominator = np.sum(centered_positions * centered_positions, dtype=np.float32)

    noise_values = values[:, :, noise_indices]
    slope = np.sum(
        noise_values * centered_positions[None, None, :], axis=2, dtype=np.float32
    ) / denominator
    intercept = noise_values.mean(axis=2, dtype=np.float32) - slope * position_mean

    signal_positions = signal_indices.astype(np.float32)
    signal_residual = values[:, :, signal_indices] - (
        intercept[:, :, None] + slope[:, :, None] * signal_positions[None, None, :]
    )
    noise_residual = noise_values - (
        intercept[:, :, None] + slope[:, :, None] * noise_positions[None, None, :]
    )
    signal_power = np.mean(signal_residual * signal_residual, axis=2, dtype=np.float32)
    noise_power = np.mean(noise_residual * noise_residual, axis=2, dtype=np.float32)
    excess_power = np.maximum(signal_power - noise_power, np.float32(0.0))
    return np.sqrt(excess_power, out=excess_power).astype(np.float32, copy=False)


def load_packed12_excess_rms_projection(
    source: str | Path,
    *,
    height: int,
    width: int,
    depth: int,
    signal_window: tuple[int, int],
    noise_windows: tuple[tuple[int, int], ...],
    spatial_sigma: float = 0.0,
    chunk_rows: int = 50,
) -> np.ndarray:
    """Decode a packed little-endian unsigned-12 BIN and project it by rows."""
    dimensions = (height, width, depth)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
        raise ValueError(f"height、width、depth 必须是正整数，实际为 {dimensions}。")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows 必须是正整数。")
    if not np.isfinite(spatial_sigma) or spatial_sigma < 0:
        raise ValueError("spatial_sigma 必须是非负有限数值。")
    _validate_windows(depth, signal_window, noise_windows)

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")
    sample_count = height * width * depth
    expected_bytes = (sample_count * BITS_PER_SAMPLE + 7) // 8
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"文件大小与 {height}×{width}×{depth} 个 12 位值不匹配："
            f"应为 {expected_bytes} 字节，实际为 {actual_bytes} 字节。"
        )
    samples_per_row = width * depth
    if samples_per_row % 2:
        raise ValueError("流式解码要求 width×depth 为偶数。")

    projection = np.empty((height, width), dtype=np.float32)
    with path.open("rb") as stream:
        for row_start in range(0, height, chunk_rows):
            row_stop = min(row_start + chunk_rows, height)
            row_count = row_stop - row_start
            pair_count = row_count * samples_per_row // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
            if packed.size != pair_count * 3:
                raise EOFError(f"读取 {path} 时意外到达文件末尾。")
            packed = packed.reshape(-1, 3)
            decoded = np.empty(pair_count * 2, dtype=np.uint16)
            decoded[0::2] = packed[:, 0].astype(np.uint16) | (
                (packed[:, 1] & 0x0F).astype(np.uint16) << 8
            )
            decoded[1::2] = (packed[:, 1] >> 4).astype(np.uint16) | (
                packed[:, 2].astype(np.uint16) << 4
            )
            volume = decoded.reshape(row_count, width, depth)
            projection[row_start:row_stop] = excess_rms_projection(
                volume,
                signal_window=signal_window,
                noise_windows=noise_windows,
            )

    if spatial_sigma > 0:
        projection = gaussian_filter(
            projection, sigma=float(spatial_sigma), mode="reflect"
        ).astype(np.float32, copy=False)
    if not np.isfinite(projection).all() or float(projection.min()) < 0:
        raise RuntimeError("去噪投影产生了无效数值。")
    return projection
