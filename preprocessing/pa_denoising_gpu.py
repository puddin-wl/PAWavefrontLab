"""CuPy acceleration for motor-crosstalk template-xcorr denoising.

Keeps the NumPy/SciPy implementation as the CPU reference while moving the
expensive batched A-line correlation, normalization, template selection,
subtraction and final MIP to the GPU.

CuPy is imported lazily so CPU-only environments can still import the project.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import numpy as np

from preprocessing.pa_denoising import TemplateXcorrProjectionResult
from preprocessing.packed12 import decode_packed12, packed12_byte_count


_PRELOADED_CUDA_LIBRARIES: list[ctypes.CDLL] = []
_CUDA_LIBRARIES_PRELOADED = False


def _preload_pip_cuda_libraries() -> None:
    """Make pip-provided CUDA libraries visible without LD_LIBRARY_PATH changes."""
    global _CUDA_LIBRARIES_PRELOADED
    if _CUDA_LIBRARIES_PRELOADED:
        return
    libraries = (
        ("nvidia.cuda_nvrtc", "libnvrtc.so.*"),
        ("nvidia.nvjitlink", "libnvJitLink.so.*"),
        ("nvidia.cublas", "libcublasLt.so.*"),
        ("nvidia.cublas", "libcublas.so.*"),
        ("nvidia.cufft", "libcufft.so.*"),
    )
    for package, pattern in libraries:
        spec = importlib.util.find_spec(package)
        if spec is None or spec.submodule_search_locations is None:
            continue
        for package_dir in spec.submodule_search_locations:
            candidates = sorted((Path(package_dir) / "lib").glob(pattern))
            if not candidates:
                continue
            try:
                loaded = ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _PRELOADED_CUDA_LIBRARIES.append(loaded)
            break
    _CUDA_LIBRARIES_PRELOADED = True


def _require_cupy():
    _preload_pip_cuda_libraries()
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as exc:
        raise RuntimeError(
            "CUDA 后端需要 CuPy 及其运行库；请安装 requirements-cuda.txt。"
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("CuPy 已安装，但 CUDA 运行时或 NVIDIA 驱动不可用。") from exc
    if device_count < 1:
        raise RuntimeError("未检测到可用的 CUDA GPU。")
    try:
        cp.arange(1, dtype=cp.float32).sum().get()
    except Exception as exc:
        raise RuntimeError("CuPy 已安装且检测到 GPU，但 CUDA 内核执行失败。") from exc
    return cp, fftconvolve


def cuda_backend_info() -> dict[str, str]:
    """Return reproducibility metadata for the active CuPy CUDA device."""
    cp, _ = _require_cupy()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "backend": "cuda",
        "cupy_version": str(cp.__version__),
        "gpu_name": str(name),
    }


def _validate_parameters(
    raw: np.ndarray,
    template: np.ndarray,
    *,
    correlation_threshold: float,
    minimum_fitted_peak_adc: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw)
    kernel = np.asarray(template, dtype=np.float32)

    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("raw 必须是二维数值 A-line 数组。")
    if not np.isfinite(values).all():
        raise ValueError("raw 必须是有限的二维 A-line 数组。")
    if kernel.ndim != 1 or kernel.size % 2 != 1 or not np.isfinite(kernel).all():
        raise ValueError("template 必须是有限的一维奇数长度数组。")
    if kernel.size < 3 or float(np.max(np.abs(kernel))) <= 0:
        raise ValueError("template 必须包含至少 3 个采样点和非零数值。")
    if not np.isfinite(correlation_threshold) or not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold 必须位于 (0, 1]。")
    if not np.isfinite(minimum_fitted_peak_adc) or minimum_fitted_peak_adc < 0:
        raise ValueError("minimum_fitted_peak_adc 必须是非负有限数值。")
    return values, kernel


def _template_energy_cpu(template: np.ndarray, depth: int) -> np.ndarray:
    """Precompute overlap template energy once for all chunks."""
    kernel = np.asarray(template, dtype=np.float32)
    half_width = kernel.size // 2
    full = np.convolve(
        np.ones(depth, dtype=np.float32),
        (kernel[::-1] * kernel[::-1]).astype(np.float32),
        mode="full",
    )
    return full[half_width : half_width + depth].astype(np.float32, copy=False)


def _sliding_signal_energy_gpu(traces, kernel_size: int, cp):
    """Centered overlap sum of squares using O(N) prefix sums instead of FFT."""
    half_width = kernel_size // 2
    squared = traces * traces
    padded = cp.pad(
        squared,
        ((0, 0), (half_width, half_width)),
        mode="constant",
    )
    prefix = cp.concatenate(
        (
            cp.zeros((padded.shape[0], 1), dtype=cp.float32),
            cp.cumsum(padded, axis=1, dtype=cp.float32),
        ),
        axis=1,
    )
    return prefix[:, kernel_size:] - prefix[:, :-kernel_size]


def _correlation_terms_gpu(centered, kernel, template_energy, cp, fftconvolve):
    depth = int(centered.shape[1])
    half_width = int(kernel.size // 2)

    dot_full = fftconvolve(
        centered,
        kernel[::-1][None, :],
        mode="full",
        axes=1,
    )
    dot = dot_full[:, half_width : half_width + depth].astype(cp.float32, copy=False)

    signal_energy = _sliding_signal_energy_gpu(centered, int(kernel.size), cp)
    denominator = cp.sqrt(
        cp.maximum(
            signal_energy * template_energy[None, :],
            cp.float32(1e-12),
        )
    )
    correlation = dot / denominator
    cp.clip(correlation, -1.0, 1.0, out=correlation)
    return correlation.astype(cp.float32, copy=False), dot


def _clean_traces_gpu_arrays(
    values,
    kernel,
    template_energy,
    *,
    correlation_threshold: float,
    minimum_fitted_peak_adc: float,
    cp,
    fftconvolve,
):
    baselines = cp.median(values, axis=1, keepdims=True)
    centered = values - baselines
    depth = int(values.shape[1])

    correlation, dot = _correlation_terms_gpu(
        centered, kernel, template_energy, cp, fftconvolve
    )

    coefficient_by_center = dot / cp.maximum(
        template_energy[None, :],
        cp.float32(1e-12),
    )
    fitted_peak_by_center = cp.abs(coefficient_by_center) * cp.max(cp.abs(kernel))
    valid_amplitude = fitted_peak_by_center >= cp.float32(minimum_fitted_peak_adc)

    selection_score = cp.where(
        valid_amplitude,
        cp.abs(correlation),
        cp.float32(-1.0),
    )
    has_valid_amplitude = cp.any(valid_amplitude, axis=1)
    best_valid_centers = cp.argmax(selection_score, axis=1)
    best_global_centers = cp.argmax(cp.abs(correlation), axis=1)
    best_centers = cp.where(
        has_valid_amplitude,
        best_valid_centers,
        best_global_centers,
    ).astype(cp.int16)

    rows = cp.arange(values.shape[0])
    best_correlation = correlation[rows, best_centers]
    coefficients = coefficient_by_center[rows, best_centers]
    matched = has_valid_amplitude & (
        cp.abs(best_correlation) >= cp.float32(correlation_threshold)
    )

    cleaned = values.copy()
    candidate_rows = cp.flatnonzero(matched)
    if int(candidate_rows.size):
        half_width = int(kernel.size // 2)
        centers = best_centers[candidate_rows].astype(cp.int32)

        template_indices = (
            cp.arange(depth, dtype=cp.int32)[None, :]
            - centers[:, None]
            + half_width
        )
        valid = (template_indices >= 0) & (template_indices < kernel.size)
        clipped = cp.clip(template_indices, 0, kernel.size - 1)
        shifted = cp.where(valid, kernel[clipped], cp.float32(0.0))

        cleaned[candidate_rows] -= (
            coefficients[candidate_rows, None] * shifted
        )

    return cleaned, best_correlation, best_centers, coefficients, matched


def clean_traces_gpu(
    raw: np.ndarray,
    template: np.ndarray,
    *,
    correlation_threshold: float,
    minimum_fitted_peak_adc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU equivalent of clean_traces, returning NumPy arrays."""
    values, kernel_np = _validate_parameters(
        raw,
        template,
        correlation_threshold=correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
    )
    cp, fftconvolve = _require_cupy()

    depth = int(values.shape[1])
    template_energy_np = _template_energy_cpu(kernel_np, depth)

    values_gpu = cp.asarray(values, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_np)
    template_energy_gpu = cp.asarray(template_energy_np)

    outputs = _clean_traces_gpu_arrays(
        values_gpu,
        kernel_gpu,
        template_energy_gpu,
        correlation_threshold=correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
        cp=cp,
        fftconvolve=fftconvolve,
    )
    return tuple(cp.asnumpy(value) for value in outputs)


def clean_traces_gpu_adaptive(
    raw: np.ndarray,
    template: np.ndarray,
    *,
    correlation_threshold: float,
    minimum_fitted_peak_adc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU template subtraction with one amplitude threshold per A-line.

    This is the adaptive pipeline counterpart of :func:`clean_traces_gpu`.
    It intentionally returns only the cleaned traces and accepted-match mask,
    which avoids copying unused diagnostic maps back from the GPU.
    """
    values = np.asarray(raw)
    kernel_np = np.asarray(template, dtype=np.float32)
    thresholds = np.asarray(minimum_fitted_peak_adc, dtype=np.float32)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("raw 必须是二维数值 A-line 数组。")
    if not np.isfinite(values).all():
        raise ValueError("raw 必须是有限的二维 A-line 数组。")
    if (
        kernel_np.ndim != 1
        or kernel_np.size < 3
        or kernel_np.size % 2 != 1
        or not np.isfinite(kernel_np).all()
    ):
        raise ValueError("template 必须是有限的一维奇数长度数组。")
    if not np.isfinite(correlation_threshold) or not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold 必须位于 (0, 1]。")
    if thresholds.shape != (values.shape[0],):
        raise ValueError(
            f"minimum_fitted_peak_adc 必须为每条 A-line 提供一个阈值，"
            f"应为 {(values.shape[0],)}，实际为 {thresholds.shape}。"
        )
    if not np.isfinite(thresholds).all() or float(np.min(thresholds)) < 0:
        raise ValueError("minimum_fitted_peak_adc 必须是非负有限数组。")

    cp, fftconvolve = _require_cupy()
    depth = int(values.shape[1])
    values_gpu = cp.asarray(values, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel_np)
    template_energy_gpu = cp.asarray(_template_energy_cpu(kernel_np, depth))

    # Adaptive callers already provide robustly detrended residuals.  Keeping
    # those values unchanged here preserves CPU/CUDA coefficient parity.
    centered = values_gpu
    correlation, dot = _correlation_terms_gpu(
        centered, kernel_gpu, template_energy_gpu, cp, fftconvolve
    )
    half_width = int(kernel_np.size // 2)
    if half_width:
        correlation[:, :half_width] = cp.float32(0.0)
        correlation[:, depth - half_width :] = cp.float32(0.0)
    coefficient_by_center = dot / cp.maximum(
        template_energy_gpu[None, :], cp.float32(1e-12)
    )
    fitted_peak = cp.abs(coefficient_by_center) * cp.max(cp.abs(kernel_gpu))
    thresholds_gpu = cp.asarray(thresholds)[:, None]
    accepted = (
        cp.abs(correlation) >= cp.float32(correlation_threshold)
    ) & (fitted_peak >= thresholds_gpu)
    matched = cp.any(accepted, axis=1)
    score = cp.where(accepted, cp.abs(correlation), cp.float32(-1.0))
    centers = cp.argmax(score, axis=1).astype(cp.int32)

    cleaned = values_gpu.copy()
    rows = cp.flatnonzero(matched)
    if int(rows.size):
        chosen_centers = centers[rows]
        template_indices = (
            cp.arange(depth, dtype=cp.int32)[None, :]
            - chosen_centers[:, None]
            + half_width
        )
        valid = (template_indices >= 0) & (template_indices < kernel_gpu.size)
        clipped = cp.clip(template_indices, 0, kernel_gpu.size - 1)
        shifted = cp.where(valid, kernel_gpu[clipped], cp.float32(0.0))
        coefficients = coefficient_by_center[rows, chosen_centers]
        cleaned[rows] -= coefficients[:, None] * shifted

    return cp.asnumpy(cleaned), cp.asnumpy(matched)


def load_packed12_template_xcorr_mip_projection_gpu(
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
    """Stream packed-12 data and perform template-xcorr denoising on CUDA."""
    dimensions = (height, width, depth)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError(
            f"height、width、depth 必须是正整数，实际为 {dimensions}。"
        )
    if (
        not isinstance(chunk_rows, int)
        or isinstance(chunk_rows, bool)
        or chunk_rows <= 0
    ):
        raise ValueError("chunk_rows 必须是正整数。")
    if not np.isfinite(baseline_adc):
        raise ValueError("baseline_adc 必须是有限数值。")

    kernel_np = np.asarray(template, dtype=np.float32)
    _validate_parameters(
        np.zeros((1, depth), dtype=np.float32),
        kernel_np,
        correlation_threshold=correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
    )

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")

    sample_count = height * width * depth
    expected_bytes = packed12_byte_count(sample_count)
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"文件大小与 {height}×{width}×{depth} 个 12 位值不匹配："
            f"应为 {expected_bytes} 字节，实际为 {path.stat().st_size} 字节。"
        )

    samples_per_row = width * depth
    if samples_per_row % 2:
        raise ValueError("流式解码要求 width×depth 为偶数。")

    cp, fftconvolve = _require_cupy()
    kernel_gpu = cp.asarray(kernel_np)
    template_energy_gpu = cp.asarray(
        _template_energy_cpu(kernel_np, depth)
    )

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

            packed = np.fromfile(
                stream,
                dtype=np.uint8,
                count=pair_count * 3,
            )
            if packed.size != pair_count * 3:
                raise EOFError(f"读取 {path} 时意外到达文件末尾。")

            raw = decode_packed12(packed).reshape(
                row_count * width,
                depth,
            )
            raw_gpu = cp.asarray(raw, dtype=cp.float32)

            (
                cleaned_gpu,
                correlation_gpu,
                centers_gpu,
                coefficients_gpu,
                matched_gpu,
            ) = _clean_traces_gpu_arrays(
                raw_gpu,
                kernel_gpu,
                template_energy_gpu,
                correlation_threshold=correlation_threshold,
                minimum_fitted_peak_adc=minimum_fitted_peak_adc,
                cp=cp,
                fftconvolve=fftconvolve,
            )

            original_mip_gpu = cp.maximum(
                cp.max(raw_gpu, axis=1) - cp.float32(baseline_adc),
                cp.float32(0.0),
            )
            cleaned_mip_gpu = cp.maximum(
                cp.max(cleaned_gpu, axis=1) - cp.float32(baseline_adc),
                cp.float32(0.0),
            )

            output_shape = (row_count, width)
            original_projection[row_start:row_stop] = cp.asnumpy(
                original_mip_gpu
            ).reshape(output_shape)
            projection[row_start:row_stop] = cp.asnumpy(
                cleaned_mip_gpu
            ).reshape(output_shape)
            correlation_map[row_start:row_stop] = cp.asnumpy(
                correlation_gpu
            ).reshape(output_shape)
            center_map[row_start:row_stop] = cp.asnumpy(
                centers_gpu
            ).reshape(output_shape).astype(np.uint16, copy=False)
            coefficient_map[row_start:row_stop] = cp.asnumpy(
                coefficients_gpu
            ).reshape(output_shape)
            matched_map[row_start:row_stop] = cp.asnumpy(
                matched_gpu
            ).reshape(output_shape)

    arrays = (
        projection,
        original_projection,
        correlation_map,
        coefficient_map,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("GPU 去相关投影产生了 NaN 或无穷值。")
    if float(projection.min()) < 0:
        raise RuntimeError("GPU 去相关投影产生了负值。")

    return TemplateXcorrProjectionResult(
        projection=projection,
        original_projection=original_projection,
        correlation_map=correlation_map,
        center_map=center_map,
        coefficient_map=coefficient_map,
        matched_map=matched_map,
    )
