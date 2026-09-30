"""三维光声 TIFF 的基线校正和最大强度投影。

旧采集代码的物理约定是：交流信号以 2048 为零点，先减去 2048，
负值直接置零，然后始终沿数组第 0 维做最大强度投影。本模块保留这个
约定，但不再假设第 0 维必须恰好有 512 层。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"光声数据必须是三维数组，实际形状为 {array.shape}。")
    if any(length <= 0 for length in array.shape):
        raise ValueError(f"光声数据的每一维都必须非空，实际形状为 {array.shape}。")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"光声数据必须是数值类型，实际类型为 {array.dtype}。")
    if not np.isfinite(array).all():
        raise ValueError("光声数据包含 NaN 或无穷值。")
    return array


def subtract_photoacoustic_baseline(
    volume: np.ndarray, baseline: float = 2048.0
) -> np.ndarray:
    """减去交流零点并将负值置零，返回 float32，避免无符号整数下溢。"""
    array = _validate_volume(volume)
    if not np.isfinite(baseline):
        raise ValueError("光声基线必须是有限数值。")
    corrected = array.astype(np.float32) - np.float32(baseline)
    return np.maximum(corrected, 0.0, out=corrected)


def maximum_intensity_projection(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """沿指定维度取最大投影；本项目入口固定传入 axis=0。"""
    array = _validate_volume(volume)
    if axis not in (0, 1, 2):
        raise ValueError(f"投影维度必须是 0、1 或 2，实际为 {axis}。")
    return np.asarray(np.max(array, axis=axis), dtype=np.float32)


def preprocess_photoacoustic_volume(
    volume: np.ndarray, baseline: float = 2048.0, projection_axis: int = 0
) -> np.ndarray:
    """执行 `max(volume - baseline, 0)`，再做最大强度投影。"""
    if projection_axis != 0:
        raise ValueError("本项目按照旧采集代码固定沿第 0 维投影，projection_axis 必须为 0。")
    corrected = subtract_photoacoustic_baseline(volume, baseline)
    return maximum_intensity_projection(corrected, axis=0)


def load_photoacoustic_projection(
    path_value: str | Path, baseline: float = 2048.0, projection_axis: int = 0
) -> tuple[np.ndarray, dict]:
    """读取一个三维 TIFF，返回未单独归一化的二维投影和可追溯元数据。"""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"三维光声 TIFF 不存在：{path}")
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"三维光声输入必须是 TIFF 文件：{path}")
    volume = tifffile.imread(path)
    projection = preprocess_photoacoustic_volume(volume, baseline, projection_axis)
    metadata = {
        "source_file": str(path),
        "source_shape": [int(value) for value in volume.shape],
        "source_dtype": str(volume.dtype),
        "baseline": float(baseline),
        "negative_policy": "clip_to_zero_after_baseline_subtraction",
        "projection": "maximum_intensity_projection",
        "projection_axis": 0,
        "layer_count_required": None,
        "projected_shape": [int(value) for value in projection.shape],
    }
    return projection, metadata
