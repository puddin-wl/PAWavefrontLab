"""Image preprocessing and lossless numeric export helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def normalize_square_array(array: np.ndarray, size: int) -> np.ndarray:
    """将二维数组中心裁剪、整体归一化并缩放到指定正方形尺寸。"""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
    if size <= 0:
        raise ValueError("Output size must be positive.")
    if not np.isfinite(array).all():
        raise ValueError("Input image contains NaN or infinite values.")
    height, width = array.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    array = array[top : top + side, left : left + side].astype(np.float32)
    minimum, maximum = float(array.min()), float(array.max())
    if maximum <= minimum:
        raise ValueError("Input image must have a non-zero intensity range.")
    array = (array - minimum) / (maximum - minimum)
    resized = Image.fromarray(array).resize((size, size), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def read_normalized_square(path_value: str | Path, size: int) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype(np.float32)
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif array.ndim > 3:
        array = np.asarray(array[0])
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {array.shape}.")
    return normalize_square_array(array, size)


def write_unit_png(path_value: str | Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Cannot save an image containing NaN or infinite values.")
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(encoded).save(path_value)


def write_wrapped_phase_png(path_value: str | Path, phase: np.ndarray) -> None:
    phase = np.asarray(phase, dtype=np.float32)
    if not np.isfinite(phase).all():
        raise ValueError("Cannot save a phase containing NaN or infinite values.")
    wrapped = np.mod(phase, 2.0 * math.pi)
    encoded = np.rint(wrapped * (65535.0 / (2.0 * math.pi))).astype(np.uint16)
    Image.fromarray(encoded).save(path_value)


def write_wrapped_phase_png_uint8(path_value: str | Path, phase: np.ndarray) -> None:
    """Save wrapped phase [0, 2pi) as an 8-bit SLM command image [0, 255]."""
    phase = np.asarray(phase, dtype=np.float32)
    if not np.isfinite(phase).all():
        raise ValueError("Cannot save a phase containing NaN or infinite values.")
    wrapped = np.mod(phase, 2.0 * math.pi)
    encoded = np.rint(wrapped * (255.0 / (2.0 * math.pi))).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path_value)
