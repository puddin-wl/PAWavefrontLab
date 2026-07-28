"""Parse explicit Noll-indexed Zernike coefficient specifications."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_NUM_MODES = 28


def _validate(values, num_modes: int) -> np.ndarray:
    coefficients = np.asarray(values, dtype=np.float32).reshape(-1)
    if coefficients.size > num_modes:
        raise ValueError(
            f"At most {num_modes} coefficients are supported, got {coefficients.size}."
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("Zernike coefficients must be finite numbers.")
    padded = np.zeros(num_modes, dtype=np.float32)
    padded[: coefficients.size] = coefficients
    return padded


def parse_dense_coefficients(text: str, num_modes: int = DEFAULT_NUM_MODES) -> np.ndarray:
    """Parse a comma/space-separated Noll 1..N coefficient list and zero-pad it."""
    pieces = [piece for piece in re.split(r"[\s,;]+", text.strip()) if piece]
    if not pieces:
        raise ValueError("The coefficient list is empty.")
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError(
            "Coefficients must be comma- or space-separated numbers in radians."
        ) from exc
    return _validate(values, num_modes)


def parse_sparse_coefficients(
    specifications: list[str], num_modes: int = DEFAULT_NUM_MODES
) -> np.ndarray:
    """Parse repeated ``NOLL=RADIANS`` assignments into a dense coefficient vector."""
    coefficients = np.zeros(num_modes, dtype=np.float32)
    seen = set()
    for specification in specifications:
        match = re.fullmatch(
            r"\s*(\d+)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*",
            specification,
        )
        if not match:
            raise ValueError(
                f"Invalid coefficient {specification!r}; expected NOLL=RADIANS, for example 4=1.5."
            )
        noll_index, value = int(match.group(1)), float(match.group(2))
        if not 1 <= noll_index <= num_modes:
            raise ValueError(f"Noll index must be between 1 and {num_modes}, got {noll_index}.")
        if noll_index in seen:
            raise ValueError(f"Noll index {noll_index} was specified more than once.")
        if not np.isfinite(value):
            raise ValueError("Zernike coefficients must be finite numbers.")
        seen.add(noll_index)
        coefficients[noll_index - 1] = value
    if not seen:
        raise ValueError("At least one NOLL=RADIANS coefficient is required.")
    return coefficients


def load_coefficients_file(
    path_value: str | Path, num_modes: int = DEFAULT_NUM_MODES
) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Coefficient file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _validate(np.load(path, allow_pickle=False), num_modes)
    if suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, list):
            return _validate(values, num_modes)
        if isinstance(values, dict):
            specifications = [f"{key}={value}" for key, value in values.items()]
            return parse_sparse_coefficients(specifications, num_modes)
        raise ValueError("JSON coefficient files must contain a list or Noll-indexed object.")
    return parse_dense_coefficients(path.read_text(encoding="utf-8"), num_modes)


def resolve_coefficients(
    *,
    dense: Optional[str] = None,
    file: Optional[str | Path] = None,
    sparse: Optional[list[str]] = None,
    num_modes: int = DEFAULT_NUM_MODES,
    required: bool = False,
) -> tuple[Optional[np.ndarray], str]:
    supplied = sum(value is not None for value in (dense, file, sparse))
    if supplied > 1:
        raise ValueError(
            "Use only one coefficient input: dense list, coefficient file, or repeated NOLL=RADIANS."
        )
    if dense is not None:
        return parse_dense_coefficients(dense, num_modes), "dense-cli"
    if file is not None:
        return load_coefficients_file(file, num_modes), str(Path(file).expanduser().resolve())
    if sparse is not None:
        return parse_sparse_coefficients(sparse, num_modes), "sparse-cli"
    if required:
        raise ValueError("Explicit Zernike coefficients are required.")
    return None, "random"
