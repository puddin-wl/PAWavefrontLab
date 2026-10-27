"""Strict loader for the real 1080 x 1080 OR-PAM SLM phase library."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import scipy.io as sio
import torch
from PIL import Image


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _load_mat(path: Path, variable: str) -> np.ndarray:
    try:
        values = sio.loadmat(path)
        if variable not in values:
            raise KeyError(f"{path} does not contain {variable!r}.")
        return np.asarray(values[variable]).squeeze()
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"{path} does not contain {variable!r}.")
            return np.asarray(handle[variable]).T.squeeze()


def _resolve_template(root: Path, specification: str) -> tuple[Path, str, str]:
    if ":" not in specification:
        raise ValueError(f"MAT specification must be path:variable, got {specification!r}.")
    relative, variable = specification.rsplit(":", 1)
    path = Path(relative)
    filename = path.name
    if filename.count("N") != 1:
        raise ValueError(f"Phase filename template must contain one N, got {filename!r}.")
    return root / path.parent, filename, variable


def _indexed_paths(directory: Path, filename_template: str) -> dict[int, Path]:
    pattern = re.compile(
        "^" + re.escape(filename_template).replace("N", r"(\d+)") + "$"
    )
    indexed: dict[int, Path] = {}
    if not directory.is_dir():
        raise FileNotFoundError(f"Phase directory does not exist: {directory}")
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"Duplicate phase frame {index} in {directory}.")
            indexed[index] = path
    return indexed


class HardwareSLMPhaseLibrary:
    """Lazy access to exact hardware-grid MAT phases and wrapped PNG checks.

    The class requires the dual-grid manifest.  It never resizes the legacy
    model-grid phase and never synthesizes a replacement random phase.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"SLM phase manifest does not exist: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("phase_units") != "radians":
            raise ValueError("Hardware MAT phases must be recorded in radians.")
        if not bool(self.manifest.get("same_coefficients_on_both_grids", False)):
            raise ValueError("Manifest does not confirm shared coefficients on both grids.")
        convention = str(self.manifest.get("phase_sign_convention", ""))
        if "exp(-1j" not in convention.replace(" ", ""):
            raise ValueError(
                "Manifest does not confirm the project convention exp(-1j * phase)."
            )
        shape = tuple(int(value) for value in self.manifest.get("hardware_shape", ()))
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 0:
            raise ValueError(f"Invalid hardware_shape in manifest: {shape}.")
        self.shape = shape
        self.phase_sign = -1
        self.num_frames = int(self.manifest.get("num_frames", 0))
        if self.num_frames <= 0:
            raise ValueError("Manifest num_frames must be positive.")

        output = self.manifest.get("hardware_output")
        if not isinstance(output, dict) or "mat" not in output:
            raise ValueError("Manifest must define hardware_output.mat.")
        self.mat_directory, self.mat_template, self.mat_variable = _resolve_template(
            self.root, str(output["mat"])
        )
        self.mat_paths = _indexed_paths(self.mat_directory, self.mat_template)
        expected = set(range(1, self.num_frames + 1))
        if set(self.mat_paths) != expected:
            missing = sorted(expected - set(self.mat_paths))
            extra = sorted(set(self.mat_paths) - expected)
            raise ValueError(f"Hardware phase numbering mismatch; missing {missing}, extra {extra}.")

        self.png_paths: dict[int, Path] = {}
        png_specification = output.get("png")
        if png_specification:
            png_path = Path(str(png_specification))
            png_directory = self.root / png_path.parent
            self.png_paths = _indexed_paths(png_directory, png_path.name)

    def _validate_index(self, index: int) -> int:
        index = int(index)
        if index < 1 or index > self.num_frames:
            raise IndexError(f"Frame index must be in [1,{self.num_frames}], got {index}.")
        return index

    def load_numpy(self, index: int, *, validate_png: bool = False) -> np.ndarray:
        index = self._validate_index(index)
        phase = _load_mat(self.mat_paths[index], self.mat_variable)
        if phase.shape != self.shape:
            raise ValueError(
                f"Frame {index} has shape {phase.shape}; expected hardware shape {self.shape}."
            )
        if not np.issubdtype(phase.dtype, np.number) or not np.isfinite(phase).all():
            raise ValueError(f"Frame {index} MAT phase must contain finite numbers.")
        phase = np.asarray(phase, dtype=np.float32)
        if validate_png:
            self.validate_png_correspondence(index, phase=phase)
        return phase

    def load(
        self,
        indices: int | Iterable[int],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        validate_png: bool = False,
    ) -> torch.Tensor:
        if isinstance(indices, int):
            selected = [indices]
        else:
            selected = list(indices)
        if not selected:
            raise ValueError("At least one phase index is required.")
        phases = [self.load_numpy(index, validate_png=validate_png) for index in selected]
        return torch.as_tensor(np.stack(phases), dtype=dtype, device=device)

    def _png_integer_maximum(self, png: np.ndarray) -> int:
        encoding = self.manifest.get("png_phase_encoding", {})
        value_range = encoding.get("hardware_integer_range", encoding.get("integer_range"))
        if isinstance(value_range, list) and len(value_range) == 2:
            return int(value_range[1])
        if np.issubdtype(png.dtype, np.integer):
            return int(np.iinfo(png.dtype).max)
        raise ValueError("Cannot determine wrapped PNG integer range from the manifest.")

    def validate_png_correspondence(
        self, index: int, *, phase: Optional[np.ndarray] = None
    ) -> float:
        """Check MAT-to-PNG value and orientation correspondence in radians."""

        index = self._validate_index(index)
        if index not in self.png_paths:
            raise FileNotFoundError(f"No wrapped hardware PNG for phase frame {index}.")
        if phase is None:
            phase = self.load_numpy(index)
        png = np.asarray(Image.open(self.png_paths[index]))
        if png.ndim != 2 or png.shape != self.shape:
            raise ValueError(
                f"Frame {index} PNG shape {png.shape} does not match {self.shape}."
            )
        maximum = self._png_integer_maximum(png)
        if maximum <= 0 or float(png.min()) < 0 or float(png.max()) > maximum:
            raise ValueError(f"Frame {index} PNG values are outside [0,{maximum}].")
        recovered = png.astype(np.float64) * (2.0 * math.pi / maximum)
        wrapped = np.mod(np.asarray(phase, dtype=np.float64), 2.0 * math.pi)
        circular_error = np.angle(np.exp(1j * (recovered - wrapped)))
        maximum_error = float(np.max(np.abs(circular_error)))
        # One quantization interval plus a small float32 allowance.  A flipped
        # or transposed image fails by orders of magnitude.
        tolerance = 1.1 * 2.0 * math.pi / maximum + 2e-7
        if maximum_error > tolerance:
            raise ValueError(
                f"Frame {index} MAT/PNG circular phase error {maximum_error:.3g} rad "
                f"exceeds {tolerance:.3g} rad; values or orientation do not correspond."
            )
        return maximum_error

    def validate_measurement_mapping(self, measurement_dir: str | Path) -> list[dict]:
        """Validate that measurement frame N maps to this library's phase N."""

        measurement_dir = Path(measurement_dir).expanduser().resolve()
        manifest_path = measurement_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Measurement manifest does not exist: {manifest_path}")
        manifest = _read_json(manifest_path)
        if int(manifest.get("num_frames", 0)) != self.num_frames:
            raise ValueError("Measurement and hardware phase frame counts differ.")
        frames = manifest.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Measurement manifest must contain a frames list.")
        indices = [int(frame.get("frame_index", -1)) for frame in frames]
        expected = list(range(1, self.num_frames + 1))
        if indices != expected:
            raise ValueError(
                "Measurement frames must be stored in exact continuous frame-index order."
            )

        model_output = self.manifest.get("model_output", {}).get("mat")
        phase_source = manifest.get("phase_source_directory")
        expected_model_dir: Optional[Path] = None
        model_template: Optional[str] = None
        model_variable: Optional[str] = None
        if model_output and phase_source:
            expected_model_dir, model_template, model_variable = _resolve_template(
                self.root, str(model_output)
            )
            if Path(phase_source).expanduser().resolve() != expected_model_dir.resolve():
                raise ValueError(
                    "Measurement manifest points to a different model-grid phase library; "
                    "hardware phase numbering cannot be certified."
                )

        mapping = []
        for frame in frames:
            index = int(frame["frame_index"])
            measurement_phase = measurement_dir / f"SLM_sim{index}.mat"
            measurement_image = measurement_dir / f"SLM_raw{index}.mat"
            if not measurement_phase.is_file() or not measurement_image.is_file():
                raise FileNotFoundError(
                    f"Measurement frame {index} must contain both SLM_sim{index}.mat "
                    f"and SLM_raw{index}.mat."
                )
            if expected_model_dir is not None and model_template and model_variable:
                source_phase = expected_model_dir / model_template.replace("N", str(index))
                source_values = _load_mat(source_phase, model_variable)
                copied_values = _load_mat(measurement_phase, model_variable)
                if source_values.shape != copied_values.shape or not np.array_equal(
                    source_values, copied_values
                ):
                    raise ValueError(
                        f"Measurement SLM_sim{index}.mat is not an exact copy of its "
                        "declared model-grid phase source."
                    )
            mapping.append(
                {
                    "frame_index": index,
                    "measurement_source": frame.get("measurement_source"),
                    "measurement_mat": str(measurement_image),
                    "copied_model_phase_mat": str(measurement_phase),
                    "hardware_phase_mat": str(self.mat_paths[index]),
                    "model_phase_index": index,
                }
            )
        return mapping
