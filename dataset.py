# Copyright (c) 2023 Brandon Y. Feng, University of Maryland, College Park and Rice University.
"""Strict, lazy-loading dataset for NeuWS MATLAB measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import scipy.io as sio
import torch

from optics import PAPER_PHASE_SIGN, aperture_mask, pad_from_aperture, validate_geometry


def _load_mat_variable(path: Path, variable: str) -> np.ndarray:
    try:
        values = sio.loadmat(path)
        if variable not in values:
            raise KeyError(f"{path} does not contain variable {variable!r}.")
        return np.asarray(values[variable])
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"{path} does not contain variable {variable!r}.")
            # MATLAB v7.3 stores dimensions in column-major order.
            return np.asarray(handle[variable]).T


def _indexed_files(data_dir: Path, prefix: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.mat$")
    indexed = {}
    for path in data_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            index = int(match.group(1))
            if index in indexed:
                raise ValueError(f"Duplicate sample index {index} for prefix {prefix!r}.")
            indexed[index] = path
    return indexed


class BatchDataset(torch.utils.data.Dataset):
    """NeuWS measurements loaded one frame at a time instead of cached on the GPU."""

    def __init__(
        self,
        data_dir,
        im_prefix: str = "SLM_raw",
        slm_prefix: str = "SLM_sim",
        num: Optional[int] = None,
        max_intensity: float = 0,
        zero_freq: int = -1,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        self.im_prefix = im_prefix
        self.slm_prefix = slm_prefix
        self.zero_freq = int(zero_freq)
        self.manifest = self._load_manifest()
        self.phase_sign = int(self.manifest.get("phase_sign", PAPER_PHASE_SIGN))
        if self.phase_sign not in (-1, 1):
            raise ValueError(f"manifest.json phase_sign must be -1 or 1, got {self.phase_sign}.")

        image_files = _indexed_files(self.data_dir, self.im_prefix)
        phase_files = _indexed_files(self.data_dir, self.slm_prefix)
        if not image_files:
            raise FileNotFoundError(
                f"No {self.im_prefix}N.mat measurement files found in {self.data_dir}."
            )
        if not phase_files:
            raise FileNotFoundError(
                f"No {self.slm_prefix}N.mat phase files found in {self.data_dir}."
            )
        if set(image_files) != set(phase_files):
            missing_images = sorted(set(phase_files) - set(image_files))
            missing_phases = sorted(set(image_files) - set(phase_files))
            raise ValueError(
                f"Measurement/phase indices do not match; missing measurements "
                f"{missing_images}, missing phases {missing_phases}."
            )
        available = sorted(image_files)
        expected = list(range(1, available[-1] + 1))
        if available != expected:
            missing = sorted(set(expected) - set(available))
            raise ValueError(f"Sample numbering must be continuous from 1; missing {missing}.")
        if num is None:
            selected = available
        else:
            if num <= 0:
                raise ValueError(f"num must be positive or None, got {num}.")
            if num > len(available):
                raise ValueError(
                    f"Requested {num} frames, but only {len(available)} continuous frames exist."
                )
            selected = available[:num]
        self.indices = selected
        self.image_files = [image_files[index] for index in selected]
        self.phase_files = [phase_files[index] for index in selected]

        self.width = 0
        self.aperture_height = 0
        observed_max = 0.0
        for position, (image_path, phase_path) in enumerate(
            zip(self.image_files, self.phase_files), start=1
        ):
            phase = self._validate_phase(phase_path, position)
            image = self._validate_image(self._measurement_path(image_path, position), position)
            if self.width == 0:
                self.width = int(image.shape[0])
                self.aperture_height = int(phase.shape[0])
                validate_geometry(self.width, self.aperture_height)
            if image.shape != (self.width, self.width):
                raise ValueError(
                    f"{image_path} has shape {image.shape}; expected {(self.width, self.width)}."
                )
            if phase.shape != (self.aperture_height, self.width):
                raise ValueError(
                    f"{phase_path} has shape {phase.shape}; expected "
                    f"{(self.aperture_height, self.width)}."
                )
            observed_max = max(observed_max, float(image.max()))

        manifest_size = self.manifest.get("size")
        if manifest_size is not None and int(manifest_size) != self.width:
            raise ValueError(
                f"manifest.json size is {manifest_size}, but measurements are {self.width}x{self.width}."
            )
        manifest_height = self.manifest.get("aperture_height")
        if manifest_height is not None and int(manifest_height) != self.aperture_height:
            raise ValueError(
                f"manifest.json aperture_height is {manifest_height}, but phase files use "
                f"{self.aperture_height}."
            )

        manifest_max = float(self.manifest.get("measurement_max", 0) or 0)
        requested_max = float(max_intensity)
        self.max_intensity = requested_max if requested_max > 0 else manifest_max
        if self.max_intensity <= 0:
            self.max_intensity = observed_max
        if not np.isfinite(self.max_intensity) or self.max_intensity <= 0:
            raise ValueError("Measurement normalization maximum must be finite and positive.")
        tolerance = max(1e-6, abs(self.max_intensity) * 1e-6)
        if observed_max > self.max_intensity + tolerance:
            raise ValueError(
                f"Normalization maximum {self.max_intensity} is below observed value {observed_max}."
            )

        self.a_slm = aperture_mask(self.width, self.aperture_height)
        self.num = len(self.indices)
        print(
            f"Training with {self.num} frames at {self.width}x{self.width}; "
            f"active aperture {self.aperture_height}x{self.width}, phase sign {self.phase_sign}."
        )

    def _load_manifest(self) -> dict:
        path = self.data_dir / "manifest.json"
        if not path.exists():
            return {}
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse {path}: {exc}") from exc
        if not isinstance(values, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return values

    def _measurement_path(self, regular_path: Path, position: int) -> Path:
        if self.zero_freq > 0 and (position - 1) % self.zero_freq == 0:
            path = self.data_dir.parent / "Zero" / regular_path.name
            if not path.is_file():
                raise FileNotFoundError(f"Zero-SLM measurement does not exist: {path}")
            return path
        return regular_path

    @staticmethod
    def _validate_phase(path: Path, position: int) -> np.ndarray:
        phase = np.asarray(_load_mat_variable(path, "proj_sim")).squeeze()
        if phase.ndim != 2 or phase.size == 0:
            raise ValueError(f"Frame {position}: {path} proj_sim must be a non-empty 2-D array.")
        if not np.issubdtype(phase.dtype, np.number) or not np.isfinite(phase).all():
            raise ValueError(f"Frame {position}: {path} proj_sim must contain finite numbers.")
        return phase

    @staticmethod
    def _validate_image(path: Path, position: int) -> np.ndarray:
        image = np.asarray(_load_mat_variable(path, "imsdata")).squeeze()
        if image.ndim != 2 or image.size == 0:
            raise ValueError(f"Frame {position}: {path} imsdata must be a non-empty 2-D array.")
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"Frame {position}: {path} imsdata must be square, got {image.shape}.")
        if image.shape[0] <= 0 or image.shape[0] % 2:
            raise ValueError(
                f"Frame {position}: {path} size must be a positive even number, got {image.shape[0]}."
            )
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(f"Frame {position}: {path} imsdata must contain finite numbers.")
        if float(image.min()) < 0:
            raise ValueError(f"Frame {position}: {path} imsdata cannot contain negative values.")
        return image

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        position = int(idx) + 1
        phase = self._validate_phase(self.phase_files[idx], position).astype(np.float32)
        if self.zero_freq > 0 and idx % self.zero_freq == 0:
            phase = np.zeros_like(phase)
        phase_tensor = torch.from_numpy(phase)
        padded_phase = pad_from_aperture(phase_tensor, self.width)
        x_train = self.a_slm * torch.exp(1j * self.phase_sign * padded_phase)

        image_path = self._measurement_path(self.image_files[idx], position)
        image = self._validate_image(image_path, position).astype(np.float32)
        y_train = torch.from_numpy(image / self.max_intensity)
        if float(y_train.min()) < -1e-6 or float(y_train.max()) > 1.0 + 1e-6:
            raise ValueError(f"Normalized frame {position} is outside [0,1].")
        return x_train.unsqueeze(0), y_train, int(idx)
