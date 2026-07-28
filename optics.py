"""Shared optical primitives for NeuWS data generation and reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from aotools.functions import zernikeArray
from torch.fft import fft2, fftshift, irfftn, rfftn


PAPER_APERTURE_RATIO = 9.0 / 16.0
PAPER_PHASE_SIGN = -1


@dataclass(frozen=True)
class Geometry:
    size: int
    aperture_height: int

    @property
    def top(self) -> int:
        return (self.size - self.aperture_height) // 2

    @property
    def bottom(self) -> int:
        return self.top + self.aperture_height


def _positive_even(value: int, name: str) -> int:
    value = int(value)
    if value <= 0 or value % 2:
        raise ValueError(f"{name} must be a positive even integer, got {value}.")
    return value


def default_aperture_height(size: int) -> int:
    """Return the nearest even height to the paper's 9/16 aperture ratio."""
    size = _positive_even(size, "size")
    height = int(2 * math.floor((size * PAPER_APERTURE_RATIO) / 2 + 0.5))
    return max(2, min(size, height))


def validate_geometry(size: int, aperture_height: Optional[int] = None) -> Geometry:
    size = _positive_even(size, "size")
    if aperture_height is None:
        aperture_height = default_aperture_height(size)
    aperture_height = _positive_even(aperture_height, "aperture_height")
    if aperture_height > size:
        raise ValueError(
            f"aperture_height ({aperture_height}) cannot exceed size ({size})."
        )
    return Geometry(size=size, aperture_height=aperture_height)


def centered_crop(array, target_height: int, target_width: int):
    """Center-crop the final two dimensions of a NumPy array or tensor."""
    height, width = array.shape[-2:]
    if target_height > height or target_width > width:
        raise ValueError(
            f"Cannot crop shape {(height, width)} to {(target_height, target_width)}."
        )
    top = (height - target_height + 1 - height % 2) // 2
    left = (width - target_width + 1 - width % 2) // 2
    return array[..., top : top + target_height, left : left + target_width]


def zernike_basis_numpy(num_modes: int, size: int) -> np.ndarray:
    """AOtools Noll-normalized modes center-cropped to a square field."""
    size = _positive_even(size, "size")
    if num_modes <= 0:
        raise ValueError(f"num_modes must be positive, got {num_modes}.")
    diameter = int(math.ceil(math.sqrt(2.0 * size * size)))
    basis = zernikeArray(int(num_modes), diameter)
    return np.asarray(centered_crop(basis, size, size), dtype=np.float32)


def zernike_basis_torch(
    num_modes: int,
    size: int,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(
        zernike_basis_numpy(num_modes, size), dtype=dtype, device=device
    )


def crop_to_aperture(full_field, aperture_height: int):
    size = full_field.shape[-1]
    geometry = validate_geometry(size, aperture_height)
    if full_field.shape[-2] != size:
        raise ValueError(f"Expected a square field, got {full_field.shape[-2:]}")
    return full_field[..., geometry.top : geometry.bottom, :]


def pad_from_aperture(active_field, size: int, value=0):
    geometry = validate_geometry(size, active_field.shape[-2])
    if active_field.shape[-1] != size:
        raise ValueError(
            f"Active field width must equal size ({size}), got {active_field.shape[-1]}."
        )
    shape = (*active_field.shape[:-2], size, size)
    if torch.is_tensor(active_field):
        output = torch.full(
            shape, value, dtype=active_field.dtype, device=active_field.device
        )
    else:
        output = np.full(shape, value, dtype=active_field.dtype)
    output[..., geometry.top : geometry.bottom, :] = active_field
    return output


def aperture_mask(
    size: int,
    aperture_height: Optional[int] = None,
    *,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    geometry = validate_geometry(size, aperture_height)
    mask = torch.zeros((size, size), dtype=torch.float32, device=device)
    mask[geometry.top : geometry.bottom, :] = 1.0
    return mask


def sample_slm_coefficients(
    num_frames: int,
    num_modes: int = 15,
    sigma: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if num_modes <= 0:
        raise ValueError(f"num_modes must be positive, got {num_modes}.")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(num_frames, num_modes)).astype(np.float32)


def synthesize_slm_patterns(
    coefficients: np.ndarray,
    basis: np.ndarray,
    aperture_height: int,
) -> np.ndarray:
    if coefficients.ndim != 2 or basis.ndim != 3:
        raise ValueError("coefficients and basis must have shapes [frames,modes] and [modes,H,W].")
    if coefficients.shape[1] != basis.shape[0]:
        raise ValueError("Coefficient and basis mode counts do not match.")
    full = np.einsum("fm,mhw->fhw", coefficients, basis, optimize=True)
    return np.asarray(crop_to_aperture(full, aperture_height), dtype=np.float32)


def slm_complex_field(
    active_phase: torch.Tensor,
    size: int,
    phase_sign: int = PAPER_PHASE_SIGN,
) -> torch.Tensor:
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be -1 or 1, got {phase_sign}.")
    phase = pad_from_aperture(active_phase, size)
    mask = aperture_mask(size, active_phase.shape[-2], device=active_phase.device)
    return mask * torch.exp(1j * phase_sign * phase)


def make_static_aberration(
    mode: str,
    size: int,
    aperture_height: int,
    *,
    sigma: float = 1.0,
    seed: int = 0,
    num_modes: int = 28,
    device: Optional[torch.device | str] = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Create one static complex pupil field and return it with its coefficients."""
    geometry = validate_geometry(size, aperture_height)
    rng = np.random.default_rng(seed)
    mask = aperture_mask(size, geometry.aperture_height, device=device)

    if mode == "zernike":
        coefficients = rng.normal(0.0, sigma, size=num_modes).astype(np.float32)
        basis = zernike_basis_torch(num_modes, size, device=device)
        coeff_tensor = torch.as_tensor(coefficients, device=device)
        phase = torch.einsum("m,mhw->hw", coeff_tensor, basis)
        field = mask * torch.exp(1j * phase)
        return field.to(torch.complex64), coefficients

    if mode == "complex-gaussian":
        real = rng.normal(size=(size, size)).astype(np.float32)
        imag = rng.normal(size=(size, size)).astype(np.float32)
        field = torch.complex(
            torch.as_tensor(real, device=device), torch.as_tensor(imag, device=device)
        ) * mask
        energy = field.abs().square()[mask.bool()].mean()
        field = field / torch.sqrt(energy)
        return field.to(torch.complex64), np.empty((0,), dtype=np.float32)

    raise ValueError(
        f"Unknown aberration mode {mode!r}; expected 'zernike' or 'complex-gaussian'."
    )


def pupil_psf(aberration: torch.Tensor, slm_field: torch.Tensor) -> torch.Tensor:
    """Normalized incoherent PSF used by both simulation and reconstruction."""
    pupil = aberration * slm_field
    psf = fftshift(fft2(pupil, norm="forward"), dim=(-2, -1)).abs().square()
    denominator = psf.sum(dim=(-2, -1), keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("The pupil produced a zero-energy PSF.")
    return psf / denominator


def fft_linear_convolution(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """NeuWS same-size linear convolution with a spatially centered kernel."""
    if signal.ndim != 4 or kernel.ndim != 4:
        raise ValueError("signal and kernel must have shape [batch,channels,height,width].")
    if signal.shape[-2] != signal.shape[-1]:
        raise ValueError("Only square signals are supported.")
    size = signal.shape[-1]
    flipped = kernel.flip(-2, -1)
    signal_fr = rfftn(signal, dim=(-2, -1), s=(2 * size, 2 * size))
    kernel_fr = rfftn(flipped, dim=(-2, -1), s=(2 * size, 2 * size))
    output = irfftn(signal_fr * kernel_fr, dim=(-2, -1), s=(2 * size, 2 * size))
    half = size // 2
    return output[..., half:-half, half:-half]


def simulate_measurements(
    object_image: torch.Tensor,
    aberration: torch.Tensor,
    slm_fields: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if object_image.ndim == 2:
        object_image = object_image[None, None]
    if aberration.ndim == 2:
        aberration = aberration[None, None]
    if slm_fields.ndim == 3:
        slm_fields = slm_fields[:, None]
    psf = pupil_psf(aberration, slm_fields)
    object_batch = object_image.expand(slm_fields.shape[0], -1, -1, -1)
    return fft_linear_convolution(object_batch, psf), psf
