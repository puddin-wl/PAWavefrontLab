"""Differentiable physical forward model B for point-scanning OR-PAM.

The model deliberately stops at an ideal, paraxial Fourier lens.  It uses a
square, uniformly illuminated 1080 x 1080 SLM pupil and does not include the
SLM-to-lens path, defocus propagation, a circular stop, or acoustic effects.

Tensor axes follow the PyTorch image convention: the penultimate axis is the
physical y/row coordinate and the last axis is the physical x/column
coordinate.  A discrete kernel entry ``K[k, l]`` is the integral of the focal
intensity over the 5 um object cell centred at row offset ``k`` and column
offset ``l``.  Passing that kernel directly to :func:`torch.nn.functional.conv2d`
therefore implements the correlation derived in the model-B technical note;
the kernel must not be flipped first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ORPAMModelBConfig:
    """Physical and numerical parameters for the ideal Fourier model."""

    wavelength_m: float = 532e-9
    focal_length_m: float = 50e-3
    slm_pixel_pitch_m: float = 6.4e-6
    slm_size: int = 1080
    fft_size: int = 4320
    scan_pitch_m: float = 5e-6
    phase_sign: int = -1
    quadrature_order: int = 12
    kernel_radius: int = 32

    def __post_init__(self) -> None:
        positive = {
            "wavelength_m": self.wavelength_m,
            "focal_length_m": self.focal_length_m,
            "slm_pixel_pitch_m": self.slm_pixel_pitch_m,
            "scan_pitch_m": self.scan_pitch_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        if self.slm_size <= 0 or self.slm_size % 2:
            raise ValueError("slm_size must be a positive even integer.")
        if self.fft_size < self.slm_size or self.fft_size % 2:
            raise ValueError("fft_size must be even and no smaller than slm_size.")
        if (self.fft_size - self.slm_size) % 2:
            raise ValueError("fft_size - slm_size must be even for centred padding.")
        if self.phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1.")
        if self.quadrature_order <= 0:
            raise ValueError("quadrature_order must be positive.")
        if self.kernel_radius < 0:
            raise ValueError("kernel_radius must be non-negative.")

    @property
    def slm_width_m(self) -> float:
        return self.slm_size * self.slm_pixel_pitch_m

    @property
    def focus_pixel_pitch_m(self) -> float:
        return (
            self.wavelength_m
            * self.focal_length_m
            / (self.fft_size * self.slm_pixel_pitch_m)
        )

    @property
    def focus_field_width_m(self) -> float:
        # Independent of zero padding: lambda*f/p.
        return self.fft_size * self.focus_pixel_pitch_m

    @property
    def kernel_size(self) -> int:
        return 2 * self.kernel_radius + 1

    def to_record(self) -> dict[str, object]:
        values: dict[str, object] = asdict(self)
        values.update(
            {
                "slm_width_m": self.slm_width_m,
                "focus_pixel_pitch_m": self.focus_pixel_pitch_m,
                "focus_field_width_m": self.focus_field_width_m,
                "pupil_amplitude": "uniform square",
                "phase_convention": "exp(+1j * aberration) * exp(phase_sign * 1j * SLM)",
                "psf_normalization": "continuous-area integral equals one",
                "object_basis": "5 um x 5 um piecewise constant cells",
                "scan_boundary": "zero extension outside the finite object",
                "kernel_application": "PyTorch conv2d cross-correlation; no kernel flip",
                "coordinate_convention": "tensor row=y, column=x; zero at FFT index N//2",
            }
        )
        return values


@dataclass
class OpticalPSF:
    """Focal-plane PSF and diagnostics on a physical Cartesian grid."""

    intensity: torch.Tensor
    raw_intensity: torch.Tensor
    raw_energy: torch.Tensor
    x_m: torch.Tensor
    y_m: torch.Tensor
    pixel_pitch_m: float


@dataclass
class ModelBFrameOutput:
    """Outputs for one or more sequentially evaluated phase frames."""

    image: torch.Tensor
    kernel: torch.Tensor
    psf: Optional[torch.Tensor] = None
    raw_psf_energy: Optional[torch.Tensor] = None


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache the immutable Gauss-Legendre rule on [-1, 1]."""

    if order <= 0:
        raise ValueError("quadrature order must be positive.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _phase_nchw(phase: torch.Tensor, name: str) -> torch.Tensor:
    if phase.is_complex():
        raise ValueError(f"{name} must be a real phase in radians.")
    if phase.ndim == 2:
        phase = phase[None, None]
    elif phase.ndim == 3:
        phase = phase[:, None]
    if phase.ndim != 4 or phase.shape[1] != 1:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W].")
    if not torch.isfinite(phase).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return phase


def _image_nchw(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[:, None]
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W].")
    return image


def ideal_fourier_psf(
    slm_phase: torch.Tensor,
    aberration_phase: Optional[torch.Tensor],
    config: ORPAMModelBConfig,
) -> OpticalPSF:
    """Compute a physically sampled focal-plane intensity PSF.

    The returned ``intensity`` has units of inverse square metres and is
    normalized so that its discrete physical-area integral is one.  The
    unnormalized intensity retains the thin-lens scale factor and is returned
    with its integrated energy for diagnostics.
    """

    slm_phase = _phase_nchw(slm_phase, "slm_phase")
    if slm_phase.shape[-2:] != (config.slm_size, config.slm_size):
        raise ValueError(
            f"slm_phase must be {config.slm_size}x{config.slm_size}, "
            f"got {tuple(slm_phase.shape[-2:])}."
        )
    if aberration_phase is None:
        aberration_phase = torch.zeros_like(slm_phase[:1])
    else:
        aberration_phase = _phase_nchw(aberration_phase, "aberration_phase")
        if aberration_phase.shape[-2:] != slm_phase.shape[-2:]:
            raise ValueError("aberration_phase and slm_phase spatial shapes must match.")
        aberration_phase = aberration_phase.to(
            device=slm_phase.device, dtype=slm_phase.dtype
        )
    if aberration_phase.shape[0] not in (1, slm_phase.shape[0]):
        raise ValueError("aberration_phase batch must be one or match the SLM batch.")

    combined_phase = aberration_phase + config.phase_sign * slm_phase
    pupil = torch.exp(1j * combined_phase)
    padding = (config.fft_size - config.slm_size) // 2
    if padding:
        pupil = F.pad(pupil, (padding, padding, padding, padding))

    # ifftshift makes array index zero the spatial origin expected by fft2;
    # fftshift then places zero focal frequency at output index N//2.
    field = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    field = field * (
        config.slm_pixel_pitch_m**2
        / (config.wavelength_m * config.focal_length_m)
    )
    raw_intensity = field.abs().square()
    dx = config.focus_pixel_pitch_m
    raw_energy = raw_intensity.sum(dim=(-2, -1), keepdim=True) * dx**2
    if torch.any(raw_energy <= 0) or not torch.isfinite(raw_energy).all():
        raise ValueError("The pupil produced a non-finite or zero-energy PSF.")
    intensity = raw_intensity / raw_energy

    coordinate = (
        torch.arange(
            config.fft_size, device=slm_phase.device, dtype=slm_phase.dtype
        )
        - config.fft_size // 2
    ) * dx
    return OpticalPSF(
        intensity=intensity,
        raw_intensity=raw_intensity,
        raw_energy=raw_energy.reshape(-1),
        x_m=coordinate,
        y_m=coordinate,
        pixel_pitch_m=dx,
    )


def integrate_equivalent_kernel(
    psf: torch.Tensor,
    *,
    focus_pixel_pitch_m: float,
    scan_pitch_m: float = 5e-6,
    kernel_radius: int,
    quadrature_order: int = 12,
) -> torch.Tensor:
    """Integrate a focal PSF over piecewise-constant object cells.

    For row/column offsets ``k,l`` this evaluates

    ``K[k,l] = integral h(k*Delta+u, l*Delta+v) du dv``

    over ``u,v in [-Delta/2, Delta/2]``.  Bilinear ``grid_sample`` is used on
    the physical FFT grid, and samples outside that grid are zero.  No tensor
    depending on the PSF is detached, so gradients propagate to the pupil.
    """

    psf = _image_nchw(psf, "psf")
    height, width = psf.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("psf height and width must be even so index N//2 is zero.")
    if not np.isfinite(focus_pixel_pitch_m) or focus_pixel_pitch_m <= 0:
        raise ValueError("focus_pixel_pitch_m must be finite and positive.")
    if not np.isfinite(scan_pitch_m) or scan_pitch_m <= 0:
        raise ValueError("scan_pitch_m must be finite and positive.")
    if kernel_radius < 0:
        raise ValueError("kernel_radius must be non-negative.")

    nodes_np, weights_np = _legendre_rule(int(quadrature_order))
    nodes = torch.as_tensor(nodes_np.copy(), device=psf.device, dtype=psf.dtype)
    weights = torch.as_tensor(weights_np.copy(), device=psf.device, dtype=psf.dtype)
    offsets = torch.arange(
        -kernel_radius, kernel_radius + 1, device=psf.device, dtype=psf.dtype
    )
    row_offset, column_offset = torch.meshgrid(offsets, offsets, indexing="ij")

    sample_y = (
        row_offset[..., None, None] * scan_pitch_m
        + 0.5 * scan_pitch_m * nodes[None, None, :, None]
    )
    sample_x = (
        column_offset[..., None, None] * scan_pitch_m
        + 0.5 * scan_pitch_m * nodes[None, None, None, :]
    )
    sample_y, sample_x = torch.broadcast_tensors(sample_y, sample_x)

    pixel_y = sample_y / focus_pixel_pitch_m + height // 2
    pixel_x = sample_x / focus_pixel_pitch_m + width // 2
    normalized_y = 2.0 * pixel_y / (height - 1) - 1.0
    normalized_x = 2.0 * pixel_x / (width - 1) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    kernel_size = 2 * kernel_radius + 1
    grid = grid.reshape(1, kernel_size * kernel_size * quadrature_order**2, 1, 2)
    grid = grid.expand(psf.shape[0], -1, -1, -1)

    values = F.grid_sample(
        psf,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    values = values.reshape(
        psf.shape[0], 1, kernel_size, kernel_size, quadrature_order, quadrature_order
    )
    quadrature_weights = weights[:, None] * weights[None, :]
    return (
        values * quadrature_weights[None, None, None, None]
    ).sum(dim=(-2, -1)) * (scan_pitch_m**2 / 4.0)


def scan_correlation(object_image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply the model-B response kernel as same-size cross-correlation.

    Outside the finite object field the absorption is set to zero.  A singleton
    object or kernel batch is broadcast to the other batch; otherwise batches
    must match.  Odd kernels preserve the image size exactly.
    """

    object_image = _image_nchw(object_image, "object_image")
    kernel = _image_nchw(kernel, "kernel")
    if object_image.is_complex() or kernel.is_complex():
        raise ValueError("object_image and kernel must be real.")
    if kernel.shape[-2] != kernel.shape[-1] or kernel.shape[-1] % 2 != 1:
        raise ValueError("kernel must be square with odd spatial dimensions.")
    batch = max(object_image.shape[0], kernel.shape[0])
    if object_image.shape[0] not in (1, batch) or kernel.shape[0] not in (1, batch):
        raise ValueError("object and kernel batches must match or be singleton.")
    object_image = object_image.expand(batch, -1, -1, -1)
    kernel = kernel.expand(batch, -1, -1, -1)
    if object_image.device != kernel.device:
        raise ValueError("object_image and kernel must be on the same device.")
    kernel = kernel.to(dtype=object_image.dtype)

    # Treat each frame as a convolution group.  PyTorch conv2d is correlation,
    # precisely sum O[p+k,q+l] K[k,l], so there is intentionally no flip.
    grouped_input = object_image[:, 0].unsqueeze(0)
    output = F.conv2d(
        grouped_input,
        kernel,
        padding=kernel.shape[-1] // 2,
        groups=batch,
    )
    return output.squeeze(0)[:, None]


def kernel_energy_diagnostics(kernel: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return retained and missing energy for an area-normalized PSF kernel."""

    kernel = _image_nchw(kernel, "kernel")
    retained = kernel.sum(dim=(-2, -1)).reshape(-1)
    return {
        "retained_energy": retained,
        "truncation_error": 1.0 - retained,
    }


class ORPAMForwardModelB(nn.Module):
    """Independent differentiable simulator for physical forward model B."""

    def __init__(self, config: ORPAMModelBConfig = ORPAMModelBConfig()):
        super().__init__()
        self.config = config

    def optical_psf(
        self,
        slm_phase: torch.Tensor,
        aberration_phase: Optional[torch.Tensor] = None,
    ) -> OpticalPSF:
        return ideal_fourier_psf(slm_phase, aberration_phase, self.config)

    def equivalent_kernel(
        self, psf: torch.Tensor, *, kernel_radius: Optional[int] = None
    ) -> torch.Tensor:
        radius = self.config.kernel_radius if kernel_radius is None else kernel_radius
        return integrate_equivalent_kernel(
            psf,
            focus_pixel_pitch_m=self.config.focus_pixel_pitch_m,
            scan_pitch_m=self.config.scan_pitch_m,
            kernel_radius=radius,
            quadrature_order=self.config.quadrature_order,
        )

    def scan_forward(self, object_image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return scan_correlation(object_image, kernel)

    def forward(
        self,
        object_image: torch.Tensor,
        slm_phase: torch.Tensor,
        aberration_phase: Optional[torch.Tensor] = None,
        *,
        retain_psf: bool = False,
    ) -> ModelBFrameOutput:
        """Evaluate frames sequentially to avoid a simultaneous large FFT batch."""

        slm_phase = _phase_nchw(slm_phase, "slm_phase")
        object_image = _image_nchw(object_image, "object_image")
        frame_count = slm_phase.shape[0]
        if object_image.shape[0] not in (1, frame_count):
            raise ValueError("object_image batch must be one or match the SLM batch.")
        prepared_aberration = None
        if aberration_phase is not None:
            prepared_aberration = _phase_nchw(
                aberration_phase, "aberration_phase"
            )
            if prepared_aberration.shape[0] not in (1, frame_count):
                raise ValueError(
                    "aberration_phase batch must be one or match the SLM batch."
                )
        outputs = []
        kernels = []
        psfs = []
        raw_energies = []
        for frame in range(frame_count):
            frame_aberration = prepared_aberration
            if prepared_aberration is not None and prepared_aberration.shape[0] > 1:
                frame_aberration = prepared_aberration[frame : frame + 1]
            result = self.optical_psf(slm_phase[frame : frame + 1], frame_aberration)
            kernel = self.equivalent_kernel(result.intensity)
            object_frame = object_image
            if object_image.shape[0] > 1:
                object_frame = object_image[frame : frame + 1]
            outputs.append(self.scan_forward(object_frame, kernel))
            kernels.append(kernel)
            raw_energies.append(result.raw_energy)
            if retain_psf:
                psfs.append(result.intensity)
        return ModelBFrameOutput(
            image=torch.cat(outputs, dim=0),
            kernel=torch.cat(kernels, dim=0),
            psf=torch.cat(psfs, dim=0) if retain_psf else None,
            raw_psf_energy=torch.cat(raw_energies, dim=0),
        )
