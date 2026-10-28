from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from networks.neuws import StaticDiffuseNet
from optics.orpam_model_b import (
    ORPAMModelBConfig,
    ideal_fourier_psf,
    pupil_amplitude_mask,
)
from tools.orpam_pupil_mask_loss_sweep import (
    CURVE_SPECS,
    SLM_WIDTH_M,
    circular_pupil_mask,
    model_a_diagnostic_forward,
    validate_dataset_alignment,
)


def _small_config(**changes) -> ORPAMModelBConfig:
    values = dict(
        wavelength_m=532e-9,
        focal_length_m=50e-3,
        slm_pixel_pitch_m=6.4e-6,
        slm_size=32,
        fft_size=128,
        scan_pitch_m=5e-6,
        kernel_radius=3,
        quadrature_order=4,
    )
    values.update(changes)
    return ORPAMModelBConfig(**values)


def _legacy_square_psf(phase: torch.Tensor, config: ORPAMModelBConfig) -> torch.Tensor:
    pupil = torch.exp(1j * phase)[None, None]
    padding = (config.fft_size - config.slm_size) // 2
    pupil = torch.nn.functional.pad(pupil, (padding,) * 4)
    field = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    field = field * (
        config.slm_pixel_pitch_m**2
        / (config.wavelength_m * config.focal_length_m)
    )
    intensity = field.abs().square()
    return intensity / (intensity.sum(dim=(-2, -1), keepdim=True) * config.focus_pixel_pitch_m**2)


def test_default_square_pupil_is_numerically_identical_to_legacy_formula() -> None:
    torch.manual_seed(4)
    config = _small_config()
    phase = torch.randn(config.slm_size, config.slm_size, dtype=torch.float64)

    observed = ideal_fourier_psf(torch.zeros_like(phase), phase, config).intensity
    expected = _legacy_square_psf(phase, config)

    assert config.pupil_shape == "square"
    assert config.pupil_diameter_m is None
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    ("diameter_mm", "expected_central_pixels"), ((6.9, 1078), (6.5, 1016))
)
def test_physical_circle_mask_is_symmetric_and_has_expected_diameter(
    diameter_mm: float, expected_central_pixels: int
) -> None:
    config = ORPAMModelBConfig(
        fft_size=1080,
        pupil_shape="circle",
        pupil_diameter_m=diameter_mm * 1e-3,
    )
    mask = pupil_amplitude_mask(config)
    diameter_m = diameter_mm * 1e-3
    active = torch.nonzero(mask[config.slm_size // 2] > 0, as_tuple=False).flatten()
    coordinate = (
        torch.arange(config.slm_size, dtype=torch.float64)
        - (config.slm_size - 1) / 2.0
    ) * config.slm_pixel_pitch_m
    maximum_active_radius = float(coordinate[active].abs().max())

    assert torch.equal(mask, mask.flip(0))
    assert torch.equal(mask, mask.flip(1))
    assert int(active.numel()) == expected_central_pixels
    assert 2.0 * maximum_active_radius <= diameter_m
    assert 2.0 * (maximum_active_radius + config.slm_pixel_pitch_m) > diameter_m
    pixel_area = float(mask.sum()) * config.slm_pixel_pitch_m**2
    assert pixel_area == pytest.approx(math.pi * (diameter_m / 2.0) ** 2, rel=3e-4)


def test_circle_psf_is_finite_and_area_normalized() -> None:
    config = _small_config(
        pupil_shape="circle",
        pupil_diameter_m=28 * 6.4e-6,
    )
    result = ideal_fourier_psf(
        torch.zeros(config.slm_size, config.slm_size), None, config
    )
    integral = result.intensity.sum() * config.focus_pixel_pitch_m**2

    assert torch.isfinite(result.intensity).all()
    assert float(result.intensity.min()) >= 0
    assert float(integral) == pytest.approx(1.0, rel=2e-6)


def test_model_a_square_matches_original_neuws_forward_formula() -> None:
    torch.manual_seed(5)
    size = 16
    batch = 2
    obj = torch.rand(1, 1, size, size)
    known_phase = torch.randn(batch, size, size)
    aberration = torch.randn(size, size)
    slm_field = torch.exp(-1j * known_phase[:, None])
    aberration_field = torch.exp(1j * aberration)[None, None].expand(batch, -1, -1, -1)
    network = StaticDiffuseNet(
        size,
        size,
        phs_layers=1,
        bsize=batch,
        static_phase=True,
        zernike_features=4,
    )

    def fixed_estimates(self, time):
        return obj, aberration_field, aberration[None, None].expand(batch, -1, -1, -1)

    network.get_estimates = types.MethodType(fixed_estimates, network)
    expected_prediction, expected_kernel, *_ = network(
        slm_field, torch.zeros(batch)
    )
    observed_prediction, observed_kernel = model_a_diagnostic_forward(
        obj, known_phase, aberration
    )

    assert torch.allclose(observed_kernel, expected_kernel, rtol=0, atol=0)
    assert torch.allclose(observed_prediction, expected_prediction, rtol=0, atol=0)


def test_model_a_circle_only_adds_a_pre_fft_amplitude_mask() -> None:
    torch.manual_seed(6)
    size = 16
    obj = torch.rand(1, 1, size, size)
    known_phase = torch.randn(2, size, size)
    aberration = torch.randn(size, size)
    mask = circular_pupil_mask(size, physical_width_m=1.0, diameter_m=0.75)

    circle_prediction, circle_kernel = model_a_diagnostic_forward(
        obj, known_phase, aberration, pupil_mask=mask
    )
    field = (
        mask[None, None]
        * torch.exp(1j * aberration[None, None])
        * torch.exp(-1j * known_phase[:, None])
    )
    expected_kernel = torch.fft.fftshift(
        torch.fft.fft2(field, norm="forward"), dim=(-2, -1)
    ).abs().square()
    expected_kernel = expected_kernel / expected_kernel.sum(
        dim=(-2, -1), keepdim=True
    )
    expected_kernel = expected_kernel.flip(-2).flip(-1)
    square_prediction, _ = model_a_diagnostic_forward(obj, known_phase, aberration)

    assert torch.allclose(circle_kernel, expected_kernel, rtol=0, atol=0)
    assert circle_prediction.shape == (2, size, size)
    assert not torch.allclose(circle_prediction, square_prediction)


def _write_aligned_dataset(path: Path, *, phase_offset: float = 0.0) -> None:
    path.mkdir()
    np.save(path / "object_true_normalized.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
    np.save(path / "phi_true_600.npy", np.ones((4, 4), dtype=np.float32))
    for frame in (1, 2):
        phase = np.full((4, 4), frame + phase_offset, dtype=np.float32)
        sio.savemat(path / f"SLM_sim{frame}.mat", {"proj_sim": phase})
        sio.savemat(
            path / f"SLM_raw{frame}.mat",
            {"imsdata": np.ones((4, 4), dtype=np.float32)},
        )
    (path / "manifest.json").write_text(
        json.dumps({"num_frames": 2}), encoding="utf-8"
    )


def test_five_curve_spec_and_dataset_alignment_use_identical_sources(
    tmp_path: Path,
) -> None:
    directories = [tmp_path / name for name in ("square", "circle69", "circle65")]
    for directory in directories:
        _write_aligned_dataset(directory)

    report = validate_dataset_alignment(directories, num_frames=2)

    assert len(CURVE_SPECS) == 5
    assert {spec[1] for spec in CURVE_SPECS} == {
        "B-square",
        "B-circle-6.9mm",
        "B-circle-6.5mm",
    }
    assert report["num_frames"] == 2
    changed = tmp_path / "changed"
    _write_aligned_dataset(changed, phase_offset=0.5)
    with pytest.raises(ValueError, match="Phase frame 1 differs"):
        validate_dataset_alignment((directories[0], changed), num_frames=2)


def test_model_a_circle_uses_same_physical_diameter_on_600_grid() -> None:
    for diameter_mm in (6.5, 6.9):
        mask = circular_pupil_mask(600, SLM_WIDTH_M, diameter_mm * 1e-3)
        assert torch.equal(mask, mask.flip(0))
        assert torch.equal(mask, mask.flip(1))
        pixel_area = float(mask.sum()) * (SLM_WIDTH_M / 600) ** 2
        expected = math.pi * (diameter_mm * 0.5e-3) ** 2
        assert pixel_area == pytest.approx(expected, rel=8e-4)
