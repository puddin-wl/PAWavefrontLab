from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
from PIL import Image

from dataio.orpam_slm import HardwareSLMPhaseLibrary
from optics.orpam_model_b import (
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    ideal_fourier_psf,
    integrate_equivalent_kernel,
    kernel_energy_diagnostics,
    scan_correlation,
)


def _small_config(*, fft_size: int = 256) -> ORPAMModelBConfig:
    return ORPAMModelBConfig(
        wavelength_m=532e-9,
        focal_length_m=50e-3,
        slm_pixel_pitch_m=6.4e-6,
        slm_size=32,
        fft_size=fft_size,
        scan_pitch_m=5e-6,
        kernel_radius=3,
        quadrature_order=4,
    )


def _positive_half_maximum(x: torch.Tensor, profile: torch.Tensor) -> float:
    centre = profile.numel() // 2
    normalized = profile / profile[centre]
    below = torch.nonzero(normalized[centre:] <= 0.5, as_tuple=False)[0, 0] + centre
    left = int(below - 1)
    right = int(below)
    fraction = float((0.5 - normalized[left]) / (normalized[right] - normalized[left]))
    return float(x[left] + fraction * (x[right] - x[left]))


def test_square_pupil_sinc_squared_and_physical_width() -> None:
    config = _small_config(fft_size=512)
    result = ideal_fourier_psf(torch.zeros(config.slm_size, config.slm_size), None, config)
    profile = result.intensity[0, 0, config.fft_size // 2]
    profile = profile / profile.max()
    x = result.x_m
    aperture_width = config.slm_width_m
    theoretical = torch.sinc(aperture_width * x / (config.wavelength_m * config.focal_length_m)).square()
    first_zero_m = config.wavelength_m * config.focal_length_m / aperture_width
    central = x.abs() <= 1.8 * first_zero_m
    assert torch.max(torch.abs(profile[central] - theoretical[central])) < 0.012

    zero_index = config.fft_size // 2 + config.fft_size // config.slm_size
    assert float(profile[zero_index]) < 1e-10
    measured_fwhm = 2.0 * _positive_half_maximum(x, profile)
    expected_fwhm = 0.8859 * first_zero_m
    assert measured_fwhm == pytest.approx(expected_fwhm, rel=0.015)
    assert float(result.intensity.sum() * config.focus_pixel_pitch_m**2) == pytest.approx(
        1.0, rel=2e-6
    )


def test_fft_padding_converges_after_physical_cell_integration() -> None:
    phase = torch.zeros(32, 32)
    kernels = []
    for fft_size in (128, 256, 512):
        config = _small_config(fft_size=fft_size)
        psf = ideal_fourier_psf(phase, None, config).intensity
        kernels.append(
            integrate_equivalent_kernel(
                psf,
                focus_pixel_pitch_m=config.focus_pixel_pitch_m,
                scan_pitch_m=config.scan_pitch_m,
                kernel_radius=3,
                quadrature_order=6,
            )
        )
    coarse_error = torch.linalg.vector_norm(kernels[0] - kernels[2])
    fine_error = torch.linalg.vector_norm(kernels[1] - kernels[2])
    assert float(fine_error) < float(coarse_error) * 0.45
    assert float(fine_error / torch.linalg.vector_norm(kernels[2])) < 0.015


def test_region_integral_is_exact_for_constant_psf() -> None:
    psf = torch.full((1, 1, 64, 64), 2.0, dtype=torch.float64)
    scan_pitch = 4e-6
    kernel = integrate_equivalent_kernel(
        psf,
        focus_pixel_pitch_m=1e-6,
        scan_pitch_m=scan_pitch,
        kernel_radius=2,
        quadrature_order=3,
    )
    assert torch.allclose(
        kernel,
        torch.full_like(kernel, 2.0 * scan_pitch**2),
        rtol=1e-12,
        atol=1e-24,
    )


def test_single_absorbing_cell_response() -> None:
    kernel = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]
    )
    obj = torch.zeros(1, 1, 9, 9)
    obj[0, 0, 4, 4] = 1.0
    image = scan_correlation(obj, kernel)
    assert torch.equal(image[0, 0, 3:6, 3:6], kernel[0, 0].flip(-2, -1))


def test_asymmetric_kernel_uses_correlation_without_hidden_flip() -> None:
    obj = torch.arange(25, dtype=torch.float64).reshape(1, 1, 5, 5)
    kernel = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 1.0, 10.0], [0.0, 0.0, 0.0]]]],
        dtype=torch.float64,
    )
    image = scan_correlation(obj, kernel)
    # At [2,2], K centre weights O[2,2] and K right weights O[2,3].
    assert image[0, 0, 2, 2] == obj[0, 0, 2, 2] + 10.0 * obj[0, 0, 2, 3]
    assert image[0, 0, 2, 2] != obj[0, 0, 2, 2] + 10.0 * obj[0, 0, 2, 1]


def test_tilted_focus_and_point_scan_move_in_opposite_directions() -> None:
    """A shifted illumination peak requires the sample to scan oppositely."""

    slm_size = 32
    fft_size = 128
    config = ORPAMModelBConfig(
        wavelength_m=1.0,
        focal_length_m=1.0,
        slm_pixel_pitch_m=1.0,
        slm_size=slm_size,
        fft_size=fft_size,
        # One pupil tilt period moves the focus four FFT pixels.  Make that
        # exactly one scan cell so every expected index is unambiguous.
        scan_pitch_m=4.0 / fft_size,
        kernel_radius=3,
        quadrature_order=8,
    )
    model = ORPAMForwardModelB(config)
    column = torch.arange(slm_size, dtype=torch.float64)[None].expand(slm_size, -1)
    positive_tilt = 2.0 * math.pi * column / slm_size
    point_object = torch.zeros(1, 1, 15, 15, dtype=torch.float64)
    point_object[0, 0, 7, 7] = 1.0

    def peak_indices(slm_phase: torch.Tensor, aberration: torch.Tensor):
        psf = model.optical_psf(slm_phase, aberration).intensity
        kernel = model.equivalent_kernel(psf)
        scan = model.scan_forward(point_object, kernel)
        psf_peak = torch.nonzero(psf[0, 0] == psf.max())[0]
        kernel_peak = torch.nonzero(kernel[0, 0] == kernel.max())[0]
        scan_peak = torch.nonzero(scan[0, 0] == scan.max())[0]
        return psf_peak.tolist(), kernel_peak.tolist(), scan_peak.tolist()

    # exp(+i Phi): focus goes right, while the delta-object scan peak goes left.
    assert peak_indices(torch.zeros_like(positive_tilt), positive_tilt) == (
        [64, 68],
        [3, 4],
        [7, 6],
    )
    # exp(-i Gamma): the known SLM tilt reverses both shifts.
    assert peak_indices(positive_tilt, torch.zeros_like(positive_tilt)) == (
        [64, 60],
        [3, 2],
        [7, 8],
    )


def test_uniform_object_response_and_zero_extended_boundary() -> None:
    kernel = torch.tensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]])
    kernel = kernel / kernel.sum()
    image = scan_correlation(torch.ones(1, 1, 21, 21), kernel)
    assert torch.allclose(image[..., 1:-1, 1:-1], torch.ones(1, 1, 19, 19))
    assert float(image[0, 0, 0, 0]) < 1.0


def test_kernel_support_reports_truncation_energy() -> None:
    size = 256
    dx = 1e-6
    coordinate = (torch.arange(size, dtype=torch.float64) - size // 2) * dx
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    sigma = 4e-6
    psf = torch.exp(-(x.square() + y.square()) / (2.0 * sigma**2))[None, None]
    psf = psf / (psf.sum() * dx**2)
    small = integrate_equivalent_kernel(
        psf,
        focus_pixel_pitch_m=dx,
        scan_pitch_m=5e-6,
        kernel_radius=1,
        quadrature_order=6,
    )
    large = integrate_equivalent_kernel(
        psf,
        focus_pixel_pitch_m=dx,
        scan_pitch_m=5e-6,
        kernel_radius=4,
        quadrature_order=6,
    )
    small_report = kernel_energy_diagnostics(small)
    large_report = kernel_energy_diagnostics(large)
    assert float(large_report["retained_energy"]) > float(small_report["retained_energy"])
    assert float(large_report["truncation_error"].abs()) < 2e-4


def test_autograd_matches_finite_difference_through_psf_kernel_and_scan() -> None:
    torch.manual_seed(3)
    config = ORPAMModelBConfig(
        wavelength_m=1.0e-6,
        focal_length_m=1.0e-3,
        slm_pixel_pitch_m=2.0e-6,
        slm_size=8,
        fft_size=32,
        scan_pitch_m=12e-6,
        kernel_radius=1,
        quadrature_order=3,
    )
    slm = 0.2 * torch.randn(8, 8, dtype=torch.float64)
    coordinate = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    basis = x + 0.37 * y + 0.21 * x * y
    obj = torch.rand(1, 1, 9, 9, dtype=torch.float64)
    loss_weight = torch.randn(1, 1, 9, 9, dtype=torch.float64)

    def objective(coefficient: torch.Tensor) -> torch.Tensor:
        psf = ideal_fourier_psf(slm, coefficient * basis, config).intensity
        kernel = integrate_equivalent_kernel(
            psf,
            focus_pixel_pitch_m=config.focus_pixel_pitch_m,
            scan_pitch_m=config.scan_pitch_m,
            kernel_radius=config.kernel_radius,
            quadrature_order=config.quadrature_order,
        )
        return (scan_correlation(obj, kernel) * loss_weight).sum()

    coefficient = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    analytical = torch.autograd.grad(objective(coefficient), coefficient)[0]
    epsilon = 1e-5
    numerical = (
        objective(torch.tensor(0.23 + epsilon, dtype=torch.float64))
        - objective(torch.tensor(0.23 - epsilon, dtype=torch.float64))
    ) / (2.0 * epsilon)
    assert float(analytical.abs()) > 1e-5
    assert torch.allclose(analytical, numerical, rtol=3e-4, atol=2e-7)


def test_scan_output_is_exactly_600_by_600() -> None:
    obj = torch.zeros(1, 1, 600, 600)
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    output = scan_correlation(obj, kernel)
    assert output.shape == (1, 1, 600, 600)


def _write_phase_fixture(root: Path) -> None:
    mat_dir = root / "hardware_4/mat"
    png_dir = root / "hardware_4/png"
    model_dir = root / "model_4/mat"
    mat_dir.mkdir(parents=True)
    png_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    for index in (1, 2):
        phase = np.arange(16, dtype=np.float32).reshape(4, 4) * (0.1 * index)
        sio.savemat(mat_dir / f"SLM_hw{index}.mat", {"phase_hw": phase})
        sio.savemat(model_dir / f"SLM_sim{index}.mat", {"proj_sim": phase})
        encoded = np.rint(np.mod(phase, 2 * np.pi) * 65535 / (2 * np.pi)).astype(np.uint16)
        Image.fromarray(encoded).save(png_dir / f"SLM_hw{index}.png")
    manifest = {
        "num_frames": 2,
        "hardware_shape": [4, 4],
        "model_shape": [4, 4],
        "same_coefficients_on_both_grids": True,
        "phase_units": "radians",
        "phase_sign_convention": "NeuWS uses aperture * exp(-1j * proj_sim)",
        "hardware_output": {
            "mat": "hardware_4/mat/SLM_hwN.mat:phase_hw",
            "png": "hardware_4/png/SLM_hwN.png",
        },
        "model_output": {"mat": "model_4/mat/SLM_simN.mat:proj_sim"},
        "png_phase_encoding": {"hardware_integer_range": [0, 65535]},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_phase_library_mat_png_and_measurement_numbering(tmp_path: Path) -> None:
    phase_root = tmp_path / "phases"
    _write_phase_fixture(phase_root)
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    measurement_manifest = {
        "num_frames": 2,
        "phase_source_directory": str((phase_root / "model_4/mat").resolve()),
        "frames": [
            {"frame_index": 1, "measurement_source": "s01.bin"},
            {"frame_index": 2, "measurement_source": "s02.bin"},
        ],
    }
    (measurements / "manifest.json").write_text(
        json.dumps(measurement_manifest), encoding="utf-8"
    )
    for index in (1, 2):
        phase = sio.loadmat(phase_root / f"model_4/mat/SLM_sim{index}.mat")["proj_sim"]
        sio.savemat(measurements / f"SLM_sim{index}.mat", {"proj_sim": phase})
        sio.savemat(
            measurements / f"SLM_raw{index}.mat",
            {"imsdata": np.zeros((4, 4), dtype=np.float32)},
        )
    library = HardwareSLMPhaseLibrary(phase_root)
    assert library.load([1, 2], validate_png=True).shape == (2, 4, 4)
    mapping = library.validate_measurement_mapping(measurements)
    assert [item["frame_index"] for item in mapping] == [1, 2]
    assert mapping[1]["hardware_phase_mat"].endswith("SLM_hw2.mat")


def test_repository_real_phase_library_mapping_when_available() -> None:
    project_root = Path(__file__).resolve().parents[1]
    phase_root = project_root / "data/slm50_noll4_28_decay4x_600_1080"
    measurement_dir = project_root / "data/2026-10-13_object_mmes"
    if not phase_root.is_dir() or not measurement_dir.is_dir():
        pytest.skip("The local real experiment data are not part of a minimal source checkout.")
    library = HardwareSLMPhaseLibrary(phase_root)
    assert library.shape == (1080, 1080)
    assert library.num_frames == 50
    assert library.validate_png_correspondence(1) < 5e-5
    assert library.validate_png_correspondence(50) < 5e-5
    mapping = library.validate_measurement_mapping(measurement_dir)
    assert mapping[0]["frame_index"] == 1
    assert mapping[-1]["frame_index"] == 50
