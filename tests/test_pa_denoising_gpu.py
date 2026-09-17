from __future__ import annotations

import numpy as np
import pytest

from preprocessing.pa_denoising import (
    clean_traces,
    load_packed12_template_xcorr_mip_projection,
)
from preprocessing.pa_denoising_gpu import (
    clean_traces_gpu,
    clean_traces_gpu_adaptive,
    load_packed12_template_xcorr_mip_projection_gpu,
)
from preprocessing.pa_adaptive_denoising import (
    AdaptiveCalibration,
    AdaptiveTemplate,
    clean_adaptive_traces,
)


def _cupy_available() -> bool:
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cupy_available(),
    reason="CUDA/CuPy unavailable",
)


def _shifted_template(template: np.ndarray, *, depth: int, center: int) -> np.ndarray:
    half_width = template.size // 2
    indices = np.arange(depth) - center + half_width
    shifted = np.zeros(depth, dtype=np.float32)
    valid = (indices >= 0) & (indices < template.size)
    shifted[valid] = template[indices[valid]]
    return shifted


def _pack_packed12(values: np.ndarray) -> bytes:
    samples = np.asarray(values, dtype=np.uint16).reshape(-1, 2)
    packed = np.empty((samples.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = samples[:, 0] & 0xFF
    packed[:, 1] = ((samples[:, 0] >> 8) & 0x0F) | (
        (samples[:, 1] & 0x0F) << 4
    )
    packed[:, 2] = samples[:, 1] >> 4
    return packed.tobytes()


@pytest.mark.parametrize("coefficient", [40.0, -40.0])
@pytest.mark.parametrize("center", [0, 6, 11])
def test_gpu_matches_cpu_for_signed_edge_matches(
    coefficient: float,
    center: int,
) -> None:
    template = np.asarray(
        [0.0, 1.0, 3.0, 1.0, 0.0],
        dtype=np.float32,
    )
    raw = (
        2048.0
        + coefficient
        * _shifted_template(template, depth=12, center=center)
    )[None, :]

    cpu = clean_traces(
        raw,
        template,
        correlation_threshold=0.99,
        minimum_fitted_peak_adc=80.0,
    )
    gpu = clean_traces_gpu(
        raw,
        template,
        correlation_threshold=0.99,
        minimum_fitted_peak_adc=80.0,
    )

    np.testing.assert_allclose(gpu[0], cpu[0], atol=2e-3, rtol=1e-5)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=2e-4, rtol=1e-5)
    np.testing.assert_array_equal(gpu[2], cpu[2])
    np.testing.assert_allclose(gpu[3], cpu[3], atol=2e-3, rtol=1e-5)
    np.testing.assert_array_equal(gpu[4], cpu[4])


def test_gpu_matches_cpu_on_random_batch() -> None:
    rng = np.random.default_rng(12345)
    template = np.asarray(
        [0.0, 0.5, 2.0, 4.0, 2.0, 0.5, 0.0],
        dtype=np.float32,
    )
    raw = rng.normal(2048.0, 8.0, size=(64, 64)).astype(np.float32)

    for row, center, coefficient in (
        (3, 8, 50.0),
        (17, 30, -45.0),
        (41, 60, 55.0),
    ):
        raw[row] += coefficient * _shifted_template(
            template,
            depth=64,
            center=center,
        )

    cpu = clean_traces(
        raw,
        template,
        correlation_threshold=0.70,
        minimum_fitted_peak_adc=80.0,
    )
    gpu = clean_traces_gpu(
        raw,
        template,
        correlation_threshold=0.70,
        minimum_fitted_peak_adc=80.0,
    )

    np.testing.assert_allclose(gpu[0], cpu[0], atol=5e-3, rtol=1e-5)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=5e-4, rtol=1e-5)
    np.testing.assert_array_equal(gpu[2], cpu[2])
    np.testing.assert_allclose(gpu[3], cpu[3], atol=5e-3, rtol=1e-5)
    np.testing.assert_array_equal(gpu[4], cpu[4])


def test_gpu_packed12_projection_matches_cpu(tmp_path) -> None:
    height, width, depth = 2, 3, 12
    template = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float32)
    raw = np.full((height * width, depth), 2048, dtype=np.uint16)
    raw[1] += (40 * _shifted_template(template, depth=depth, center=6)).astype(
        np.uint16
    )
    raw[4] -= (40 * _shifted_template(template, depth=depth, center=10)).astype(
        np.uint16
    )
    source = tmp_path / "synthetic.bin"
    source.write_bytes(_pack_packed12(raw))
    options = dict(
        template=template,
        height=height,
        width=width,
        depth=depth,
        correlation_threshold=0.99,
        minimum_fitted_peak_adc=80.0,
        chunk_rows=1,
    )

    cpu = load_packed12_template_xcorr_mip_projection(source, **options)
    gpu = load_packed12_template_xcorr_mip_projection_gpu(source, **options)

    np.testing.assert_allclose(gpu.projection, cpu.projection, atol=2e-3)
    np.testing.assert_array_equal(gpu.original_projection, cpu.original_projection)
    np.testing.assert_allclose(gpu.correlation_map, cpu.correlation_map, atol=2e-4)
    np.testing.assert_array_equal(gpu.center_map, cpu.center_map)
    np.testing.assert_allclose(gpu.coefficient_map, cpu.coefficient_map, atol=2e-3)
    np.testing.assert_array_equal(gpu.matched_map, cpu.matched_map)


def test_adaptive_gpu_matches_cpu_with_per_aline_thresholds() -> None:
    rng = np.random.default_rng(9876)
    template = np.asarray(
        [0.0, 0.2, -0.6, 1.0, -0.6, 0.2, 0.0], dtype=np.float32
    )
    raw = rng.normal(0.0, 2.0, size=(32, 64)).astype(np.float32)
    raw[3] += 20.0 * _shifted_template(template, depth=64, center=24)
    raw[19] -= 22.0 * _shifted_template(template, depth=64, center=37)
    thresholds = np.linspace(8.0, 14.0, raw.shape[0], dtype=np.float32)

    cpu_calibration = AdaptiveCalibration(
        height=1,
        width=32,
        depth=64,
        signal_window=(20, 44),
        noise_windows=((0, 16), (48, 64)),
        templates=(AdaptiveTemplate(values=template, correlation_threshold=0.90),),
    )
    cpu, _, _, _ = clean_adaptive_traces(raw, cpu_calibration, backend="cpu")
    # The public GPU primitive is also checked directly because the adaptive
    # pipeline derives its thresholds from each trace's robust noise scale.
    gpu_direct, matched = clean_traces_gpu_adaptive(
        raw,
        template,
        correlation_threshold=0.90,
        minimum_fitted_peak_adc=thresholds,
    )

    assert matched.shape == (raw.shape[0],)
    assert np.isfinite(gpu_direct).all()
    gpu, _, _, _ = clean_adaptive_traces(raw, cpu_calibration, backend="cuda")
    np.testing.assert_allclose(gpu, cpu, atol=5e-3, rtol=1e-5)
