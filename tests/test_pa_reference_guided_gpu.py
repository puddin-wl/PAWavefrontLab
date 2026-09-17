from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from preprocessing.pa_adaptive_denoising import (
    adaptive_excess_rms_projection,
    robust_linear_detrend,
)
from preprocessing.pa_denoising import clean_traces
from preprocessing.packed12 import decode_packed12
from preprocessing.pa_reference_guided_denoising import ReferenceGuidedTemplate


def _template() -> np.ndarray:
    x = np.arange(-20, 21, dtype=np.float32)
    values = (
        np.exp(-0.5 * (x / 2.0) ** 2)
        - 0.55 * np.exp(-0.5 * ((x - 5.0) / 2.8) ** 2)
        + 0.28 * np.exp(-0.5 * ((x + 7.0) / 3.0) ** 2)
    )
    return (values / np.max(np.abs(values))).astype(np.float32)


def _pack_u12(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    if flat.size % 2:
        raise ValueError("test helper requires an even number of samples")
    first = flat[0::2]
    second = flat[1::2]
    packed = np.empty(first.size * 3, dtype=np.uint8)
    packed[0::3] = (first & 0xFF).astype(np.uint8)
    packed[1::3] = (((first >> 8) & 0x0F) | ((second & 0x0F) << 4)).astype(
        np.uint8
    )
    packed[2::3] = ((second >> 4) & 0xFF).astype(np.uint8)
    return packed


def _calibration(depth: int = 128):
    return SimpleNamespace(
        depth=depth,
        signal_window=(70, 92),
        noise_windows=((8, 28), (102, 122)),
        template=ReferenceGuidedTemplate(
            values=_template(),
            correlation_threshold=0.70,
            minimum_fitted_peak_adc=80.0,
        ),
    )


def _pipeline():
    try:
        from preprocessing.pa_reference_guided_gpu import ReferenceGuidedGpuPipeline

        return ReferenceGuidedGpuPipeline(_calibration())
    except RuntimeError as exc:
        pytest.skip(f"CUDA/CuPy unavailable: {exc}")


def _synthetic_raw(calibration):
    rng = np.random.default_rng(11)
    raw = np.rint(
        2048.0 + rng.normal(0.0, 3.0, size=(96, calibration.depth))
    ).astype(np.int32)
    template = calibration.template.values
    half = template.size // 2
    for row in range(0, 60, 2):
        center = 45 + (row % 7)
        amplitude = 100.0 + (row % 5) * 12.0
        raw[row, center - half : center + half + 1] += np.rint(
            amplitude * template
        ).astype(np.int32)
    for row in range(1, 50, 5):
        raw[row, 78:83] += np.asarray([20, 55, 110, 55, 20])
    return np.clip(raw, 0, 4095).astype(np.uint16)


def _cpu_reference(raw, calibration):
    cleaned, _corr, _centers, _coeff, matched = clean_traces(
        raw,
        calibration.template.values,
        correlation_threshold=calibration.template.correlation_threshold,
        minimum_fitted_peak_adc=calibration.template.minimum_fitted_peak_adc,
    )
    before_residual, _ = robust_linear_detrend(raw)
    after_residual, _ = robust_linear_detrend(cleaned)
    signal = slice(*calibration.signal_window)
    before_power = float(np.sum(before_residual[:, signal] ** 2, dtype=np.float64))
    after_power = float(np.sum(after_residual[:, signal] ** 2, dtype=np.float64))
    projection = adaptive_excess_rms_projection(cleaned, calibration)
    return projection, int(np.sum(matched)), before_power, after_power


def _assert_matches_cpu(result, raw, calibration):
    projection, count, before, after = _cpu_reference(raw, calibration)
    np.testing.assert_allclose(result.projection, projection, rtol=2e-4, atol=2e-3)
    assert result.matched_count == count
    assert result.signal_power_before == pytest.approx(before, rel=2e-4)
    assert result.signal_power_after == pytest.approx(after, rel=2e-4)


def test_gpu_packed12_decode_matches_cpu() -> None:
    pipeline = _pipeline()
    rng = np.random.default_rng(7)
    values = rng.integers(0, 4096, size=(32, 128), dtype=np.uint16)
    packed = _pack_u12(values)
    gpu = pipeline._decode_packed12_gpu(packed, sample_count=values.size).get()
    np.testing.assert_array_equal(gpu, decode_packed12(packed))


def test_gpu_compatibility_path_matches_cpu_reference() -> None:
    pipeline = _pipeline()
    calibration = _calibration()
    raw = _synthetic_raw(calibration)
    result = pipeline.process_packed_chunk(_pack_u12(raw), aline_count=raw.shape[0])
    _assert_matches_cpu(result, raw, calibration)


def test_gpu_pinned_preloaded_slot_matches_cpu_reference() -> None:
    pipeline = _pipeline()
    calibration = _calibration()
    raw = _synthetic_raw(calibration)
    packed = _pack_u12(raw)
    slots = pipeline.prepare_streaming(
        max_packed_bytes=packed.size,
        max_alines=raw.shape[0],
    )
    slot = slots[0]
    slot.host[: packed.size] = packed
    pipeline.enqueue_h2d(slot, nbytes=packed.size)
    result = pipeline.process_preloaded_slot(slot, aline_count=raw.shape[0])
    _assert_matches_cpu(result, raw, calibration)


def test_gpu_workspace_is_reused_across_volumes() -> None:
    """The same experiment pipeline must keep its large buffers resident."""
    pipeline = _pipeline()
    calibration = _calibration()
    raw1 = _synthetic_raw(calibration)
    raw2 = raw1.copy()
    raw2[:, 75:80] = np.clip(raw2[:, 75:80].astype(np.int32) + 3, 0, 4095)
    packed1 = _pack_u12(raw1)
    packed2 = _pack_u12(raw2)

    slots1 = pipeline.prepare_streaming(
        max_packed_bytes=max(packed1.size, packed2.size),
        max_alines=raw1.shape[0],
    )
    slot_ptrs_before = tuple(int(slot.device.data.ptr) for slot in slots1)
    values_ptr_before = int(pipeline._values_buffer.data.ptr)
    centers_ptr_before = int(pipeline._centers_buffer.data.ptr)
    coefficients_ptr_before = int(pipeline._coefficients_buffer.data.ptr)
    matched_ptr_before = int(pipeline._matched_buffer.data.ptr)

    slots1[0].host[: packed1.size] = packed1
    pipeline.enqueue_h2d(slots1[0], nbytes=packed1.size)
    result1 = pipeline.process_preloaded_slot(slots1[0], aline_count=raw1.shape[0])
    _assert_matches_cpu(result1, raw1, calibration)

    # Preparing the next volume with identical dimensions must be a no-op for
    # the persistent transfer/decode workspace.
    slots2 = pipeline.prepare_streaming(
        max_packed_bytes=max(packed1.size, packed2.size),
        max_alines=raw2.shape[0],
    )
    assert slots2 is slots1
    assert tuple(int(slot.device.data.ptr) for slot in slots2) == slot_ptrs_before
    assert int(pipeline._values_buffer.data.ptr) == values_ptr_before
    assert int(pipeline._centers_buffer.data.ptr) == centers_ptr_before
    assert int(pipeline._coefficients_buffer.data.ptr) == coefficients_ptr_before
    assert int(pipeline._matched_buffer.data.ptr) == matched_ptr_before

    slots2[1].host[: packed2.size] = packed2
    pipeline.enqueue_h2d(slots2[1], nbytes=packed2.size)
    result2 = pipeline.process_preloaded_slot(slots2[1], aline_count=raw2.shape[0])
    _assert_matches_cpu(result2, raw2, calibration)

    pipeline.shutdown()
