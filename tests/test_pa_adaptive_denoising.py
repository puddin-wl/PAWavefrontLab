from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.pa_adaptive_denoising import (
    AdaptiveCalibration,
    AdaptiveTemplate,
    CalibrationError,
    adaptive_excess_rms_projection,
    calibrate_from_samples,
    clean_adaptive_traces,
    detect_signal_window,
    load_packed12_adaptive_projection,
)


def _pack_ubit12_little(values: np.ndarray) -> bytes:
    samples = np.asarray(values, dtype=np.uint16).reshape(-1)
    assert samples.size % 2 == 0
    pairs = samples.reshape(-1, 2)
    packed = np.empty((pairs.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = pairs[:, 0] & 0xFF
    packed[:, 1] = ((pairs[:, 0] >> 8) & 0x0F) | (
        (pairs[:, 1] & 0x0F) << 4
    )
    packed[:, 2] = pairs[:, 1] >> 4
    return packed.tobytes()


def _synthetic_frames(*, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    depth = 256
    positions = np.arange(depth, dtype=np.float32)
    transient = np.asarray([0, 0.2, -0.6, 1.0, -0.6, 0.2, 0], dtype=np.float32)
    frames = []
    for frame_index in range(10):
        count = 240
        values = (
            2000.0
            + 0.04 * positions[None, :]
            + rng.normal(0.0, 2.0, size=(count, depth))
        ).astype(np.float32)
        signal_rows = rng.choice(count, size=12, replace=False)
        for row in signal_rows:
            center = int(rng.integers(132, 144))
            values[row, center - 2 : center + 3] += np.asarray(
                [5, 18, 36, 18, 5], dtype=np.float32
            )
        transient_rows = rng.choice(count, size=150, replace=False)
        for row in transient_rows:
            center = 35 + int(rng.integers(-1, 2))
            amplitude = float(rng.uniform(18.0, 26.0))
            values[row, center - 3 : center + 4] += amplitude * transient
        frames.append(values)
    return frames


def test_detect_signal_window_finds_sparse_depth_band() -> None:
    traces = np.concatenate(_synthetic_frames(), axis=0)
    window, diagnostics = detect_signal_window(traces)

    assert window[0] <= 132
    assert window[1] >= 144
    assert 32 <= window[1] - window[0] <= 160
    assert diagnostics["significant_energy_coverage"] >= 0.995


def test_experiment_calibration_is_frozen_and_discovers_template() -> None:
    frames = _synthetic_frames()
    calibration = calibrate_from_samples(
        frames,
        height=4,
        width=60,
        depth=256,
        grid_size=4,
    )

    assert calibration.signal_window[0] <= 132
    assert calibration.signal_window[1] >= 144
    assert len(calibration.noise_windows) == 2
    assert len(calibration.templates) >= 1
    assert calibration.qc["minimum_fold_window_iou"] >= 0.85
    assert calibration.qc["passed"] is True
    assert calibration.fingerprint() == calibration.fingerprint()


def test_calibration_rejects_too_few_frames() -> None:
    with pytest.raises(CalibrationError, match="至少需要 5"):
        calibrate_from_samples(
            _synthetic_frames()[:4],
            height=4,
            width=60,
            depth=256,
        )


def test_adaptive_projection_subtracts_interpolated_noise_power() -> None:
    calibration = AdaptiveCalibration(
        height=1,
        width=2,
        depth=20,
        signal_window=(8, 12),
        noise_windows=((0, 4), (16, 20)),
    )
    residual = np.zeros((2, 20), dtype=np.float32)
    residual[0, 8:12] = 5.0
    residual[1, 8:12] = 9.0

    projection = adaptive_excess_rms_projection(residual, calibration)

    np.testing.assert_allclose(projection, [5.0, 9.0], atol=1e-5)
    assert projection.dtype == np.float32


def test_streaming_projection_matches_in_memory_and_has_no_spatial_filter(
    tmp_path: Path,
) -> None:
    height, width, depth = 2, 3, 20
    calibration = AdaptiveCalibration(
        height=height,
        width=width,
        depth=depth,
        signal_window=(8, 12),
        noise_windows=((0, 4), (16, 20)),
    )
    volume = np.full((height * width, depth), 2048, dtype=np.uint16)
    for row in range(height * width):
        volume[row, 8:12] += np.uint16(row + 1)
    source = tmp_path / "known.bin"
    source.write_bytes(_pack_ubit12_little(volume))

    cleaned, _, _, _ = clean_adaptive_traces(volume, calibration)
    expected = adaptive_excess_rms_projection(cleaned, calibration).reshape(
        height, width
    )
    actual = load_packed12_adaptive_projection(
        source, calibration=calibration, chunk_rows=1
    ).projection

    np.testing.assert_allclose(actual, expected, atol=1e-5)
    # An isolated A-line remains isolated: there is no 2-D smoothing stage.
    assert np.count_nonzero(actual) == np.count_nonzero(expected)


def test_signed_template_is_removed_once_per_aline() -> None:
    template = np.asarray([0, 0.2, -0.6, 1.0, -0.6, 0.2, 0], dtype=np.float32)
    model = AdaptiveTemplate(values=template, correlation_threshold=0.95)
    calibration = AdaptiveCalibration(
        height=1,
        width=2,
        depth=40,
        signal_window=(16, 24),
        noise_windows=((0, 8), (32, 40)),
        templates=(model,),
    )
    traces = np.full((2, 40), 2048.0, dtype=np.float32)
    traces[0, 17:24] += 30.0 * template
    traces[1, 17:24] -= 30.0 * template

    cleaned, counts, _, _ = clean_adaptive_traces(traces, calibration)

    assert counts == (2,)
    np.testing.assert_allclose(cleaned, 0.0, atol=1e-3)
