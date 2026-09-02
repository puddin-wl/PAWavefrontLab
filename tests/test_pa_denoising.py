from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.pa_denoising import (
    clean_traces,
    excess_rms_projection,
    load_packed12_excess_rms_projection,
    load_packed12_template_xcorr_mip_projection,
    load_template_csv,
)


def _pack_ubit12_little(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    assert values.size % 2 == 0
    pairs = values.reshape(-1, 2)
    packed = np.empty((pairs.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = pairs[:, 0] & 0xFF
    packed[:, 1] = ((pairs[:, 0] >> 8) & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
    packed[:, 2] = pairs[:, 1] >> 4
    return packed.tobytes()


def test_excess_rms_removes_linear_baseline_and_returns_signal_amplitude() -> None:
    depth = np.arange(12, dtype=np.float32)
    volume = np.broadcast_to(100.0 + 2.0 * depth, (2, 3, 12)).copy()
    volume[:, :, 4:8] += 5.0

    projection = excess_rms_projection(
        volume,
        signal_window=(4, 8),
        noise_windows=((0, 3), (9, 12)),
    )

    np.testing.assert_allclose(projection, 5.0, atol=1e-5)
    assert projection.dtype == np.float32


def test_streaming_packed12_projection_matches_in_memory_result(tmp_path: Path) -> None:
    height, width, depth = 2, 3, 12
    positions = np.arange(depth, dtype=np.uint16)
    volume = np.broadcast_to(1000 + positions, (height, width, depth)).copy()
    volume[:, :, 4:8] += np.arange(1, height * width + 1, dtype=np.uint16).reshape(
        height, width, 1
    )
    source = tmp_path / "known.bin"
    source.write_bytes(_pack_ubit12_little(volume))

    expected = excess_rms_projection(
        volume,
        signal_window=(4, 8),
        noise_windows=((0, 3), (9, 12)),
    )
    actual = load_packed12_excess_rms_projection(
        source,
        height=height,
        width=width,
        depth=depth,
        signal_window=(4, 8),
        noise_windows=((0, 3), (9, 12)),
        spatial_sigma=0,
        chunk_rows=1,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_rejects_overlapping_signal_and_noise_windows() -> None:
    with pytest.raises(ValueError, match="重叠"):
        excess_rms_projection(
            np.zeros((1, 1, 12), dtype=np.float32),
            signal_window=(4, 8),
            noise_windows=((0, 5), (9, 12)),
        )


def _shifted_template(template: np.ndarray, *, depth: int, center: int) -> np.ndarray:
    half_width = template.size // 2
    indices = np.arange(depth) - center + half_width
    shifted = np.zeros(depth, dtype=np.float32)
    valid = (indices >= 0) & (indices < template.size)
    shifted[valid] = template[indices[valid]]
    return shifted


@pytest.mark.parametrize("coefficient", [40.0, -40.0])
@pytest.mark.parametrize("center", [0, 6, 11])
def test_template_xcorr_removes_signed_and_edge_truncated_matches(
    coefficient: float, center: int
) -> None:
    template = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float32)
    raw = 2048.0 + coefficient * _shifted_template(template, depth=12, center=center)

    cleaned, correlation, centers, fitted, matched = clean_traces(
        raw[None, :],
        template,
        correlation_threshold=0.99,
        minimum_fitted_peak_adc=80.0,
    )

    assert matched.tolist() == [True]
    assert centers.tolist() == [center]
    assert correlation[0] == pytest.approx(np.sign(coefficient), abs=1e-5)
    assert fitted[0] == pytest.approx(coefficient, abs=1e-4)
    np.testing.assert_allclose(cleaned, 2048.0, atol=1e-4)


def test_template_xcorr_amplitude_gate_prevents_low_energy_false_match() -> None:
    template = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float32)
    raw = 2048.0 + 5.0 * _shifted_template(template, depth=12, center=6)

    cleaned, _, _, _, matched = clean_traces(
        raw[None, :],
        template,
        correlation_threshold=0.70,
        minimum_fitted_peak_adc=80.0,
    )

    assert matched.tolist() == [False]
    np.testing.assert_array_equal(cleaned[0], raw)


def test_template_xcorr_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="correlation_threshold"):
        clean_traces(
            np.zeros((1, 12), dtype=np.float32),
            np.ones(5, dtype=np.float32),
            correlation_threshold=1.000001,
            minimum_fitted_peak_adc=80.0,
        )


def test_load_template_csv_requires_centered_consecutive_samples(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "relative_sample,template_relative_adc\n-1,0\n0,3\n1,0\n",
        encoding="utf-8",
    )
    np.testing.assert_array_equal(
        load_template_csv(valid), np.asarray([0.0, 3.0, 0.0], dtype=np.float32)
    )

    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "relative_sample,template_relative_adc\n-2,0\n0,3\n1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="连续"):
        load_template_csv(invalid)


def test_streaming_template_xcorr_projection_matches_trace_cleaning(
    tmp_path: Path,
) -> None:
    height, width, depth = 2, 3, 12
    template = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float32)
    volume = np.full((height, width, depth), 2048, dtype=np.uint16)
    volume[0, 1] += (
        40.0 * _shifted_template(template, depth=depth, center=6)
    ).astype(np.uint16)
    volume[1, 2] += (
        50.0 * _shifted_template(template, depth=depth, center=11)
    ).astype(np.uint16)
    source = tmp_path / "known.bin"
    source.write_bytes(_pack_ubit12_little(volume))

    result = load_packed12_template_xcorr_mip_projection(
        source,
        template=template,
        height=height,
        width=width,
        depth=depth,
        correlation_threshold=0.99,
        minimum_fitted_peak_adc=80.0,
        chunk_rows=1,
    )

    np.testing.assert_array_equal(
        result.original_projection,
        np.maximum(volume.max(axis=2).astype(np.float32) - 2048.0, 0.0),
    )
    np.testing.assert_allclose(result.projection, 0.0, atol=1e-4)
    assert int(result.matched_map.sum()) == 2
