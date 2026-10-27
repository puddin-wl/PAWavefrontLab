from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tools.orpam_b_to_a_experiment import (
    _array_metrics,
    _phase_metrics,
    fixed_aberration,
    load_vessel_crop,
)


def test_vessel_crop_preserves_uint16_and_exact_slice(tmp_path: Path) -> None:
    image = np.arange(1_000_000, dtype=np.uint32).reshape(1000, 1000)
    image = (image % 60000).astype(np.uint16)
    path = tmp_path / "vessel.tif"
    tifffile.imwrite(path, image[None])

    crop, record = load_vessel_crop(path)

    assert crop.dtype == np.uint16
    assert np.array_equal(crop, image[200:800, 200:800])
    assert record["stored_shape"] == [1, 1000, 1000]
    assert record["dtype"] == "uint16"
    assert record["linear_scale_divisor"] == float(crop.max())


def test_fixed_aberration_is_grid_independent_and_has_expected_centre() -> None:
    small = fixed_aberration(20).numpy()
    large = fixed_aberration(40).numpy()
    assert small.shape == (20, 20)
    assert large.shape == (40, 40)
    assert np.isfinite(small).all()
    # The even-grid origin lies between four pixels, so the centre samples are
    # small but not forced artificially to zero.
    assert abs(float(small[9:11, 9:11].mean())) < 0.01
    assert float(small.std()) == pytest.approx(float(large.std()), rel=0.03)


def test_phase_metrics_remove_only_piston_and_preserve_sign() -> None:
    truth = fixed_aberration(32).numpy()
    same_with_piston = truth + 3.2
    matched = _phase_metrics(same_with_piston, truth)
    reversed_sign = _phase_metrics(-truth, truth)

    assert matched["rmse_rad"] < 2e-7
    assert matched["relative_rmse_to_truth_rms"] < 2e-6
    assert matched["correlation"] == pytest.approx(1.0, abs=1e-7)
    assert reversed_sign["correlation"] == pytest.approx(-1.0, abs=1e-7)
    assert reversed_sign["rmse_rad"] > 0.1


def test_object_metrics_report_direct_and_shape_only_scales_separately() -> None:
    truth = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    estimate = 2.0 * truth
    metrics = _array_metrics(estimate, truth)

    assert metrics["rmse"] > 0
    assert metrics["correlation"] == pytest.approx(1.0)
    assert metrics["normalized_l2"] < 1e-12
    assert metrics["shape_only_fitted_gain_truth_per_estimate"] == pytest.approx(0.5)
    assert metrics["shape_only_gain_fitted_rmse"] < 1e-12
