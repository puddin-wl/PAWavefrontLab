from __future__ import annotations

import numpy as np

from preprocessing.pa_reference_guided_denoising import (
    fit_experiment_template_from_teacher,
)
from preprocessing.pa_denoising import clean_traces


def _teacher() -> np.ndarray:
    x = np.arange(-20, 21, dtype=np.float32)
    values = (
        1.00 * np.exp(-0.5 * (x / 2.0) ** 2)
        - 0.55 * np.exp(-0.5 * ((x - 5.0) / 2.8) ** 2)
        + 0.28 * np.exp(-0.5 * ((x + 7.0) / 3.0) ** 2)
    )
    return (values / np.max(np.abs(values))).astype(np.float32)


def _current_template() -> np.ndarray:
    teacher = _teacher()
    # A modest experiment-to-experiment waveform change: still the same family,
    # but not identical to the teacher.
    changed = teacher.copy()
    changed += 0.10 * np.roll(teacher, 1)
    changed -= 0.04 * np.roll(teacher, -2)
    changed /= np.max(np.abs(changed))
    return changed.astype(np.float32)


def test_teacher_guides_refit_to_current_experiment() -> None:
    rng = np.random.default_rng(12)
    teacher = _teacher()
    current = _current_template()
    depth = 160
    frame_count = 10
    traces_per_frame = 100
    frames = []
    ids = []
    for frame in range(frame_count):
        traces = (2048 + rng.normal(0, 3, size=(traces_per_frame, depth))).astype(np.float32)
        for row in range(30):
            center = int(rng.integers(25, 60))
            amp = float(rng.uniform(100, 160))
            half = current.size // 2
            traces[row, center-half:center+half+1] += amp * current
        # Sparse true PA signal in a separate depth band.
        for row in range(30, 40):
            center = int(rng.integers(105, 120))
            traces[row, center-2:center+3] += np.array([20, 55, 100, 55, 20], dtype=np.float32)
        frames.append(traces)
        ids.append(np.full(traces_per_frame, frame, dtype=np.int32))

    all_traces = np.concatenate(frames)
    frame_ids = np.concatenate(ids)
    model, diagnostics = fit_experiment_template_from_teacher(
        all_traces,
        frame_ids,
        signal_window=(100, 125),
        teacher_template=teacher,
        teacher_correlation_threshold=0.70,
        teacher_refit_threshold=0.80,
        final_correlation_threshold=0.70,
        minimum_fitted_peak_adc=80.0,
        minimum_events=50,
        minimum_frames=5,
    )

    assert model.source_event_count >= 50
    assert model.source_frame_count >= 5
    assert model.teacher_to_refined_ncc >= 0.75
    assert diagnostics["candidate_event_count"] >= diagnostics["refit_event_count"]

    def ncc(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert abs(ncc(model.values, current)) >= abs(ncc(teacher, current)) - 0.03


def test_refined_template_reuses_old_subtraction_core() -> None:
    template = _teacher()
    traces = np.full((2, 100), 2048.0, dtype=np.float32)
    half = template.size // 2
    traces[0, 50-half:50+half+1] += 120.0 * template
    traces[1, 60-half:60+half+1] -= 140.0 * template

    cleaned, corr, centers, coeff, matched = clean_traces(
        traces,
        template,
        correlation_threshold=0.70,
        minimum_fitted_peak_adc=80.0,
    )

    assert matched.tolist() == [True, True]
    assert abs(int(centers[0]) - 50) <= 1
    assert abs(int(centers[1]) - 60) <= 1
    np.testing.assert_allclose(cleaned, 2048.0, atol=2e-2)
