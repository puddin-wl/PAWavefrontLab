from pathlib import Path

import pytest

from workflows.real_experiment.run_pipeline import (
    _build_preprocess_command,
    _preprocess_effective_config,
    _preprocess_inputs_equivalent,
)


def test_pipeline_uses_fixed_template_decorrelation_without_depth_calibration() -> None:
    config = {
        "source_dir": "/tmp/raw",
        "origin_source": "/tmp/raw/origin_1_600_600_512_PA1.bin",
        "phase_dir": "/tmp/phases",
        "preprocess": {
            "method": "fixed-template-decorrelation-full-depth-mip",
            "backend": "cuda",
            "chunk_rows": 10,
            "measurement_glob": "s*_600_600_512_PA1.bin",
            "baseline_adc": 2048.0,
            "correlation_threshold": 0.7,
            "minimum_fitted_peak_adc": 80.0,
        },
    }

    command = _build_preprocess_command(config, Path("/tmp/output"), "brain")
    joined = " ".join(command)

    assert "prepare_decorrelated_real_dataset.py" in joined
    assert "--phase-dir /tmp/phases" in joined
    assert "--baseline-adc 2048.0" in joined
    assert "--correlation-threshold 0.7" in joined
    assert "--noise-template" in joined
    assert "grid-size" not in joined
    assert "teacher-refit" not in joined
    assert "signal-window" not in joined


def test_fixed_template_preprocess_defaults_use_small_chunks() -> None:
    effective = _preprocess_effective_config({})

    assert effective["method"] == "fixed-template-decorrelation-full-depth-mip"
    assert effective["chunk_rows"] == 10
    assert effective["baseline_adc"] == 2048.0


@pytest.mark.parametrize(
    "legacy_key",
    (
        "grid_size",
        "seed",
        "teacher_template",
        "teacher_correlation_threshold",
        "teacher_refit_threshold",
        "final_correlation_threshold",
    ),
)
def test_automatic_depth_parameters_are_rejected(legacy_key: str) -> None:
    with pytest.raises(ValueError, match="已停用的自动深度"):
        _preprocess_effective_config({"preprocess": {legacy_key: 1}})


def test_non_fixed_template_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="只支持"):
        _preprocess_effective_config(
            {"preprocess": {"method": "reference-guided-auto-depth"}}
        )


def test_preprocess_fingerprint_ignores_training_scene_label_only() -> None:
    stored = {
        "inputs": {
            "schema_version": 2,
            "scene_name": "original_training",
            "source_dir": "/raw",
            "noise_template": {"path": "/template.csv", "size": 1},
        }
    }
    current = {
        "inputs": {
            "schema_version": 3,
            "source_dir": "/raw",
            "noise_template": {"path": "/template.csv", "size": 1},
        }
    }

    assert _preprocess_inputs_equivalent(stored, current)

    changed_template = {"inputs": dict(current["inputs"])}
    changed_template["inputs"]["noise_template"] = {
        "path": "/other.csv",
        "size": 1,
    }
    assert not _preprocess_inputs_equivalent(stored, changed_template)
