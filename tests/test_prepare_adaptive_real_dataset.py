from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch, sentinel

import pytest

from preprocessing.pa_adaptive_denoising import CalibrationError
from tools import prepare_adaptive_real_dataset as prepare_module


def test_parser_defaults_to_automatic_backend() -> None:
    required = [
        "--source-dir",
        "source",
        "--origin-source",
        "origin.bin",
        "--template-data-dir",
        "template",
        "--output-dir",
        "output",
        "--scene-name",
        "scene",
    ]
    args = prepare_module.build_parser().parse_args(required)
    assert args.backend == "auto"
    assert args.grid_size == 20
    assert args.seed == 0


def test_auto_backend_falls_back_to_cpu() -> None:
    with patch.object(
        prepare_module,
        "cuda_backend_info",
        side_effect=RuntimeError("no CUDA"),
    ):
        backend, info = prepare_module._resolve_backend("auto")

    assert backend == "cpu"
    assert info["selection"] == "auto_fallback"
    assert "no CUDA" in info["cuda_unavailable_reason"]


def test_project_passes_frozen_calibration() -> None:
    with patch.object(
        prepare_module,
        "load_packed12_adaptive_projection",
        return_value=sentinel.result,
    ) as loader:
        result = prepare_module._project(
            Path("source.bin"),
            sentinel.calibration,
            backend="cpu",
            chunk_rows=7,
        )

    assert result is sentinel.result
    loader.assert_called_once_with(
        Path("source.bin"),
        calibration=sentinel.calibration,
        backend="cpu",
        chunk_rows=7,
    )


def test_prepare_is_atomic_on_success(tmp_path: Path) -> None:
    output = tmp_path / "adaptive"
    args = Namespace(output_dir=str(output))

    def fake_prepare(work_dir: Path, _: Namespace) -> None:
        (work_dir / "complete.txt").write_text("ok", encoding="utf-8")

    with patch.object(prepare_module, "_prepare_into", side_effect=fake_prepare):
        actual = prepare_module.prepare(args)

    assert actual == output
    assert (output / "complete.txt").read_text(encoding="utf-8") == "ok"
    assert not list(tmp_path.glob(".adaptive.tmp-*"))


def test_failed_calibration_keeps_report_not_partial_dataset(tmp_path: Path) -> None:
    output = tmp_path / "adaptive"
    args = Namespace(output_dir=str(output))
    failure = CalibrationError("unsafe", diagnostics={"metric": 1.2})

    with (
        patch.object(prepare_module, "_prepare_into", side_effect=failure),
        pytest.raises(RuntimeError, match="诊断已保存"),
    ):
        prepare_module.prepare(args)

    assert not output.exists()
    report = json.loads(
        (tmp_path / "adaptive_failed" / "calibration_failed.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is False
    assert report["error_type"] == "CalibrationError"
    assert report["diagnostics"] == {"metric": 1.2}
