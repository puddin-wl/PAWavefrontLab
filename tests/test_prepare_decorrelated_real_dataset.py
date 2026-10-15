from argparse import Namespace
from pathlib import Path
from unittest.mock import patch, sentinel

import numpy as np
import pytest

from tools import prepare_decorrelated_real_dataset as prepare_module


def _args(backend: str) -> Namespace:
    return Namespace(
        backend=backend,
        height=9,
        width=11,
        depth=13,
        baseline_adc=2048.0,
        correlation_threshold=0.7,
        minimum_fitted_peak_adc=80.0,
        chunk_rows=3,
    )


@pytest.mark.parametrize(
    ("backend", "selected_name", "unselected_name"),
    (
        (
            "cpu",
            "load_packed12_template_xcorr_mip_projection",
            "load_packed12_template_xcorr_mip_projection_gpu",
        ),
        (
            "cuda",
            "load_packed12_template_xcorr_mip_projection_gpu",
            "load_packed12_template_xcorr_mip_projection",
        ),
    ),
)
def test_project_dispatches_to_selected_backend(
    backend: str,
    selected_name: str,
    unselected_name: str,
) -> None:
    template = np.ones(5, dtype=np.float32)
    with (
        patch.object(
            prepare_module, selected_name, return_value=sentinel.result
        ) as selected,
        patch.object(prepare_module, unselected_name) as unselected,
    ):
        result = prepare_module._project(
            Path("input.bin"), template=template, args=_args(backend)
        )

    assert result is sentinel.result
    selected.assert_called_once_with(
        Path("input.bin"),
        template=template,
        height=9,
        width=11,
        depth=13,
        baseline_adc=2048.0,
        correlation_threshold=0.7,
        minimum_fitted_peak_adc=80.0,
        chunk_rows=3,
    )
    unselected.assert_not_called()


def test_parser_defaults_to_cuda_and_accepts_cpu() -> None:
    required = [
        "--template-data-dir",
        "template",
        "--source-dir",
        "source",
        "--origin-source",
        "origin.bin",
        "--output-dir",
        "output",
        "--scene-name",
        "scene",
    ]
    assert prepare_module.build_parser().parse_args(required).backend == "cuda"
    assert (
        prepare_module.build_parser().parse_args([*required, "--backend", "cpu"]).backend
        == "cpu"
    )


def test_parser_accepts_direct_phase_directory_mode() -> None:
    args = prepare_module.build_parser().parse_args(
        [
            "--phase-dir",
            "phases",
            "--source-dir",
            "source",
            "--origin-source",
            "origin.bin",
            "--output-dir",
            "output",
            "--scene-name",
            "scene",
        ]
    )

    assert args.phase_dir == "phases"
    assert args.template_data_dir is None
    assert args.measurement_glob == "s*_600_600_512_PA1.bin"


def test_discover_direct_inputs_pairs_measurements_and_phases(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    phases = tmp_path / "phases"
    source.mkdir()
    phases.mkdir()
    origin = source / "origin_20261013-090113_1_6_6_8_PA1.bin"
    origin.touch()
    for index in (1, 2):
        (source / f"s{index:02d}_20261013-090{index}_1_6_6_8_PA1.bin").touch()
        (phases / f"SLM_sim{index}.mat").touch()
    args = Namespace(
        source_dir=str(source),
        phase_dir=str(phases),
        origin_source=str(origin),
        measurement_glob="s*_6_6_8_PA1.bin",
        scene_name="brain",
    )

    manifest, measurements, phase_files, shape = prepare_module._discover_direct_inputs(
        args
    )

    assert shape == (6, 6, 8)
    assert [path.name for path in measurements] == [
        "s01_20261013-0901_1_6_6_8_PA1.bin",
        "s02_20261013-0902_1_6_6_8_PA1.bin",
    ]
    assert sorted(phase_files) == [1, 2]
    assert manifest["num_frames"] == 2
    assert manifest["sample_prefix"] == "s"
