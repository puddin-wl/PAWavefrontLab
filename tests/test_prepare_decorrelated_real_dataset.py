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
