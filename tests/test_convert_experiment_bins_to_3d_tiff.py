from pathlib import Path

import pytest

from tools.convert_experiment_bins_to_3d_tiff import (
    _experiment_date,
    _infer_shape,
    _source_sort_key,
)


def test_infers_shape_from_pa_filename() -> None:
    assert _infer_shape(Path("s01_time_1_600_600_512_PA1.bin")) == (600, 600, 512)


def test_orders_origin_before_numbered_measurements() -> None:
    names = [
        "s10_x_1_600_600_512_PA1.bin",
        "s02_x_1_600_600_512_PA1.bin",
        "origin_x_1_600_600_512_PA1.bin",
    ]
    ordered = sorted((Path(name) for name in names), key=_source_sort_key)
    assert [path.name for path in ordered] == [names[2], names[1], names[0]]


def test_uses_source_directory_name_as_experiment_date() -> None:
    assert _experiment_date(Path("/tmp/2026-10-13"), None) == "2026-10-13"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _experiment_date(Path("/tmp/raw"), None)
