from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from savetif.bin_to_tiff import (
    convert_bin_to_mip_tiff,
    convert_bin_to_tiff,
    convert_directory,
    convert_directory_to_mip,
)


def _pack_ubit12_little(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    assert values.size % 2 == 0
    assert np.all(values <= 4095)
    pairs = values.reshape(-1, 2)
    packed = np.empty((pairs.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = pairs[:, 0] & 0xFF
    packed[:, 1] = ((pairs[:, 0] >> 8) & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
    packed[:, 2] = pairs[:, 1] >> 4
    return packed.tobytes()


def test_convert_bin_to_float32_multipage_tiff(tmp_path: Path) -> None:
    height, width, depth = 2, 3, 4
    volume = np.array(
        [
            [[0, 1, 15, 16], [255, 256, 4094, 4095], [3, 17, 300, 400]],
            [[500, 600, 700, 800], [900, 1000, 1100, 1200], [1300, 1400, 1500, 1600]],
        ],
        dtype=np.uint16,
    )
    source = tmp_path / "known.bin"
    destination = tmp_path / "known.tif"
    source.write_bytes(_pack_ubit12_little(volume))

    result = convert_bin_to_tiff(
        source, destination, height=height, width=width, depth=depth, chunk_bytes=7
    )

    assert result == destination.resolve()
    actual = tifffile.imread(destination)
    np.testing.assert_array_equal(actual, volume.transpose(2, 0, 1).astype(np.float32))
    assert actual.dtype == np.float32
    with tifffile.TiffFile(destination) as tiff:
        assert len(tiff.pages) == depth
        assert tiff.pages[0].photometric == tifffile.PHOTOMETRIC.MINISBLACK
        assert tiff.pages[0].compression == tifffile.COMPRESSION.NONE


def test_rejects_wrong_bin_size(tmp_path: Path) -> None:
    source = tmp_path / "short.bin"
    source.write_bytes(b"\x00" * 5)

    with pytest.raises(ValueError, match="文件大小"):
        convert_bin_to_tiff(source, tmp_path / "result.tif", height=1, width=1, depth=4)


def test_directory_conversion_preserves_matlab_filename(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    volume = np.arange(8, dtype=np.uint16).reshape(1, 2, 4)
    (input_dir / "sample.bin").write_bytes(_pack_ubit12_little(volume))

    outputs = convert_directory(input_dir, output_dir, height=1, width=2, depth=4)

    assert outputs == [(output_dir / "sample.bin.tif").resolve()]
    np.testing.assert_array_equal(
        tifffile.imread(outputs[0]), volume.transpose(2, 0, 1).astype(np.float32)
    )


def test_direct_mip_matches_full_volume_preprocessing(tmp_path: Path) -> None:
    volume = np.array(
        [
            [[2047, 2048, 2050, 2000], [100, 3000, 200, 400]],
            [[4095, 0, 2048, 2049], [2000, 2001, 2002, 2003]],
        ],
        dtype=np.uint16,
    )
    source = tmp_path / "sample_PA1.bin"
    destination = tmp_path / "sample_mip.tif"
    source.write_bytes(_pack_ubit12_little(volume))

    convert_bin_to_mip_tiff(
        source,
        destination,
        height=2,
        width=2,
        depth=4,
        baseline=2048,
        chunk_bytes=3,
    )

    expected = np.maximum(volume.astype(np.int32) - 2048, 0).max(axis=2).astype(np.uint16)
    actual = tifffile.imread(destination)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 2)
    assert actual.dtype == np.uint16
    with tifffile.TiffFile(destination) as tiff:
        page = tiff.pages[0]
        assert page.compression == tifffile.COMPRESSION.PACKBITS
        assert page.rowsperstrip == 2
        assert page.tags[274].value == tifffile.ORIENTATION.TOPLEFT


def test_windows_display_scaling_uses_explicit_shared_maximum(tmp_path: Path) -> None:
    volume = np.array([[[2048, 2050], [2048, 2052]]], dtype=np.uint16)
    source = tmp_path / "display.bin"
    destination = tmp_path / "display.tif"
    source.write_bytes(_pack_ubit12_little(volume))

    convert_bin_to_mip_tiff(
        source,
        destination,
        height=1,
        width=2,
        depth=2,
        display_max=4,
    )

    np.testing.assert_array_equal(
        tifffile.imread(destination), np.array([[32768, 65535]], dtype=np.uint16)
    )


def test_mip_directory_defaults_to_pa_channel(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    volume = np.arange(8, dtype=np.uint16).reshape(1, 2, 4)
    packed = _pack_ubit12_little(volume)
    (input_dir / "capture_PA1.bin").write_bytes(packed)
    (input_dir / "capture_PD1.bin").write_bytes(packed)

    outputs = convert_directory_to_mip(
        input_dir, output_dir, height=1, width=2, depth=4
    )

    assert [path.name for path in outputs] == ["capture_PA1_mip.tif"]
