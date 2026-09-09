#!/usr/bin/env python3
"""将 MATLAB ``fread(..., 'ubit12')`` 格式的 BIN 转为多页 TIFF。"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin
from tqdm import tqdm

from preprocessing.packed12 import (
    BITS_PER_SAMPLE,
    decode_packed12,
    packed12_byte_count,
)

DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1000
DEFAULT_DEPTH = 512


def _expected_file_size(sample_count: int) -> int:
    return packed12_byte_count(sample_count)


def _validate_dimensions(height: int, width: int, depth: int) -> tuple[int, int, int]:
    dimensions = (height, width, depth)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
        raise ValueError(f"height、width、depth 必须是正整数，实际为 {dimensions}。")
    return dimensions


def _decode_into_volume(
    source_path: Path,
    volume: np.memmap,
    *,
    chunk_bytes: int,
    show_progress: bool,
) -> None:
    """按 MATLAB 原生小端 ``ubit12`` 规则解码到 C-order 三维数组。"""
    sample_count = int(volume.size)
    pair_count, has_last_sample = divmod(sample_count, 2)
    groups_per_chunk = max(1, chunk_bytes // 3)
    flat = volume.reshape(-1)
    write_start = 0

    progress = tqdm(
        total=sample_count,
        unit="value",
        desc=f"解码 {source_path.name}",
        disable=not show_progress,
    )
    try:
        with source_path.open("rb") as stream:
            remaining_pairs = pair_count
            while remaining_pairs:
                group_count = min(remaining_pairs, groups_per_chunk)
                packed = np.fromfile(stream, dtype=np.uint8, count=group_count * 3)
                if packed.size != group_count * 3:
                    raise EOFError(f"读取 {source_path} 时意外到达文件末尾。")
                decoded = decode_packed12(packed)
                flat[write_start : write_start + decoded.size] = decoded
                write_start += decoded.size
                remaining_pairs -= group_count
                progress.update(decoded.size)

            if has_last_sample:
                tail = np.frombuffer(stream.read(2), dtype=np.uint8)
                if tail.size != 2:
                    raise EOFError(f"读取 {source_path} 的最后一个 12 位值时数据不足。")
                flat[write_start] = decode_packed12(tail, sample_count=1)[0]
                progress.update(1)
    finally:
        progress.close()
    volume.flush()


def _write_tiff(
    volume: np.memmap,
    destination_path: Path,
    *,
    show_progress: bool,
) -> None:
    """按 MATLAB ``sample(:, :, depth)`` 的页顺序写为 float32 TIFF。"""
    height, width, depth = volume.shape
    output_bytes = height * width * depth * np.dtype(np.float32).itemsize
    use_bigtiff = output_bytes > 2**32 - 1

    with tifffile.TiffWriter(destination_path, bigtiff=use_bigtiff) as writer:
        pages: Iterable[int] = tqdm(
            range(depth),
            unit="page",
            desc=f"写入 {destination_path.name}",
            disable=not show_progress,
        )
        for index in pages:
            page = np.asarray(volume[:, :, index], dtype=np.float32)
            writer.write(
                page,
                photometric="minisblack",
                compression=None,
                rowsperstrip=min(512, height),
                metadata=None,
            )


def _decode_rows_and_project(
    source_path: Path,
    *,
    height: int,
    width: int,
    depth: int,
    baseline: float,
    chunk_bytes: int,
    show_progress: bool,
) -> np.ndarray:
    """分行解码 packed-12 数据，并直接沿深度维计算最大值投影。"""
    samples_per_row = width * depth
    if samples_per_row % 2:
        raise ValueError(
            "直接投影要求 width×depth 为偶数，以保持每行位于完整的 3 字节边界。"
        )
    bytes_per_row = samples_per_row * BITS_PER_SAMPLE // 8
    rows_per_chunk = max(1, chunk_bytes // bytes_per_row)
    projection = np.empty((height, width), dtype=np.uint16)

    progress = tqdm(
        total=height,
        unit="row",
        desc=f"投影 {source_path.name}",
        disable=not show_progress,
    )
    try:
        with source_path.open("rb") as stream:
            for row_start in range(0, height, rows_per_chunk):
                row_stop = min(row_start + rows_per_chunk, height)
                row_count = row_stop - row_start
                group_count = row_count * samples_per_row // 2
                packed = np.fromfile(stream, dtype=np.uint8, count=group_count * 3)
                if packed.size != group_count * 3:
                    raise EOFError(f"读取 {source_path} 时意外到达文件末尾。")
                decoded = decode_packed12(packed)
                raw_max = decoded.reshape(row_count, width, depth).max(axis=2)
                corrected = raw_max.astype(np.float32) - np.float32(baseline)
                corrected = np.clip(np.rint(corrected), 0, np.iinfo(np.uint16).max)
                projection[row_start:row_stop] = corrected.astype(np.uint16)
                progress.update(row_count)
    finally:
        progress.close()
    return projection


def _write_windows_uint16_tiff(
    destination_path: Path,
    image: np.ndarray,
    *,
    display_max: float | None,
) -> None:
    """按 Windows 查看器兼容的经典 TIFF 标签写入单层 16 位灰度图。"""
    output = np.asarray(image, dtype=np.uint16)
    if display_max is not None:
        if not np.isfinite(display_max) or display_max <= 0:
            raise ValueError("display_max 必须是正的有限数值。")
        output = np.round(
            np.clip(output.astype(np.float64) / display_max, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)

    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[274] = 1  # Orientation: top-left
    tags[278] = min(15, output.shape[0])  # 与参考 1.tif 一致的小条带布局
    tags[282] = 72.0
    tags[283] = 72.0
    tags[296] = 2  # ResolutionUnit: inch
    Image.fromarray(output).save(
        destination_path,
        format="TIFF",
        compression="packbits",
        tiffinfo=tags,
    )


def convert_bin_to_mip_tiff(
    source: str | Path,
    destination: str | Path,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    depth: int = DEFAULT_DEPTH,
    baseline: float = 2048.0,
    display_max: float | None = None,
    overwrite: bool = False,
    chunk_bytes: int = 64 * 1024 * 1024,
    show_progress: bool = False,
) -> Path:
    """将 packed-12 BIN 直接转换成基线校正后的单层 uint16 最大值投影 TIFF。"""
    height, width, depth = _validate_dimensions(height, width, depth)
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 3:
        raise ValueError("chunk_bytes 必须是不小于 3 的整数。")
    if not np.isfinite(baseline):
        raise ValueError("baseline 必须是有限数值。")

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{source_path}")
    if source_path == destination_path:
        raise ValueError("输入 BIN 和输出 TIFF 不能是同一个文件。")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请使用 --overwrite：{destination_path}")

    sample_count = height * width * depth
    expected_bytes = _expected_file_size(sample_count)
    actual_bytes = source_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"BIN 文件大小与 {height}×{width}×{depth} 个 12 位值不匹配："
            f"应为 {expected_bytes} 字节，实际为 {actual_bytes} 字节。"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_fd, tiff_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp.tif", dir=destination_path.parent
    )
    os.close(tiff_fd)
    temporary_tiff_path = Path(tiff_name)
    try:
        projection = _decode_rows_and_project(
            source_path,
            height=height,
            width=width,
            depth=depth,
            baseline=float(baseline),
            chunk_bytes=chunk_bytes,
            show_progress=show_progress,
        )
        _write_windows_uint16_tiff(
            temporary_tiff_path,
            projection,
            display_max=display_max,
        )
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"写入期间输出文件已出现，未覆盖：{destination_path}")
        os.replace(temporary_tiff_path, destination_path)
    finally:
        temporary_tiff_path.unlink(missing_ok=True)
    return destination_path


def convert_bin_to_tiff(
    source: str | Path,
    destination: str | Path,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    depth: int = DEFAULT_DEPTH,
    overwrite: bool = False,
    chunk_bytes: int = 64 * 1024 * 1024,
    temporary_directory: str | Path | None = None,
    show_progress: bool = False,
) -> Path:
    """转换单个 packed-12 BIN，返回生成的 TIFF 绝对路径。

    BIN 的线性排列与原 MATLAB 程序一致：``depth`` 最快，其次是
    ``width``，最后是 ``height``。TIFF 中每个深度位置保存为一页。
    """
    height, width, depth = _validate_dimensions(height, width, depth)
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 3:
        raise ValueError("chunk_bytes 必须是不小于 3 的整数。")

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{source_path}")
    if source_path == destination_path:
        raise ValueError("输入 BIN 和输出 TIFF 不能是同一个文件。")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请使用 --overwrite：{destination_path}")

    sample_count = height * width * depth
    expected_bytes = _expected_file_size(sample_count)
    actual_bytes = source_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"BIN 文件大小与 {height}×{width}×{depth} 个 12 位值不匹配："
            f"应为 {expected_bytes} 字节，实际为 {actual_bytes} 字节。"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = (
        Path(temporary_directory).expanduser().resolve()
        if temporary_directory is not None
        else destination_path.parent
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)

    volume_fd, volume_name = tempfile.mkstemp(
        prefix=f".{source_path.stem}.", suffix=".uint16.tmp", dir=scratch_parent
    )
    os.close(volume_fd)
    tiff_fd, tiff_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp.tif", dir=destination_path.parent
    )
    os.close(tiff_fd)
    volume_path = Path(volume_name)
    temporary_tiff_path = Path(tiff_name)

    volume: np.memmap | None = None
    try:
        volume = np.memmap(
            volume_path,
            mode="w+",
            dtype=np.uint16,
            shape=(height, width, depth),
            order="C",
        )
        _decode_into_volume(
            source_path,
            volume,
            chunk_bytes=chunk_bytes,
            show_progress=show_progress,
        )
        _write_tiff(volume, temporary_tiff_path, show_progress=show_progress)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"写入期间输出文件已出现，未覆盖：{destination_path}")
        os.replace(temporary_tiff_path, destination_path)
    finally:
        if volume is not None:
            del volume
        volume_path.unlink(missing_ok=True)
        temporary_tiff_path.unlink(missing_ok=True)

    return destination_path


def convert_directory(
    input_directory: str | Path,
    output_directory: str | Path,
    **options,
) -> list[Path]:
    """转换目录中的全部 ``.bin`` 文件，命名方式与 MATLAB 脚本一致。"""
    input_path = Path(input_directory).expanduser().resolve()
    output_path = Path(output_directory).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{input_path}")
    sources = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".bin"
    )
    if not sources:
        raise FileNotFoundError(f"输入目录中没有 .bin 文件：{input_path}")

    results = []
    for source_path in sources:
        # MATLAB 中 Files(i).name 已包含 .bin，因此输出为 name.bin.tif。
        destination_path = output_path / f"{source_path.name}.tif"
        results.append(convert_bin_to_tiff(source_path, destination_path, **options))
    return results


def convert_directory_to_mip(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    filename_suffix: str = "_PA1.bin",
    **options,
) -> list[Path]:
    """批量将目录中匹配后缀的 BIN 转成单层最大值投影 TIFF。"""
    input_path = Path(input_directory).expanduser().resolve()
    output_path = Path(output_directory).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{input_path}")
    suffix_lower = filename_suffix.lower()
    sources = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.name.lower().endswith(suffix_lower)
    )
    if not sources:
        raise FileNotFoundError(
            f"输入目录中没有名称以 {filename_suffix!r} 结尾的文件：{input_path}"
        )

    results = []
    for source_path in sources:
        destination_path = output_path / f"{source_path.stem}_mip.tif"
        results.append(convert_bin_to_mip_tiff(source_path, destination_path, **options))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 MATLAB ubit12（小端位序）BIN 转换为 float32 多页 TIFF。"
    )
    parser.add_argument("input", help="单个 .bin 文件或包含 .bin 文件的目录")
    parser.add_argument("output", help="单文件模式下的 .tif 路径，或批量模式下的输出目录")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="图像高度，默认 1000")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="图像宽度，默认 1000")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="TIFF 页数，默认 512")
    parser.add_argument(
        "--chunk-mb", type=int, default=64, help="每次解码的数据量（MiB），默认 64"
    )
    parser.add_argument("--temp-dir", help="约 1 GiB 临时体数据的存放目录")
    parser.add_argument(
        "--mip",
        action="store_true",
        help="直接输出减基线、截零后的单层最大值投影，不生成三维 TIFF",
    )
    parser.add_argument("--baseline", type=float, default=2048.0, help="光声零点，默认 2048")
    parser.add_argument(
        "--display-max",
        type=float,
        help="将该强度线性映射为 65535；同组数据应使用同一个值",
    )
    parser.add_argument(
        "--filename-suffix",
        default="_PA1.bin",
        help="--mip 批量模式匹配的文件名后缀，默认 _PA1.bin",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 TIFF")
    parser.add_argument("--quiet", action="store_true", help="不显示进度条")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    common_options = {
        "height": args.height,
        "width": args.width,
        "depth": args.depth,
        "overwrite": args.overwrite,
        "chunk_bytes": args.chunk_mb * 1024 * 1024,
        "show_progress": not args.quiet,
    }

    if args.mip:
        common_options["baseline"] = args.baseline
        common_options["display_max"] = args.display_max
        if input_path.is_dir():
            outputs = convert_directory_to_mip(
                input_path,
                args.output,
                filename_suffix=args.filename_suffix,
                **common_options,
            )
        else:
            outputs = [convert_bin_to_mip_tiff(input_path, args.output, **common_options)]
    else:
        common_options["temporary_directory"] = args.temp_dir
        if input_path.is_dir():
            outputs = convert_directory(input_path, args.output, **common_options)
        else:
            outputs = [convert_bin_to_tiff(input_path, args.output, **common_options)]
    for output in outputs:
        print(f"已保存：{output}")


if __name__ == "__main__":
    main()
