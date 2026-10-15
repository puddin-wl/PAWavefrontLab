#!/usr/bin/env python3
"""Batch-convert one PA experiment's packed-12 BIN files to 3-D TIFF stacks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.bin_to_tiff import convert_bin_to_tiff  # noqa: E402


SHAPE_PATTERN = re.compile(
    r"_(?P<height>\d+)_(?P<width>\d+)_(?P<depth>\d+)_PA1\.bin$",
    re.IGNORECASE,
)


def _infer_shape(path: Path) -> tuple[int, int, int]:
    match = SHAPE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"无法从文件名推断 height/width/depth：{path.name}")
    return tuple(
        int(match.group(name)) for name in ("height", "width", "depth")
    )


def _experiment_date(source_dir: Path, requested: str | None) -> str:
    value = requested or source_dir.name
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"实验日期必须为 YYYY-MM-DD；当前为 {value!r}。"
        ) from exc


def _source_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name.lower().startswith("origin_"):
        return (0, 0, path.name.lower())
    match = re.match(r"^[sS](\d+)_", path.name)
    if match:
        return (1, int(match.group(1)), path.name.lower())
    return (2, 0, path.name.lower())


def _valid_tiff(path: Path, shape: tuple[int, int, int]) -> bool:
    if not path.is_file():
        return False
    height, width, depth = shape
    try:
        with tifffile.TiffFile(path) as tif:
            return (
                len(tif.pages) == depth
                and all(page.shape == (height, width) for page in tif.pages)
                and all(page.dtype.name == "float32" for page in tif.pages)
            )
    except (OSError, ValueError, tifffile.TiffFileError):
        return False


def convert_experiment(
    source_dir: Path,
    output_root: Path,
    *,
    experiment_date: str | None = None,
    pattern: str = "*_PA1.bin",
    overwrite: bool = False,
    chunk_mb: int = 64,
    temporary_directory: Path | None = None,
    show_progress: bool = True,
    dry_run: bool = False,
) -> Path:
    source_dir = source_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"BIN 输入目录不存在：{source_dir}")
    date = _experiment_date(source_dir, experiment_date)
    output_dir = output_root / date
    sources = sorted(source_dir.glob(pattern), key=_source_sort_key)
    if not sources:
        raise FileNotFoundError(f"没有找到匹配 {pattern!r} 的 BIN：{source_dir}")

    shapes = {path: _infer_shape(path) for path in sources}
    unique_shapes = sorted(set(shapes.values()))
    if len(unique_shapes) != 1:
        raise ValueError(f"同批 BIN 的尺寸不一致：{unique_shapes}")
    shape = unique_shapes[0]

    print(f"输入：{source_dir}", flush=True)
    print(f"输出：{output_dir}", flush=True)
    print(f"数量：{len(sources)}；体数据尺寸：{shape}", flush=True)
    if dry_run:
        for source in sources:
            print(f"  {source.name} -> {source.name}.tif")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, source in enumerate(sources, start=1):
        destination = output_dir / f"{source.name}.tif"
        if not overwrite and _valid_tiff(destination, shape):
            print(f"[{position}/{len(sources)}] 已完整，跳过：{destination.name}", flush=True)
            status = "skipped_valid"
        else:
            print(f"[{position}/{len(sources)}] 转换：{source.name}", flush=True)
            convert_bin_to_tiff(
                source,
                destination,
                height=shape[0],
                width=shape[1],
                depth=shape[2],
                overwrite=overwrite or destination.exists(),
                chunk_bytes=chunk_mb * 1024 * 1024,
                temporary_directory=temporary_directory,
                show_progress=show_progress,
            )
            if not _valid_tiff(destination, shape):
                raise RuntimeError(f"转换后 TIFF 完整性检查失败：{destination}")
            status = "converted"
        records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "status": status,
                "bytes": destination.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment_date": date,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "shape_height_width_depth": list(shape),
        "encoding": "packed unsigned 12-bit little-endian",
        "tiff": {
            "pages": shape[2],
            "page_shape": [shape[0], shape[1]],
            "dtype": "float32",
            "baseline_subtracted": False,
            "projection": None,
        },
        "file_count": len(records),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"全部完成：{output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--experiment-date", help="YYYY-MM-DD；默认使用输入目录名。")
    parser.add_argument("--pattern", default="*_PA1.bin")
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_experiment(
        args.source_dir,
        args.output_root,
        experiment_date=args.experiment_date,
        pattern=args.pattern,
        overwrite=args.overwrite,
        chunk_mb=args.chunk_mb,
        temporary_directory=args.temp_dir,
        show_progress=not args.quiet,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
