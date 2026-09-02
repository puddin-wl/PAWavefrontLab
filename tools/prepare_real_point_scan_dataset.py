#!/usr/bin/env python3
"""Prepare a square real point-scan NeuWS dataset from packed-12 PA BIN files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_unit_png  # noqa: E402
from savetif.bin_to_tiff import convert_bin_to_mip_tiff  # noqa: E402


SAMPLE_PATTERN = re.compile(r"^s(\d+)_.*_PA1\.bin$", re.IGNORECASE)
ORIGIN_PATTERN = re.compile(r"^origin_.*_PA1\.bin$", re.IGNORECASE)
DEFOCUS_PATTERN = re.compile(r"^defocus_.*_PA1\.bin$", re.IGNORECASE)


def _single_match(files: list[Path], pattern: re.Pattern[str], label: str) -> Path:
    matches = [path for path in files if pattern.match(path.name)]
    if len(matches) != 1:
        raise ValueError(f"{label} 必须且只能匹配一个文件，实际匹配 {len(matches)} 个。")
    return matches[0]


def _sample_pattern(prefix: str) -> re.Pattern[str]:
    prefix = prefix.strip()
    if not prefix or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", prefix):
        raise ValueError(f"样本前缀无效：{prefix!r}")
    return re.compile(rf"^{re.escape(prefix)}(\d+)_.*_PA1\.bin$", re.IGNORECASE)


def _indexed_samples(
    files: list[Path], num_frames: int, pattern: re.Pattern[str] = SAMPLE_PATTERN
) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in files:
        match = pattern.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"s{index} 匹配到多个文件。")
        indexed[index] = path
    expected = set(range(1, num_frames + 1))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"样本 1–{num_frames} 不完整；缺少 {missing}，额外编号 {extra}。")
    return indexed


def _single_phase_file(
    phase_dir: Path, index: int, *, height: int, width: int
) -> Path:
    path = phase_dir / f"SLM_sim{index}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"缺少第 {index} 张计算相位：{path}")
    values = sio.loadmat(path)
    if "proj_sim" not in values:
        raise KeyError(f"{path} 中没有 proj_sim 变量。")
    phase = np.asarray(values["proj_sim"]).squeeze()
    if phase.shape != (height, width) or not np.isfinite(phase).all():
        raise ValueError(
            f"{path} 的 proj_sim 必须是有限的 {height}×{width} 数组，"
            f"实际为 {phase.shape}。"
        )
    return path


def _convert_projection(
    source: Path,
    destination: Path,
    *,
    baseline: float,
    chunk_bytes: int,
    height: int = 600,
    width: int = 600,
    depth: int = 512,
) -> np.ndarray:
    convert_bin_to_mip_tiff(
        source,
        destination,
        height=height,
        width=width,
        depth=depth,
        baseline=baseline,
        overwrite=False,
        chunk_bytes=chunk_bytes,
        show_progress=False,
    )
    image = np.asarray(tifffile.imread(destination))
    if image.shape != (height, width):
        raise ValueError(f"{destination} 投影尺寸错误：{image.shape}")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError(f"{destination} 包含无效数据。")
    return image.astype(np.float32)


def _normalize_ground_truth(image: np.ndarray) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        raise ValueError("origin 投影没有有效的强度范围，不能作为 ground truth。")
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def write_quality_control(
    output_dir: Path,
    origin: np.ndarray,
    defocus: np.ndarray | None,
    measurements: np.ndarray,
) -> None:
    """Write contrast-stretched diagnostics without modifying reconstruction inputs."""
    selected = [
        (origin, "origin (no added SLM phase)"),
    ]
    if defocus is not None:
        selected.append((defocus, "defocus"))
    selected.extend((
        (measurements[0], "s1"),
        (measurements[len(measurements) // 2], f"s{len(measurements) // 2 + 1}"),
        (measurements[-1], f"s{len(measurements)}"),
    ))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (image, title) in zip(axes.flat, selected):
        low, high = np.percentile(image, (1.0, 99.5))
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"{title}\nper-panel P1–P99.5 display")
        axis.axis("off")

    means = np.asarray(measurements).mean(axis=(1, 2))
    maxima = np.asarray(measurements).max(axis=(1, 2))
    frame_axis = np.arange(1, len(measurements) + 1)
    trend_axis = axes.flat[-1]
    trend_axis.plot(frame_axis, means, label="mean intensity")
    trend_axis.plot(frame_axis, maxima, label="maximum intensity", alpha=0.75)
    trend_axis.set_xlabel("SLM frame")
    trend_axis.set_ylabel("baseline-corrected value")
    trend_axis.set_title("Cross-frame intensity trend (unscaled)")
    trend_axis.grid(True, alpha=0.3)
    trend_axis.legend()
    fig.suptitle(
        "Real point-scan data quality control; contrast stretch is display-only",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "quality_control.png", dpi=150)
    plt.close(fig)

    report = {
        "display_only": True,
        "display_stretch": "each image independently mapped from percentile 1 to percentile 99.5",
        "training_arrays_modified": False,
        "origin_quantiles": {
            str(value): float(np.percentile(origin, value))
            for value in (0, 1, 50, 95, 99, 99.5, 100)
        },
        "measurement_frame_means": [float(value) for value in means],
        "measurement_frame_maxima": [float(value) for value in maxima],
    }
    if defocus is not None:
        report["defocus_quantiles"] = {
            str(value): float(np.percentile(defocus, value))
            for value in (0, 1, 50, 95, 99, 99.5, 100)
        }
    (output_dir / "quality_control.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_measurement_quality_control(
    output_dir: Path, measurements: np.ndarray
) -> None:
    """Write samples-only diagnostics without requiring reference acquisitions."""
    selected_indices = (0, len(measurements) // 2, len(measurements) - 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, index in zip(axes.flat[:3], selected_indices):
        image = np.asarray(measurements[index])
        low, high = np.percentile(image, (1.0, 99.5))
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"s{index + 1}\nper-panel P1–P99.5 display")
        axis.axis("off")
    means = np.asarray(measurements).mean(axis=(1, 2))
    maxima = np.asarray(measurements).max(axis=(1, 2))
    frame_axis = np.arange(1, len(measurements) + 1)
    axes.flat[3].plot(frame_axis, means, label="mean intensity")
    axes.flat[3].plot(frame_axis, maxima, label="maximum intensity", alpha=0.75)
    axes.flat[3].set_xlabel("SLM frame")
    axes.flat[3].set_ylabel("baseline-corrected value")
    axes.flat[3].set_title("Cross-frame intensity trend (unscaled)")
    axes.flat[3].grid(True, alpha=0.3)
    axes.flat[3].legend()
    fig.suptitle("Samples-only real data quality control", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_control.png", dpi=150)
    plt.close(fig)
    report = {
        "samples_only": True,
        "display_only": True,
        "training_arrays_modified": False,
        "measurement_frame_means": [float(value) for value in means],
        "measurement_frame_maxima": [float(value) for value in maxima],
    }
    (output_dir / "quality_control.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase_dir = Path(args.phase_dir).expanduser().resolve()
    height = int(getattr(args, "height", 600))
    width = int(getattr(args, "width", 600))
    depth = int(getattr(args, "depth", 512))
    if height <= 0 or width <= 0 or depth <= 0:
        raise ValueError("height、width、depth 必须为正整数。")
    if height != width or width % 2:
        raise ValueError(
            f"NeuWS 当前要求正偶数方形网格，实际为 {height}×{width}。"
        )
    if not source_dir.is_dir():
        raise NotADirectoryError(f"原始数据目录不存在：{source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已有结果已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    samples_only = bool(getattr(args, "samples_only", False))
    origin_only = bool(getattr(args, "origin_only", False))
    sample_prefix = str(getattr(args, "sample_prefix", "s"))
    sample_pattern = _sample_pattern(sample_prefix)
    if samples_only and origin_only:
        raise ValueError("--samples-only 与 --origin-only 不能同时使用。")
    requested_origin = getattr(args, "origin_source", None)
    if samples_only:
        origin_source = None
    elif requested_origin:
        origin_source = Path(requested_origin).expanduser().resolve()
        if not origin_source.is_file():
            raise FileNotFoundError(f"指定的 origin 文件不存在：{origin_source}")
        if not origin_source.name.lower().endswith("_pa1.bin"):
            raise ValueError(f"指定的 origin 文件必须是 PA1 BIN：{origin_source.name}")
    else:
        origin_source = _single_match(files, ORIGIN_PATTERN, "origin")
    defocus_source = (
        None
        if samples_only or origin_only
        else _single_match(files, DEFOCUS_PATTERN, "defocus")
    )
    sample_sources = _indexed_samples(files, args.num_frames, sample_pattern)
    phase_sources = {
        index: _single_phase_file(
            phase_dir, index, height=height, width=width
        )
        for index in range(1, args.num_frames + 1)
    }

    reference_dir = output_dir / "reference"
    mip_dir = output_dir / "measurement_mip_tiff"
    preview_dir = output_dir / "measurement_png"
    if not samples_only:
        reference_dir.mkdir(parents=True, exist_ok=True)
    mip_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    origin = defocus = clear_object = None
    reference_max = None
    if samples_only:
        print(
            f"[1/3] samples-only：忽略所有非 {sample_prefix}1–{sample_prefix}{args.num_frames} 文件。",
            flush=True,
        )
    else:
        label = "origin（无额外 SLM 相位）" if origin_only else "origin 和 defocus"
        print(f"[1/3] 正在处理 {label}……", flush=True)
        origin = _convert_projection(
            origin_source,
            reference_dir / "origin_mip.tif",
            baseline=args.baseline,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            height=height,
            width=width,
            depth=depth,
        )
        if defocus_source is not None:
            defocus = _convert_projection(
                defocus_source,
                reference_dir / "defocus_mip.tif",
                baseline=args.baseline,
                chunk_bytes=args.chunk_mb * 1024 * 1024,
                height=height,
                width=width,
                depth=depth,
            )
        clear_object = _normalize_ground_truth(origin)
        reference_max = max(
            float(origin.max()),
            float(defocus.max()) if defocus is not None else float(origin.max()),
        )
        if reference_max <= 0:
            raise ValueError("参考采集在基线校正后没有正信号。")
        np.save(reference_dir / "origin_projection.npy", origin)
        np.save(reference_dir / "clear_object.npy", clear_object)
        write_unit_png(reference_dir / "origin_ground_truth.png", clear_object)
        write_unit_png(reference_dir / "origin_shared_scale.png", origin / reference_max)
        if defocus is not None:
            np.save(reference_dir / "defocus_projection.npy", defocus)
            write_unit_png(reference_dir / "defocus_shared_scale.png", defocus / reference_max)

    print(
        f"[2/3] 正在处理 {sample_prefix}1–{sample_prefix}{args.num_frames}……",
        flush=True,
    )
    measurements = np.lib.format.open_memmap(
        output_dir / "measurements.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.num_frames, height, width),
    )
    measurement_max = 0.0
    source_records = []
    for index in range(1, args.num_frames + 1):
        source = sample_sources[index]
        projection = _convert_projection(
            source,
            mip_dir / f"{sample_prefix}{index}_mip.tif",
            baseline=args.baseline,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            height=height,
            width=width,
            depth=depth,
        )
        measurements[index - 1] = projection
        measurement_max = max(measurement_max, float(projection.max()))
        shutil.copy2(phase_sources[index], output_dir / f"SLM_sim{index}.mat")
        sio.savemat(
            output_dir / f"SLM_raw{index}.mat",
            {"imsdata": projection},
            do_compression=True,
        )
        source_records.append(
            {
                "frame": index,
                "measurement_source": source.name,
                "phase_source": str(phase_sources[index]),
            }
        )
        print(f"  完成 {index:02d}/{args.num_frames}: {source.name}", flush=True)
    measurements.flush()
    if not np.isfinite(measurement_max) or measurement_max <= 0:
        raise ValueError("s1–s50 在基线校正后没有正信号，无法用于训练。")

    for index in range(1, args.num_frames + 1):
        write_unit_png(
            preview_dir / f"modulated_measurement_{index:04d}.png",
            np.asarray(measurements[index - 1]) / measurement_max,
        )
    if samples_only:
        write_measurement_quality_control(output_dir, measurements)
    else:
        write_quality_control(output_dir, origin, defocus, measurements)
        ground_truth_values = {
            "clear_object": clear_object,
            "object_image": clear_object,
            "origin_projection": origin,
        }
        if defocus is not None:
            ground_truth_values.update(
                {
                    "defocus_projection": defocus,
                    "baseline_aberrated_measurement": np.clip(
                        defocus / reference_max, 0.0, 1.0
                    ).astype(np.float32),
                }
            )
        sio.savemat(
            output_dir / "ground_truth.mat",
            ground_truth_values,
            do_compression=True,
        )

    ignored_df = sorted(
        path.name for path in files if re.match(r"^df\d+_", path.name, re.IGNORECASE)
    )
    manifest = {
        "schema_version": 1,
        "workflow": (
            "real_point_scan_neuws_samples_only"
            if samples_only
            else (
                "real_point_scan_neuws_origin_only"
                if origin_only
                else "real_point_scan_neuws"
            )
        ),
        "scene_name": args.scene_name,
        "size": width,
        "measurement_shape": [height, width],
        "aperture_height": height,
        "num_frames": args.num_frames,
        "phase_sign": -1,
        "source_directory": str(source_dir),
        "sample_prefix": sample_prefix,
        "photoacoustic_preprocessing": {
            "input_encoding": "packed unsigned 12-bit little-endian",
            "source_shape": [height, width, depth],
            "baseline": float(args.baseline),
            "negative_policy": "clip_to_zero_after_baseline_subtraction",
            "projection": "maximum_intensity_projection",
            "projection_axis": 2,
            "formula": "max(max(raw - baseline, 0), axis=depth)",
        },
        "samples_only": samples_only,
        "origin_only": origin_only,
        "ignored_non_sample_files": sorted(
            path.name
            for path in files
            if not sample_pattern.match(path.name)
            and path != origin_source
            and path != defocus_source
        ),
        "ignored_repeat_defocus_files": ignored_df,
        "measurement_max": measurement_max,
        "measurement_normalization": (
            f"one shared maximum across {sample_prefix}1-{sample_prefix}{args.num_frames}, "
            "applied by BatchDataset"
        ),
        "measurement_generation": (
            f"Real {height}x{width} point-scan PA data with one fixed SLM pattern per frame."
        ),
        "slm": {
            "hardware_grid": [1080, 1080],
            "model_grid": [height, width],
            "model_phase_directory": str(phase_dir),
            "model_phase_files": "SLM_simN.mat:proj_sim",
            "same_frame_mapping": (
                f"{sample_prefix}N measurement corresponds to SLM_simN phase"
            ),
        },
        "completed_steps": {
            "ground_truth": not samples_only,
            "patterns": True,
            "measurements": True,
        },
        "outputs": {
            "measurements": "measurements.npy",
            "measurement_mat": "SLM_rawN.mat:imsdata",
            "slm_phase_mat": "SLM_simN.mat:proj_sim",
            "measurement_previews": "measurement_png/modulated_measurement_NNNN.png",
            "measurement_mip_tiff": (
                f"measurement_mip_tiff/{sample_prefix}N_mip.tif"
            ),
            "quality_control_figure": "quality_control.png",
            "quality_control_report": "quality_control.json",
        },
        "frames": source_records,
    }
    if not samples_only:
        manifest["ground_truth"] = {
            "source": origin_source.name,
            "normalized_variable": "ground_truth.mat:clear_object",
            "raw_projection": "reference/origin_projection.npy",
            "normalization": "individual min-max to [0,1] after projection",
        }
        if defocus_source is not None:
            manifest["defocus_baseline"] = {
                "source": defocus_source.name,
                "raw_projection": "reference/defocus_projection.npy",
                "purpose": "deliberately defocused reference image",
            }
        manifest["outputs"]["ground_truth"] = "ground_truth.mat"
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[3/3] 数据集及共享强度预览已写入完成。", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--scene-name", default="real_2026_08_12_600")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument(
        "--sample-prefix",
        default="s",
        help="Raw measurement prefix before the frame number, for example s or d.",
    )
    parser.add_argument(
        "--origin-source",
        help="Optional origin BIN path when the reference is stored outside --source-dir.",
    )
    parser.add_argument("--baseline", type=float, default=2048.0)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Only process S1–S50 and ignore origin, defocus and every other file.",
    )
    parser.add_argument(
        "--origin-only",
        action="store_true",
        help="Use origin as the no-added-SLM-phase reference without requiring defocus.",
    )
    return parser


def main() -> None:
    output = prepare(build_parser().parse_args())
    print(f"处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
