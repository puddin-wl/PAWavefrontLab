#!/usr/bin/env python3
"""Prepare a real 600x600 point-scan NeuWS dataset from packed-12 PA BIN files."""

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


def _indexed_samples(files: list[Path], num_frames: int) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in files:
        match = SAMPLE_PATTERN.match(path.name)
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
        raise ValueError(f"s1–s{num_frames} 不完整；缺少 {missing}，额外编号 {extra}。")
    return indexed


def _single_phase_file(phase_dir: Path, index: int) -> Path:
    path = phase_dir / f"SLM_sim{index}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"缺少第 {index} 张计算相位：{path}")
    values = sio.loadmat(path)
    if "proj_sim" not in values:
        raise KeyError(f"{path} 中没有 proj_sim 变量。")
    phase = np.asarray(values["proj_sim"]).squeeze()
    if phase.shape != (600, 600) or not np.isfinite(phase).all():
        raise ValueError(f"{path} 的 proj_sim 必须是有限的 600×600 数组，实际为 {phase.shape}。")
    return path


def _convert_projection(
    source: Path,
    destination: Path,
    *,
    baseline: float,
    chunk_bytes: int,
) -> np.ndarray:
    convert_bin_to_mip_tiff(
        source,
        destination,
        height=600,
        width=600,
        depth=512,
        baseline=baseline,
        overwrite=False,
        chunk_bytes=chunk_bytes,
        show_progress=False,
    )
    image = np.asarray(tifffile.imread(destination))
    if image.shape != (600, 600):
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
    defocus: np.ndarray,
    measurements: np.ndarray,
) -> None:
    """Write contrast-stretched diagnostics without modifying reconstruction inputs."""
    selected = (
        (origin, "origin (ground truth)"),
        (defocus, "defocus"),
        (measurements[0], "s1"),
        (measurements[len(measurements) // 2], f"s{len(measurements) // 2 + 1}"),
        (measurements[-1], f"s{len(measurements)}"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (image, title) in zip(axes.flat[:5], selected):
        low, high = np.percentile(image, (1.0, 99.5))
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"{title}\nper-panel P1–P99.5 display")
        axis.axis("off")

    means = np.asarray(measurements).mean(axis=(1, 2))
    maxima = np.asarray(measurements).max(axis=(1, 2))
    frame_axis = np.arange(1, len(measurements) + 1)
    trend_axis = axes.flat[5]
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
        "defocus_quantiles": {
            str(value): float(np.percentile(defocus, value))
            for value in (0, 1, 50, 95, 99, 99.5, 100)
        },
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
    if not source_dir.is_dir():
        raise NotADirectoryError(f"原始数据目录不存在：{source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已有结果已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    origin_source = _single_match(files, ORIGIN_PATTERN, "origin")
    defocus_source = _single_match(files, DEFOCUS_PATTERN, "defocus")
    sample_sources = _indexed_samples(files, args.num_frames)
    phase_sources = {
        index: _single_phase_file(phase_dir, index)
        for index in range(1, args.num_frames + 1)
    }

    reference_dir = output_dir / "reference"
    mip_dir = output_dir / "measurement_mip_tiff"
    preview_dir = output_dir / "measurement_png"
    reference_dir.mkdir(parents=True, exist_ok=True)
    mip_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 正在处理 origin 和 defocus……", flush=True)
    origin = _convert_projection(
        origin_source,
        reference_dir / "origin_mip.tif",
        baseline=args.baseline,
        chunk_bytes=args.chunk_mb * 1024 * 1024,
    )
    defocus = _convert_projection(
        defocus_source,
        reference_dir / "defocus_mip.tif",
        baseline=args.baseline,
        chunk_bytes=args.chunk_mb * 1024 * 1024,
    )
    clear_object = _normalize_ground_truth(origin)
    reference_max = max(float(origin.max()), float(defocus.max()))
    if reference_max <= 0:
        raise ValueError("origin 和 defocus 在基线校正后都没有正信号。")
    np.save(reference_dir / "origin_projection.npy", origin)
    np.save(reference_dir / "defocus_projection.npy", defocus)
    np.save(reference_dir / "clear_object.npy", clear_object)
    write_unit_png(reference_dir / "origin_ground_truth.png", clear_object)
    write_unit_png(reference_dir / "origin_shared_scale.png", origin / reference_max)
    write_unit_png(reference_dir / "defocus_shared_scale.png", defocus / reference_max)

    print(f"[2/3] 正在处理 s1–s{args.num_frames}……", flush=True)
    measurements = np.lib.format.open_memmap(
        output_dir / "measurements.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.num_frames, 600, 600),
    )
    measurement_max = 0.0
    source_records = []
    for index in range(1, args.num_frames + 1):
        source = sample_sources[index]
        projection = _convert_projection(
            source,
            mip_dir / f"s{index}_mip.tif",
            baseline=args.baseline,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
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
    write_quality_control(output_dir, origin, defocus, measurements)

    sio.savemat(
        output_dir / "ground_truth.mat",
        {
            "clear_object": clear_object,
            "object_image": clear_object,
            "origin_projection": origin,
            "defocus_projection": defocus,
            "baseline_aberrated_measurement": np.clip(
                defocus / reference_max, 0.0, 1.0
            ).astype(np.float32),
        },
        do_compression=True,
    )

    ignored_df = sorted(
        path.name for path in files if re.match(r"^df\d+_", path.name, re.IGNORECASE)
    )
    manifest = {
        "schema_version": 1,
        "workflow": "real_point_scan_neuws",
        "scene_name": args.scene_name,
        "size": 600,
        "measurement_shape": [600, 600],
        "aperture_height": 600,
        "num_frames": args.num_frames,
        "phase_sign": -1,
        "source_directory": str(source_dir),
        "photoacoustic_preprocessing": {
            "input_encoding": "packed unsigned 12-bit little-endian",
            "source_shape": [600, 600, 512],
            "baseline": float(args.baseline),
            "negative_policy": "clip_to_zero_after_baseline_subtraction",
            "projection": "maximum_intensity_projection",
            "projection_axis": 2,
            "formula": "max(max(raw - baseline, 0), axis=depth)",
        },
        "ground_truth": {
            "source": origin_source.name,
            "normalized_variable": "ground_truth.mat:clear_object",
            "raw_projection": "reference/origin_projection.npy",
            "normalization": "individual min-max to [0,1] after projection",
        },
        "defocus_baseline": {
            "source": defocus_source.name,
            "raw_projection": "reference/defocus_projection.npy",
            "purpose": "deliberately defocused reference image",
        },
        "ignored_repeat_defocus_files": ignored_df,
        "measurement_max": measurement_max,
        "measurement_normalization": "one shared maximum across s1-s50, applied by BatchDataset",
        "measurement_generation": "Real 600x600 point-scan PA data with one fixed SLM pattern per frame.",
        "slm": {
            "hardware_grid": [1080, 1080],
            "model_grid": [600, 600],
            "model_phase_directory": str(phase_dir),
            "model_phase_files": "SLM_simN.mat:proj_sim",
            "same_frame_mapping": "sN measurement corresponds to SLM_simN phase",
        },
        "completed_steps": {"ground_truth": True, "patterns": True, "measurements": True},
        "outputs": {
            "ground_truth": "ground_truth.mat",
            "measurements": "measurements.npy",
            "measurement_mat": "SLM_rawN.mat:imsdata",
            "slm_phase_mat": "SLM_simN.mat:proj_sim",
            "measurement_previews": "measurement_png/modulated_measurement_NNNN.png",
            "measurement_mip_tiff": "measurement_mip_tiff/sN_mip.tif",
            "quality_control_figure": "quality_control.png",
            "quality_control_report": "quality_control.json",
        },
        "frames": source_records,
    }
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
    parser.add_argument("--baseline", type=float, default=2048.0)
    parser.add_argument("--chunk-mb", type=int, default=64)
    return parser


def main() -> None:
    output = prepare(build_parser().parse_args())
    print(f"处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
