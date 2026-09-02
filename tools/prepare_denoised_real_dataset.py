#!/usr/bin/env python3
"""Rebuild a real NeuWS dataset with noise-aware excess-RMS PA projections."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_unit_png  # noqa: E402
from preprocessing.pa_denoising import (  # noqa: E402
    load_packed12_excess_rms_projection,
)


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        raise ValueError("图像没有有效强度范围。")
    return np.asarray((values - minimum) / (maximum - minimum), dtype=np.float32)


def _project(path: Path, args: argparse.Namespace) -> np.ndarray:
    return load_packed12_excess_rms_projection(
        path,
        height=600,
        width=600,
        depth=512,
        signal_window=(args.signal_start, args.signal_stop),
        noise_windows=(
            (args.noise_left_start, args.noise_left_stop),
            (args.noise_right_start, args.noise_right_stop),
        ),
        spatial_sigma=args.spatial_sigma,
        chunk_rows=args.chunk_rows,
    )


def _write_quality_control(
    output_dir: Path,
    template_measurements: np.ndarray,
    measurements: np.ndarray,
    template_origin: np.ndarray,
    origin: np.ndarray,
) -> dict:
    indices = (0, len(measurements) // 2, len(measurements) - 1)
    old_images = [template_origin, *(template_measurements[index] for index in indices)]
    new_images = [origin, *(measurements[index] for index in indices)]
    labels = ["origin", *(f"d{index + 1}" for index in indices)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row, (images, prefix) in enumerate(
        ((old_images, "old all-depth MIP"), (new_images, "denoised excess-RMS"))
    ):
        for column, (image, label) in enumerate(zip(images, labels)):
            low, high = np.percentile(image, (1.0, 99.5))
            axes[row, column].imshow(image, cmap="gray", vmin=low, vmax=high)
            axes[row, column].set_title(f"{label}: {prefix}\nP1–P99.5 display")
            axes[row, column].axis("off")
    fig.suptitle("2026-08-20 PA preprocessing comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_control.png", dpi=150)
    plt.close(fig)

    old_means = np.asarray(template_measurements).mean(axis=(1, 2))
    new_means = np.asarray(measurements).mean(axis=(1, 2))
    old_maxima = np.asarray(template_measurements).max(axis=(1, 2))
    new_maxima = np.asarray(measurements).max(axis=(1, 2))
    frames = np.arange(1, len(measurements) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frames, old_means, label="old MIP")
    axes[0].plot(frames, new_means, label="denoised")
    axes[0].set_title("Frame mean")
    axes[1].plot(frames, old_maxima, label="old MIP")
    axes[1].plot(frames, new_maxima, label="denoised")
    axes[1].set_title("Frame maximum")
    for axis in axes:
        axis.set_xlabel("frame")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_trends.png", dpi=150)
    plt.close(fig)

    def summary(values: np.ndarray) -> dict:
        array = np.asarray(values)
        frame_means = array.mean(axis=(1, 2))
        return {
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "mean": float(array.mean()),
            "frame_mean_cv": float(frame_means.std() / frame_means.mean()),
            "quantiles": {
                str(value): float(np.percentile(array, value))
                for value in (0, 1, 50, 95, 99, 99.5, 100)
            },
            "frame_means": [float(value) for value in frame_means],
            "frame_maxima": [float(value) for value in array.max(axis=(1, 2))],
        }

    report = {
        "old_all_depth_mip": summary(template_measurements),
        "denoised_excess_rms": summary(measurements),
        "origin_old_quantiles": {
            str(value): float(np.percentile(template_origin, value))
            for value in (0, 1, 50, 95, 99, 99.5, 100)
        },
        "origin_denoised_quantiles": {
            str(value): float(np.percentile(origin, value))
            for value in (0, 1, 50, 95, 99, 99.5, 100)
        },
    }
    (output_dir / "quality_control.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def prepare(args: argparse.Namespace) -> Path:
    template_dir = Path(args.template_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    origin_source = Path(args.origin_source).expanduser().resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"模板数据集不存在：{template_dir}")
    if not origin_source.is_file():
        raise FileNotFoundError(f"origin BIN 不存在：{origin_source}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    template_manifest = json.loads(
        (template_dir / "manifest.json").read_text(encoding="utf-8")
    )
    num_frames = int(template_manifest["num_frames"])
    frames = template_manifest["frames"]
    if len(frames) != num_frames:
        raise ValueError("模板 manifest 的帧记录数量不一致。")
    source_dir = Path(template_manifest["source_directory"]).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"原始测量目录不存在：{source_dir}")

    reference_dir = output_dir / "reference"
    projection_dir = output_dir / "measurement_projection_tiff"
    preview_dir = output_dir / "measurement_png"
    for directory in (reference_dir, projection_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/3] 正在处理去噪 origin……", flush=True)
    origin = _project(origin_source, args)
    tifffile.imwrite(
        reference_dir / "origin_excess_rms.tif", origin, photometric="minisblack"
    )
    clear_object = _normalize(origin)
    np.save(reference_dir / "origin_projection.npy", origin)
    np.save(reference_dir / "clear_object.npy", clear_object)
    write_unit_png(reference_dir / "origin_ground_truth.png", clear_object)
    sio.savemat(
        output_dir / "ground_truth.mat",
        {
            "clear_object": clear_object,
            "object_image": clear_object,
            "origin_projection": origin,
        },
        do_compression=True,
    )

    measurements = np.lib.format.open_memmap(
        output_dir / "measurements.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_frames, 600, 600),
    )
    print(f"[2/3] 正在处理 {num_frames} 份 D 测量……", flush=True)
    for position, record in enumerate(frames, start=1):
        source = source_dir / record["measurement_source"]
        if not source.is_file():
            raise FileNotFoundError(f"缺少第 {position} 帧原始 BIN：{source}")
        projection = _project(source, args)
        measurements[position - 1] = projection
        tifffile.imwrite(
            projection_dir / f"d{position}_excess_rms.tif",
            projection,
            photometric="minisblack",
        )
        sio.savemat(
            output_dir / f"SLM_raw{position}.mat",
            {"imsdata": projection},
            do_compression=True,
        )
        phase_source = template_dir / f"SLM_sim{position}.mat"
        if not phase_source.is_file():
            raise FileNotFoundError(f"模板缺少第 {position} 帧相位：{phase_source}")
        shutil.copy2(phase_source, output_dir / phase_source.name)
        if position == 1 or position % 5 == 0 or position == num_frames:
            print(f"  已完成 {position}/{num_frames}", flush=True)
    measurements.flush()

    measurement_max = float(np.max(measurements))
    if not np.isfinite(measurement_max) or measurement_max <= 0:
        raise ValueError("去噪测量没有正的有限最大值。")
    for position in range(1, num_frames + 1):
        write_unit_png(
            preview_dir / f"modulated_measurement_{position:04d}.png",
            np.asarray(measurements[position - 1]) / measurement_max,
        )

    template_measurements = np.load(
        template_dir / "measurements.npy", mmap_mode="r", allow_pickle=False
    )
    template_origin = np.load(
        template_dir / "reference" / "origin_projection.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    _write_quality_control(
        output_dir, template_measurements, measurements, template_origin, origin
    )

    manifest = copy.deepcopy(template_manifest)
    manifest.update(
        {
            "workflow": "real_point_scan_neuws_origin_only_denoised_excess_rms",
            "scene_name": args.scene_name,
            "template_dataset": str(template_dir),
            "measurement_max": measurement_max,
            "measurement_normalization": (
                "one shared maximum across all denoised D1-D50 frames, applied by BatchDataset"
            ),
            "photoacoustic_preprocessing": {
                "input_encoding": "packed unsigned 12-bit little-endian",
                "source_shape": [600, 600, 512],
                "pd_correction": False,
                "baseline": "per-A-line least-squares linear fit over both noise windows",
                "signal_window_start_inclusive": args.signal_start,
                "signal_window_stop_exclusive": args.signal_stop,
                "noise_windows_start_inclusive_stop_exclusive": [
                    [args.noise_left_start, args.noise_left_stop],
                    [args.noise_right_start, args.noise_right_stop],
                ],
                "projection": "noise-subtracted excess RMS amplitude",
                "formula": "sqrt(max(mean(signal_residual^2) - mean(noise_residual^2), 0))",
                "spatial_filter": "Gaussian",
                "spatial_sigma_pixels": float(args.spatial_sigma),
                "cross_frame_scaling": "none before dataset-wide shared-maximum normalization",
            },
        }
    )
    manifest["ground_truth"] = {
        "source": str(origin_source),
        "normalized_variable": "ground_truth.mat:clear_object",
        "raw_projection": "reference/origin_projection.npy",
        "normalization": "individual min-max to [0,1] after identical denoising",
    }
    manifest["outputs"].update(
        {
            "measurement_projection_tiff": "measurement_projection_tiff/dN_excess_rms.tif",
            "quality_control_figure": "quality_control.png",
            "quality_trends": "quality_trends.png",
            "quality_control_report": "quality_control.json",
        }
    )
    manifest["outputs"].pop("measurement_mip_tiff", None)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[3/3] 去噪数据集与共享尺度预览已完成。", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--origin-source", required=True)
    parser.add_argument("--signal-start", type=int, default=250)
    parser.add_argument("--signal-stop", type=int, default=320)
    parser.add_argument("--noise-left-start", type=int, default=220)
    parser.add_argument("--noise-left-stop", type=int, default=245)
    parser.add_argument("--noise-right-start", type=int, default=330)
    parser.add_argument("--noise-right-stop", type=int, default=355)
    parser.add_argument("--spatial-sigma", type=float, default=0.8)
    parser.add_argument("--chunk-rows", type=int, default=50)
    return parser


def main() -> None:
    output = prepare(build_parser().parse_args())
    print(f"处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
