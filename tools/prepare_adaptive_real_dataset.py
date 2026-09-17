#!/usr/bin/env python3
"""Build a real NeuWS dataset with experiment-calibrated A-line denoising."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_unit_png  # noqa: E402
from preprocessing.pa_adaptive_denoising import (  # noqa: E402
    AdaptiveCalibration,
    AdaptiveProjectionResult,
    CalibrationError,
    calibrate_experiment,
    detect_signal_window,
    load_packed12_adaptive_projection,
    sample_packed12_alines,
    verify_source_records,
)
from preprocessing.pa_denoising_gpu import cuda_backend_info  # noqa: E402


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        raise ValueError("投影没有有效强度范围，不能生成 ground truth。")
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _resolve_backend(requested: str) -> tuple[str, dict]:
    if requested == "cpu":
        return "cpu", {
            "backend": "cpu",
            "selection": "explicit",
            "cupy_version": None,
            "gpu_name": None,
        }
    if requested == "cuda":
        return "cuda", {**cuda_backend_info(), "selection": "explicit"}
    if requested != "auto":
        raise ValueError(f"未知 backend：{requested}")
    try:
        return "cuda", {**cuda_backend_info(), "selection": "auto"}
    except RuntimeError as exc:
        return "cpu", {
            "backend": "cpu",
            "selection": "auto_fallback",
            "cuda_unavailable_reason": str(exc),
            "cupy_version": None,
            "gpu_name": None,
        }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in (0, 1, 50, 95, 99, 99.5, 99.7, 100)
    }


def _summarize_projection(
    result: AdaptiveProjectionResult, *, label: str, source: Path
) -> dict:
    retention = result.signal_power_after / max(result.signal_power_before, 1e-12)
    return {
        "label": label,
        "source": str(source),
        "projection_quantiles": _quantiles(result.projection),
        "projection_mean": float(np.mean(result.projection)),
        "template_match_counts": list(result.template_match_counts),
        "template_match_fractions": list(result.template_match_fractions),
        "sampled_signal_power_before_after_ratio": float(retention),
    }


def _write_template_csv(path: Path, values: np.ndarray) -> None:
    half = values.size // 2
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_sample", "template_relative_adc"))
        for relative, value in zip(range(-half, half + 1), values):
            writer.writerow((relative, f"{float(value):.9g}"))


def _write_calibration(
    output_dir: Path, calibration: AdaptiveCalibration
) -> tuple[str, dict]:
    template_dir = output_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    metadata = calibration.metadata()
    calibration_hash = calibration.fingerprint()
    for index, (model, record) in enumerate(
        zip(calibration.templates, metadata["templates"]), start=1
    ):
        relative = Path("templates") / f"transient_template_{index}.csv"
        destination = output_dir / relative
        _write_template_csv(destination, model.values)
        record["path"] = str(relative)
        record["file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata["calibration_sha256"] = calibration_hash
    (output_dir / "calibration.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return calibration_hash, metadata


def _write_calibration_figure(
    output_dir: Path, calibration: AdaptiveCalibration
) -> None:
    profile = np.asarray(getattr(calibration, "_signal_profile"), dtype=np.float32)
    evidence = np.asarray(getattr(calibration, "_signal_evidence"), dtype=np.float32)
    depth = np.arange(calibration.depth)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(depth, profile, color="#1f77b4", label="depth signal score")
    axes[1].plot(depth, evidence, color="#d62728", label="significant evidence")
    for axis in axes:
        axis.axvspan(
            *calibration.signal_window,
            color="#2ca02c",
            alpha=0.18,
            label="signal window",
        )
        for position, window in enumerate(calibration.noise_windows):
            axis.axvspan(
                *window,
                color="#7f7f7f",
                alpha=0.18,
                label="noise windows" if position == 0 else None,
            )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    axes[1].set_xlabel("A-line depth sample")
    axes[0].set_ylabel("score")
    axes[1].set_ylabel("excess score")
    fig.suptitle("Automatic experiment-level A-line calibration")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_depth_profile.png", dpi=160)
    plt.close(fig)

    if calibration.templates:
        fig, axes = plt.subplots(
            len(calibration.templates), 1, figsize=(10, 3 * len(calibration.templates))
        )
        axes = np.atleast_1d(axes)
        for index, (axis, model) in enumerate(
            zip(axes, calibration.templates), start=1
        ):
            half = model.values.size // 2
            axis.plot(np.arange(-half, half + 1), model.values)
            axis.set_title(
                f"template {index}: |NCC|≥{model.correlation_threshold:.4f}, "
                f"events={model.source_event_count}"
            )
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_templates.png", dpi=160)
        plt.close(fig)


def _write_quality_figure(
    output_dir: Path,
    selected: dict[str, np.ndarray],
    frame_summaries: list[dict],
) -> None:
    labels = list(selected)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 4.8))
    axes = np.atleast_1d(axes)
    for axis, label in zip(axes, labels):
        image = selected[label]
        low, high = np.percentile(image, (1.0, 99.7))
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"{label}\nadaptive excess-RMS, P1–P99.7")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_control.png", dpi=160)
    plt.close(fig)

    frames = np.arange(1, len(frame_summaries) + 1)
    means = [item["projection_mean"] for item in frame_summaries]
    maxima = [item["projection_quantiles"]["100"] for item in frame_summaries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frames, means)
    axes[0].set_title("Frame mean")
    axes[1].plot(frames, maxima)
    axes[1].set_title("Frame maximum")
    for axis in axes:
        axis.set_xlabel("measurement frame")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_trends.png", dpi=160)
    plt.close(fig)


def _project(
    source: Path,
    calibration: AdaptiveCalibration,
    *,
    backend: str,
    chunk_rows: int,
) -> AdaptiveProjectionResult:
    return load_packed12_adaptive_projection(
        source,
        calibration=calibration,
        backend=backend,
        chunk_rows=chunk_rows,
    )


def _prepare_into(work_dir: Path, args: argparse.Namespace) -> None:
    template_dir = Path(args.template_data_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    origin_source = Path(args.origin_source).expanduser().resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"现有数据集不存在：{template_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"原始实验目录不存在：{source_dir}")
    if not origin_source.is_file():
        raise FileNotFoundError(f"origin BIN 不存在：{origin_source}")

    template_manifest = json.loads(
        (template_dir / "manifest.json").read_text(encoding="utf-8")
    )
    frames = list(template_manifest["frames"])
    num_frames = int(template_manifest["num_frames"])
    if len(frames) != num_frames or num_frames < 5:
        raise ValueError("现有数据集的帧清单无效或不足 5 帧。")
    source_shape = template_manifest.get("photoacoustic_preprocessing", {}).get(
        "source_shape", [600, 600, 512]
    )
    if len(source_shape) != 3:
        raise ValueError(f"source_shape 无效：{source_shape}")
    height, width, depth = (int(value) for value in source_shape)
    sample_prefix = str(template_manifest.get("sample_prefix", "s"))
    measurement_sources = [
        source_dir / str(record["measurement_source"]) for record in frames
    ]
    missing = [str(path) for path in measurement_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少原始测量 BIN：{missing[:3]}")

    backend, backend_info = _resolve_backend(args.backend)
    print("[1/5] 正在从本次实验自动标定 A-line 参数……", flush=True)
    calibration = calibrate_experiment(
        measurement_sources,
        height=height,
        width=width,
        depth=depth,
        grid_size=args.grid_size,
        seed=args.seed,
    )
    origin_stat = origin_source.stat()
    origin_record = {
        "path": str(origin_source),
        "size": int(origin_stat.st_size),
        "mtime_ns": int(origin_stat.st_mtime_ns),
    }

    # Origin is independent of parameter selection.  It must reproduce the
    # experiment-level depth gate when its signal is detectable.
    origin_sample = sample_packed12_alines(
        origin_source,
        height=height,
        width=width,
        depth=depth,
        grid_size=args.grid_size,
    )
    origin_window, _ = detect_signal_window(origin_sample)
    intersection = max(
        0,
        min(origin_window[1], calibration.signal_window[1])
        - max(origin_window[0], calibration.signal_window[0]),
    )
    union = max(origin_window[1], calibration.signal_window[1]) - min(
        origin_window[0], calibration.signal_window[0]
    )
    origin_iou = float(intersection / max(union, 1))
    if origin_iou < 0.50:
        raise CalibrationError(
            f"origin 深度信号与测量标定不一致（IoU={origin_iou:.3f}）。",
            diagnostics={
                "origin_signal_window": list(origin_window),
                "measurement_signal_window": list(calibration.signal_window),
                "origin_window_iou": origin_iou,
            },
        )

    calibration_hash, calibration_metadata = _write_calibration(
        work_dir, calibration
    )
    _write_calibration_figure(work_dir, calibration)
    print(
        f"  信号窗 {calibration.signal_window}；噪声窗 "
        f"{calibration.noise_windows}；模板 {len(calibration.templates)} 个。",
        flush=True,
    )

    reference_dir = work_dir / "reference"
    projection_dir = work_dir / "measurement_projection_tiff"
    preview_dir = work_dir / "measurement_png"
    for directory in (reference_dir, projection_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("[2/5] 正在处理 origin……", flush=True)
    origin_result = _project(
        origin_source,
        calibration,
        backend=backend,
        chunk_rows=args.chunk_rows,
    )
    origin = origin_result.projection
    tifffile.imwrite(
        reference_dir / "origin_adaptive_excess_rms.tif",
        origin,
        photometric="minisblack",
    )
    np.save(reference_dir / "origin_projection.npy", origin)
    clear_object = _normalize(origin)
    np.save(reference_dir / "clear_object.npy", clear_object)
    write_unit_png(reference_dir / "origin_ground_truth.png", clear_object)
    sio.savemat(
        work_dir / "ground_truth.mat",
        {
            "clear_object": clear_object,
            "object_image": clear_object,
            "origin_projection": origin,
        },
        do_compression=True,
    )

    measurements = np.lib.format.open_memmap(
        work_dir / "measurements.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_frames, height, width),
    )
    frame_summaries = []
    selected = {"origin": origin}
    selected_positions = {1, (num_frames + 1) // 2, num_frames}
    print(f"[3/5] 正在处理 {num_frames} 份测量……", flush=True)
    for position, (record, source) in enumerate(
        zip(frames, measurement_sources), start=1
    ):
        result = _project(
            source,
            calibration,
            backend=backend,
            chunk_rows=args.chunk_rows,
        )
        measurements[position - 1] = result.projection
        tifffile.imwrite(
            projection_dir / f"{sample_prefix}{position}_adaptive_excess_rms.tif",
            result.projection,
            photometric="minisblack",
        )
        sio.savemat(
            work_dir / f"SLM_raw{position}.mat",
            {"imsdata": result.projection},
            do_compression=True,
        )
        phase_source = template_dir / f"SLM_sim{position}.mat"
        if not phase_source.is_file():
            raise FileNotFoundError(f"现有数据集缺少相位文件：{phase_source}")
        shutil.copy2(phase_source, work_dir / phase_source.name)
        record["adaptive_calibration_sha256"] = calibration_hash
        frame_summaries.append(
            _summarize_projection(
                result, label=f"{sample_prefix}{position}", source=source
            )
        )
        if position in selected_positions:
            selected[f"{sample_prefix}{position}"] = result.projection.copy()
        if position == 1 or position % 5 == 0 or position == num_frames:
            print(f"  已完成 {position}/{num_frames}", flush=True)
    measurements.flush()

    verify_source_records(calibration.source_records)
    current_origin = origin_source.stat()
    if (
        int(current_origin.st_size),
        int(current_origin.st_mtime_ns),
    ) != (origin_record["size"], origin_record["mtime_ns"]):
        raise RuntimeError(f"处理期间 origin BIN 发生变化：{origin_source}")

    measurement_max = float(np.max(measurements))
    if not np.isfinite(measurement_max) or measurement_max <= 0:
        raise RuntimeError("自适应投影没有正的有限测量值。")
    print("[4/5] 正在生成共享尺度预览和自动质控报告……", flush=True)
    for position in range(1, num_frames + 1):
        write_unit_png(
            preview_dir / f"modulated_measurement_{position:04d}.png",
            np.asarray(measurements[position - 1]) / measurement_max,
        )
    _write_quality_figure(work_dir, selected, frame_summaries)
    origin_summary = _summarize_projection(
        origin_result, label="origin", source=origin_source
    )
    quality_report = {
        "passed": True,
        "automatic_release": True,
        "manual_image_review_required": False,
        "calibration_sha256": calibration_hash,
        "calibration": calibration_metadata["quality_control"],
        "origin_independent_validation": {
            "signal_window": list(origin_window),
            "measurement_window_iou": origin_iou,
        },
        "origin": origin_summary,
        "frames": frame_summaries,
        "aggregate": {
            "frame_mean_cv": float(
                np.std([item["projection_mean"] for item in frame_summaries])
                / max(
                    np.mean([item["projection_mean"] for item in frame_summaries]),
                    1e-12,
                )
            ),
            "template_match_fraction_mean": [
                float(
                    np.mean(
                        [item["template_match_fractions"][index] for item in frame_summaries]
                    )
                )
                for index in range(len(calibration.templates))
            ],
        },
    }
    (work_dir / "quality_control.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = copy.deepcopy(template_manifest)
    manifest.update(
        {
            "workflow": "real_point_scan_neuws_experiment_adaptive_aline_denoised",
            "scene_name": args.scene_name,
            "source_directory": str(source_dir),
            "template_dataset": str(template_dir),
            "measurement_max": measurement_max,
            "measurement_normalization": (
                f"one shared maximum across adaptive {sample_prefix}1-"
                f"{sample_prefix}{num_frames}, applied by BatchDataset"
            ),
            "frames": frames,
            "photoacoustic_preprocessing": {
                "input_encoding": "packed unsigned 12-bit little-endian",
                "source_shape": [height, width, depth],
                **backend_info,
                "calibration_scope": "one frozen calibration for the full experiment",
                "calibration_file": "calibration.json",
                "calibration_sha256": calibration_hash,
                "origin_used_for_calibration": False,
                "origin_independent_window_iou": origin_iou,
                "signal_window_start_inclusive_stop_exclusive": list(
                    calibration.signal_window
                ),
                "noise_windows_start_inclusive_stop_exclusive": [
                    list(window) for window in calibration.noise_windows
                ],
                "template_count": len(calibration.templates),
                "template_amplitude_threshold": "6 times per-A-line robust noise sigma",
                "baseline": "per-A-line robust linear fit",
                "noise_power": "left/right robust powers linearly interpolated over signal depth",
                "projection": "noise-power-subtracted excess RMS amplitude",
                "formula": "sqrt(max(mean(signal_residual^2)-mean(interpolated_noise_power),0))",
                "spatial_filter": None,
                "spatial_sigma_pixels": 0.0,
                "pd_correction": False,
                "source_modified": False,
            },
        }
    )
    manifest["ground_truth"] = {
        "source": str(origin_source),
        "adaptive_calibration_sha256": calibration_hash,
        "normalized_variable": "ground_truth.mat:clear_object",
        "raw_projection": "reference/origin_projection.npy",
        "normalization": "individual min-max to [0,1] after identical adaptive A-line denoising",
    }
    manifest["outputs"].update(
        {
            "measurement_projection_tiff": (
                f"measurement_projection_tiff/{sample_prefix}N_adaptive_excess_rms.tif"
            ),
            "calibration": "calibration.json",
            "calibration_templates": "templates/transient_template_N.csv",
            "calibration_depth_figure": "calibration_depth_profile.png",
            "quality_control_figure": "quality_control.png",
            "quality_trends": "quality_trends.png",
            "quality_control_report": "quality_control.json",
        }
    )
    manifest["outputs"].pop("measurement_mip_tiff", None)
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[5/5] 自适应 A-line 去噪数据集已通过自动质控。", flush=True)


def _failure_directory(output_dir: Path) -> Path:
    candidate = output_dir.with_name(f"{output_dir.name}_failed")
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir.with_name(f"{output_dir.name}_failed_{stamp}")


def prepare(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _prepare_into(temporary, args)
        temporary.replace(output_dir)
    except Exception as exc:
        report = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "diagnostics": getattr(exc, "diagnostics", {}),
        }
        (temporary / "calibration_failed.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        failed_dir = _failure_directory(output_dir)
        temporary.replace(failed_dir)
        raise RuntimeError(
            f"自适应标定或质控失败；诊断已保存至 {failed_dir}"
        ) from exc
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--origin-source", required=True)
    parser.add_argument("--template-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=10)
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser


def main() -> None:
    output = prepare(build_parser().parse_args())
    print(f"处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
