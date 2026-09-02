#!/usr/bin/env python3
"""Rebuild a real NeuWS dataset after shifted-template crosstalk subtraction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_NOISE_TEMPLATE = (
    PROJECT_ROOT
    / "motor_crosstalk_denoise"
    / "template"
    / "motor_crosstalk_template_v1.csv"
)

from image_utils import write_unit_png  # noqa: E402
from preprocessing.pa_denoising import (  # noqa: E402
    TemplateXcorrProjectionResult,
    load_packed12_template_xcorr_mip_projection,
    load_template_csv,
)


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        raise ValueError("图像没有有效强度范围。")
    return np.asarray((values - minimum) / (maximum - minimum), dtype=np.float32)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in (0, 1, 50, 95, 99, 99.5, 99.7, 100)
    }


def _summarize_result(
    result: TemplateXcorrProjectionResult,
    *,
    label: str,
    source: Path,
) -> dict:
    matched = result.matched_map
    matched_count = int(matched.sum())
    correlation = np.abs(result.correlation_map[matched])
    centers = result.center_map[matched]
    coefficients = result.coefficient_map[matched]
    positive_removed = np.maximum(
        result.original_projection - result.projection, np.float32(0.0)
    )
    return {
        "label": label,
        "source": str(source),
        "matched_pixel_count": matched_count,
        "matched_pixel_fraction": float(matched.mean()),
        "matched_abs_ncc_p10_p50_p90": (
            [float(value) for value in np.percentile(correlation, (10, 50, 90))]
            if matched_count
            else []
        ),
        "matched_center_p1_p50_p99": (
            [float(value) for value in np.percentile(centers, (1, 50, 99))]
            if matched_count
            else []
        ),
        "matched_coefficient_p1_p50_p99": (
            [float(value) for value in np.percentile(coefficients, (1, 50, 99))]
            if matched_count
            else []
        ),
        "projection_before_quantiles_adc": _percentiles(result.original_projection),
        "projection_after_quantiles_adc": _percentiles(result.projection),
        "projection_mean_before_after_adc": [
            float(result.original_projection.mean()),
            float(result.projection.mean()),
        ],
        "projection_median_before_after_adc": [
            float(np.median(result.original_projection)),
            float(np.median(result.projection)),
        ],
        "positive_removed_mean_adc": float(positive_removed.mean()),
        "changed_projection_pixel_fraction": float(
            np.mean(result.projection != result.original_projection)
        ),
    }


def _save_diagnostics(
    output_dir: Path,
    *,
    label: str,
    result: TemplateXcorrProjectionResult,
) -> None:
    tifffile.imwrite(
        output_dir / f"{label}_max_abs_ncc_float32.tif",
        np.abs(result.correlation_map).astype(np.float32),
    )
    tifffile.imwrite(
        output_dir / f"{label}_matched_center_uint16.tif", result.center_map
    )
    tifffile.imwrite(
        output_dir / f"{label}_subtraction_mask_uint8.tif",
        result.matched_map.astype(np.uint8),
    )


def _save_quality_figures(
    output_dir: Path,
    *,
    selected: dict[str, TemplateXcorrProjectionResult],
    frame_summaries: list[dict],
    sample_prefix: str,
) -> None:
    labels = (
        "origin",
        f"{sample_prefix}1",
        f"{sample_prefix}25",
        f"{sample_prefix}50",
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for column, label in enumerate(labels):
        result = selected[label]
        for row, (image, title) in enumerate(
            (
                (result.original_projection, "Original full-depth MIP"),
                (result.projection, "Template-xcorr cleaned MIP"),
            )
        ):
            low, high = np.percentile(image, (1.0, 99.5))
            axes[row, column].imshow(image, cmap="gray", vmin=low, vmax=high)
            axes[row, column].set_title(f"{label}: {title}\nP1–P99.5 display")
            axes[row, column].axis("off")
    fig.suptitle("Motor-crosstalk decorrelation comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_control.png", dpi=150)
    plt.close(fig)

    frames = np.arange(1, len(frame_summaries) + 1)
    mean_before = [item["projection_mean_before_after_adc"][0] for item in frame_summaries]
    mean_after = [item["projection_mean_before_after_adc"][1] for item in frame_summaries]
    p997_before = [
        item["projection_before_quantiles_adc"]["99.7"] for item in frame_summaries
    ]
    p997_after = [
        item["projection_after_quantiles_adc"]["99.7"] for item in frame_summaries
    ]
    matched_fraction = [item["matched_pixel_fraction"] for item in frame_summaries]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(frames, mean_before, label="before")
    axes[0].plot(frames, mean_after, label="after")
    axes[0].set_title("Frame mean")
    axes[1].plot(frames, p997_before, label="before")
    axes[1].plot(frames, p997_after, label="after")
    axes[1].set_title("Frame P99.7")
    axes[2].plot(frames, np.asarray(matched_fraction) * 100.0, color="#1f77b4")
    axes[2].set_title("Matched A-lines")
    axes[2].set_ylabel("Percent")
    for axis in axes:
        axis.set_xlabel(f"{sample_prefix} frame")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_trends.png", dpi=150)
    plt.close(fig)


def _project(
    source: Path,
    *,
    template: np.ndarray,
    args: argparse.Namespace,
) -> TemplateXcorrProjectionResult:
    return load_packed12_template_xcorr_mip_projection(
        source,
        template=template,
        height=int(getattr(args, "height", 600)),
        width=int(getattr(args, "width", 600)),
        depth=int(getattr(args, "depth", 512)),
        baseline_adc=args.baseline_adc,
        correlation_threshold=args.correlation_threshold,
        minimum_fitted_peak_adc=args.minimum_fitted_peak_adc,
        chunk_rows=args.chunk_rows,
    )


def _prepare_into(work_dir: Path, args: argparse.Namespace) -> None:
    template_dir = Path(args.template_data_dir).expanduser().resolve()
    origin_source = Path(args.origin_source).expanduser().resolve()
    template_path = Path(args.noise_template).expanduser().resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"模板数据集不存在：{template_dir}")
    if not origin_source.is_file():
        raise FileNotFoundError(f"origin BIN 不存在：{origin_source}")
    template = load_template_csv(template_path)
    if template.size != 361:
        raise ValueError(f"冻结模板必须为 361 点，实际为 {template.size}。")

    template_manifest = json.loads(
        (template_dir / "manifest.json").read_text(encoding="utf-8")
    )
    num_frames = int(template_manifest["num_frames"])
    frames = template_manifest["frames"]
    if num_frames != 50 or len(frames) != 50:
        raise ValueError("本次实验要求连续的 1–50 共 50 帧。")
    source_shape = template_manifest.get("photoacoustic_preprocessing", {}).get(
        "source_shape", [600, 600, 512]
    )
    if len(source_shape) != 3:
        raise ValueError(f"模板数据集 source_shape 无效：{source_shape}")
    height, width, depth = (int(value) for value in source_shape)
    if height <= 0 or width <= 0 or depth <= 0 or height != width or width % 2:
        raise ValueError(f"模板数据集尺寸无效：{source_shape}")
    args.height, args.width, args.depth = height, width, depth
    sample_prefix = str(template_manifest.get("sample_prefix", "d"))
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"原始 D 数据目录不存在：{source_dir}")

    reference_dir = work_dir / "reference"
    projection_dir = work_dir / "measurement_projection_tiff"
    preview_dir = work_dir / "measurement_png"
    diagnostic_dir = work_dir / "diagnostics"
    for directory in (reference_dir, projection_dir, preview_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    template_measurements = np.load(
        template_dir / "measurements.npy", mmap_mode="r", allow_pickle=False
    )
    template_origin = np.load(
        template_dir / "reference" / "origin_projection.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if template_measurements.shape != (num_frames, height, width):
        raise ValueError(
            f"模板 measurements.npy 应为 {(num_frames, height, width)}，"
            f"实际为 {template_measurements.shape}。"
        )
    if template_origin.shape != (height, width):
        raise ValueError(
            f"模板 origin 投影应为 {(height, width)}，实际为 {template_origin.shape}。"
        )
    selected: dict[str, TemplateXcorrProjectionResult] = {}

    print("[1/4] 正在处理去相关 origin……", flush=True)
    origin_result = _project(origin_source, template=template, args=args)
    if not np.array_equal(origin_result.original_projection, template_origin):
        raise RuntimeError("重新解码的 origin MIP 与原数据集不一致，已停止。")
    selected["origin"] = origin_result
    _save_diagnostics(diagnostic_dir, label="origin", result=origin_result)
    origin = origin_result.projection
    tifffile.imwrite(
        reference_dir / "origin_decorrelated_mip.tif", origin, photometric="minisblack"
    )
    clear_object = _normalize(origin)
    np.save(reference_dir / "origin_projection.npy", origin)
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
    selected_positions = {1, 25, 50}
    source_state = {}
    print(f"[2/4] 正在处理 50 份 {sample_prefix} 测量……", flush=True)
    for position, record in enumerate(frames, start=1):
        source = source_dir / record["measurement_source"]
        if not source.is_file():
            raise FileNotFoundError(f"缺少第 {position} 帧原始 BIN：{source}")
        source_state[str(source)] = {
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
        result = _project(source, template=template, args=args)
        if not np.array_equal(
            result.original_projection, template_measurements[position - 1]
        ):
            raise RuntimeError(
                f"{sample_prefix}{position} 重新解码 MIP 与原数据集不一致。"
            )
        measurements[position - 1] = result.projection
        tifffile.imwrite(
            projection_dir / f"{sample_prefix}{position}_decorrelated_mip.tif",
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
            raise FileNotFoundError(f"模板缺少第 {position} 帧相位：{phase_source}")
        shutil.copy2(phase_source, work_dir / phase_source.name)
        frame_summaries.append(
            _summarize_result(
                result, label=f"{sample_prefix}{position}", source=source
            )
        )
        if position in selected_positions:
            selected[f"{sample_prefix}{position}"] = result
            _save_diagnostics(
                diagnostic_dir,
                label=f"{sample_prefix}{position}",
                result=result,
            )
        if position == 1 or position % 5 == 0:
            print(f"  已完成 {position}/50", flush=True)
    measurements.flush()

    for source_name, initial in source_state.items():
        source = Path(source_name)
        current = {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
        if current != initial:
            raise RuntimeError(f"原始 BIN 在处理期间发生变化：{source}")

    measurement_max = float(np.max(measurements))
    if not np.isfinite(measurement_max) or measurement_max <= 0:
        raise ValueError("去相关测量没有正的有限最大值。")
    print("[3/4] 正在生成共享尺度预览和质量报告……", flush=True)
    for position in range(1, num_frames + 1):
        write_unit_png(
            preview_dir / f"modulated_measurement_{position:04d}.png",
            np.asarray(measurements[position - 1]) / measurement_max,
        )
    _save_quality_figures(
        work_dir,
        selected=selected,
        frame_summaries=frame_summaries,
        sample_prefix=sample_prefix,
    )
    origin_summary = _summarize_result(
        origin_result, label="origin", source=origin_source
    )
    quality_report = {
        "method": "shifted template normalized-xcorr + least-squares subtraction",
        "origin": origin_summary,
        "frames": frame_summaries,
        "aggregate": {
            "matched_fraction_mean_min_max": [
                float(np.mean([item["matched_pixel_fraction"] for item in frame_summaries])),
                float(np.min([item["matched_pixel_fraction"] for item in frame_summaries])),
                float(np.max([item["matched_pixel_fraction"] for item in frame_summaries])),
            ],
            "matched_abs_ncc_median_mean": float(
                np.mean(
                    [item["matched_abs_ncc_p10_p50_p90"][1] for item in frame_summaries]
                )
            ),
            "frame_mean_cv_after": float(
                np.std([item["projection_mean_before_after_adc"][1] for item in frame_summaries])
                / np.mean([item["projection_mean_before_after_adc"][1] for item in frame_summaries])
            ),
        },
    }
    (work_dir / "quality_control.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = copy.deepcopy(template_manifest)
    template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    manifest.update(
        {
            "workflow": "real_point_scan_neuws_origin_only_template_decorrelated",
            "scene_name": args.scene_name,
            "source_directory": str(source_dir),
            "template_dataset": str(template_dir),
            "measurement_max": measurement_max,
            "measurement_normalization": (
                f"one shared maximum across all decorrelated {sample_prefix}1-"
                f"{sample_prefix}50 frames, applied by BatchDataset"
            ),
            "photoacoustic_preprocessing": {
                "input_encoding": "packed unsigned 12-bit little-endian",
                "source_shape": [height, width, depth],
                "pd_correction": False,
                "time_gate": None,
                "matching_baseline": "per-A-line median",
                "template_path": str(template_path),
                "template_sha256": template_hash,
                "template_relative_samples_inclusive": [-180, 180],
                "template_length_samples": int(template.size),
                "correlation": "maximum absolute normalized cross-correlation",
                "correlation_threshold_absolute": float(args.correlation_threshold),
                "minimum_fitted_peak_adc": float(args.minimum_fitted_peak_adc),
                "amplitude_fit": "least squares on the valid non-circular overlap",
                "subtraction": "one strongest accepted shifted template per A-line",
                "baseline_adc": float(args.baseline_adc),
                "negative_policy": "clip_to_zero_after_baseline_subtraction",
                "projection": "full-depth maximum intensity projection after subtraction",
                "projection_axis": 2,
                "cross_frame_scaling": "none before dataset-wide shared-maximum normalization",
                "source_modified": False,
            },
        }
    )
    manifest["ground_truth"] = {
        "source": str(origin_source),
        "normalized_variable": "ground_truth.mat:clear_object",
        "raw_projection": "reference/origin_projection.npy",
        "normalization": "individual min-max to [0,1] after identical decorrelation",
    }
    manifest["outputs"].update(
        {
            "measurement_projection_tiff": (
                f"measurement_projection_tiff/{sample_prefix}N_decorrelated_mip.tif"
            ),
            "quality_control_figure": "quality_control.png",
            "quality_trends": "quality_trends.png",
            "quality_control_report": "quality_control.json",
            "diagnostics": (
                f"diagnostics/{{origin,{sample_prefix}1,{sample_prefix}25,"
                f"{sample_prefix}50}}_*.tif"
            ),
        }
    )
    manifest["outputs"].pop("measurement_mip_tiff", None)
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[4/4] 去相关数据集已完成。", flush=True)


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
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-data-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--origin-source", required=True)
    parser.add_argument("--noise-template", default=str(DEFAULT_NOISE_TEMPLATE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--baseline-adc", type=float, default=2048.0)
    parser.add_argument("--correlation-threshold", type=float, default=0.70)
    parser.add_argument("--minimum-fitted-peak-adc", type=float, default=80.0)
    parser.add_argument("--chunk-rows", type=int, default=10)
    return parser


def main() -> None:
    output = prepare(build_parser().parse_args())
    print(f"处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
