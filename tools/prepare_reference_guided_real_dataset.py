#!/usr/bin/env python3
"""Build a NeuWS dataset with reference-guided adaptive PA denoising.

Inputs are intentionally minimal:
- current experiment PA1 BINs named s01...sNN;
- one origin PA1 BIN;
- SLM_sim1.mat...SLM_simN.mat phase files;
- historical motor-crosstalk teacher template.

No pre-existing manifest.json is required. A temporary compatibility manifest is
created internally and never modifies the user's phase directory.

With --backend cuda, packed-12 bytes are uploaded before decoding and the full
per-chunk signal pipeline stays on CUDA until the final 2-D projection.  One
experiment-level GPU workspace is shared by origin and all sNN volumes, so
CuPy/FFT pools and transfer buffers are not destroyed and rebuilt per file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# This is a headless batch-processing tool.  Force Matplotlib away from TkAgg
# before importing the shared dataset writer, which imports pyplot at module load.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pa_denoising import load_template_csv  # noqa: E402
from preprocessing.pa_reference_guided_denoising import (  # noqa: E402
    ALGORITHM_VERSION,
    calibrate_experiment_reference_guided,
    load_packed12_reference_guided_projection,
)
from preprocessing.pa_reference_guided_gpu import ReferenceGuidedGpuPipeline  # noqa: E402
from tools import prepare_adaptive_real_dataset as base  # noqa: E402


DEFAULT_TEACHER_TEMPLATE = (
    PROJECT_ROOT
    / "motor_crosstalk_denoise"
    / "template"
    / "motor_crosstalk_template_v1.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--origin-source", required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=600)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--measurement-glob",
        default="s*_600_600_512_PA1.bin",
        help="当前实验测量文件匹配模式；默认识别 s01...sNN 的 600x600x512 PA1 BIN。",
    )
    parser.add_argument(
        "--teacher-template",
        default=str(DEFAULT_TEACHER_TEMPLATE),
        help="历史电机串扰 teacher 模板 CSV。",
    )
    parser.add_argument(
        "--teacher-correlation-threshold",
        type=float,
        default=0.70,
        help="teacher 候选筛选 |NCC| 阈值。",
    )
    parser.add_argument(
        "--teacher-refit-threshold",
        type=float,
        default=0.80,
        help="允许进入本实验模板重拟合的高可信 |NCC| 阈值。",
    )
    parser.add_argument(
        "--final-correlation-threshold",
        type=float,
        default=0.70,
        help="本实验模板最终去相关 |NCC| 阈值。",
    )
    parser.add_argument(
        "--minimum-fitted-peak-adc",
        type=float,
        default=80.0,
        help="teacher 筛选和最终去相关的最小拟合峰值 ADC。",
    )
    return parser


def _measurement_index(path: Path) -> int:
    match = re.match(r"^[sS](\d+)_", path.name)
    if match is None:
        raise ValueError(f"无法从测量文件名解析 sNN 编号：{path.name}")
    return int(match.group(1))


def _infer_shape(path: Path) -> tuple[int, int, int]:
    match = re.search(r"_(\d+)_(\d+)_(\d+)_PA1\.bin$", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"无法从文件名推断 height/width/depth：{path.name}")
    return tuple(int(value) for value in match.groups())


def _phase_index(path: Path) -> int | None:
    match = re.fullmatch(r"SLM_sim(\d+)\.mat", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _discover_inputs(args) -> tuple[list[Path], Path, tuple[int, int, int], dict[int, Path]]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    phase_dir = Path(args.phase_dir).expanduser().resolve()
    origin = Path(args.origin_source).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"当前实验目录不存在：{source_dir}")
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"相位目录不存在：{phase_dir}")
    if not origin.is_file():
        raise FileNotFoundError(f"origin BIN 不存在：{origin}")

    measurements = sorted(source_dir.glob(args.measurement_glob), key=_measurement_index)
    if not measurements:
        raise FileNotFoundError(
            f"没有按 {args.measurement_glob!r} 找到测量 PA1 BIN：{source_dir}"
        )
    indices = [_measurement_index(path) for path in measurements]
    expected = list(range(1, len(measurements) + 1))
    if indices != expected:
        raise ValueError(f"测量文件编号必须连续为 1..N，实际为：{indices}")

    shape = _infer_shape(measurements[0])
    inconsistent = [path.name for path in measurements if _infer_shape(path) != shape]
    if inconsistent:
        raise ValueError(f"测量文件尺寸不一致，例如：{inconsistent[:3]}")
    if _infer_shape(origin) != shape:
        raise ValueError(
            f"origin 尺寸 {_infer_shape(origin)} 与测量尺寸 {shape} 不一致。"
        )

    phases: dict[int, Path] = {}
    for path in phase_dir.glob("SLM_sim*.mat"):
        index = _phase_index(path)
        if index is not None:
            phases[index] = path.resolve()
    missing = [index for index in expected if index not in phases]
    if missing:
        raise FileNotFoundError(
            f"相位目录缺少 SLM_simN.mat；缺失编号前几个：{missing[:10]}"
        )

    return measurements, origin, shape, phases


def _make_adapter_dataset(
    measurements: list[Path],
    shape: tuple[int, int, int],
    phases: dict[int, Path],
) -> tempfile.TemporaryDirectory:
    holder = tempfile.TemporaryDirectory(prefix="neuws_reference_guided_adapter_")
    directory = Path(holder.name)
    frames = [
        {
            "measurement_source": path.name,
            "reference_guided_source_position": position,
        }
        for position, path in enumerate(measurements, start=1)
    ]
    manifest = {
        "workflow": "reference_guided_temporary_adapter",
        "num_frames": len(measurements),
        "sample_prefix": "s",
        "frames": frames,
        "photoacoustic_preprocessing": {"source_shape": list(shape)},
        "outputs": {},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for position in range(1, len(measurements) + 1):
        (directory / f"SLM_sim{position}.mat").symlink_to(phases[position])
    return holder


def _rewrite_manifest(output: Path, args, *, phase_dir: Path, shape: tuple[int, int, int]) -> None:
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pa = manifest.setdefault("photoacoustic_preprocessing", {})
    manifest["workflow"] = "real_point_scan_neuws_reference_guided_adaptive_aline_denoised"
    manifest["template_dataset"] = None
    manifest["phase_source_directory"] = str(phase_dir)
    pa.update(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "source_shape": list(shape),
            "calibration_strategy": "historical-template-guided experiment refit",
            "teacher_template": str(Path(args.teacher_template).expanduser().resolve()),
            "teacher_template_role": (
                "identify current-experiment crosstalk candidates; final subtraction "
                "uses the refitted experiment template"
            ),
            "teacher_correlation_threshold_absolute": float(args.teacher_correlation_threshold),
            "teacher_refit_threshold_absolute": float(args.teacher_refit_threshold),
            "final_correlation_threshold_absolute": float(args.final_correlation_threshold),
            "minimum_fitted_peak_adc": float(args.minimum_fitted_peak_adc),
            "decorrelation": (
                "validated historical normalized-xcorr + least-squares "
                "non-circular shifted-template subtraction"
            ),
            "template_count": 1,
            "template_amplitude_threshold": (
                f"fixed fitted template peak >= {float(args.minimum_fitted_peak_adc):g} ADC"
            ),
            "baseline": (
                "per-A-line median for template matching; robust linear fit for "
                "noise-power/excess-RMS projection"
            ),
            "cuda_pipeline": (
                "persistent experiment-level CUDA workspace; double pinned packed-byte "
                "buffers + overlapped H2D; direct GPU u12->float32 decode; "
                "serial NCC/FFT/LS subtraction; window-only robust QC/excess-RMS; "
                "final projection D2H; workspace released only after all volumes"
                if args.backend in {"auto", "cuda"}
                else None
            ),
            "chunk_rows_requested": int(args.chunk_rows),
            "cuda_read_ahead": bool(args.backend in {"auto", "cuda"}),
            "cuda_pinned_double_buffer": bool(args.backend in {"auto", "cuda"}),
            "cuda_h2d_compute_overlap": bool(args.backend in {"auto", "cuda"}),
            "cuda_persistent_workspace_across_volumes": bool(
                args.backend in {"auto", "cuda"}
            ),
            "cuda_oom_split_fallback": False,
        }
    )
    outputs = manifest.setdefault("outputs", {})
    outputs["calibration_templates"] = "templates/transient_template_1.csv"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare(args) -> Path:
    teacher_path = Path(args.teacher_template).expanduser().resolve()
    teacher_template = load_template_csv(teacher_path)
    if not 0 < args.teacher_correlation_threshold <= 1:
        raise ValueError("teacher-correlation-threshold 必须位于 (0, 1]。")
    if not 0 < args.teacher_refit_threshold <= 1:
        raise ValueError("teacher-refit-threshold 必须位于 (0, 1]。")
    if args.teacher_refit_threshold < args.teacher_correlation_threshold:
        raise ValueError("teacher-refit-threshold 不应低于 teacher-correlation-threshold。")
    if not 0 < args.final_correlation_threshold <= 1:
        raise ValueError("final-correlation-threshold 必须位于 (0, 1]。")
    if args.minimum_fitted_peak_adc < 0:
        raise ValueError("minimum-fitted-peak-adc 必须为非负数。")

    measurements, origin, shape, phases = _discover_inputs(args)
    phase_dir = Path(args.phase_dir).expanduser().resolve()
    print(
        f"识别到 {len(measurements)} 帧测量：{measurements[0].name} -> {measurements[-1].name}",
        flush=True,
    )
    print(f"采集尺寸：{shape[0]} x {shape[1]} x {shape[2]}", flush=True)
    print(f"相位目录：{phase_dir}", flush=True)

    def guided_calibrator(
        sources,
        *,
        height: int,
        width: int,
        depth: int,
        grid_size: int = 20,
        seed: int = 0,
    ):
        return calibrate_experiment_reference_guided(
            sources,
            teacher_template=teacher_template,
            height=height,
            width=width,
            depth=depth,
            grid_size=grid_size,
            seed=seed,
            teacher_correlation_threshold=args.teacher_correlation_threshold,
            teacher_refit_threshold=args.teacher_refit_threshold,
            final_correlation_threshold=args.final_correlation_threshold,
            minimum_fitted_peak_adc=args.minimum_fitted_peak_adc,
        )

    base.calibrate_experiment = guided_calibrator

    # Keep one CUDA workspace alive for the entire experiment.  The shared
    # dataset writer resolves auto -> cpu/cuda before invoking this function,
    # so we lazily create the pipeline on the first CUDA projection after the
    # frozen calibration is available.
    shared_pipeline = None
    shared_calibration = None

    def guided_projector(
        source,
        *,
        calibration,
        backend: str = "cpu",
        chunk_rows: int = 600,
    ):
        nonlocal shared_pipeline, shared_calibration
        if backend == "cuda":
            if shared_pipeline is None:
                shared_calibration = calibration
                shared_pipeline = ReferenceGuidedGpuPipeline(calibration)
                info = shared_pipeline.device_info()
                print(
                    "  CUDA persistent sync-optimized workspace: "
                    f"{info['gpu_name']}，总显存 {info['total_memory_gib']:.2f} GiB；"
                    "将在 origin + 全部测量帧间复用。",
                    flush=True,
                )
            elif calibration is not shared_calibration:
                # This should never happen in one frozen experiment.  Refuse
                # silent reuse with a different template/window calibration.
                if calibration.fingerprint() != shared_calibration.fingerprint():
                    raise RuntimeError("检测到不同 calibration，不能复用同一 GPU workspace。")
            return load_packed12_reference_guided_projection(
                source,
                calibration=calibration,
                backend="cuda",
                chunk_rows=chunk_rows,
                gpu_pipeline=shared_pipeline,
            )
        return load_packed12_reference_guided_projection(
            source,
            calibration=calibration,
            backend=backend,
            chunk_rows=chunk_rows,
        )

    base.load_packed12_adaptive_projection = guided_projector

    holder = _make_adapter_dataset(measurements, shape, phases)
    try:
        compat_args = argparse.Namespace(
            source_dir=str(Path(args.source_dir).expanduser().resolve()),
            origin_source=str(origin),
            template_data_dir=holder.name,
            output_dir=args.output_dir,
            scene_name=args.scene_name,
            grid_size=args.grid_size,
            seed=args.seed,
            chunk_rows=args.chunk_rows,
            backend=args.backend,
        )
        output = base.prepare(compat_args)
    finally:
        if shared_pipeline is not None:
            print("  释放整批实验的 CUDA persistent sync-optimized workspace……", flush=True)
            shared_pipeline.shutdown()
        holder.cleanup()

    _rewrite_manifest(output, args, phase_dir=phase_dir, shape=shape)
    return output


def main() -> None:
    args = build_parser().parse_args()
    output = prepare(args)
    print(f"reference-guided 处理完成：{output}", flush=True)


if __name__ == "__main__":
    main()
