#!/usr/bin/env python3
"""Run the real PA experiment pipeline from raw BIN files to SLM correction.

This is an orchestration layer only. Each stage remains an independent program:

1. reference-guided CUDA/CPU denoising and NeuWS dataset preparation;
2. NeuWS training/reconstruction;
3. immediate full-phase SLM correction export (experiment-critical output);
4. Zernike fitting/analysis (runs after the hardware correction is available).

The stages are launched in separate Python processes so CuPy/CUDA resources from
preprocessing are released before PyTorch training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "real_experiment.json"
DEFAULT_MEASUREMENT_GLOB = "s*_600_600_512_PA1.bin"
DEFAULT_TEACHER_TEMPLATE = (
    PROJECT_ROOT
    / "motor_crosstalk_denoise"
    / "template"
    / "motor_crosstalk_template_v1.csv"
)
PREPROCESS_FINGERPRINT_NAME = "pipeline_preprocess_fingerprint.json"


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在：{path}\n"
            "请复制 configs/examples/real_experiment_pipeline.json 为 "
            "configs/real_experiment.json，并填写本机实验路径。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pipeline 配置顶层必须是 JSON object。")
    return data


def _required(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"配置缺少必填字段：{key}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_command(command: list[str]) -> None:
    quoted = [shlex.quote(str(part)) for part in command]
    print("  $", *quoted, sep=" ", flush=True)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"输入文件不存在：{resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _measurement_index(path: Path) -> int:
    name = path.name
    if not name.lower().startswith("s") or "_" not in name:
        raise ValueError(f"无法从测量文件名解析 sNN 编号：{name}")
    prefix = name.split("_", 1)[0][1:]
    if not prefix.isdigit():
        raise ValueError(f"无法从测量文件名解析 sNN 编号：{name}")
    return int(prefix)


def _preprocess_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("preprocess", {})
    return {
        "backend": str(section.get("backend", "cuda")),
        "chunk_rows": int(section.get("chunk_rows", 600)),
        "grid_size": int(section.get("grid_size", 20)),
        "seed": int(section.get("seed", 0)),
        "measurement_glob": str(section.get("measurement_glob", DEFAULT_MEASUREMENT_GLOB)),
        "teacher_template": str(
            _resolve_path(section.get("teacher_template", DEFAULT_TEACHER_TEMPLATE))
        ),
        "teacher_correlation_threshold": float(
            section.get("teacher_correlation_threshold", 0.70)
        ),
        "teacher_refit_threshold": float(section.get("teacher_refit_threshold", 0.80)),
        "final_correlation_threshold": float(
            section.get("final_correlation_threshold", 0.70)
        ),
        "minimum_fitted_peak_adc": float(section.get("minimum_fitted_peak_adc", 80.0)),
    }


def _build_preprocess_fingerprint(
    config: dict[str, Any],
    *,
    scene_name: str,
    dataset_dir: Path,
) -> dict[str, Any]:
    source_dir = _resolve_path(_required(config, "source_dir"))
    origin = _resolve_path(_required(config, "origin_source"))
    phase_dir = _resolve_path(_required(config, "phase_dir"))
    effective = _preprocess_effective_config(config)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir 不存在：{source_dir}")
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"phase_dir 不存在：{phase_dir}")

    measurements = sorted(
        source_dir.glob(effective["measurement_glob"]),
        key=_measurement_index,
    )
    if not measurements:
        raise FileNotFoundError(
            f"没有按 {effective['measurement_glob']!r} 找到测量 BIN：{source_dir}"
        )
    indices = [_measurement_index(path) for path in measurements]
    expected_indices = list(range(1, len(measurements) + 1))
    if indices != expected_indices:
        raise ValueError(f"测量文件编号必须连续为 1..N，实际为：{indices}")

    phase_files: list[Path] = []
    for index in expected_indices:
        phase_path = phase_dir / f"SLM_sim{index}.mat"
        if not phase_path.is_file():
            raise FileNotFoundError(f"相位文件不存在：{phase_path}")
        phase_files.append(phase_path)

    teacher = Path(effective["teacher_template"])
    payload = {
        "schema_version": 1,
        "scene_name": scene_name,
        "dataset_output_dir": str(dataset_dir.resolve()),
        "source_dir": str(source_dir),
        "origin_source": _file_identity(origin),
        "measurement_files": [_file_identity(path) for path in measurements],
        "phase_dir": str(phase_dir),
        "phase_files": [_file_identity(path) for path in phase_files],
        "teacher_template": _file_identity(teacher),
        "preprocess": effective,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "inputs": payload,
    }


def _preprocess_expected_outputs(dataset_dir: Path, num_frames: int) -> list[Path]:
    outputs = [dataset_dir / "manifest.json"]
    outputs.extend(dataset_dir / f"SLM_raw{index}.mat" for index in range(1, num_frames + 1))
    outputs.extend(dataset_dir / f"SLM_sim{index}.mat" for index in range(1, num_frames + 1))
    return outputs


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层不是 object：{path}")
    return data


def _write_preprocess_fingerprint(path: Path, fingerprint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fingerprint)
    payload["written_at"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mark_preprocess_skipped(
    *,
    state: dict[str, Any],
    state_path: Path,
    expected_outputs: list[Path],
    fingerprint: dict[str, Any],
    fingerprint_path: Path,
    status: str,
    message: str,
) -> None:
    stage = state.setdefault("stages", {}).setdefault("01_preprocess", {})
    stage.update(
        {
            "status": status,
            "checked_at": _utc_now(),
            "expected_outputs": [str(path) for path in expected_outputs],
            "input_fingerprint_digest": fingerprint["digest"],
            "input_fingerprint_file": str(fingerprint_path),
        }
    )
    _write_state(state_path, state)
    print(f"\n[01_preprocess] {message}", flush=True)


def _check_preprocess_resume(
    *,
    dataset_dir: Path,
    expected_outputs: list[Path],
    fingerprint: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    adopt_existing: bool,
    dry_run: bool,
) -> bool:
    fingerprint_path = dataset_dir / PREPROCESS_FINGERPRINT_NAME
    complete_on_disk = all(path.exists() for path in expected_outputs)
    dataset_nonempty = dataset_dir.is_dir() and any(dataset_dir.iterdir())

    if not complete_on_disk:
        if dataset_nonempty:
            missing = [str(path) for path in expected_outputs if not path.exists()]
            preview = "\n  ".join(missing[:8])
            raise RuntimeError(
                "发现已有但不完整的预处理目录，--resume 为避免混用数据而拒绝继续。\n"
                f"目录：{dataset_dir}\n"
                f"缺失输出（前几个）：\n  {preview}\n"
                "请确认后删除该旧目录，或为本次实验更换 dataset_output_dir。"
            )
        return False

    if fingerprint_path.is_file():
        stored = _read_json(fingerprint_path)
        stored_digest = stored.get("digest")
        if stored_digest != fingerprint["digest"]:
            raise RuntimeError(
                "发现完整的旧预处理结果，但输入指纹与当前配置不一致，拒绝 --resume。\n"
                f"旧指纹：{stored_digest}\n"
                f"当前指纹：{fingerprint['digest']}\n"
                f"指纹文件：{fingerprint_path}\n"
                "说明当前 source_dir / origin_source / phase_dir / teacher / 预处理参数中的至少一项已变化。\n"
                "请为新实验使用新的 scene_name 和 dataset_output_dir；若要重跑同一目录，请先人工确认并清理旧输出。"
            )
        _mark_preprocess_skipped(
            state=state,
            state_path=state_path,
            expected_outputs=expected_outputs,
            fingerprint=fingerprint,
            fingerprint_path=fingerprint_path,
            status="skipped_verified",
            message="输入指纹一致且数据集完整，--resume 安全跳过。",
        )
        return True

    if not adopt_existing:
        raise RuntimeError(
            "发现完整的旧预处理结果，但没有输入指纹文件，无法确认它是否属于当前 BIN/SLM 数据。\n"
            f"目录：{dataset_dir}\n"
            "如果这是你已经人工确认过的历史数据，可仅首次执行：\n"
            "  python workflows/real_experiment/run_pipeline.py --resume --adopt-existing-preprocess\n"
            "该选项只会为当前已存在的数据集登记当前输入指纹，不会重新处理 BIN。\n"
            "对来源不确定的旧目录不要使用该选项。"
        )

    if dry_run:
        _mark_preprocess_skipped(
            state=state,
            state_path=state_path,
            expected_outputs=expected_outputs,
            fingerprint=fingerprint,
            fingerprint_path=fingerprint_path,
            status="would_adopt_existing",
            message="旧数据完整；dry-run 将模拟登记输入指纹并跳过（不会写入指纹文件）。",
        )
        return True

    _write_preprocess_fingerprint(fingerprint_path, fingerprint)
    _mark_preprocess_skipped(
        state=state,
        state_path=state_path,
        expected_outputs=expected_outputs,
        fingerprint=fingerprint,
        fingerprint_path=fingerprint_path,
        status="adopted_existing",
        message="已登记当前输入指纹；现有预处理数据被明确认领，--resume 跳过。",
    )
    return True


def _run_stage(
    *,
    name: str,
    command: list[str],
    expected_outputs: list[Path],
    state: dict[str, Any],
    state_path: Path,
    resume: bool,
    dry_run: bool,
) -> None:
    stage = state.setdefault("stages", {}).setdefault(name, {})
    complete_on_disk = all(path.exists() for path in expected_outputs)
    if resume and complete_on_disk:
        stage.update(
            {
                "status": "skipped_existing",
                "checked_at": _utc_now(),
                "expected_outputs": [str(path) for path in expected_outputs],
            }
        )
        _write_state(state_path, state)
        print(f"\n[{name}] 已完成，--resume 跳过。", flush=True)
        return

    print(f"\n[{name}] 开始", flush=True)
    _print_command(command)
    stage.update(
        {
            "status": "dry_run" if dry_run else "running",
            "started_at": _utc_now(),
            "command": command,
            "expected_outputs": [str(path) for path in expected_outputs],
        }
    )
    _write_state(state_path, state)
    if dry_run:
        return

    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        stage.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "returncode": int(exc.returncode),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_state(state_path, state)
        raise RuntimeError(f"阶段 {name} 失败，退出码 {exc.returncode}。") from exc

    missing = [path for path in expected_outputs if not path.exists()]
    if missing:
        stage.update(
            {
                "status": "failed_missing_output",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "missing_outputs": [str(path) for path in missing],
            }
        )
        _write_state(state_path, state)
        raise RuntimeError(
            f"阶段 {name} 返回成功，但缺少预期输出："
            + ", ".join(str(path) for path in missing)
        )

    stage.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    _write_state(state_path, state)
    print(f"[{name}] 完成", flush=True)


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_bool_flag(command: list[str], flag: str, value: Any) -> None:
    if bool(value):
        command.append(flag)


def _build_preprocess_command(config: dict[str, Any], dataset_dir: Path, scene_name: str) -> list[str]:
    section = config.get("preprocess", {})
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "prepare_reference_guided_real_dataset.py"),
        "--source-dir",
        str(_resolve_path(_required(config, "source_dir"))),
        "--origin-source",
        str(_resolve_path(_required(config, "origin_source"))),
        "--phase-dir",
        str(_resolve_path(_required(config, "phase_dir"))),
        "--output-dir",
        str(dataset_dir),
        "--scene-name",
        scene_name,
        "--backend",
        str(section.get("backend", "cuda")),
        "--chunk-rows",
        str(section.get("chunk_rows", 600)),
    ]
    option_map = {
        "grid_size": "--grid-size",
        "seed": "--seed",
        "measurement_glob": "--measurement-glob",
        "teacher_template": "--teacher-template",
        "teacher_correlation_threshold": "--teacher-correlation-threshold",
        "teacher_refit_threshold": "--teacher-refit-threshold",
        "final_correlation_threshold": "--final-correlation-threshold",
        "minimum_fitted_peak_adc": "--minimum-fitted-peak-adc",
    }
    for key, flag in option_map.items():
        _append_option(command, flag, section.get(key))
    return command


def _build_training_command(config: dict[str, Any], dataset_dir: Path, scene_name: str) -> list[str]:
    section = config.get("training", {})
    root_dir = _resolve_path(section.get("root_dir", "."))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "recon_exp_data.py"),
        "--root_dir",
        str(root_dir),
        "--data_dir",
        str(dataset_dir),
        "--scene_name",
        scene_name,
    ]
    option_map = {
        "num_epochs": "--num_epochs",
        "num_t": "--num_t",
        "batch_size": "--batch_size",
        "width": "--width",
        "vis_freq": "--vis_freq",
        "log_freq": "--log_freq",
        "early_stop_patience": "--early_stop_patience",
        "early_stop_min_delta": "--early_stop_min_delta",
        "early_stop_warmup": "--early_stop_warmup",
        "early_stop_window": "--early_stop_window",
        "init_lr": "--init_lr",
        "final_lr": "--final_lr",
        "num_workers": "--num_workers",
        "max_intensity": "--max_intensity",
        "normalization": "--normalization",
        "im_prefix": "--im_prefix",
        "slm_prefix": "--slm_prefix",
        "zero_freq": "--zero_freq",
        "phs_layers": "--phs_layers",
        "zernike_features": "--zernike_features",
        "device": "--device",
        "seed": "--seed",
    }
    for key, flag in option_map.items():
        _append_option(command, flag, section.get(key))
    for key, flag in {
        "silence_tqdm": "--silence_tqdm",
        "save_per_frame": "--save_per_frame",
        "static_phase": "--static_phase",
        "dynamic_scene": "--dynamic_scene",
    }.items():
        _append_bool_flag(command, flag, section.get(key, False))
    return command


def _build_slm_export_command(config: dict[str, Any], phase_path: Path, output_dir: Path) -> list[str]:
    section = config.get("slm_export", {})
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "export_slm_correction.py"),
        "--input-phase",
        str(phase_path),
        "--output-dir",
        str(output_dir),
    ]
    _append_option(command, "--hardware-size", section.get("hardware_size", 1080))
    return command


def _build_zernike_command(config: dict[str, Any], phase_path: Path, output_dir: Path) -> list[str]:
    section = config.get("zernike", {})
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "fit_recovered_zernike.py"),
        "--input-phase",
        str(phase_path),
        "--output-dir",
        str(output_dir),
    ]
    _append_option(command, "--num-modes", section.get("num_modes", 28))
    _append_option(command, "--hardware-size", section.get("hardware_size", 1080))
    return command


def _preflight(config: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    source_dir = _resolve_path(_required(config, "source_dir"))
    origin = _resolve_path(_required(config, "origin_source"))
    phase_dir = _resolve_path(_required(config, "phase_dir"))
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir 不存在：{source_dir}")
    if not origin.is_file():
        raise FileNotFoundError(f"origin_source 不存在：{origin}")
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"phase_dir 不存在：{phase_dir}")


def run(
    config_path: Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    adopt_existing_preprocess: bool = False,
) -> Path:
    config = _load_config(config_path)
    if adopt_existing_preprocess and not resume:
        raise ValueError("--adopt-existing-preprocess 必须与 --resume 一起使用。")
    _preflight(config, dry_run=dry_run)

    scene_name = str(_required(config, "scene_name"))
    dataset_dir = _resolve_path(_required(config, "dataset_output_dir"))
    training = config.get("training", {})
    root_dir = _resolve_path(training.get("root_dir", "."))
    scene_dir = root_dir / "vis" / scene_name
    final_dir = scene_dir / "final"
    recovered_phase = final_dir / "final_aberration.mat"
    training_summary = final_dir / "training_summary.json"

    slm_section = config.get("slm_export", {})
    slm_output_dir = _resolve_path(
        slm_section.get("output_dir", str(scene_dir / "slm_correction"))
    )
    zernike_section = config.get("zernike", {})
    zernike_output_dir = _resolve_path(
        zernike_section.get("output_dir", str(scene_dir / "zernike_fit"))
    )
    state_path = scene_dir / "pipeline_state.json"

    state: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "real_experiment",
        "scene_name": scene_name,
        "config": str(config_path),
        "updated_at": _utc_now(),
        "stages": {},
    }
    if resume and state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                state.update(previous)
                state["config"] = str(config_path)
        except (OSError, json.JSONDecodeError):
            pass

    preprocess_fingerprint: dict[str, Any] | None = None
    preprocess_outputs = [
        dataset_dir / "manifest.json",
        dataset_dir / "SLM_raw1.mat",
        dataset_dir / "SLM_sim1.mat",
    ]
    if resume or not dry_run:
        preprocess_fingerprint = _build_preprocess_fingerprint(
            config,
            scene_name=scene_name,
            dataset_dir=dataset_dir,
        )
        num_frames = len(preprocess_fingerprint["inputs"]["measurement_files"])
        preprocess_outputs = _preprocess_expected_outputs(dataset_dir, num_frames)

    preprocess_skipped = False
    if resume and preprocess_fingerprint is not None:
        preprocess_skipped = _check_preprocess_resume(
            dataset_dir=dataset_dir,
            expected_outputs=preprocess_outputs,
            fingerprint=preprocess_fingerprint,
            state=state,
            state_path=state_path,
            adopt_existing=adopt_existing_preprocess,
            dry_run=dry_run,
        )

    if not preprocess_skipped:
        _run_stage(
            name="01_preprocess",
            command=_build_preprocess_command(config, dataset_dir, scene_name),
            expected_outputs=preprocess_outputs,
            state=state,
            state_path=state_path,
            resume=False,
            dry_run=dry_run,
        )
        if not dry_run and preprocess_fingerprint is not None:
            fingerprint_path = dataset_dir / PREPROCESS_FINGERPRINT_NAME
            _write_preprocess_fingerprint(fingerprint_path, preprocess_fingerprint)
            stage = state.setdefault("stages", {}).setdefault("01_preprocess", {})
            stage.update(
                {
                    "input_fingerprint_digest": preprocess_fingerprint["digest"],
                    "input_fingerprint_file": str(fingerprint_path),
                }
            )
            _write_state(state_path, state)
    _run_stage(
        name="02_train",
        command=_build_training_command(config, dataset_dir, scene_name),
        expected_outputs=[recovered_phase, training_summary],
        state=state,
        state_path=state_path,
        resume=resume,
        dry_run=dry_run,
    )
    _run_stage(
        name="03_export_slm",
        command=_build_slm_export_command(config, recovered_phase, slm_output_dir),
        expected_outputs=[
            slm_output_dir / "SLM_final_correction_1080.png",
            slm_output_dir / "SLM_final_correction_1080.mat",
            slm_output_dir / "manifest.json",
        ],
        state=state,
        state_path=state_path,
        resume=resume,
        dry_run=dry_run,
    )

    if not dry_run:
        slm_png = slm_output_dir / "SLM_final_correction_1080.png"
        print("\n" + "=" * 72, flush=True)
        print("SLM 校正相位已生成，可以立即加载到 SLM：", flush=True)
        print(str(slm_png), flush=True)
        print("=" * 72 + "\n", flush=True)

    if bool(zernike_section.get("enabled", True)):
        _run_stage(
            name="04_zernike_analysis",
            command=_build_zernike_command(config, recovered_phase, zernike_output_dir),
            expected_outputs=[zernike_output_dir / "zernike_fit_report.json"],
            state=state,
            state_path=state_path,
            resume=resume,
            dry_run=dry_run,
        )

    state["status"] = "dry_run" if dry_run else "completed"
    state["updated_at"] = _utc_now()
    _write_state(state_path, state)
    print(f"\n真实实验 pipeline 完成。状态文件：{state_path}", flush=True)
    return state_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="JSON 配置文件；默认 configs/real_experiment.json。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "恢复运行。预处理阶段只有在输出完整且输入指纹一致时才会跳过；"
            "其余阶段按关键输出恢复。"
        ),
    )
    parser.add_argument(
        "--adopt-existing-preprocess",
        action="store_true",
        help=(
            "仅用于升级前已经人工确认过的旧预处理数据：与 --resume 联用，"
            "首次登记当前输入指纹后跳过预处理。来源不确定的数据禁止使用。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印/记录将要执行的命令，不运行耗时任务。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        _resolve_path(args.config),
        resume=args.resume,
        dry_run=args.dry_run,
        adopt_existing_preprocess=args.adopt_existing_preprocess,
    )


if __name__ == "__main__":
    main()
