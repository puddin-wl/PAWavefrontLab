"""Reusable stages for the static NeuWS simulation and reconstruction workflow."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import tifffile

from evaluation import evaluate_images, evaluate_phases
from image_utils import (
    normalize_square_array,
    read_normalized_square,
    resolve_device,
    write_unit_png,
    write_wrapped_phase_png,
)
from preprocessing.photoacoustic import (
    load_photoacoustic_projection,
    preprocess_photoacoustic_volume,
)
from optics import (
    aperture_mask,
    sample_slm_coefficients,
    simulate_measurements,
    slm_complex_field,
    validate_geometry,
    zernike_aberration_from_coefficients,
    zernike_basis_numpy,
)


WORKFLOW_SCHEMA_VERSION = 1
SYSTEM_NUM_MODES = 28


@dataclass(frozen=True)
class SimulationSettings:
    """All settings shared by the five independently runnable stages."""

    project_root: Path
    input_image: Path
    data_dir: Path
    result_root: Path
    scene_name: str
    size: int = 256
    aperture_height: Optional[int] = None
    num_frames: int = 50
    system_noll_start: int = 4
    system_noll_end: int = 15
    system_num_modes: int = SYSTEM_NUM_MODES
    system_disabled_noll_indices: tuple[int, ...] = ()
    system_sigma: float = 0.6
    system_seed: int = 20260730
    slm_num_modes: int = 15
    slm_disabled_noll_indices: tuple[int, ...] = ()
    slm_sigma: float = 5.0
    slm_seed: int = 20260731
    phase_sign: int = -1
    noise_std: float = 0.0
    noise_seed: int = 20260732
    simulation_batch_size: int = 8
    generation_device: str = "cuda"
    training_device: str = "cuda"
    training_epochs: int = 1000
    training_batch_size: int = 8
    network_zernike_features: int = SYSTEM_NUM_MODES
    phase_layers: int = 4
    initial_learning_rate: float = 1e-3
    final_learning_rate: float = 1e-3
    visualization_frequency: int = 1000
    input_mode: str = "auto"
    photoacoustic_baseline: float = 2048.0
    photoacoustic_projection_axis: int = 0
    raw_measurement_dir: Optional[Path] = None
    overwrite: bool = False
    silence_tqdm: bool = False

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def reconstruction_dir(self) -> Path:
        return self.result_root / "vis" / self.scene_name / "final"

    @property
    def report_dir(self) -> Path:
        return self.result_root / "outputs" / self.scene_name / "evaluation"

    def validate(self) -> None:
        validate_geometry(self.size, self.aperture_height)
        if not Path(self.input_image).is_file():
            raise FileNotFoundError(f"清晰物体图片不存在：{self.input_image}")
        if not self.scene_name.strip():
            raise ValueError("scene_name 不能为空。")
        if self.num_frames <= 0:
            raise ValueError("num_frames 必须为正数。")
        if self.system_num_modes <= 0:
            raise ValueError("system_num_modes 必须为正数。")
        if not 1 <= self.system_noll_start <= self.system_noll_end <= self.system_num_modes:
            raise ValueError(
                f"系统像差 Noll 范围必须位于 1 到 {self.system_num_modes}。"
            )
        if self.system_sigma < 0 or self.slm_sigma < 0 or self.noise_std < 0:
            raise ValueError("所有标准差都必须为非负数。")
        if self.slm_num_modes <= 0:
            raise ValueError("slm_num_modes 必须为正数。")
        if self.network_zernike_features <= 0:
            raise ValueError("network_zernike_features 必须为正数。")
        self._validate_disabled_noll_indices(
            self.system_disabled_noll_indices,
            self.system_num_modes,
            "system_disabled_noll_indices",
        )
        self._validate_disabled_noll_indices(
            self.slm_disabled_noll_indices,
            self.slm_num_modes,
            "slm_disabled_noll_indices",
        )
        if self.phase_sign not in (-1, 1):
            raise ValueError("phase_sign 必须为 -1 或 1。")
        if self.simulation_batch_size <= 0 or self.training_batch_size <= 0:
            raise ValueError("batch size 必须为正数。")
        if self.training_epochs <= 0 or self.phase_layers <= 0:
            raise ValueError("训练轮数和相位网络层数必须为正数。")
        if self.input_mode not in ("auto", "image", "photoacoustic-volume"):
            raise ValueError("input_mode 必须是 auto、image 或 photoacoustic-volume。")
        if not np.isfinite(self.photoacoustic_baseline):
            raise ValueError("photoacoustic_baseline 必须是有限数值。")
        if self.photoacoustic_projection_axis != 0:
            raise ValueError("按照旧代码约定，photoacoustic_projection_axis 必须固定为 0。")

    @staticmethod
    def _validate_disabled_noll_indices(
        indices: tuple[int, ...], num_modes: int, name: str
    ) -> None:
        values = tuple(indices)
        if len(values) != len(set(values)):
            raise ValueError(f"{name} 不能包含重复项。")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise ValueError(f"{name} 必须只包含整数 Noll 编号。")
        if any(index < 1 or index > num_modes for index in values):
            raise ValueError(f"{name} 必须位于 1 到 {num_modes}。")

    def dataset_signature(self) -> dict:
        geometry = validate_geometry(self.size, self.aperture_height)
        signature = {
            "input_image": str(Path(self.input_image).expanduser().resolve()),
            "size": geometry.size,
            "aperture_height": geometry.aperture_height,
            "num_frames": self.num_frames,
            "system_num_modes": self.system_num_modes,
            "system_noll_range": [self.system_noll_start, self.system_noll_end],
            "system_sigma_rad": self.system_sigma,
            "system_seed": self.system_seed,
            "slm_num_modes": self.slm_num_modes,
            "slm_sigma_rad": self.slm_sigma,
            "slm_seed": self.slm_seed,
            "phase_sign": self.phase_sign,
            "noise_std": self.noise_std,
            "noise_seed": self.noise_seed,
        }
        # Keep the default signature byte-for-byte compatible with existing schema-v1 runs.
        if self.system_disabled_noll_indices:
            signature["system_disabled_noll_indices"] = list(
                self.system_disabled_noll_indices
            )
        if self.slm_disabled_noll_indices:
            signature["slm_disabled_noll_indices"] = list(
                self.slm_disabled_noll_indices
            )
        return signature


def sample_system_aberration_coefficients(settings: SimulationSettings) -> np.ndarray:
    """Draw one reproducible system aberration and leave all other Noll terms at zero."""
    settings.validate()
    coefficients = np.zeros(settings.system_num_modes, dtype=np.float32)
    rng = np.random.default_rng(settings.system_seed)
    start = settings.system_noll_start - 1
    stop = settings.system_noll_end
    coefficients[start:stop] = rng.normal(
        0.0, settings.system_sigma, size=stop - start
    ).astype(np.float32)
    if settings.system_disabled_noll_indices:
        disabled = np.asarray(settings.system_disabled_noll_indices, dtype=np.int64) - 1
        coefficients[disabled] = 0.0
    return coefficients


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _manifest_path(settings: SimulationSettings) -> Path:
    return settings.data_dir / "manifest.json"


def _write_manifest(settings: SimulationSettings, manifest: dict) -> None:
    _manifest_path(settings).write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(settings: SimulationSettings, required_step: Optional[str] = None) -> dict:
    path = _manifest_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"缺少工作流清单：{path}，请先运行步骤一。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"不支持的工作流清单版本：{manifest.get('schema_version')}")
    if manifest.get("dataset_settings") != settings.dataset_signature():
        raise ValueError("当前静态仿真 config.py 与已生成数据的配置不一致，请更换运行目录。")
    if required_step and not manifest.get("completed_steps", {}).get(required_step, False):
        raise RuntimeError(f"前置步骤 {required_step} 尚未完成。")
    return manifest


def _refuse_existing(paths: list[Path], settings: SimulationSettings, stage: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not settings.overwrite:
        names = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"{stage} 的输出已经存在：{names}。为防止覆盖，请更换运行目录；"
            "确认需要覆盖时再将 overwrite 设置为 True。"
        )


def _load_ground_truth(settings: SimulationSettings) -> dict:
    path = settings.data_dir / "ground_truth.mat"
    if not path.is_file():
        raise FileNotFoundError(f"缺少真值文件：{path}")
    return sio.loadmat(path)


def _read_clear_object(
    settings: SimulationSettings, size: int
) -> tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """读取普通二维图，或按旧约定预处理三维光声 TIFF。"""
    path = Path(settings.input_image).expanduser().resolve()
    is_tiff = path.suffix.lower() in (".tif", ".tiff")
    use_volume = settings.input_mode == "photoacoustic-volume"
    inspected = None
    if settings.input_mode == "auto" and is_tiff:
        inspected = tifffile.imread(path)
        # H×W×3/4 更可能是普通 RGB(A) TIFF；其他三维 TIFF 按光声体数据处理。
        use_volume = inspected.ndim == 3 and inspected.shape[-1] not in (3, 4)
    if settings.input_mode == "image" or not use_volume:
        clear_object = read_normalized_square(path, size)
        return clear_object, {
            "mode": "ordinary_image",
            "source_file": str(path),
            "output_shape": [size, size],
            "normalization": "individual_min_max_to_unit_interval",
        }, None
    if not is_tiff:
        raise ValueError("photoacoustic-volume 模式只支持 .tif 或 .tiff 文件。")
    if inspected is None:
        projection, metadata = load_photoacoustic_projection(
            path,
            settings.photoacoustic_baseline,
            settings.photoacoustic_projection_axis,
        )
    else:
        projection = preprocess_photoacoustic_volume(
            inspected,
            settings.photoacoustic_baseline,
            settings.photoacoustic_projection_axis,
        )
        metadata = {
            "source_file": str(path),
            "source_shape": [int(value) for value in inspected.shape],
            "source_dtype": str(inspected.dtype),
            "baseline": float(settings.photoacoustic_baseline),
            "negative_policy": "clip_to_zero_after_baseline_subtraction",
            "projection": "maximum_intensity_projection",
            "projection_axis": 0,
            "layer_count_required": None,
            "projected_shape": [int(value) for value in projection.shape],
        }
    clear_object = normalize_square_array(projection, size)
    metadata.update(
        {
            "mode": "photoacoustic_volume",
            "output_shape": [size, size],
            "normalization": "individual_min_max_to_unit_interval_after_projection",
        }
    )
    return clear_object, metadata, projection


def prepare_ground_truth(settings: SimulationSettings) -> dict:
    """Stage 1: save the clear object, fixed aberration and flat-SLM baseline."""
    settings.validate()
    _refuse_existing(
        [_manifest_path(settings), settings.data_dir / "ground_truth.mat", settings.reference_dir],
        settings,
        "步骤一",
    )
    geometry = validate_geometry(settings.size, settings.aperture_height)
    settings.reference_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(settings.generation_device)
    clear_object, input_preprocessing, photoacoustic_projection = _read_clear_object(
        settings, geometry.size
    )
    coefficients = sample_system_aberration_coefficients(settings)
    system_field, system_phase = zernike_aberration_from_coefficients(
        coefficients,
        geometry.size,
        geometry.aperture_height,
        device=device,
    )
    flat_phase = torch.zeros(
        (1, geometry.aperture_height, geometry.size), dtype=torch.float32, device=device
    )
    flat_slm = slm_complex_field(flat_phase, geometry.size, settings.phase_sign)
    object_tensor = torch.as_tensor(clear_object, dtype=torch.float32, device=device)
    baseline, psfs = simulate_measurements(object_tensor, system_field, flat_slm)
    baseline_np = baseline[0, 0].detach().cpu().numpy().astype(np.float32)
    psf_np = psfs[0, 0].detach().cpu().numpy().astype(np.float32)
    phase_np = system_phase.detach().cpu().numpy().astype(np.float32)
    field_np = system_field.detach().cpu().numpy().astype(np.complex64)
    if not all(np.isfinite(value).all() for value in (clear_object, baseline_np, psf_np, phase_np, field_np)):
        raise RuntimeError("步骤一产生了 NaN 或无穷值。")

    np.save(settings.reference_dir / "clear_object.npy", clear_object)
    np.save(settings.reference_dir / "system_aberration_coefficients.npy", coefficients)
    np.save(settings.reference_dir / "system_aberration_phase.npy", phase_np)
    np.save(settings.reference_dir / "system_aberration_field.npy", field_np)
    np.save(settings.reference_dir / "baseline_aberrated_measurement.npy", baseline_np)
    np.save(settings.reference_dir / "baseline_psf.npy", psf_np)
    if photoacoustic_projection is not None:
        np.save(
            settings.reference_dir / "photoacoustic_projection_raw.npy",
            photoacoustic_projection,
        )
    write_unit_png(settings.reference_dir / "clear_object.png", clear_object)
    write_wrapped_phase_png(
        settings.reference_dir / "system_aberration_phase.png", phase_np
    )
    write_unit_png(
        settings.reference_dir / "baseline_aberrated_measurement.png", baseline_np
    )
    psf_display = np.log1p(psf_np / max(float(psf_np.max()), 1e-12)) / math.log(2.0)
    write_unit_png(settings.reference_dir / "baseline_psf.png", psf_display)
    ground_truth_values = {
        "clear_object": clear_object,
        "object_image": clear_object,
        "system_aberration_coefficients": coefficients,
        "aberration_coefficients": coefficients,
        "system_aberration_phase": phase_np,
        "aberration_phase": phase_np,
        "system_aberration_field": field_np,
        "aberration_field": field_np,
        "system_aberration_amplitude": np.abs(field_np).astype(np.float32),
        "baseline_aberrated_measurement": baseline_np,
        "baseline_psf": psf_np,
    }
    if photoacoustic_projection is not None:
        ground_truth_values["photoacoustic_projection_raw"] = photoacoustic_projection
    sio.savemat(
        settings.data_dir / "ground_truth.mat",
        ground_truth_values,
        do_compression=True,
    )
    manifest = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow": "static_neuws_simulation",
        "scene_name": settings.scene_name,
        "size": geometry.size,
        "aperture_height": geometry.aperture_height,
        "num_frames": settings.num_frames,
        "phase_sign": settings.phase_sign,
        "dataset_settings": settings.dataset_signature(),
        "phase_convention": "system exp(+1j*phase), SLM exp(phase_sign*1j*phase)",
        "system_aberration_coefficients_rad": coefficients.tolist(),
        "input_preprocessing": input_preprocessing,
        "completed_steps": {"ground_truth": True, "patterns": False, "measurements": False},
        "outputs": {
            "ground_truth": "ground_truth.mat",
            "reference_directory": "reference",
        },
    }
    if photoacoustic_projection is not None:
        manifest["outputs"]["photoacoustic_projection_raw"] = (
            "reference/photoacoustic_projection_raw.npy"
        )
    _write_manifest(settings, manifest)
    print(f"步骤一完成：固定系统像差和基准模糊图已保存到 {settings.reference_dir}")
    return manifest


def generate_slm_patterns(settings: SimulationSettings) -> dict:
    """Stage 2: generate and persist the known SLM modulation phases."""
    settings.validate()
    manifest = _load_manifest(settings, "ground_truth")
    png_dir = settings.data_dir / "slm_png"
    _refuse_existing(
        [settings.data_dir / "slm_patterns.npy", settings.data_dir / "SLM_sim1.mat", png_dir],
        settings,
        "步骤二",
    )
    geometry = validate_geometry(settings.size, settings.aperture_height)
    png_dir.mkdir(parents=True, exist_ok=True)
    coefficients = sample_slm_coefficients(
        settings.num_frames, settings.slm_num_modes, settings.slm_sigma, settings.slm_seed
    )
    if settings.slm_disabled_noll_indices:
        disabled = np.asarray(settings.slm_disabled_noll_indices, dtype=np.int64) - 1
        coefficients[:, disabled] = 0.0
    basis = zernike_basis_numpy(settings.slm_num_modes, geometry.size)
    full_patterns = np.einsum("fm,mhw->fhw", coefficients, basis, optimize=True)
    top = (geometry.size - geometry.aperture_height) // 2
    patterns = np.asarray(
        full_patterns[:, top : top + geometry.aperture_height, :], dtype=np.float32
    )
    np.save(settings.data_dir / "slm_coefficients.npy", coefficients)
    np.save(settings.data_dir / "slm_patterns.npy", patterns)
    for frame, phase in enumerate(patterns, start=1):
        sio.savemat(
            settings.data_dir / f"SLM_sim{frame}.mat",
            {"proj_sim": phase},
            do_compression=True,
        )
        write_wrapped_phase_png(png_dir / f"slm_phase_{frame:04d}.png", phase)
    manifest["completed_steps"]["patterns"] = True
    manifest["outputs"].update(
        {
            "slm_patterns": "slm_patterns.npy",
            "slm_coefficients": "slm_coefficients.npy",
            "slm_phase_mat": "SLM_simN.mat:proj_sim",
            "slm_phase_previews": "slm_png/slm_phase_NNNN.png",
        }
    )
    _write_manifest(settings, manifest)
    print(f"步骤二完成：已生成 {settings.num_frames} 张已知 SLM 相位，保存在 {settings.data_dir}")
    return manifest


def simulate_modulated_measurements(settings: SimulationSettings) -> dict:
    """Stage 3: synthesize every frame directly from the clear object and total pupil."""
    settings.validate()
    manifest = _load_manifest(settings, "patterns")
    png_dir = settings.data_dir / "measurement_png"
    _refuse_existing(
        [settings.data_dir / "measurements.npy", settings.data_dir / "SLM_raw1.mat", png_dir],
        settings,
        "步骤三",
    )
    geometry = validate_geometry(settings.size, settings.aperture_height)
    patterns = np.load(settings.data_dir / "slm_patterns.npy", allow_pickle=False)
    expected_shape = (settings.num_frames, geometry.aperture_height, geometry.size)
    if patterns.shape != expected_shape or not np.isfinite(patterns).all():
        raise ValueError(f"SLM 相位数组应为 {expected_shape} 且全部有限，实际为 {patterns.shape}。")
    truth = _load_ground_truth(settings)
    clear_object = np.asarray(truth["clear_object"], dtype=np.float32)
    system_field = np.asarray(truth["system_aberration_field"], dtype=np.complex64)
    if clear_object.shape != (geometry.size, geometry.size):
        raise ValueError("清晰物体真值尺寸与当前配置不一致。")
    if system_field.shape != clear_object.shape:
        raise ValueError("系统像差复场尺寸与清晰物体不一致。")
    device = resolve_device(settings.generation_device)
    object_tensor = torch.as_tensor(clear_object, dtype=torch.float32, device=device)
    field_tensor = torch.as_tensor(system_field, dtype=torch.complex64, device=device)
    measurements = np.empty(
        (settings.num_frames, geometry.size, geometry.size), dtype=np.float32
    )
    noise_rng = np.random.default_rng(settings.noise_seed)
    for start in range(0, settings.num_frames, settings.simulation_batch_size):
        stop = min(start + settings.simulation_batch_size, settings.num_frames)
        active_phase = torch.as_tensor(patterns[start:stop], dtype=torch.float32, device=device)
        slm_fields = slm_complex_field(active_phase, geometry.size, settings.phase_sign)
        batch, _ = simulate_measurements(object_tensor, field_tensor, slm_fields)
        batch_np = batch[:, 0].detach().cpu().numpy().astype(np.float32)
        if settings.noise_std:
            batch_np += noise_rng.normal(
                0.0, settings.noise_std, size=batch_np.shape
            ).astype(np.float32)
            batch_np = np.clip(batch_np, 0.0, None)
        if not np.isfinite(batch_np).all():
            raise RuntimeError("步骤三产生了 NaN 或无穷值。")
        measurements[start:stop] = batch_np
    png_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.data_dir / "measurements.npy", measurements)
    for frame, measurement in enumerate(measurements, start=1):
        sio.savemat(
            settings.data_dir / f"SLM_raw{frame}.mat",
            {"imsdata": measurement},
            do_compression=True,
        )
        write_unit_png(
            png_dir / f"modulated_measurement_{frame:04d}.png", measurement
        )
    manifest["measurement_max"] = float(measurements.max())
    manifest["measurement_generation"] = (
        "Each frame is generated directly from clear_object and the combined "
        "fixed-system/known-SLM pupil; baseline_aberrated_measurement is never reused."
    )
    manifest["completed_steps"]["measurements"] = True
    manifest["outputs"].update(
        {
            "measurements": "measurements.npy",
            "measurement_mat": "SLM_rawN.mat:imsdata",
            "measurement_previews": "measurement_png/modulated_measurement_NNNN.png",
        }
    )
    _write_manifest(settings, manifest)
    print(f"步骤三完成：已直接从清晰物体生成 {settings.num_frames} 张调制测量图。")
    return manifest


def _natural_filename_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def import_photoacoustic_measurements(settings: SimulationSettings) -> dict:
    """Alternative stage 3: convert acquired 3-D TIFF frames to NeuWS MAT inputs."""
    settings.validate()
    manifest = _load_manifest(settings, "patterns")
    if settings.raw_measurement_dir is None:
        raise ValueError("请先在 config.py 中设置 raw_measurement_dir。")
    source_dir = Path(settings.raw_measurement_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"三维光声测量目录不存在：{source_dir}")
    files = sorted(
        [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
        ],
        key=_natural_filename_key,
    )
    if len(files) != settings.num_frames:
        raise ValueError(
            f"测量目录必须恰好包含 {settings.num_frames} 个 TIFF，实际找到 {len(files)} 个。"
        )
    png_dir = settings.data_dir / "measurement_png"
    _refuse_existing(
        [settings.data_dir / "measurements.npy", settings.data_dir / "SLM_raw1.mat", png_dir],
        settings,
        "真实光声步骤三",
    )
    expected_shape = (settings.size, settings.size)
    measurements = np.empty((settings.num_frames, *expected_shape), dtype=np.float32)
    source_records = []
    for index, path in enumerate(files):
        projection, metadata = load_photoacoustic_projection(
            path,
            settings.photoacoustic_baseline,
            settings.photoacoustic_projection_axis,
        )
        if projection.shape != expected_shape:
            raise ValueError(
                f"第 {index + 1} 帧 {path.name} 投影后为 {projection.shape}，"
                f"必须与 SLM/网络尺寸 {expected_shape} 一致；真实测量不会自动裁剪或缩放。"
            )
        measurements[index] = projection
        source_records.append(
            {
                "frame": index + 1,
                "source_file": path.name,
                "source_shape": metadata["source_shape"],
                "source_dtype": metadata["source_dtype"],
            }
        )
    measurement_max = float(measurements.max())
    if not np.isfinite(measurement_max) or measurement_max <= 0:
        raise ValueError("全部测量在减去基线并置零后没有正信号，无法归一化训练。")
    png_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.data_dir / "measurements.npy", measurements)
    for frame, measurement in enumerate(measurements, start=1):
        sio.savemat(
            settings.data_dir / f"SLM_raw{frame}.mat",
            {"imsdata": measurement},
            do_compression=True,
        )
        # 所有预览共享同一个最大值，避免每帧独立拉伸后产生误导。
        write_unit_png(
            png_dir / f"modulated_measurement_{frame:04d}.png",
            measurement / measurement_max,
        )
    manifest["measurement_max"] = measurement_max
    manifest["measurement_generation"] = (
        "Acquired 3-D photoacoustic TIFF: max(raw - baseline, 0), then axis-0 "
        "maximum-intensity projection. MAT arrays retain cross-frame intensity scale."
    )
    manifest["photoacoustic_preprocessing"] = {
        "source_directory": str(source_dir),
        "baseline": float(settings.photoacoustic_baseline),
        "negative_policy": "clip_to_zero_after_baseline_subtraction",
        "projection": "maximum_intensity_projection",
        "projection_axis": 0,
        "layer_count_required": None,
        "preview_normalization": "one_dataset_wide_maximum",
        "mat_normalization": "none",
        "frames": source_records,
    }
    manifest["completed_steps"]["measurements"] = True
    manifest["outputs"].update(
        {
            "measurements": "measurements.npy",
            "measurement_mat": "SLM_rawN.mat:imsdata",
            "measurement_previews": "measurement_png/modulated_measurement_NNNN.png",
        }
    )
    _write_manifest(settings, manifest)
    print(
        f"真实光声步骤三完成：已将 {settings.num_frames} 个三维 TIFF 沿第 0 维投影并写入 {settings.data_dir}"
    )
    return manifest


def reconstruct_static_scene(settings: SimulationSettings) -> Path:
    """Stage 4: run the existing static NeuWS inverse model and add semantic outputs."""
    settings.validate()
    manifest = _load_manifest(settings, "measurements")
    final_dir = settings.reconstruction_dir
    _refuse_existing([final_dir], settings, "步骤四")
    resolve_device(settings.training_device)
    command = [
        sys.executable,
        str(settings.project_root / "recon_exp_data.py"),
        "--root_dir",
        str(settings.result_root),
        "--data_dir",
        str(settings.data_dir),
        "--scene_name",
        settings.scene_name,
        "--num_epochs",
        str(settings.training_epochs),
        "--batch_size",
        str(settings.training_batch_size),
        "--phs_layers",
        str(settings.phase_layers),
        "--zernike_features",
        str(settings.network_zernike_features),
        "--init_lr",
        str(settings.initial_learning_rate),
        "--final_lr",
        str(settings.final_learning_rate),
        "--vis_freq",
        str(settings.visualization_frequency),
        "--device",
        settings.training_device,
        "--seed",
        str(settings.system_seed),
        "--static_phase",
    ]
    if settings.silence_tqdm:
        command.append("--silence_tqdm")
    environment = os.environ.copy()
    mpl_cache = settings.data_dir / ".matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    environment["MPLCONFIGDIR"] = str(mpl_cache)
    subprocess.run(command, cwd=settings.project_root, env=environment, check=True)

    image_values = sio.loadmat(final_dir / "final_I_est.mat")
    network_image_path = final_dir / "final_I_est_network_units.mat"
    if network_image_path.is_file():
        network_image_values = sio.loadmat(network_image_path)
        reconstructed_object_network_units = np.asarray(
            network_image_values["image"], dtype=np.float32
        ).squeeze()
    else:
        reconstructed_object_network_units = np.asarray(
            image_values["image"], dtype=np.float32
        ).squeeze()
    aberration_values = sio.loadmat(final_dir / "final_aberration.mat")
    measurement_scale = float(manifest.get("measurement_max", 1.0))
    if not np.isfinite(measurement_scale) or measurement_scale <= 0:
        raise ValueError("manifest.json 中的 measurement_max 必须为有限正数。")
    reconstructed_object = np.clip(
        reconstructed_object_network_units * measurement_scale, 0.0, 1.0
    ).astype(np.float32)
    if "phase" in aberration_values:
        reconstructed_phase = np.asarray(aberration_values["phase"], dtype=np.float32).squeeze()
    else:
        reconstructed_phase = np.angle(aberration_values["field"]).astype(np.float32).squeeze()
    reconstructed_field = np.asarray(aberration_values["field"]).squeeze().astype(np.complex64)
    if not all(
        np.isfinite(value).all()
        for value in (
            reconstructed_object_network_units,
            reconstructed_object,
            reconstructed_phase,
            reconstructed_field,
        )
    ):
        raise RuntimeError("网络恢复结果包含 NaN 或无穷值。")
    sio.savemat(
        final_dir / "reconstructed_object.mat",
        {"reconstructed_object": reconstructed_object},
        do_compression=True,
    )
    np.save(final_dir / "reconstructed_object.npy", reconstructed_object)
    np.save(
        final_dir / "reconstructed_object_network_units.npy",
        reconstructed_object_network_units,
    )
    write_unit_png(final_dir / "reconstructed_object.png", reconstructed_object)
    sio.savemat(
        final_dir / "reconstructed_aberration.mat",
        {
            "reconstructed_aberration_field": reconstructed_field,
            "reconstructed_aberration_phase": reconstructed_phase,
        },
        do_compression=True,
    )
    np.save(final_dir / "reconstructed_aberration_field.npy", reconstructed_field)
    np.save(final_dir / "reconstructed_aberration_phase.npy", reconstructed_phase)
    write_wrapped_phase_png(
        final_dir / "reconstructed_aberration_phase.png", reconstructed_phase
    )
    manifest["reconstruction"] = {
        "completed": True,
        "directory": str(final_dir),
        "device": settings.training_device,
        "epochs": settings.training_epochs,
        "batch_size": settings.training_batch_size,
        "phase_layers": settings.phase_layers,
        "network_zernike_features": settings.network_zernike_features,
        "object_radiometric_scale": measurement_scale,
    }
    _write_manifest(settings, manifest)
    print(f"步骤四完成：网络恢复结果保存在 {final_dir}")
    return final_dir


def evaluate_reconstruction(settings: SimulationSettings) -> dict:
    """Stage 5: compare the reconstruction with image and aberration ground truth."""
    settings.validate()
    manifest = _load_manifest(settings, "measurements")
    if not manifest.get("reconstruction", {}).get("completed", False):
        raise RuntimeError("步骤四尚未完成，无法评估网络恢复结果。")
    output_dir = settings.report_dir
    _refuse_existing([output_dir], settings, "步骤五")
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = _load_ground_truth(settings)
    clear_object = np.asarray(truth["clear_object"], dtype=np.float32).squeeze()
    baseline = np.asarray(
        truth["baseline_aberrated_measurement"], dtype=np.float32
    ).squeeze()
    reference_phase = np.asarray(
        truth["system_aberration_phase"], dtype=np.float32
    ).squeeze()
    final_dir = settings.reconstruction_dir
    reconstructed_object = np.load(final_dir / "reconstructed_object.npy", allow_pickle=False)
    reconstructed_phase = np.load(
        final_dir / "reconstructed_aberration_phase.npy", allow_pickle=False
    )
    baseline_metrics, _ = evaluate_images(clear_object, baseline, register=False)
    reconstruction_metrics, registered = evaluate_images(
        clear_object, reconstructed_object, register=True
    )
    geometry = validate_geometry(settings.size, settings.aperture_height)
    mask = aperture_mask(geometry.size, geometry.aperture_height).numpy().astype(bool)
    phase_metrics, primary_error, diagnostic_error = evaluate_phases(
        reference_phase, reconstructed_phase, mask
    )
    summary_path = final_dir / "training_summary.json"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loss_history = [float(value) for value in training_summary.get("loss_history", [])]
    report = {
        "scene_name": settings.scene_name,
        "baseline_image": baseline_metrics,
        "reconstructed_image": reconstruction_metrics,
        "reconstructed_system_aberration": phase_metrics,
        "training": {
            "epochs": training_summary.get("num_epochs"),
            "elapsed_seconds": training_summary.get("elapsed_seconds"),
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "minimum_loss": min(loss_history) if loss_history else None,
            "batch_size": training_summary.get("batch_size"),
            "network_zernike_features": training_summary.get(
                "network_zernike_features"
            ),
            "peak_cuda_memory_bytes": training_summary.get(
                "peak_cuda_memory_bytes"
            ),
            "peak_cuda_memory_reserved_bytes": training_summary.get(
                "peak_cuda_memory_reserved_bytes"
            ),
            "object_radiometric_scale": manifest.get("measurement_max"),
        },
    }
    (output_dir / "reconstruction_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sio.savemat(
        output_dir / "phase_errors.mat",
        {
            "aperture_mask": mask,
            "piston_tip_tilt_removed_error": primary_error,
            "diagnostic_error": diagnostic_error,
        },
        do_compression=True,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    image_panels = (
        (clear_object, "Clear object ground truth"),
        (baseline, "Flat-SLM aberrated baseline"),
        (reconstructed_object, "Reconstructed object"),
    )
    for axis, (panel, title) in zip(axes[0], image_panels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    phase_panels = (
        (reference_phase, "System aberration ground truth"),
        (reconstructed_phase, "Reconstructed aberration"),
        (np.ma.array(primary_error, mask=~mask), "PTT-removed wrapped error"),
    )
    for axis, (panel, title) in zip(axes[1], phase_panels):
        shown = axis.imshow(panel, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_comparison.png", dpi=150)
    plt.close(fig)

    if registered is not None:
        ref_view, estimate_view = registered
        np.save(output_dir / "registered_clear_object.npy", ref_view)
        np.save(output_dir / "registered_reconstructed_object.npy", estimate_view)
    if loss_history:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(np.arange(1, len(loss_history) + 1), loss_history)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean squared error")
        axis.set_title("NeuWS training loss")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss.png", dpi=150)
        plt.close(fig)
    manifest["evaluation"] = {
        "completed": True,
        "directory": str(output_dir),
        "report": "reconstruction_report.json",
    }
    _write_manifest(settings, manifest)
    print(f"步骤五完成：定量指标和总览图保存在 {output_dir}")
    return report


def compare_reconstruction_with_baseline(
    settings: SimulationSettings, baseline_report_path: Path
) -> dict:
    """Write a compact metric comparison after stage five has completed."""
    report_path = settings.report_dir / "reconstruction_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"缺少当前实验评价报告：{report_path}")
    baseline_report_path = Path(baseline_report_path).expanduser().resolve()
    if not baseline_report_path.is_file():
        raise FileNotFoundError(f"缺少基线评价报告：{baseline_report_path}")
    current_report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))

    def headline_metrics(report: dict) -> dict:
        image = report["reconstructed_image"]["registered"]
        phase = report["reconstructed_system_aberration"][
            "primary_piston_tip_tilt_removed"
        ]
        return {
            "psnr_db": float(image["psnr_db"]),
            "ssim": float(image["ssim"]),
            "phase_rmse_rad": float(phase["rmse_rad"]),
            "training_elapsed_seconds": report.get("training", {}).get(
                "elapsed_seconds"
            ),
        }

    baseline = headline_metrics(baseline_report)
    current = headline_metrics(current_report)
    delta = {
        key: (
            float(current[key]) - float(baseline[key])
            if current[key] is not None and baseline[key] is not None
            else None
        )
        for key in current
    }
    comparison = {
        "baseline_scene_name": baseline_report.get("scene_name"),
        "current_scene_name": current_report.get("scene_name"),
        "baseline": baseline,
        "current": current,
        "delta_current_minus_baseline": delta,
        "comparison_note": (
            "System-aberration expected total coefficient RMS is approximately "
            "matched. The SLM uses the radial-order-15 preview sigma of 1.22 rad, "
            "so this is not a fully equal-strength controlled comparison."
        ),
    }
    comparison_path = settings.report_dir / "comparison_to_test_static_zernike_50.json"
    comparison_path.write_text(
        json.dumps(_json_safe(comparison), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Noll 1–15", "Radial n=15"]
    panels = (
        ("psnr_db", "PSNR (dB)", True),
        ("ssim", "SSIM", True),
        ("phase_rmse_rad", "PTT-removed phase RMSE (rad)", False),
    )
    for axis, (key, title, higher_is_better) in zip(axes, panels):
        values = [baseline[key], current[key]]
        colors = ["#4C78A8", "#F58518"]
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_xlabel(direction)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.5g}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(settings.report_dir / "radial15_vs_baseline.png", dpi=150)
    plt.close(fig)
    return comparison
