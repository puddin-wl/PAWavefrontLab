#!/usr/bin/env python3
"""Export presentation assets for a real-experiment optical-correction review.

The origin acquisition is an uncorrected, system-aberrated baseline rather than
ground truth.  Origin/corrected PSNR and SSIM are exported as similarity metrics
only; they are not evidence of optical-quality improvement.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image


def _minmax(image: np.ndarray, label: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).squeeze()
    minimum, maximum = float(image.min()), float(image.max())
    if image.ndim != 2 or not np.isfinite(image).all() or maximum <= minimum:
        raise ValueError(
            f"无法归一化 {label}，形状/范围为 {image.shape}, {minimum}, {maximum}。"
        )
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def _write_grayscale(path: Path, image: np.ndarray) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path)


def _load_origin(data_dir: Path) -> tuple[np.ndarray, Path]:
    candidates = (
        data_dir / "reference" / "origin_projection.npy",
        data_dir / "reference" / "clear_object.npy",
    )
    for path in candidates:
        if path.is_file():
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32), path
    raise FileNotFoundError(
        "找不到 origin baseline；需要 reference/origin_projection.npy，或兼容文件 "
        "reference/clear_object.npy。"
    )


def _load_corrected(data_dir: Path) -> tuple[np.ndarray, Path]:
    candidates = (
        data_dir / "reference" / "optically_corrected_projection.npy",
        # Legacy output name retained only so old completed runs remain exportable.
        data_dir / "reference" / "restored_projection.npy",
    )
    for path in candidates:
        if path.is_file():
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32), path
    raise FileNotFoundError(
        "找不到 optically corrected acquisition；请先运行 "
        "tools/evaluate_optical_restoration.py。"
    )


def _load_reconstruction(reconstruction_dir: Path) -> np.ndarray:
    normalized_path = reconstruction_dir / "reconstructed_object_normalized.npy"
    if normalized_path.is_file():
        return _minmax(
            np.load(normalized_path, allow_pickle=False),
            "computational NeuWS reconstruction",
        )
    mat_path = reconstruction_dir / "final_I_est_network_units.mat"
    return _minmax(
        sio.loadmat(mat_path)["image"], "computational NeuWS reconstruction"
    )


def _load_phase(reconstruction_dir: Path) -> np.ndarray:
    phase_path = reconstruction_dir / "reconstructed_aberration_phase.npy"
    if phase_path.is_file():
        return np.asarray(np.load(phase_path, allow_pickle=False), dtype=np.float32)
    return np.asarray(
        sio.loadmat(reconstruction_dir / "final_aberration.mat")["phase"],
        dtype=np.float32,
    ).squeeze()


def export(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    reconstruction_dir = Path(args.reconstruction_dir).expanduser().resolve()
    validation_dir = Path(args.validation_dir).expanduser().resolve()
    correction_dir = Path(args.correction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"PPT素材目录非空，为避免覆盖已停止：{output_dir}")

    origin_raw, origin_path = _load_origin(data_dir)
    corrected_raw, corrected_path = _load_corrected(data_dir)
    origin = _minmax(origin_raw, "origin baseline")
    corrected = _minmax(corrected_raw, "optically corrected acquisition")
    computational = _load_reconstruction(reconstruction_dir)
    phase = _load_phase(reconstruction_dir)
    slm_phase = np.asarray(
        np.load(correction_dir / "SLM_final_correction_1080.npy", allow_pickle=False),
        dtype=np.float32,
    )
    training = json.loads(
        (reconstruction_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (validation_dir / "final_validation_report.json").read_text(encoding="utf-8")
    )
    similarity = metrics["origin_corrected_similarity"]
    selected = similarity["selected"]
    quality = metrics["optical_quality_metrics"]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_grayscale(output_dir / "01_origin_baseline.png", origin)
    _write_grayscale(output_dir / "02_computational_reconstruction.png", computational)
    _write_grayscale(output_dir / "03_optically_corrected_acquisition.png", corrected)

    wrapped_phase = np.angle(np.exp(1j * phase))
    fig, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(wrapped_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axis.set_title("Recovered system-aberration phase", fontsize=15)
    axis.axis("off")
    colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Phase (rad)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_recovered_system_aberration_phase.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(slm_phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    axis.set_title("Final 1080×1080 SLM correction phase", fontsize=15)
    axis.axis("off")
    colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Wrapped command phase (rad)")
    fig.tight_layout()
    fig.savefig(output_dir / "05_final_slm_correction_phase.png", dpi=200)
    plt.close(fig)

    loss = np.asarray(training["loss_history"], dtype=np.float64)
    early = training.get("early_stopping", {})
    best_epoch = early.get("best_epoch")
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, loss.size + 1)
    axis.plot(epochs, loss, color="#1769aa", linewidth=1.3, label="Epoch mean MSE")
    if best_epoch and 1 <= int(best_epoch) <= loss.size:
        axis.axvline(best_epoch, color="#d32f2f", linestyle="--", linewidth=1.5)
        axis.scatter(
            [best_epoch],
            [loss[int(best_epoch) - 1]],
            color="#d32f2f",
            zorder=3,
            label=f"Best smoothed state: epoch {best_epoch}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title(f"NeuWS training loss ({loss.size} epochs run)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_training_loss.png", dpi=200)
    plt.close(fig)

    mode_label = (
        "Registered overlap"
        if selected["mode"] == "registered_overlap"
        else "Same coordinates"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    psnr_bar = axes[0].bar([mode_label], [selected["psnr_db"]], color="#5e35b1")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Origin/corrected similarity")
    axes[0].bar_label(psnr_bar, fmt="%.2f", padding=3)
    ssim_bar = axes[1].bar([mode_label], [selected["ssim"]], color="#00897b")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Origin/corrected similarity")
    axes[1].bar_label(ssim_bar, fmt="%.3f", padding=3)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Similarity only — not an optical-quality improvement score", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "07_origin_corrected_similarity.png", dpi=200)
    plt.close(fig)

    shutil.copy2(
        validation_dir / "final_validation_comparison.png",
        output_dir / "08_origin_corrected_similarity_overview.png",
    )
    shutil.copy2(
        correction_dir / "SLM_final_correction_preview.png",
        output_dir / "09_slm_correction_derivation.png",
    )

    shared_max = max(float(origin_raw.max()), float(corrected_raw.max()))
    if not np.isfinite(shared_max) or shared_max <= 0:
        raise ValueError("origin/re 共用强度尺度的最大值必须为有限正数。")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    shared_panels = (
        (origin_raw, "Origin baseline\n(uncorrected, system-aberrated)"),
        (corrected_raw, "Optically corrected acquisition"),
    )
    for axis, (image, title) in zip(axes, shared_panels):
        axis.imshow(image / shared_max, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"Experimental acquisitions on one shared intensity scale (max={shared_max:.0f})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "10_experimental_shared_intensity_scale.png", dpi=200)
    plt.close(fig)

    shutil.copy2(
        validation_dir / "final_validation_report.json",
        output_dir / "11_optical_correction_consistency_report.json",
    )

    shift = selected["proposed_shift_yx_pixels"]
    registration_text = (
        "位移在阈值内，采用 registered-overlap comparison。"
        if selected["registration_accepted"]
        else "位移超过阈值，拒绝配准结果并采用 same-coordinate comparison。"
    )
    readme = f"""# NeuWS 真实实验 PPT 素材说明

## 物理语义

- origin 是系统自身带像差时采集的 uncorrected/system-aberrated baseline，不是 clear ground truth。
- s01–sNN 是系统像差与已知 SLM 人工调制共同作用的 NeuWS 反演输入，不是最终验证 baseline。
- computational reconstruction 只作为反演一致性参考。
- optically corrected acquisition 是加载恢复系统像差的反相校正后重新采集的 re。

## 建议展示顺序

1. `01_origin_baseline.png`：未校正、带系统像差的 origin baseline。
2. `02_computational_reconstruction.png`：NeuWS 计算重建的一致性参考。
3. `03_optically_corrected_acquisition.png`：加载反相校正后重新采集的图像。
4. `04_recovered_system_aberration_phase.png`：网络恢复的系统像差相位。
5. `05_final_slm_correction_phase.png`：最终 SLM 校正相位。
6. `06_training_loss.png`：训练损失与最佳状态。
7. `07_origin_corrected_similarity.png`：origin/re 相似度，不是改善分数。
8. `08_origin_corrected_similarity_overview.png`：origin/re 一致性总览。
9. `09_slm_correction_derivation.png`：SLM 校正相位预览。
10. `10_experimental_shared_intensity_scale.png`：两次真实采集使用同一个强度尺度。

## 当前可报告结果

- 选定比较：`{selected['mode']}`。{registration_text}
- origin/re PSNR：{selected['psnr_db']:.2f} dB；SSIM：{selected['ssim']:.3f}；候选位移：{shift} 像素。
- 上述 PSNR/SSIM 只表示 origin 与 re 的相似度，不能据此声称光学校正质量提高。
- 光学质量指标状态：`{quality['status']}`。当前没有可靠 ROI、血管分割或 line profile，因此未计算 contrast、CNR、edge sharpness、line profile 或 FWHM。
- 训练运行 {training.get('num_epochs', loss.size)} 轮；最佳平滑损失状态为第 {best_epoch} 轮。

## 显示说明

- `01`–`03` 为了显示空间结构而各自归一化，不能用其亮度比较绝对 PA 强度。
- `10` 才使用 origin/re 共用强度尺度，但仍不能代替明确 ROI/profile 的光学质量指标。
- origin 输入：`{origin_path.name}`；校正后输入：`{corrected_path.name}`。`clear_object.npy` 若出现，仅是兼容文件名，物理上仍代表 origin baseline。
"""
    (output_dir / "PPT素材说明.md").write_text(readme, encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--correction-dir", required=True)
    parser.add_argument(
        "--evaluation-dir",
        help="Deprecated compatibility option; no longer used by the exporter.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output = export(build_parser().parse_args())
    print(f"PPT素材已导出：{output}")


if __name__ == "__main__":
    main()
