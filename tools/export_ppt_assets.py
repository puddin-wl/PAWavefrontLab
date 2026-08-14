#!/usr/bin/env python3
"""Export clean, numbered PNG assets for a NeuWS experiment presentation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _minmax(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).squeeze()
    minimum, maximum = float(image.min()), float(image.max())
    if image.ndim != 2 or not np.isfinite(image).all() or maximum <= minimum:
        raise ValueError(f"无法归一化图像，形状/范围为 {image.shape}, {minimum}, {maximum}。")
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def _write_grayscale(path: Path, image: np.ndarray) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path)


def export(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    reconstruction_dir = Path(args.reconstruction_dir).expanduser().resolve()
    validation_dir = Path(args.validation_dir).expanduser().resolve()
    correction_dir = Path(args.correction_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"PPT素材目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    origin_raw = np.load(data_dir / "reference" / "origin_projection.npy")
    defocus_raw = np.load(data_dir / "reference" / "defocus_projection.npy")
    restored_raw = np.load(data_dir / "reference" / "restored_projection.npy")
    origin = _minmax(origin_raw)
    defocus = _minmax(defocus_raw)
    restored = _minmax(restored_raw)
    computational = _minmax(
        np.load(reconstruction_dir / "reconstructed_object_normalized.npy")
    )
    phase = np.load(reconstruction_dir / "reconstructed_aberration_phase.npy")
    slm_phase = np.load(correction_dir / "SLM_final_correction_1080.npy")
    training = json.loads(
        (reconstruction_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (validation_dir / "final_validation_report.json").read_text(encoding="utf-8")
    )

    _write_grayscale(output_dir / "01_origin_ground_truth.png", origin)
    _write_grayscale(output_dir / "02_defocus_baseline.png", defocus)
    _write_grayscale(output_dir / "03_computational_reconstruction.png", computational)
    _write_grayscale(output_dir / "04_optically_restored.png", restored)

    wrapped_phase = np.angle(np.exp(1j * phase))
    fig, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(wrapped_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axis.set_title("Recovered system-aberration phase", fontsize=15)
    axis.axis("off")
    colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Phase (rad)")
    fig.tight_layout()
    fig.savefig(output_dir / "05_recovered_system_aberration_phase.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    shown = axis.imshow(slm_phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    axis.set_title("Final 1080×1080 SLM correction phase", fontsize=15)
    axis.axis("off")
    colorbar = fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Wrapped command phase (rad)")
    fig.tight_layout()
    fig.savefig(output_dir / "06_final_slm_correction_phase.png", dpi=200)
    plt.close(fig)

    loss = np.asarray(training["loss_history"], dtype=np.float64)
    early = training.get("early_stopping", {})
    best_epoch = early.get("best_epoch")
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, loss.size + 1)
    axis.plot(epochs, loss, color="#1769aa", linewidth=1.3, label="Epoch mean MSE")
    if best_epoch:
        axis.axvline(best_epoch, color="#d32f2f", linestyle="--", linewidth=1.5)
        axis.scatter(
            [best_epoch], [loss[best_epoch - 1]], color="#d32f2f", zorder=3,
            label=f"Best smoothed state: epoch {best_epoch}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title(f"NeuWS training loss (early stop at epoch {loss.size})")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_training_loss.png", dpi=200)
    plt.close(fig)

    labels = ["Defocus", "Optical\nrestoration", "Computational\nreconstruction"]
    psnr = [
        metrics["defocus_baseline"]["registered"]["psnr_db"],
        metrics["optically_restored"]["registered"]["psnr_db"],
        metrics["computational_reconstruction"]["registered"]["psnr_db"],
    ]
    ssim = [
        metrics["defocus_baseline"]["registered"]["ssim"],
        metrics["optically_restored"]["registered"]["ssim"],
        metrics["computational_reconstruction"]["registered"]["ssim"],
    ]
    colors = ["#9e9e9e", "#2e7d32", "#1769aa"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(labels, psnr, color=colors)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Registered PSNR versus origin")
    axes[0].set_ylim(0, max(psnr) * 1.2)
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    bars = axes[1].bar(labels, ssim, color=colors)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Registered SSIM versus origin")
    axes[1].set_ylim(0, 1.0)
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Final optical correction validation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "08_psnr_ssim_comparison.png", dpi=200)
    plt.close(fig)

    shutil.copy2(
        validation_dir / "final_validation_comparison.png",
        output_dir / "09_final_validation_overview.png",
    )
    shutil.copy2(
        evaluation_dir / "reconstruction_comparison.png",
        output_dir / "10_computational_reconstruction_overview.png",
    )
    shutil.copy2(
        correction_dir / "SLM_final_correction_preview.png",
        output_dir / "11_slm_correction_derivation.png",
    )

    shared_max = max(float(origin_raw.max()), float(defocus_raw.max()), float(restored_raw.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    shared_panels = (
        (origin_raw, "Origin"),
        (defocus_raw, "Defocus"),
        (restored_raw, "Optically restored"),
    )
    for axis, (image, title) in zip(axes, shared_panels):
        axis.imshow(image / shared_max, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Experimental acquisitions on one shared intensity scale (max={shared_max:.0f})")
    fig.tight_layout()
    fig.savefig(output_dir / "12_experimental_shared_intensity_scale.png", dpi=200)
    plt.close(fig)

    shutil.copy2(
        validation_dir / "final_validation_report.json",
        output_dir / "13_final_metrics.json",
    )

    optical = metrics["optically_restored"]["registered"]
    baseline = metrics["defocus_baseline"]["registered"]
    digital = metrics["computational_reconstruction"]["registered"]
    readme = f"""# NeuWS 2026-08-12 PPT素材说明

## 建议展示顺序

1. `01_origin_ground_truth.png`：无故意离焦的参考真值。
2. `02_defocus_baseline.png`：故意离焦后的基准图。
3. `03_computational_reconstruction.png`：NeuWS计算恢复图。
4. `04_optically_restored.png`：把恢复相位加载到SLM后重新采集的图。
5. `05_recovered_system_aberration_phase.png`：网络恢复的系统像差相位。
6. `06_final_slm_correction_phase.png`：最终加载到1080×1080 SLM的校正相位。
7. `07_training_loss.png`：训练损失与自动早停。
8. `08_psnr_ssim_comparison.png`：定量指标柱状图。
9. `09_final_validation_overview.png`：最终验证总览，适合结论页。
10. `10_computational_reconstruction_overview.png`：计算重建过程总览。
11. `11_slm_correction_derivation.png`：600相位到1080校正相位的过程。
12. `12_experimental_shared_intensity_scale.png`：三张真实采集图使用同一个强度尺度。

## 可直接用于PPT的结论

- Defocus（配准后）：PSNR {baseline['psnr_db']:.2f} dB，SSIM {baseline['ssim']:.3f}。
- 光学校正（配准后）：PSNR {optical['psnr_db']:.2f} dB，SSIM {optical['ssim']:.3f}。
- 计算重建（配准后）：PSNR {digital['psnr_db']:.2f} dB，SSIM {digital['ssim']:.3f}。
- 光学校正相对Defocus提升：PSNR +{optical['psnr_db'] - baseline['psnr_db']:.2f} dB，SSIM +{optical['ssim'] - baseline['ssim']:.3f}。
- 训练在第{training['num_epochs']}轮停止，并恢复第{best_epoch}轮的最佳平滑损失状态。

## 显示与指标说明

- `01`–`04` 为了清楚展示空间结构，各自独立归一化到 `[0,1]`，不能用亮度直接比较绝对光声强度。
- `12` 使用同一个实验强度上限，可用于比较 `origin / defocus / restored` 的相对亮度。
- PSNR和SSIM均在独立归一化后进行平移配准计算，恢复位移为 `{optical['shift_yx_pixels']}` 像素。
"""
    (output_dir / "PPT素材说明.md").write_text(readme, encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--correction-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output = export(build_parser().parse_args())
    print(f"PPT素材已导出：{output}")


if __name__ == "__main__":
    main()
