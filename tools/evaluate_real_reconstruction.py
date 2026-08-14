#!/usr/bin/env python3
"""Postprocess and evaluate a real NeuWS reconstruction against origin and defocus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluate_images  # noqa: E402
from image_utils import write_unit_png, write_wrapped_phase_png  # noqa: E402


def _minmax(image: np.ndarray, label: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{label} 必须是有限二维数组，实际为 {image.shape}。")
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        raise ValueError(f"{label} 没有有效强度范围。")
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def evaluate(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    final_dir = Path(args.reconstruction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"评价目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = np.load(data_dir / "reference" / "clear_object.npy", allow_pickle=False)
    defocus_raw = np.load(
        data_dir / "reference" / "defocus_projection.npy", allow_pickle=False
    )
    estimate_values = sio.loadmat(final_dir / "final_I_est_network_units.mat")
    estimate_raw = np.asarray(estimate_values["image"], dtype=np.float32).squeeze()
    aberration_values = sio.loadmat(final_dir / "final_aberration.mat")
    phase = np.asarray(aberration_values["phase"], dtype=np.float32).squeeze()
    field = np.asarray(aberration_values["field"]).squeeze().astype(np.complex64)
    training = json.loads(
        (final_dir / "training_summary.json").read_text(encoding="utf-8")
    )

    defocus = _minmax(defocus_raw, "defocus")
    reconstruction = _minmax(estimate_raw, "reconstruction")
    ground_truth = _minmax(ground_truth, "origin ground truth")
    defocus_metrics, defocus_registered = evaluate_images(
        ground_truth, defocus, register=True
    )
    reconstruction_metrics, reconstruction_registered = evaluate_images(
        ground_truth, reconstruction, register=True
    )

    np.save(final_dir / "reconstructed_object_normalized.npy", reconstruction)
    sio.savemat(
        final_dir / "reconstructed_object_normalized.mat",
        {"reconstructed_object": reconstruction},
        do_compression=True,
    )
    write_unit_png(final_dir / "reconstructed_object_normalized.png", reconstruction)
    np.save(final_dir / "reconstructed_aberration_phase.npy", phase)
    np.save(final_dir / "reconstructed_aberration_field.npy", field)
    write_wrapped_phase_png(final_dir / "reconstructed_aberration_phase.png", phase)

    if defocus_registered is not None:
        np.save(output_dir / "registered_origin_for_defocus.npy", defocus_registered[0])
        np.save(output_dir / "registered_defocus.npy", defocus_registered[1])
    if reconstruction_registered is not None:
        np.save(
            output_dir / "registered_origin_for_reconstruction.npy",
            reconstruction_registered[0],
        )
        np.save(
            output_dir / "registered_reconstruction.npy", reconstruction_registered[1]
        )

    defocus_registered_values = defocus_metrics["registered"]
    reconstruction_registered_values = reconstruction_metrics["registered"]
    report = {
        "scene_name": args.scene_name,
        "comparison_normalization": (
            "origin, defocus and reconstruction are independently min-max normalized "
            "to [0,1]; this evaluates spatial reconstruction, not absolute radiometry"
        ),
        "defocus_baseline": defocus_metrics,
        "reconstruction": reconstruction_metrics,
        "registered_improvement_over_defocus": {
            "psnr_db": float(
                reconstruction_registered_values["psnr_db"]
                - defocus_registered_values["psnr_db"]
            ),
            "ssim": float(
                reconstruction_registered_values["ssim"]
                - defocus_registered_values["ssim"]
            ),
        },
        "training": {
            "epochs_completed": training.get("num_epochs"),
            "requested_epochs": training.get("requested_num_epochs"),
            "batch_size": training.get("batch_size"),
            "phase_layers": training.get("phase_layers"),
            "elapsed_seconds": training.get("elapsed_seconds"),
            "initial_loss": training.get("loss_history", [None])[0],
            "minimum_epoch_loss": min(training.get("loss_history", [float("nan")])),
            "early_stopping": training.get("early_stopping"),
        },
        "phase_evaluation": (
            "not available because this real acquisition has no system-aberration phase ground truth"
        ),
    }
    (output_dir / "reconstruction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    loss_history = np.asarray(training.get("loss_history", []), dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = (
        (ground_truth, "Origin ground truth", "gray", 0.0, 1.0),
        (defocus, "Defocus baseline", "gray", 0.0, 1.0),
        (reconstruction, "NeuWS reconstruction", "gray", 0.0, 1.0),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes[0], panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    wrapped_phase = np.angle(np.exp(1j * phase))
    shown = axes[1, 0].imshow(
        wrapped_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[1, 0].set_title("Recovered system-aberration phase (wrapped)")
    axes[1, 0].axis("off")
    fig.colorbar(shown, ax=axes[1, 0], fraction=0.046)

    if loss_history.size:
        axes[1, 1].plot(np.arange(1, loss_history.size + 1), loss_history)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Mean MSE")
        axes[1, 1].set_title("Training loss")
        axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "Registered comparison\n\n"
        f"Defocus PSNR: {defocus_registered_values['psnr_db']:.2f} dB\n"
        f"NeuWS PSNR: {reconstruction_registered_values['psnr_db']:.2f} dB\n"
        f"Improvement: {report['registered_improvement_over_defocus']['psnr_db']:.2f} dB\n\n"
        f"Defocus SSIM: {defocus_registered_values['ssim']:.3f}\n"
        f"NeuWS SSIM: {reconstruction_registered_values['ssim']:.3f}\n"
        f"Improvement: {report['registered_improvement_over_defocus']['ssim']:.3f}\n\n"
        f"Recovered shift: {reconstruction_registered_values['shift_yx_pixels']} px\n"
        f"Training stopped at epoch {training.get('num_epochs')}",
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.suptitle("Real 600×600 NeuWS reconstruction", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_comparison.png", dpi=150)
    plt.close(fig)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reconstruction"] = {
        "completed": True,
        "directory": str(final_dir),
        "device": training.get("device"),
        "epochs_completed": training.get("num_epochs"),
        "requested_epochs": training.get("requested_num_epochs"),
        "batch_size": training.get("batch_size"),
        "phase_layers": training.get("phase_layers"),
        "early_stopping": training.get("early_stopping"),
    }
    manifest["evaluation"] = {
        "completed": True,
        "directory": str(output_dir),
        "report": "reconstruction_report.json",
        "comparison": "reconstruction_comparison.png",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="real_2026_08_12_600")
    return parser


def main() -> None:
    output = evaluate(build_parser().parse_args())
    print(f"真实数据重建评价完成：{output}")


if __name__ == "__main__":
    main()
