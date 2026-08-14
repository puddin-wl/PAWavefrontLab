#!/usr/bin/env python3
"""Evaluate the final optically restored acquisition against origin and defocus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluate_images  # noqa: E402
from image_utils import write_unit_png  # noqa: E402


def _minmax(image: np.ndarray, label: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{label} 必须是有限二维图像，实际为 {image.shape}。")
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        raise ValueError(f"{label} 没有有效强度范围。")
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def run(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    reconstruction_dir = Path(args.reconstruction_dir).expanduser().resolve()
    restored_tiff = Path(args.restored_tiff).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"最终验证目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    origin = _minmax(
        np.load(data_dir / "reference" / "clear_object.npy", allow_pickle=False),
        "origin",
    )
    defocus = _minmax(
        np.load(
            data_dir / "reference" / "defocus_projection.npy", allow_pickle=False
        ),
        "defocus",
    )
    restored_raw = np.asarray(tifffile.imread(restored_tiff), dtype=np.float32)
    restored = _minmax(restored_raw, "restored")
    computational_raw = np.asarray(
        sio.loadmat(reconstruction_dir / "final_I_est_network_units.mat")["image"],
        dtype=np.float32,
    ).squeeze()
    computational = _minmax(computational_raw, "computational reconstruction")

    defocus_metrics, defocus_registered = evaluate_images(origin, defocus, register=True)
    restored_metrics, restored_registered = evaluate_images(origin, restored, register=True)
    computational_metrics, computational_registered = evaluate_images(
        origin, computational, register=True
    )

    np.save(data_dir / "reference" / "restored_projection.npy", restored_raw)
    np.save(data_dir / "reference" / "restored_normalized.npy", restored)
    sio.savemat(
        data_dir / "reference" / "restored_result.mat",
        {"restored_projection": restored_raw, "restored_normalized": restored},
        do_compression=True,
    )
    write_unit_png(data_dir / "reference" / "restored_normalized.png", restored)

    defocus_reg = defocus_metrics["registered"]
    restored_reg = restored_metrics["registered"]
    computational_reg = computational_metrics["registered"]
    report = {
        "scene_name": args.scene_name,
        "restored_source": str(restored_tiff),
        "preprocessing": "max(max(raw - 2048, 0), axis=depth), matching all other acquisitions",
        "comparison_normalization": (
            "origin, defocus, optical restoration and computational reconstruction "
            "are independently min-max normalized to [0,1]"
        ),
        "origin_raw_reference": str(data_dir / "reference" / "origin_projection.npy"),
        "restored_raw_statistics": {
            "shape": list(restored_raw.shape),
            "minimum": float(restored_raw.min()),
            "maximum": float(restored_raw.max()),
            "mean": float(restored_raw.mean()),
            "standard_deviation": float(restored_raw.std()),
        },
        "defocus_baseline": defocus_metrics,
        "optically_restored": restored_metrics,
        "computational_reconstruction": computational_metrics,
        "registered_improvement_over_defocus": {
            "optically_restored": {
                "psnr_db": float(restored_reg["psnr_db"] - defocus_reg["psnr_db"]),
                "ssim": float(restored_reg["ssim"] - defocus_reg["ssim"]),
            },
            "computational_reconstruction": {
                "psnr_db": float(
                    computational_reg["psnr_db"] - defocus_reg["psnr_db"]
                ),
                "ssim": float(computational_reg["ssim"] - defocus_reg["ssim"]),
            },
        },
        "optical_minus_computational_registered": {
            "psnr_db": float(restored_reg["psnr_db"] - computational_reg["psnr_db"]),
            "ssim": float(restored_reg["ssim"] - computational_reg["ssim"]),
        },
    }
    (output_dir / "final_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if restored_registered is not None:
        restored_origin_view, restored_view = restored_registered
        np.save(output_dir / "registered_origin_for_restored.npy", restored_origin_view)
        np.save(output_dir / "registered_optically_restored.npy", restored_view)
        restored_error = np.abs(restored_origin_view - restored_view)
    else:
        restored_error = np.abs(origin - restored)
    if computational_registered is not None:
        np.save(
            output_dir / "registered_origin_for_computational.npy",
            computational_registered[0],
        )
        np.save(
            output_dir / "registered_computational_reconstruction.npy",
            computational_registered[1],
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = (
        (origin, "Origin ground truth"),
        (defocus, "Defocus baseline"),
        (restored, "Optically restored acquisition"),
        (computational, "Computational NeuWS reconstruction"),
    )
    for axis, (image, title) in zip(axes.flat[:4], panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    error_axis = axes[1, 1]
    shown = error_axis.imshow(restored_error, cmap="magma", vmin=0, vmax=1)
    error_axis.set_title("Registered optical-restoration absolute error")
    error_axis.axis("off")
    fig.colorbar(shown, ax=error_axis, fraction=0.046)

    text_axis = axes[1, 2]
    text_axis.axis("off")
    text_axis.text(
        0.02,
        0.98,
        "Registered comparison with origin\n\n"
        f"Defocus:  {defocus_reg['psnr_db']:.2f} dB, SSIM {defocus_reg['ssim']:.3f}\n"
        f"Optical:  {restored_reg['psnr_db']:.2f} dB, SSIM {restored_reg['ssim']:.3f}\n"
        f"Digital:  {computational_reg['psnr_db']:.2f} dB, SSIM {computational_reg['ssim']:.3f}\n\n"
        f"Optical improvement over defocus:\n"
        f"  +{report['registered_improvement_over_defocus']['optically_restored']['psnr_db']:.2f} dB\n"
        f"  +{report['registered_improvement_over_defocus']['optically_restored']['ssim']:.3f} SSIM\n\n"
        f"Optical shift: {restored_reg['shift_yx_pixels']} px",
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.suptitle("Final experimental validation of SLM optical correction", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "final_validation_comparison.png", dpi=150)
    plt.close(fig)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optical_validation"] = {
        "completed": True,
        "source": str(restored_tiff),
        "output_directory": str(output_dir),
        "report": "final_validation_report.json",
        "comparison": "final_validation_comparison.png",
        "registered_psnr_db": restored_reg["psnr_db"],
        "registered_ssim": restored_reg["ssim"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--restored-tiff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="real_2026_08_12_600")
    return parser


def main() -> None:
    output = run(build_parser().parse_args())
    print(f"最终光学校正验证完成：{output}")


if __name__ == "__main__":
    main()
