#!/usr/bin/env python3
"""Compare an optical correction acquisition with the uncorrected origin baseline.

The origin acquisition is not a clear ground truth. It is the system-aberrated,
uncorrected baseline. PSNR/SSIM therefore quantify image similarity only and must
not be interpreted as optical-correction quality or improvement.
"""

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

from analysis.evaluation import evaluate_images  # noqa: E402
from analysis.image_utils import write_unit_png  # noqa: E402


# Legacy pipeline contract only: this file physically stores the uncorrected,
# system-aberrated origin baseline, not a clear object or ground truth.
LEGACY_ORIGIN_FILENAME = "clear_object.npy"


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
    corrected_value = getattr(args, "corrected_tiff", None) or getattr(
        args, "restored_tiff", None
    )
    if not corrected_value:
        raise ValueError(
            "缺少 optically corrected acquisition；请提供 --corrected-tiff。"
        )
    corrected_tiff = Path(corrected_value).expanduser().resolve()
    if not corrected_tiff.is_file():
        raise FileNotFoundError(
            f"optically corrected acquisition 不存在：{corrected_tiff}"
        )
    max_registration_shift = int(getattr(args, "max_registration_shift", 50))
    if max_registration_shift < 0:
        raise ValueError("max_registration_shift 必须为非负整数。")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"一致性评估输出目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin_path = data_dir / "reference" / LEGACY_ORIGIN_FILENAME
    if not origin_path.is_file():
        raise FileNotFoundError(
            "origin baseline 不存在；当前兼容文件名应为 "
            f"{origin_path}（legacy filename，不代表 clear ground truth）。"
        )
    origin_raw = np.asarray(np.load(origin_path, allow_pickle=False), dtype=np.float32)
    origin = _minmax(origin_raw, "origin baseline")
    corrected_raw = np.asarray(tifffile.imread(corrected_tiff), dtype=np.float32)
    corrected = _minmax(corrected_raw, "optically corrected acquisition")
    computational_raw = np.asarray(
        sio.loadmat(reconstruction_dir / "final_I_est_network_units.mat")["image"],
        dtype=np.float32,
    ).squeeze()
    computational = _minmax(computational_raw, "computational reconstruction")

    corrected_metrics, corrected_registered = evaluate_images(
        origin, corrected, register=True
    )
    computational_origin_metrics, computational_registered = evaluate_images(
        origin, computational, register=True
    )
    computational_corrected_metrics, _ = evaluate_images(
        corrected, computational, register=False
    )
    corrected_shift = corrected_metrics["registered"]["shift_yx_pixels"]
    corrected_registration_accepted = (
        max(abs(value) for value in corrected_shift) <= max_registration_shift
    )
    corrected_selected = (
        corrected_metrics["registered"]
        if corrected_registration_accepted
        else corrected_metrics["raw"]
    )
    selected_mode = (
        "registered_overlap"
        if corrected_registration_accepted
        else "same_coordinates"
    )

    np.save(
        data_dir / "reference" / "optically_corrected_projection.npy", corrected_raw
    )
    np.save(
        data_dir / "reference" / "optically_corrected_normalized.npy", corrected
    )
    sio.savemat(
        data_dir / "reference" / "optically_corrected_result.mat",
        {
            "optically_corrected_projection": corrected_raw,
            "optically_corrected_normalized": corrected,
        },
        do_compression=True,
    )
    write_unit_png(
        data_dir / "reference" / "optically_corrected_normalized.png", corrected
    )

    report = {
        "schema_version": 2,
        "scene_name": args.scene_name,
        "evaluation_scope": "similarity and consistency review; not a ground-truth quality evaluation",
        "physical_semantics": {
            "origin_baseline": (
                "uncorrected acquisition containing the system's native aberration; "
                "not a clear image and not ground truth"
            ),
            "optically_corrected_acquisition": (
                "re-acquisition after applying the inverse of the NeuWS-recovered "
                "system aberration on the SLM"
            ),
            "computational_neuws_reconstruction": (
                "computational consistency reference from the same inverse problem"
            ),
            "psnr_ssim": (
                "similarity measures only; higher similarity to origin does not prove "
                "better optical correction"
            ),
        },
        "origin_baseline_source": {
            "file": str(origin_path),
            "legacy_filename": LEGACY_ORIGIN_FILENAME,
            "note": (
                "Legacy filename retained for pipeline compatibility; this file "
                "physically represents the uncorrected, system-aberrated origin baseline."
            ),
        },
        "optically_corrected_source": str(corrected_tiff),
        "preprocessing": getattr(
            args,
            "preprocessing_description",
            "projection supplied by the acquisition workflow",
        ),
        "comparison_normalization": (
            "origin baseline, optically corrected acquisition and computational reconstruction "
            "are independently min-max normalized to [0,1]"
        ),
        "origin_baseline_raw_statistics": {
            "shape": list(origin_raw.squeeze().shape),
            "minimum": float(origin_raw.min()),
            "maximum": float(origin_raw.max()),
            "mean": float(origin_raw.mean()),
            "standard_deviation": float(origin_raw.std()),
        },
        "optically_corrected_raw_statistics": {
            "shape": list(corrected_raw.squeeze().shape),
            "minimum": float(corrected_raw.min()),
            "maximum": float(corrected_raw.max()),
            "mean": float(corrected_raw.mean()),
            "standard_deviation": float(corrected_raw.std()),
        },
        "origin_corrected_similarity": {
            "all_comparisons": corrected_metrics,
            "selected": {
                "mode": selected_mode,
                "psnr_db": corrected_selected["psnr_db"],
                "ssim": corrected_selected["ssim"],
                "registration_accepted": corrected_registration_accepted,
                "proposed_shift_yx_pixels": corrected_shift,
                "max_registration_shift_pixels": max_registration_shift,
            },
            "registration_rejection_reason": (
                None
                if corrected_registration_accepted
                else (
                    "phase-correlation shift exceeds the configured limit of "
                    f"{max_registration_shift} pixels for the same scan geometry"
                )
            ),
            "interpretation": (
                "PSNR/SSIM quantify origin-to-corrected similarity only. They are not "
                "optical-correction quality or improvement scores."
            ),
        },
        "computational_consistency_reference": {
            "to_origin_baseline": computational_origin_metrics,
            "to_optically_corrected_same_coordinates": computational_corrected_metrics[
                "raw"
            ],
            "interpretation": (
                "These comparisons are consistency references and do not treat the "
                "origin baseline as ground truth."
            ),
        },
        "optical_quality_metrics": {
            "status": "not_computed",
            "reason": (
                "No validated vessel/background ROI, vessel segmentation or line-profile "
                "definition is available for this acquisition."
            ),
            "candidate_metrics_requiring_explicit_roi_or_profile": [
                "vessel/background contrast",
                "CNR",
                "edge sharpness or gradient",
                "line profile",
                "FWHM",
            ],
        },
        "conclusion_constraint": (
            "Do not infer better optical correction from higher PSNR/SSIM relative "
            "to the origin baseline."
        ),
    }
    (output_dir / "final_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if corrected_registered is not None and corrected_registration_accepted:
        corrected_origin_view, corrected_view = corrected_registered
        np.save(
            output_dir / "registered_origin_for_optically_corrected.npy",
            corrected_origin_view,
        )
        np.save(output_dir / "registered_optically_corrected.npy", corrected_view)
        origin_corrected_difference = np.abs(corrected_origin_view - corrected_view)
    else:
        origin_corrected_difference = np.abs(origin - corrected)
    np.save(
        output_dir / "origin_corrected_absolute_difference.npy",
        origin_corrected_difference,
    )
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
        (origin, "Origin baseline\n(uncorrected, system-aberrated)"),
        (corrected, "Optically corrected acquisition"),
        (computational, "Computational NeuWS reconstruction\n(consistency reference)"),
    )
    for axis, (image, title) in zip(axes.flat[:3], panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[3].axis("off")

    error_axis = axes[1, 1]
    shown = error_axis.imshow(
        origin_corrected_difference, cmap="magma", vmin=0, vmax=1
    )
    error_axis.set_title(
        "Origin/corrected absolute difference\n"
        + (
            "registered overlap"
            if corrected_registration_accepted
            else "same coordinates"
        )
    )
    error_axis.axis("off")
    fig.colorbar(shown, ax=error_axis, fraction=0.046)

    text_axis = axes[1, 2]
    text_axis.axis("off")
    metric_text = (
        "Origin / corrected similarity\n"
        "(not a quality or improvement score)\n\n"
        f"Mode: {selected_mode}\n"
        f"PSNR: {corrected_selected['psnr_db']:.2f} dB\n"
        f"SSIM: {corrected_selected['ssim']:.3f}\n\n"
        f"Proposed shift: {corrected_shift} px\n"
        f"Registration accepted: {corrected_registration_accepted}\n\n"
        "Higher similarity to origin does not\n"
        "prove better optical correction."
    )
    text_axis.text(
        0.02,
        0.98,
        metric_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.suptitle(
        "Origin baseline vs optically corrected acquisition: similarity review",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "final_validation_comparison.png", dpi=150)
    plt.close(fig)

    manifest["optical_correction_consistency"] = {
        "completed": True,
        "origin_baseline": str(origin_path),
        "optically_corrected_source": str(corrected_tiff),
        "output_directory": str(output_dir),
        "report": "final_validation_report.json",
        "comparison": "final_validation_comparison.png",
        "selected_similarity_mode": selected_mode,
        "selected_psnr_db": corrected_selected["psnr_db"],
        "selected_ssim": corrected_selected["ssim"],
        "interpretation": "similarity only; not optical-correction improvement",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    corrected_group = parser.add_mutually_exclusive_group(required=True)
    corrected_group.add_argument(
        "--corrected-tiff",
        help="Optically corrected re-acquisition after applying inverse recovered aberration.",
    )
    corrected_group.add_argument(
        "--restored-tiff",
        help="Legacy alias for --corrected-tiff.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="real_2026_08_12_600")
    parser.add_argument(
        "--preprocessing-description",
        default="max(max(raw - 2048, 0), axis=depth), matching all other acquisitions",
    )
    parser.add_argument("--max-registration-shift", type=int, default=50)
    return parser


def main() -> None:
    output = run(build_parser().parse_args())
    print(f"光学校正一致性评估完成：{output}")


if __name__ == "__main__":
    main()
