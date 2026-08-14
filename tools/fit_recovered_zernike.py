#!/usr/bin/env python3
"""Fit a recovered NeuWS phase to AOtools Noll Zernike modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_wrapped_phase_png  # noqa: E402
from optics import zernike_basis_numpy  # noqa: E402


NOLL_NAMES = {
    1: "Piston",
    2: "Tilt X/Y 1",
    3: "Tilt X/Y 2",
    4: "Defocus",
    5: "Astigmatism 1",
    6: "Astigmatism 2",
    7: "Coma 1",
    8: "Coma 2",
    9: "Trefoil 1",
    10: "Trefoil 2",
}


def _load_phase(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        phase = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".mat":
        values = sio.loadmat(path)
        for name in ("phase", "reconstructed_aberration_phase", "angle"):
            if name in values:
                phase = values[name]
                break
        else:
            raise KeyError(f"{path} 中没有 phase/reconstructed_aberration_phase/angle。")
    else:
        raise ValueError("输入必须是 .npy 或 .mat 相位文件。")
    phase = np.asarray(phase, dtype=np.float32).squeeze()
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
        raise ValueError(f"相位必须是正方形二维数组，实际为 {phase.shape}。")
    if not np.isfinite(phase).all():
        raise ValueError("相位包含 NaN 或无穷值。")
    return phase


def _fit(phase: np.ndarray, num_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = zernike_basis_numpy(num_modes, phase.shape[0])
    design = basis.reshape(num_modes, -1).T.astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(
        design, phase.reshape(-1).astype(np.float64), rcond=None
    )
    fitted = np.einsum(
        "m,mhw->hw", coefficients.astype(np.float32), basis, optimize=True
    ).astype(np.float32)
    return coefficients.astype(np.float32), fitted, basis


def _statistics(phase: np.ndarray, fitted: np.ndarray) -> dict:
    residual = np.asarray(phase - fitted, dtype=np.float32)
    wrapped_residual = np.angle(np.exp(1j * residual)).astype(np.float32)
    centered_phase = phase - float(phase.mean())
    total_energy = float(np.sum(centered_phase.astype(np.float64) ** 2))
    residual_energy = float(np.sum(residual.astype(np.float64) ** 2))
    coherence = float(np.abs(np.mean(np.exp(1j * wrapped_residual))))
    return {
        "unwrapped_rmse_rad": float(np.sqrt(np.mean(residual**2))),
        "unwrapped_mae_rad": float(np.mean(np.abs(residual))),
        "unwrapped_max_abs_rad": float(np.max(np.abs(residual))),
        "wrapped_rmse_rad": float(np.sqrt(np.mean(wrapped_residual**2))),
        "wrapped_mae_rad": float(np.mean(np.abs(wrapped_residual))),
        "wrapped_rmse_deg": math.degrees(float(np.sqrt(np.mean(wrapped_residual**2)))),
        "wrapped_max_abs_rad": float(np.max(np.abs(wrapped_residual))),
        "phase_variance_explained_r2": (
            float(1.0 - residual_energy / total_energy) if total_energy > 0 else None
        ),
        "complex_field_coherence": coherence,
        "coherence_squared_strehl_proxy": coherence**2,
    }


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_phase).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拟合输出目录非空，为避免覆盖已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_modes <= 0:
        raise ValueError("num_modes 必须为正整数。")

    recovered = _load_phase(input_path)
    coefficients, fitted, _ = _fit(recovered, args.num_modes)
    residual = np.asarray(recovered - fitted, dtype=np.float32)
    wrapped_recovered = np.angle(np.exp(1j * recovered)).astype(np.float32)
    wrapped_fitted = np.angle(np.exp(1j * fitted)).astype(np.float32)
    wrapped_residual = np.angle(np.exp(1j * residual)).astype(np.float32)
    statistics = _statistics(recovered, fitted)

    # Build the hardware candidate directly from the fitted coefficients. Noll
    # 1-3 are removed for the command because piston is irrelevant and tilt is
    # ambiguous with image translation in NeuWS.
    hardware_basis = zernike_basis_numpy(args.num_modes, args.hardware_size)
    command_coefficients = coefficients.copy()
    command_coefficients[: min(3, args.num_modes)] = 0.0
    hardware_phase_signed = np.einsum(
        "m,mhw->hw", command_coefficients, hardware_basis, optimize=True
    ).astype(np.float32)
    hardware_phase_wrapped = np.mod(hardware_phase_signed, 2.0 * np.pi).astype(np.float32)

    np.save(output_dir / "zernike_coefficients_noll_1_10.npy", coefficients)
    np.save(output_dir / "zernike_fitted_phase_600.npy", fitted)
    np.save(output_dir / "zernike_residual_phase_600.npy", residual)
    np.save(output_dir / "zernike_fitted_slm_candidate_1080.npy", hardware_phase_wrapped)
    sio.savemat(
        output_dir / "zernike_fit_result.mat",
        {
            "noll_indices": np.arange(1, args.num_modes + 1, dtype=np.int16),
            "zernike_coefficients_rad": coefficients,
            "recovered_phase": recovered,
            "zernike_fitted_phase": fitted,
            "residual_phase": residual,
            "wrapped_residual_phase": wrapped_residual,
            "slm_candidate_phase_hw": hardware_phase_wrapped,
            "slm_candidate_coefficients_rad": command_coefficients,
        },
        do_compression=True,
    )
    write_wrapped_phase_png(
        output_dir / "zernike_fitted_slm_candidate_1080.png", hardware_phase_wrapped
    )

    with (output_dir / "zernike_coefficients_noll_1_10.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["Noll index", "Mode", "Coefficient (rad)"])
        for index, coefficient in enumerate(coefficients, start=1):
            writer.writerow(
                [index, NOLL_NAMES.get(index, f"Noll {index}"), f"{float(coefficient):.9g}"]
            )

    report = {
        "source_phase": str(input_path),
        "source_shape": list(recovered.shape),
        "basis": "AOtools Noll-normalized Zernike basis, identical to NeuWS pattern generation",
        "fit_method": "ordinary linear least squares on the saved continuous/unwrapped phase",
        "num_modes": args.num_modes,
        "coefficients_rad": [
            {
                "noll_index": index,
                "name": NOLL_NAMES.get(index, f"Noll {index}"),
                "coefficient": float(coefficient),
            }
            for index, coefficient in enumerate(coefficients, start=1)
        ],
        "fit_statistics": statistics,
        "slm_candidate": {
            "file": "zernike_fitted_slm_candidate_1080.png",
            "shape": [args.hardware_size, args.hardware_size],
            "noll_1_3_removed": True,
            "status": "experimental candidate; does not replace the already validated full-phase correction",
        },
        "interpretation_note": (
            "The 50 measurement modulations used Noll 1-10, but that does not force the "
            "unknown system aberration recovered by the MLP to lie in the same 10-mode subspace."
        ),
    }
    (output_dir / "zernike_fit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = (
        (wrapped_recovered, "Recovered phase", "twilight", -np.pi, np.pi),
        (wrapped_fitted, f"Noll 1–{args.num_modes} fitted phase", "twilight", -np.pi, np.pi),
        (wrapped_residual, "Wrapped residual", "twilight", -np.pi, np.pi),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat[:3], panels):
        shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, label="rad")

    coefficient_axis = axes.flat[3]
    indices = np.arange(1, args.num_modes + 1)
    bars = coefficient_axis.bar(indices, coefficients, color="#1769aa")
    coefficient_axis.axhline(0, color="black", linewidth=0.8)
    coefficient_axis.set_xticks(indices)
    coefficient_axis.set_xlabel("Noll index")
    coefficient_axis.set_ylabel("Coefficient (rad)")
    coefficient_axis.set_title("Fitted Zernike coefficients")
    coefficient_axis.grid(True, axis="y", alpha=0.25)
    coefficient_axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    fig.suptitle(
        f"10-mode fit: wrapped RMSE {statistics['wrapped_rmse_rad']:.3f} rad, "
        f"R² {statistics['phase_variance_explained_r2']:.3f}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "zernike_fit_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    positive = coefficients >= 0
    bar_colors = np.where(positive, "#1769aa", "#d1495b")
    bars = axis.bar(indices, coefficients, color=bar_colors, width=0.72)
    axis.axhline(0, color="black", linewidth=1.0)
    axis.set_xticks(indices)
    axis.set_xlabel("Noll index", fontsize=13)
    axis.set_ylabel("Coefficient (rad)", fontsize=13)
    axis.set_title("Fitted Zernike coefficients (Noll 1–10)", fontsize=16)
    axis.grid(True, axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    margin = max(0.5, float(np.max(np.abs(coefficients))) * 0.12)
    axis.set_ylim(float(coefficients.min()) - margin, float(coefficients.max()) + margin)
    fig.tight_layout()
    fig.savefig(output_dir / "zernike_coefficients_noll_1_10.png", dpi=240)
    plt.close(fig)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-phase", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-modes", type=int, default=10)
    parser.add_argument("--hardware-size", type=int, default=1080)
    return parser


def main() -> None:
    output = run(build_parser().parse_args())
    print(f"Zernike拟合完成：{output}")


if __name__ == "__main__":
    main()
