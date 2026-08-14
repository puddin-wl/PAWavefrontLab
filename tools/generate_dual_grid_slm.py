#!/usr/bin/env python3
"""Generate paired hardware- and model-grid SLM phase patterns.

The same Zernike coefficient vector is evaluated independently on both grids.
This avoids interpolating wrapped phase images and keeps the hardware pattern
and the NeuWS ``proj_sim`` input physically consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_wrapped_phase_png  # noqa: E402
from optics import zernike_basis_numpy  # noqa: E402


NOLL_INDICES = tuple(range(1, 11))
DISABLED_NOLL_INDICES = (2, 3)
DEFOCUS_NOLL_INDEX = 4


def _positive_even(value: int, name: str) -> int:
    value = int(value)
    if value <= 0 or value % 2:
        raise ValueError(f"{name} 必须是正偶数，实际为 {value}。")
    return value


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"输出目录非空，为避免覆盖已有 SLM 图像已停止：{path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _sample_coefficients(num_frames: int, sigma: float, seed: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames 必须为正整数。")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma 必须是非负有限数值。")
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(
        0.0, sigma, size=(num_frames, len(NOLL_INDICES))
    ).astype(np.float32)
    for noll_index in DISABLED_NOLL_INDICES:
        coefficients[:, noll_index - 1] = 0.0
    return coefficients


def _write_coefficients(output_dir: Path, coefficients: np.ndarray) -> None:
    np.save(output_dir / "slm_coefficients.npy", coefficients)
    sio.savemat(
        output_dir / "slm_coefficients.mat",
        {
            "slm_coefficients": coefficients,
            "noll_indices": np.asarray(NOLL_INDICES, dtype=np.int16),
        },
        do_compression=True,
    )
    with (output_dir / "slm_coefficients.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", *[f"noll_{index}_rad" for index in NOLL_INDICES]])
        for frame, row in enumerate(coefficients, start=1):
            writer.writerow([frame, *[f"{float(value):.9g}" for value in row]])


def _render_grid(
    output_dir: Path,
    *,
    size: int,
    coefficients: np.ndarray,
    filename_prefix: str,
    mat_variable: str,
) -> None:
    mat_dir = output_dir / "mat"
    png_dir = output_dir / "png_uint16_wrapped"
    mat_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    basis = zernike_basis_numpy(len(NOLL_INDICES), size)
    for frame, coefficient in enumerate(coefficients, start=1):
        phase = np.asarray(
            np.einsum("m,mhw->hw", coefficient, basis, optimize=True),
            dtype=np.float32,
        )
        sio.savemat(
            mat_dir / f"{filename_prefix}{frame}.mat",
            {mat_variable: phase},
            do_compression=True,
        )
        write_wrapped_phase_png(
            png_dir / f"{filename_prefix}{frame}.png",
            phase,
        )


def generate(args: argparse.Namespace) -> Path:
    hardware_size = _positive_even(args.hardware_size, "hardware_size")
    model_size = _positive_even(args.model_size, "model_size")
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_directory(output_dir)

    coefficients = _sample_coefficients(args.num_frames, args.sigma, args.seed)
    _write_coefficients(output_dir, coefficients)

    model_dir = output_dir / f"model_{model_size}"
    _render_grid(
        model_dir,
        size=model_size,
        coefficients=coefficients,
        filename_prefix="SLM_sim",
        mat_variable="proj_sim",
    )
    hardware_dir = output_dir / f"hardware_{hardware_size}"
    _render_grid(
        hardware_dir,
        size=hardware_size,
        coefficients=coefficients,
        filename_prefix="SLM_hw",
        mat_variable="phase_hw",
    )

    active_noll_indices = [
        index for index in NOLL_INDICES if index not in DISABLED_NOLL_INDICES
    ]
    manifest = {
        "schema_version": 1,
        "generator": "tools/generate_dual_grid_slm.py",
        "num_frames": int(args.num_frames),
        "hardware_shape": [hardware_size, hardware_size],
        "model_shape": [model_size, model_size],
        "noll_indices": list(NOLL_INDICES),
        "active_noll_indices": active_noll_indices,
        "disabled_noll_indices": list(DISABLED_NOLL_INDICES),
        "disabled_terms": {"2": "X/Y Tilt", "3": "X/Y Tilt"},
        "defocus_noll_index": DEFOCUS_NOLL_INDEX,
        "piston_included": True,
        "coefficient_distribution": "independent Gaussian before disabled terms are zeroed",
        "coefficient_sigma_rad": float(args.sigma),
        "seed": int(args.seed),
        "same_coefficients_on_both_grids": True,
        "phase_units": "radians",
        "phase_sign_convention": "NeuWS uses aperture * exp(-1j * proj_sim)",
        "hardware_output": {
            "mat": f"hardware_{hardware_size}/mat/SLM_hwN.mat:phase_hw",
            "png": f"hardware_{hardware_size}/png_uint16_wrapped/SLM_hwN.png",
        },
        "model_output": {
            "mat": f"model_{model_size}/mat/SLM_simN.mat:proj_sim",
            "png": f"model_{model_size}/png_uint16_wrapped/SLM_simN.png",
        },
        "png_phase_encoding": {
            "wrap_interval_rad": [0.0, 2.0 * math.pi],
            "integer_range": [0, 65535],
            "hardware_lut_applied": False,
            "warning": "Apply the device-specific SLM phase LUT before hardware use.",
        },
        "coefficients": {
            "npy": "slm_coefficients.npy",
            "mat": "slm_coefficients.mat:slm_coefficients",
            "csv": "slm_coefficients.csv",
            "shape": list(coefficients.shape),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "slm_dual_50_noll1_10_no_tilt"),
    )
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--hardware-size", type=int, default=1080)
    parser.add_argument("--model-size", type=int, default=600)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = generate(args)
    print(f"已生成 {args.num_frames} 组双网格 SLM 相位：{output_dir}")


if __name__ == "__main__":
    main()
