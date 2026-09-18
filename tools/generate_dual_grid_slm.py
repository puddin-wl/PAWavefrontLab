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
from aotools.functions.zernike import zernike_noll


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import (
    write_wrapped_phase_png,
    write_wrapped_phase_png_uint8,
)  # noqa: E402
from optics import centered_crop, zernike_basis_numpy  # noqa: E402


DEFAULT_NUM_MODES = 10
DEFAULT_DISABLED_NOLL_INDICES = (2, 3)
DEFOCUS_NOLL_INDEX = 4
FULL_BASIS_MODE_LIMIT = 32


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


def _validate_disabled_noll_indices(
    disabled_noll_indices: tuple[int, ...], num_modes: int
) -> tuple[int, ...]:
    disabled = tuple(disabled_noll_indices)
    if len(disabled) != len(set(disabled)):
        raise ValueError("disabled_noll_indices 不能包含重复项。")
    if any(index < 1 or index > num_modes for index in disabled):
        raise ValueError(f"disabled_noll_indices 必须位于 1 到 {num_modes}。")
    return disabled


def _sample_coefficients(
    num_frames: int,
    num_modes: int,
    sigma: float,
    seed: int,
    disabled_noll_indices: tuple[int, ...],
    coefficient_limit: float | None = None,
) -> np.ndarray:
    mode_sigmas = np.full(num_modes, float(sigma), dtype=np.float64)
    mode_limits = (
        None
        if coefficient_limit is None
        else np.full(num_modes, float(coefficient_limit), dtype=np.float64)
    )
    return _sample_coefficients_per_mode(
        num_frames,
        mode_sigmas,
        seed,
        disabled_noll_indices,
        mode_limits=mode_limits,
    )


def _sample_coefficients_per_mode(
    num_frames: int,
    mode_sigmas: np.ndarray,
    seed: int,
    disabled_noll_indices: tuple[int, ...],
    *,
    mode_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Sample independent truncated Gaussians with one scale per Noll mode."""
    mode_sigmas = np.asarray(mode_sigmas, dtype=np.float64)
    if mode_sigmas.ndim != 1 or mode_sigmas.size == 0:
        raise ValueError("mode_sigmas 必须是一维非空数组。")
    num_modes = int(mode_sigmas.size)
    if num_frames <= 0:
        raise ValueError("num_frames 必须为正整数。")
    if not np.isfinite(mode_sigmas).all() or np.any(mode_sigmas < 0):
        raise ValueError("所有 mode_sigmas 必须是非负有限数值。")
    if mode_limits is not None:
        mode_limits = np.asarray(mode_limits, dtype=np.float64)
        if mode_limits.shape != mode_sigmas.shape:
            raise ValueError("mode_limits 必须与 mode_sigmas 形状一致。")
        if not np.isfinite(mode_limits).all() or np.any(mode_limits <= 0):
            raise ValueError("所有 mode_limits 必须是正有限数值。")
    disabled_noll_indices = _validate_disabled_noll_indices(
        disabled_noll_indices, num_modes
    )
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(num_frames, num_modes)) * mode_sigmas[None, :]
    if mode_limits is not None:
        outside = np.abs(coefficients) > mode_limits[None, :]
        while np.any(outside):
            columns = np.nonzero(outside)[1]
            coefficients[outside] = rng.normal(size=int(outside.sum())) * mode_sigmas[
                columns
            ]
            outside = np.abs(coefficients) > mode_limits[None, :]
    coefficients = coefficients.astype(np.float32)
    for noll_index in disabled_noll_indices:
        coefficients[:, noll_index - 1] = 0.0
    return coefficients


def _noll_radial_order(noll_index: int) -> int:
    """Return the radial order n for a one-based Noll index."""
    if noll_index < 1:
        raise ValueError("noll_index 必须从 1 开始。")
    radial_order = 0
    while noll_index > (radial_order + 1) * (radial_order + 2) // 2:
        radial_order += 1
    return radial_order


def _expand_radial_order_values(
    values: list[float] | tuple[float, ...], num_modes: int, name: str
) -> np.ndarray:
    maximum_order = _noll_radial_order(num_modes)
    if len(values) != maximum_order + 1:
        raise ValueError(
            f"{name} 需要给出径向阶 0..{maximum_order} 的 {maximum_order + 1} 个值。"
        )
    values_array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values_array).all() or np.any(values_array < 0):
        raise ValueError(f"{name} 必须全部是非负有限数值。")
    return np.asarray(
        [values_array[_noll_radial_order(index)] for index in range(1, num_modes + 1)],
        dtype=np.float64,
    )


def _resolve_coefficients(
    args: argparse.Namespace,
    disabled_noll_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict]:
    """Sample coefficients or scale a saved basis and optionally extend it."""
    base_value = getattr(args, "base_coefficients", None)
    if not base_value:
        sigma_by_order = getattr(args, "sigma_by_radial_order", None)
        limit_by_order = getattr(args, "coefficient_limit_by_radial_order", None)
        if limit_by_order is not None and sigma_by_order is None:
            raise ValueError(
                "--coefficient-limit-by-radial-order 必须与 "
                "--sigma-by-radial-order 一起使用。"
            )
        if sigma_by_order is not None:
            mode_sigmas = _expand_radial_order_values(
                sigma_by_order, args.num_modes, "--sigma-by-radial-order"
            )
            mode_limits = (
                None
                if limit_by_order is None
                else _expand_radial_order_values(
                    limit_by_order,
                    args.num_modes,
                    "--coefficient-limit-by-radial-order",
                )
            )
            coefficients = _sample_coefficients_per_mode(
                args.num_frames,
                mode_sigmas,
                args.seed,
                disabled_noll_indices,
                mode_limits=mode_limits,
            )
            maximum_order = _noll_radial_order(args.num_modes)
            return coefficients, {
                "coefficient_distribution": (
                    "independent zero-mean Gaussian with radial-order-dependent "
                    "sigma, truncated by rejection before disabled terms are zeroed"
                    if mode_limits is not None
                    else "independent zero-mean Gaussian with radial-order-dependent sigma"
                ),
                "coefficient_sigma_rad": None,
                "coefficient_limit_rad": None,
                "coefficient_sigma_by_radial_order_rad": {
                    str(order): float(sigma_by_order[order])
                    for order in range(maximum_order + 1)
                },
                "coefficient_limit_by_radial_order_rad": (
                    None
                    if limit_by_order is None
                    else {
                        str(order): float(limit_by_order[order])
                        for order in range(maximum_order + 1)
                    }
                ),
                "seed": int(args.seed),
            }
        coefficient_limit = getattr(args, "coefficient_limit", None)
        coefficients = _sample_coefficients(
            args.num_frames,
            args.num_modes,
            args.sigma,
            args.seed,
            disabled_noll_indices,
            coefficient_limit,
        )
        return coefficients, {
            "coefficient_distribution": (
                "independent zero-mean Gaussian truncated by rejection before "
                "disabled terms are zeroed"
                if coefficient_limit is not None
                else "independent Gaussian before disabled terms are zeroed"
            ),
            "coefficient_sigma_rad": float(args.sigma),
            "coefficient_limit_rad": (
                float(coefficient_limit) if coefficient_limit is not None else None
            ),
            "seed": int(args.seed),
        }

    base_path = Path(base_value).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"基础系数文件不存在：{base_path}")
    if base_path.suffix.lower() != ".npy":
        raise ValueError("--base-coefficients 当前只支持 .npy 文件。")
    base = np.asarray(np.load(base_path, allow_pickle=False), dtype=np.float32)
    if base.ndim != 2 or base.shape[0] != args.num_frames:
        raise ValueError(
            f"基础系数应为 ({args.num_frames}, modes)，实际为 {base.shape}。"
        )
    if base.shape[1] > args.num_modes:
        raise ValueError(
            f"基础系数已有 {base.shape[1]} 项，超过目标 {args.num_modes} 项。"
        )
    if not np.isfinite(base).all():
        raise ValueError("基础系数包含 NaN 或无穷值。")
    scale = float(getattr(args, "coefficient_scale", 1.0))
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("--coefficient-scale 必须是非负有限数值。")

    base_mode_count = int(base.shape[1])
    coefficients = np.zeros(
        (args.num_frames, args.num_modes), dtype=np.float32
    )
    coefficients[:, :base_mode_count] = base * np.float32(scale)
    extension_count = args.num_modes - base_mode_count
    extension_sigma = getattr(args, "extension_sigma", None)
    extension_seed = getattr(args, "extension_seed", None)
    if extension_count:
        extension_sigma = (
            float(args.sigma)
            if extension_sigma is None
            else float(extension_sigma)
        )
        extension_seed = (
            int(args.seed) if extension_seed is None else int(extension_seed)
        )
        if not np.isfinite(extension_sigma) or extension_sigma < 0:
            raise ValueError("--extension-sigma 必须是非负有限数值。")
        rng = np.random.default_rng(extension_seed)
        coefficients[:, base_mode_count:] = rng.normal(
            0.0,
            extension_sigma,
            size=(args.num_frames, extension_count),
        ).astype(np.float32)
    for noll_index in disabled_noll_indices:
        coefficients[:, noll_index - 1] = 0.0

    metadata = {
        "coefficient_distribution": (
            "scaled saved coefficients with independent Gaussian extension"
            if extension_count
            else "scaled saved coefficients"
        ),
        "coefficient_sigma_rad": None,
        "seed": None,
        "base_coefficient_source": str(base_path),
        "base_mode_count": base_mode_count,
        "base_coefficient_scale": scale,
        "extension_noll_range": (
            [base_mode_count + 1, args.num_modes] if extension_count else None
        ),
        "extension_sigma_rad": extension_sigma if extension_count else None,
        "extension_seed": extension_seed if extension_count else None,
    }
    return coefficients, metadata


def _write_coefficients(
    output_dir: Path, coefficients: np.ndarray, noll_indices: tuple[int, ...]
) -> None:
    np.save(output_dir / "slm_coefficients.npy", coefficients)
    sio.savemat(
        output_dir / "slm_coefficients.mat",
        {
            "slm_coefficients": coefficients,
            "noll_indices": np.asarray(noll_indices, dtype=np.int16),
        },
        do_compression=True,
    )
    with (output_dir / "slm_coefficients.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", *[f"noll_{index}_rad" for index in noll_indices]])
        for frame, row in enumerate(coefficients, start=1):
            writer.writerow([frame, *[f"{float(value):.9g}" for value in row]])


def _synthesize_phases_low_memory(
    coefficients: np.ndarray, size: int, block_size: int
) -> np.ndarray:
    """Accumulate one Zernike mode at a time without retaining a full basis stack."""
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数。")
    diameter = int(math.ceil(math.sqrt(2.0 * size * size)))
    phases = np.zeros((coefficients.shape[0], size, size), dtype=np.float32)
    for mode_offset in range(coefficients.shape[1]):
        weights = coefficients[:, mode_offset]
        if not np.any(weights):
            continue
        mode = np.asarray(
            centered_crop(zernike_noll(mode_offset + 1, diameter), size, size),
            dtype=np.float32,
        )
        for start in range(0, coefficients.shape[0], block_size):
            stop = min(start + block_size, coefficients.shape[0])
            phases[start:stop] += weights[start:stop, None, None] * mode
    return phases


def _synthesize_phases(
    coefficients: np.ndarray, size: int, block_size: int
) -> tuple[np.ndarray, str]:
    if coefficients.shape[1] <= FULL_BASIS_MODE_LIMIT:
        basis = zernike_basis_numpy(coefficients.shape[1], size)
        phases = np.einsum("fm,mhw->fhw", coefficients, basis, optimize=True)
        return np.asarray(phases, dtype=np.float32), "full_basis_einsum"
    return (
        _synthesize_phases_low_memory(coefficients, size, block_size),
        "one_mode_at_a_time_blocked_accumulation",
    )


def _render_grid(
    output_dir: Path,
    *,
    size: int,
    coefficients: np.ndarray,
    filename_prefix: str,
    mat_variable: str,
    block_size: int,
    hardware_png_uint8: bool = False,
) -> str:
    mat_dir = output_dir / "mat"
    png_dir = output_dir / (
        "png_uint8_wrapped" if hardware_png_uint8 else "png_uint16_wrapped"
    )
    mat_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"正在计算 {size}×{size}、{coefficients.shape[1]} 项 Zernike 相位……",
        flush=True,
    )
    phases, strategy = _synthesize_phases(coefficients, size, block_size)
    for frame, phase in enumerate(phases, start=1):
        sio.savemat(
            mat_dir / f"{filename_prefix}{frame}.mat",
            {mat_variable: phase},
            do_compression=True,
        )
        if hardware_png_uint8:
            write_wrapped_phase_png_uint8(
                png_dir / f"{filename_prefix}{frame}.png",
                phase,
            )
        else:
            write_wrapped_phase_png(
                png_dir / f"{filename_prefix}{frame}.png",
                phase,
            )
    return strategy


def generate(args: argparse.Namespace) -> Path:
    hardware_size = _positive_even(args.hardware_size, "hardware_size")
    model_size = _positive_even(args.model_size, "model_size")
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_directory(output_dir)
    noll_indices = tuple(range(1, args.num_modes + 1))
    disabled_noll_indices = _validate_disabled_noll_indices(
        tuple(args.disabled_noll_indices), args.num_modes
    )

    coefficients, coefficient_metadata = _resolve_coefficients(
        args, disabled_noll_indices
    )
    _write_coefficients(output_dir, coefficients, noll_indices)

    model_dir = output_dir / f"model_{model_size}"
    model_strategy = _render_grid(
        model_dir,
        size=model_size,
        coefficients=coefficients,
        filename_prefix="SLM_sim",
        mat_variable="proj_sim",
        block_size=args.render_block_size,
    )
    hardware_dir = output_dir / f"hardware_{hardware_size}"
    hardware_strategy = _render_grid(
        hardware_dir,
        size=hardware_size,
        coefficients=coefficients,
        filename_prefix="SLM_hw",
        mat_variable="phase_hw",
        block_size=args.render_block_size,
        hardware_png_uint8=True,
    )

    active_noll_indices = [
        index for index in noll_indices if index not in disabled_noll_indices
    ]
    nonpiston_columns = [
        index - 1 for index in active_noll_indices if index >= DEFOCUS_NOLL_INDEX
    ]
    complete_radial_order = int(round((-3.0 + math.sqrt(1 + 8 * args.num_modes)) / 2.0))
    if (complete_radial_order + 1) * (complete_radial_order + 2) // 2 != args.num_modes:
        complete_radial_order = None
    manifest = {
        "schema_version": 1,
        "generator": "tools/generate_dual_grid_slm.py",
        "num_frames": int(args.num_frames),
        "hardware_shape": [hardware_size, hardware_size],
        "model_shape": [model_size, model_size],
        "noll_indices": list(noll_indices),
        "complete_maximum_radial_order": complete_radial_order,
        "active_noll_indices": active_noll_indices,
        "disabled_noll_indices": list(disabled_noll_indices),
        "disabled_terms": {
            str(index): (
                "Piston"
                if index == 1
                else (
                    "X/Y Tilt"
                    if index in (2, 3)
                    else ("Defocus" if index == DEFOCUS_NOLL_INDEX else "user-disabled")
                )
            )
            for index in disabled_noll_indices
        },
        "defocus_noll_index": DEFOCUS_NOLL_INDEX,
        "defocus_included": DEFOCUS_NOLL_INDEX not in disabled_noll_indices,
        "piston_included": 1 not in disabled_noll_indices,
        **coefficient_metadata,
        "nonpiston_coefficient_norm_mean_rad": float(
            np.linalg.norm(coefficients[:, nonpiston_columns], axis=1).mean()
        ),
        "same_coefficients_on_both_grids": True,
        "render_strategy": {
            "model": model_strategy,
            "hardware": hardware_strategy,
            "frame_block_size": int(args.render_block_size),
        },
        "phase_units": "radians",
        "phase_sign_convention": "NeuWS uses aperture * exp(-1j * proj_sim)",
        "hardware_output": {
            "mat": f"hardware_{hardware_size}/mat/SLM_hwN.mat:phase_hw",
            "png": f"hardware_{hardware_size}/png_uint8_wrapped/SLM_hwN.png",
        },
        "model_output": {
            "mat": f"model_{model_size}/mat/SLM_simN.mat:proj_sim",
            "png": f"model_{model_size}/png_uint16_wrapped/SLM_simN.png",
        },
        "png_phase_encoding": {
            "wrap_interval_rad": [0.0, 2.0 * math.pi],
            "model_integer_range": [0, 65535],
            "hardware_integer_range": [0, 255],
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
    parser.add_argument("--num-modes", type=int, default=DEFAULT_NUM_MODES)
    parser.add_argument(
        "--disabled-noll-indices",
        nargs="*",
        type=int,
        default=list(DEFAULT_DISABLED_NOLL_INDICES),
    )
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument(
        "--sigma-by-radial-order",
        nargs="+",
        type=float,
        help=(
            "Optional Gaussian sigma for every radial order n=0..n_max. "
            "Overrides --sigma for newly sampled coefficients."
        ),
    )
    parser.add_argument(
        "--coefficient-limit",
        type=float,
        help=(
            "Optional absolute coefficient limit in radians. Gaussian samples "
            "outside [-limit, limit] are rejected and resampled."
        ),
    )
    parser.add_argument(
        "--coefficient-limit-by-radial-order",
        nargs="+",
        type=float,
        help=(
            "Optional absolute coefficient limit for every radial order n=0..n_max. "
            "Used with --sigma-by-radial-order."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--base-coefficients",
        help=(
            "Optional saved [frames,modes] .npy coefficients to scale and reuse. "
            "If the target has more modes, the remaining terms are sampled separately."
        ),
    )
    parser.add_argument("--coefficient-scale", type=float, default=1.0)
    parser.add_argument("--extension-sigma", type=float)
    parser.add_argument("--extension-seed", type=int)
    parser.add_argument("--render-block-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = generate(args)
    print(f"已生成 {args.num_frames} 组双网格 SLM 相位：{output_dir}")


if __name__ == "__main__":
    main()
