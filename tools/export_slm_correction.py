#!/usr/bin/env python3
"""Export a recovered 600-grid NeuWS phase as a 1080-grid SLM correction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_utils import write_wrapped_phase_png  # noqa: E402
from optics import zernike_basis_numpy  # noqa: E402


def _load_phase(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        phase = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".mat":
        values = sio.loadmat(path)
        for variable in ("phase", "reconstructed_aberration_phase", "angle"):
            if variable in values:
                phase = values[variable]
                break
        else:
            raise KeyError(f"{path} 中没有可识别的相位变量。")
    else:
        raise ValueError("输入相位必须是 .npy 或 .mat。")
    phase = np.asarray(phase, dtype=np.float32).squeeze()
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError(f"恢复相位必须是有限二维数组，实际为 {phase.shape}。")
    return phase


def _remove_piston_tip_tilt(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = zernike_basis_numpy(3, phase.shape[0])
    design = basis.reshape(3, -1).T.astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(
        design, phase.reshape(-1).astype(np.float64), rcond=None
    )
    fitted = np.einsum(
        "m,mhw->hw", coefficients.astype(np.float32), basis, optimize=True
    )
    return np.asarray(phase - fitted, dtype=np.float32), coefficients.astype(np.float32)


def _complex_resample_phase(phase: np.ndarray, output_size: int) -> np.ndarray:
    unit_field = np.exp(1j * phase).astype(np.complex64)
    channels = torch.from_numpy(
        np.stack((unit_field.real, unit_field.imag), axis=0)
    ).unsqueeze(0)
    resized = F.interpolate(
        channels,
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=True,
    )[0]
    field = resized[0].numpy() + 1j * resized[1].numpy()
    amplitude = np.abs(field)
    if float(amplitude.min()) <= 1e-8:
        raise RuntimeError("复数相位插值产生了接近零的幅度，无法可靠恢复相位。")
    field /= amplitude
    return np.asarray(np.angle(field), dtype=np.float32)


def export(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_phase).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖校正相位已停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    recovered_phase = _load_phase(input_path)
    if recovered_phase.shape[0] != recovered_phase.shape[1]:
        raise ValueError(f"恢复相位必须是正方形，实际为 {recovered_phase.shape}。")
    corrected_model_phase, ptt_coefficients = _remove_piston_tip_tilt(recovered_phase)
    hardware_phase_signed = _complex_resample_phase(
        corrected_model_phase, args.hardware_size
    )
    hardware_phase_wrapped = np.mod(hardware_phase_signed, 2.0 * math.pi).astype(
        np.float32
    )
    model_phase_wrapped = np.mod(corrected_model_phase, 2.0 * math.pi).astype(
        np.float32
    )

    # With the measured-data convention SLM=exp(-i*Gamma), Gamma=estimated phi
    # cancels system=exp(+i*phi). The stored phase_hw is therefore not negated.
    np.save(output_dir / "SLM_final_correction_1080.npy", hardware_phase_wrapped)
    sio.savemat(
        output_dir / "SLM_final_correction_1080.mat",
        {
            "phase_hw": hardware_phase_wrapped,
            "phase_hw_signed": hardware_phase_signed,
            "slm_field": np.exp(-1j * hardware_phase_signed).astype(np.complex64),
            "removed_noll_1_3_coefficients_rad": ptt_coefficients,
        },
        do_compression=True,
    )
    write_wrapped_phase_png(
        output_dir / "SLM_final_correction_1080.png", hardware_phase_wrapped
    )

    np.save(output_dir / "SLM_final_correction_600.npy", model_phase_wrapped)
    sio.savemat(
        output_dir / "SLM_final_correction_600.mat",
        {
            "proj_sim": corrected_model_phase,
            "proj_sim_wrapped": model_phase_wrapped,
            "removed_noll_1_3_coefficients_rad": ptt_coefficients,
        },
        do_compression=True,
    )

    wrapped_recovered = np.angle(np.exp(1j * recovered_phase))
    wrapped_model = np.angle(np.exp(1j * corrected_model_phase))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = (
        (wrapped_recovered, "Recovered 600 phase"),
        (wrapped_model, "600 phase after Piston/Tip/Tilt removal"),
        (hardware_phase_signed, "Final 1080 SLM correction"),
    )
    for axis, (phase, title) in zip(axes, panels):
        shown = axis.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046)
    fig.suptitle("Final SLM correction phase (all panels wrapped to [-π, π])")
    fig.tight_layout()
    fig.savefig(output_dir / "SLM_final_correction_preview.png", dpi=150)
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "generator": "tools/export_slm_correction.py",
        "source_recovered_phase": str(input_path),
        "source_shape": list(recovered_phase.shape),
        "hardware_shape": [args.hardware_size, args.hardware_size],
        "mapping": (
            "interpolate real and imaginary parts of exp(1j*phase) with bicubic "
            "align_corners=True, renormalize to unit magnitude, then recover angle"
        ),
        "removed_modes": {
            "noll_indices": [1, 2, 3],
            "names": ["Piston", "Tilt", "Tilt"],
            "fitted_coefficients_rad": [float(value) for value in ptt_coefficients],
            "reason": "global phase is irrelevant and recovered tilt is ambiguous with image translation",
        },
        "correction_convention": {
            "estimated_system_field": "exp(+1j * recovered_phase)",
            "slm_field_used_during_reconstruction": "exp(-1j * proj_sim)",
            "command": "phase_hw = recovered_phase with Noll 1-3 removed",
            "expected_product": "system_field * exp(-1j * phase_hw) approximately cancels non-PTT aberration",
        },
        "recommended_hardware_file": "SLM_final_correction_1080.png",
        "mat_file": "SLM_final_correction_1080.mat:phase_hw",
        "phase_units": "radians",
        "phase_range": [0.0, 2.0 * math.pi],
        "png_encoding": {
            "dtype": "uint16",
            "mapping": "wrapped [0,2pi) linearly mapped to [0,65535]",
            "device_lut_applied": False,
            "note": "This matches the generic encoding used for SLM_hw1-SLM_hw50; apply a device LUT only if the acquisition patterns also used one.",
        },
        "orientation": "same orientation as the generated and acquired SLM_hw1-SLM_hw50 patterns; no flip or rotation applied",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-phase", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hardware-size", type=int, default=1080)
    return parser


def main() -> None:
    output = export(build_parser().parse_args())
    print(f"最终SLM校正相位已导出：{output}")


if __name__ == "__main__":
    main()
