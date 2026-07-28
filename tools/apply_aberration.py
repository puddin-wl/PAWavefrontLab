#!/usr/bin/env python3
"""Apply explicitly specified AOtools Zernike aberrations to one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aberration_config import DEFAULT_NUM_MODES, resolve_coefficients  # noqa: E402
from image_utils import (  # noqa: E402
    read_normalized_square,
    resolve_device,
    write_unit_png,
    write_wrapped_phase_png,
)
from optics import simulate_measurements, validate_geometry, zernike_aberration_from_coefficients  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed used only for optional noise.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--coefficients",
        help="Dense Noll 1..N coefficient list in radians; missing terms are zero-padded to 28.",
    )
    group.add_argument(
        "--coefficients-file",
        help=".npy, JSON, CSV, or text coefficient file.",
    )
    group.add_argument(
        "--coefficient",
        action="append",
        help="Sparse NOLL=RADIANS assignment; repeat this option, for example --coefficient 4=1.5.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.noise_std < 0:
        parser.error("--noise-std must be non-negative")
    try:
        geometry = validate_geometry(args.size, args.size)
        coefficients, coefficient_source = resolve_coefficients(
            dense=args.coefficients,
            file=args.coefficients_file,
            sparse=args.coefficient,
            num_modes=DEFAULT_NUM_MODES,
            required=True,
        )
        device = resolve_device(args.device)
        input_path = Path(args.input_image).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        object_image = read_normalized_square(input_path, geometry.size)
        aberration_field, aberration_phase = zernike_aberration_from_coefficients(
            coefficients,
            geometry.size,
            geometry.size,
            device=device,
        )
        flat_slm = torch.ones(
            (1, geometry.size, geometry.size), dtype=torch.complex64, device=device
        )
        object_tensor = torch.as_tensor(object_image, dtype=torch.float32, device=device)
        measurements, psfs = simulate_measurements(
            object_tensor, aberration_field, flat_slm
        )
        aberrated_image = measurements[0, 0].detach().cpu().numpy().astype(np.float32)
        aberrated_image = np.clip(aberrated_image, 0.0, None)
        if args.noise_std:
            rng = np.random.default_rng(args.seed)
            aberrated_image += rng.normal(
                0.0, args.noise_std, aberrated_image.shape
            ).astype(np.float32)
            aberrated_image = np.clip(aberrated_image, 0.0, None)
        if float(aberrated_image.max()) > 1.0 + 1e-5:
            raise RuntimeError(
                f"Aberrated image exceeds [0,1] with maximum {aberrated_image.max()}."
            )
        phase = aberration_phase.detach().cpu().numpy().astype(np.float32)
        field = aberration_field.detach().cpu().numpy().astype(np.complex64)
        psf = psfs[0, 0].detach().cpu().numpy().astype(np.float32)

        np.save(output_dir / "zernike_coefficients.npy", coefficients)
        np.save(output_dir / "aberration_phase.npy", phase)
        np.save(output_dir / "aberration_field.npy", field)
        np.save(output_dir / "psf.npy", psf)
        np.save(output_dir / "aberrated_image.npy", aberrated_image)
        write_unit_png(output_dir / "input_image.png", object_image)
        write_unit_png(output_dir / "aberrated_image.png", aberrated_image)
        write_wrapped_phase_png(output_dir / "aberration_phase_wrapped.png", phase)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = (
            (object_image, "Input image", "gray"),
            (np.angle(field), "Aberration phase", "twilight"),
            (np.log1p(psf / max(float(psf.max()), 1e-12)), "PSF (log)", "magma"),
            (aberrated_image, "Aberrated image", "gray"),
        )
        for axis, (panel, title, cmap) in zip(axes, panels):
            axis.imshow(panel, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "comparison.png", dpi=150)
        plt.close(fig)
        sio.savemat(
            output_dir / "aberration_result.mat",
            {
                "object_image": object_image,
                "aberrated_image": aberrated_image,
                "zernike_coefficients": coefficients,
                "aberration_phase": phase,
                "aberration_field": field,
                "aberration_amplitude": np.abs(field).astype(np.float32),
                "psf": psf,
            },
            do_compression=True,
        )
        manifest = {
            "schema_version": 1,
            "program": "apply_aberration.py",
            "input_image": str(input_path),
            "size": geometry.size,
            "full_field_illuminated": True,
            "aperture_height": geometry.size,
            "zernike_num_modes": DEFAULT_NUM_MODES,
            "noll_indices": list(range(1, DEFAULT_NUM_MODES + 1)),
            "coefficient_units": "radians",
            "coefficient_source": coefficient_source,
            "zernike_coefficients": coefficients.tolist(),
            "aberration_convention": "exp(+1j * sum(c_j * Z_j))",
            "noise_std": args.noise_std,
            "noise_seed": args.seed,
            "device": str(device),
            "image_preprocessing": "grayscale, center-square crop, bicubic resize, min-max [0,1]",
            "png_encoding": "16-bit linear [0,1] for intensity; wrapped [0,2pi) for phase",
            "presentation_figure": "comparison.png",
            "exact_numeric_outputs": [
                "aberration_result.mat",
                "zernike_coefficients.npy",
                "aberration_phase.npy",
                "aberration_field.npy",
                "psf.npy",
                "aberrated_image.npy",
            ],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        f"Applied 28 Noll coefficients to {input_path.name}; results saved in {output_dir}"
    )


if __name__ == "__main__":
    main()
