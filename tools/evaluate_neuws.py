#!/usr/bin/env python3
"""Evaluate NeuWS image or phase reconstruction results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluate_images, evaluate_phases  # noqa: E402
from optics import aperture_mask, validate_geometry  # noqa: E402


def _load(path_value: str, variable: str | None) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".mat":
        if not variable:
            raise ValueError(f"A variable name is required for MATLAB file {path}.")
        values = sio.loadmat(path)
        if variable not in values:
            raise KeyError(f"{path} does not contain {variable!r}.")
        return np.asarray(values[variable]).squeeze()
    with Image.open(path) as image:
        return np.asarray(image)


def _phase_and_mask(value: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    value = np.asarray(value).squeeze()
    if np.iscomplexobj(value):
        return np.angle(value), np.abs(value) > 1e-8
    return value.astype(np.float64), None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _write_json(output_dir: Path, result: dict) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8"
    )


def evaluate_image_command(args) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _load(args.ground_truth, args.ground_truth_var)
    estimate = _load(args.estimate, args.estimate_var)
    result, registered = evaluate_images(reference, estimate, register=args.register)
    reference = np.asarray(reference).squeeze()
    estimate = np.asarray(estimate).squeeze()
    if np.issubdtype(reference.dtype, np.integer):
        reference = reference / np.iinfo(reference.dtype).max
    if np.issubdtype(estimate.dtype, np.integer):
        estimate = estimate / np.iinfo(estimate.dtype).max
    columns = 6 if registered else 3
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    panels = [reference, estimate, np.abs(reference - estimate)]
    titles = ["Ground truth", "Estimate", "Absolute error"]
    if registered:
        ref_view, est_view = registered
        panels.extend([ref_view, est_view, np.abs(ref_view - est_view)])
        titles.extend(["Registered truth", "Registered estimate", "Registered error"])
    for axis, panel, title in zip(np.atleast_1d(axes), panels, titles):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)
    _write_json(output_dir, result)
    print(json.dumps(_json_safe(result), indent=2))


def evaluate_phase_command(args) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, reference_mask = _phase_and_mask(
        _load(args.ground_truth, args.ground_truth_var)
    )
    estimate, estimate_mask = _phase_and_mask(_load(args.estimate, args.estimate_var))
    if reference.shape != estimate.shape:
        raise ValueError(f"Phase shapes differ: {reference.shape} and {estimate.shape}.")
    if args.aperture_height is not None:
        geometry = validate_geometry(reference.shape[0], args.aperture_height)
        mask = aperture_mask(geometry.size, geometry.aperture_height).numpy().astype(bool)
    elif reference_mask is not None:
        mask = reference_mask
    elif estimate_mask is not None:
        mask = estimate_mask
    else:
        mask = np.ones(reference.shape, dtype=bool)
    result, primary, diagnostic = evaluate_phases(reference, estimate, mask)
    masked_primary = np.ma.array(primary, mask=~mask)
    masked_diagnostic = np.ma.array(diagnostic, mask=~mask)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [reference, estimate, masked_primary, masked_diagnostic]
    titles = ["Ground-truth phase", "Estimated phase", "PTT-removed error", "PTT+defocus+coma error"]
    for axis, panel, title in zip(axes, panels, titles):
        image = axis.imshow(panel, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_comparison.png", dpi=150)
    plt.close(fig)
    sio.savemat(
        output_dir / "phase_errors.mat",
        {"mask": mask, "primary_error": primary, "diagnostic_error": diagnostic},
    )
    _write_json(output_dir, result)
    print(json.dumps(_json_safe(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    image = subparsers.add_parser("image")
    image.add_argument("--ground-truth", required=True)
    image.add_argument("--ground-truth-var", default="object_image")
    image.add_argument("--estimate", required=True)
    image.add_argument("--estimate-var", default="image")
    image.add_argument("--output-dir", required=True)
    image.add_argument("--register", action="store_true")
    image.set_defaults(func=evaluate_image_command)
    phase = subparsers.add_parser("phase")
    phase.add_argument("--ground-truth", required=True)
    phase.add_argument("--ground-truth-var", default="aberration_field")
    phase.add_argument("--estimate", required=True)
    phase.add_argument("--estimate-var", default="field")
    phase.add_argument("--aperture-height", type=int)
    phase.add_argument("--output-dir", required=True)
    phase.set_defaults(func=evaluate_phase_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
