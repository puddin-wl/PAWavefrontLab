#!/usr/bin/env python3
"""Smoke benchmark MMES forward/backward at real experiment resolutions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks import DualMMES, StaticDiffuseMMES  # noqa: E402


def _parse_bool(value) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}.")


def _branch_config(args, seed: int) -> dict:
    return {
        "tau": args.tau,
        "ranks": [args.rank1, args.rank2, args.rank3],
        "noise_std": args.noise_std,
        "seed": seed,
        "init_scale": args.init_scale,
        "chunk_size": args.chunk_size,
        "checkpoint_chunks": args.checkpoint_chunks,
    }


def benchmark_size(args, size: int, device: torch.device) -> dict:
    torch.manual_seed(0)
    shared = {
        "width": size,
        "PSF_size": size,
        "bsize": args.batch_size,
        "static_phase": True,
        "phs_layers": args.phs_layers,
        "zernike_features": args.zernike_features,
    }
    object_config = _branch_config(args, 0)
    if args.model_mode == "object_mmes":
        model = StaticDiffuseMMES(**shared, mmes_kwargs=object_config)
    else:
        model = DualMMES(
            **shared,
            num_frames=args.batch_size,
            object_kwargs=object_config,
            aberration_kwargs=_branch_config(args, 1),
            amplitude_offset=args.amplitude_offset,
        )
    model = model.to(device).train()
    slm = torch.ones(
        args.batch_size, 1, size, size, dtype=torch.complex64, device=device
    )
    frame_time = torch.full((args.batch_size,), -0.5, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction, kernel, _, _, _ = model(slm, frame_time)
    measurement_loss = prediction.square().mean()
    object_ae_loss = model.g_im.pop_ae_loss()
    aberration_ae_loss = (
        model.g_g.pop_ae_loss()
        if args.model_mode == "dual_mmes"
        else measurement_loss.new_zeros(())
    )
    loss = measurement_loss + object_ae_loss + aberration_ae_loss
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    object_gradient_norm = float(model.g_im.z.grad.norm().detach().cpu())
    aberration_gradient_norm = (
        float(model.g_g.z.grad.norm().detach().cpu())
        if args.model_mode == "dual_mmes"
        else None
    )
    if not object_gradient_norm or (
        aberration_gradient_norm is not None and not aberration_gradient_norm
    ):
        raise RuntimeError("Backward completed but an MMES latent gradient is zero.")
    result = {
        "status": "passed",
        "size": size,
        "batch_size": args.batch_size,
        "model_mode": args.model_mode,
        "elapsed_seconds": elapsed,
        "peak_allocated_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_reserved_cuda_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "psf_sum_max_abs_error": float(
            (kernel.sum((-2, -1)) - 1).abs().max().detach().cpu()
        ),
        "loss": float(loss.detach().cpu()),
        "object_latent_gradient_norm": object_gradient_norm,
        "aberration_latent_gradient_norm": aberration_gradient_norm,
    }
    del model, slm, frame_time, prediction, kernel, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", action="append", type=int, default=[])
    parser.add_argument(
        "--model-mode", choices=("object_mmes", "dual_mmes"), default="dual_mmes"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau", type=int, default=4)
    parser.add_argument("--rank1", type=int, default=512)
    parser.add_argument("--rank2", type=int, default=16)
    parser.add_argument("--rank3", type=int, default=512)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--checkpoint-chunks", type=_parse_bool, default=True)
    parser.add_argument("--amplitude-offset", type=float, default=1.0)
    parser.add_argument("--phs-layers", type=int, default=4)
    parser.add_argument("--zernike-features", type=int, default=28)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sizes = args.size or [600]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    results = []
    for size in sizes:
        try:
            result = benchmark_size(args, size, device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            result = {
                "status": "cuda_out_of_memory",
                "size": size,
                "batch_size": args.batch_size,
                "model_mode": args.model_mode,
                "error": str(exc),
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
