#!/usr/bin/env python3
# Copyright (c) 2023 Brandon Y. Feng, University of Maryland, College Park and Rice University.
"""Reconstruct a NeuWS scene from modulated MATLAB measurements."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import tqdm
from torch.fft import fft2, fftshift
from torch.utils.data import DataLoader

from dataset import BatchDataset
from networks import MovingDiffuse, StaticDiffuseNet
from utils import ang_to_unit


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _time_coordinate(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 1:
        return torch.zeros_like(indices, dtype=torch.float32) - 0.5
    return indices.float() / (count - 1) - 0.5


def _normalize_for_display(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().float()
    minimum, maximum = value.min(), value.max()
    if float(maximum - minimum) <= 1e-12:
        return np.zeros(value.shape, dtype=np.float32)
    return ((value - minimum) / (maximum - minimum)).numpy()


def _save_progress(
    vis_dir: Path,
    epoch: int,
    iteration: int,
    y_batch,
    y,
    kernel,
    sim_g,
    sim_phs,
    image_estimate,
    aperture,
) -> None:
    with torch.no_grad():
        image = torch.clamp(image_estimate[0:1], 0, 1)
        aberrated_psf = fftshift(
            fft2(aperture * sim_g[0:1], norm="forward"), dim=(-2, -1)
        ).abs().square()
        convolved = F.conv2d(image, aberrated_psf, padding="same").squeeze()
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    panels = (
        (y_batch[0].detach().cpu().squeeze(), "Real Measurement", "gray"),
        (y[0].detach().cpu().squeeze(), "Sim Measurement", "gray"),
        (image.detach().cpu().squeeze(), "I_est", "gray"),
        (sim_phs[0].detach().cpu().squeeze(), "Estimated Phase", "rainbow"),
        (kernel[0].detach().cpu().squeeze(), "Post-SLM PSF", "gray"),
        (convolved.detach().cpu(), "Aberrated I_est", "gray"),
    )
    for axis, (panel, title, cmap) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / f"e_{epoch}_it_{iteration}.jpg", dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--data_dir", default="resized")
    parser.add_argument("--scene_name", default="0609")
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_t", type=int, default=None)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--vis_freq", default=1000, type=int)
    parser.add_argument(
        "--log_freq",
        default=0,
        type=int,
        help="Print an epoch-loss update at this interval; 0 disables periodic logging.",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=0,
        type=int,
        help="Stop after this many epochs without a meaningful smoothed-loss improvement; 0 disables.",
    )
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float)
    parser.add_argument("--early_stop_warmup", default=0, type=int)
    parser.add_argument("--early_stop_window", default=10, type=int)
    parser.add_argument("--init_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-3, type=float)
    parser.add_argument("--silence_tqdm", action="store_true")
    parser.add_argument("--save_per_frame", action="store_true")
    parser.add_argument("--static_phase", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_intensity", default=0, type=float)
    parser.add_argument(
        "--normalization",
        choices=("shared-max", "per-frame-minmax"),
        default="shared-max",
        help="Measurement scaling applied by BatchDataset.",
    )
    parser.add_argument("--im_prefix", default="SLM_raw")
    parser.add_argument("--slm_prefix", default="SLM_sim")
    parser.add_argument("--zero_freq", default=-1, type=int)
    parser.add_argument("--phs_layers", default=2, type=int)
    parser.add_argument("--zernike_features", default=28, type=int)
    parser.add_argument("--dynamic_scene", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_epochs <= 0 or args.batch_size <= 0 or args.zernike_features <= 0:
        raise ValueError("--num_epochs, --batch_size and --zernike_features must be positive.")
    if args.early_stop_patience < 0 or args.early_stop_warmup < 0:
        raise ValueError("Early-stopping patience and warmup must be non-negative.")
    if args.early_stop_window <= 0 or args.early_stop_min_delta < 0:
        raise ValueError("Early-stopping window must be positive and min_delta non-negative.")
    root_dir = Path(args.root_dir).expanduser().resolve()
    requested_data_dir = Path(args.data_dir).expanduser()
    data_dir = requested_data_dir.resolve() if requested_data_dir.is_absolute() else root_dir / requested_data_dir
    vis_dir = root_dir / "vis" / args.scene_name
    final_dir = vis_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = final_dir / "per_frame"
    if args.save_per_frame:
        per_frame_dir.mkdir(exist_ok=True)
    print(f"Saving output at: {vis_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Using PyTorch {torch.__version__} on {device}")

    dataset = BatchDataset(
        data_dir,
        num=args.num_t,
        im_prefix=args.im_prefix,
        slm_prefix=args.slm_prefix,
        max_intensity=args.max_intensity,
        zero_freq=args.zero_freq,
        normalization=args.normalization,
    )
    width = dataset.width
    if args.width is not None and args.width != width:
        raise ValueError(
            f"--width {args.width} does not match inferred measurement size {width}."
        )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    network_class = MovingDiffuse if args.dynamic_scene else StaticDiffuseNet
    network = network_class(
        width=width,
        PSF_size=width,
        use_FFT=True,
        bsize=args.batch_size,
        phs_layers=args.phs_layers,
        static_phase=args.static_phase,
        zernike_features=args.zernike_features,
    ).to(device)
    image_optimizer = torch.optim.Adam(network.g_im.parameters(), lr=args.init_lr)
    phase_optimizer = torch.optim.Adam(network.g_g.parameters(), lr=args.init_lr)
    image_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        image_optimizer, T_max=args.num_epochs, eta_min=args.final_lr
    )
    phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        phase_optimizer, T_max=args.num_epochs, eta_min=args.final_lr
    )

    total_iteration = 0
    loss_history = []
    best_smoothed_loss = float("inf")
    patience_reference_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_image_state = None
    best_phase_state = None
    stopped_early = False
    progress = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)
    start_time = time.time()
    for epoch in progress:
        epoch_losses = []
        for iteration, (x_batch, y_batch, indices) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            indices = indices.to(device)
            current_time = _time_coordinate(indices, len(dataset))
            image_optimizer.zero_grad(set_to_none=True)
            phase_optimizer.zero_grad(set_to_none=True)
            y, kernel, sim_g, sim_phs, image_estimate = network(x_batch, current_time)
            y = y.reshape_as(y_batch)
            mse_loss = F.mse_loss(y, y_batch)
            mse_loss.backward()
            phase_optimizer.step()
            image_optimizer.step()
            epoch_losses.append(float(mse_loss.detach()))
            progress.set_postfix(MSE=f"{epoch_losses[-1]:.4e}")

            if args.vis_freq > 0 and total_iteration % args.vis_freq == 0:
                _save_progress(
                    vis_dir,
                    epoch,
                    iteration,
                    y_batch,
                    y,
                    kernel,
                    sim_g,
                    sim_phs,
                    image_estimate,
                    dataset.a_slm.to(device)[None, None],
                )
                sio.savemat(
                    vis_dir / "Sim_Phase.mat",
                    {"angle": sim_phs.detach().cpu().squeeze().numpy()},
                )
            total_iteration += 1
        loss_history.append(float(np.mean(epoch_losses)))
        image_scheduler.step()
        phase_scheduler.step()
        smoothed_loss = None
        improvement = None
        smoothed_loss_available = (
            args.early_stop_patience > 0
            and len(loss_history) >= args.early_stop_window
        )
        early_stopping_active = (
            smoothed_loss_available and epoch + 1 >= args.early_stop_warmup
        )
        if smoothed_loss_available:
            smoothed_loss = float(np.mean(loss_history[-args.early_stop_window :]))
            if smoothed_loss < best_smoothed_loss:
                best_smoothed_loss = smoothed_loss
                best_epoch = epoch + 1
                best_image_state = copy.deepcopy(network.g_im.state_dict())
                best_phase_state = copy.deepcopy(network.g_g.state_dict())

        if early_stopping_active:
            if not np.isfinite(patience_reference_loss):
                patience_reference_loss = smoothed_loss
                epochs_without_improvement = 0
            else:
                improvement = patience_reference_loss - smoothed_loss
                if improvement > 0.0 and improvement >= args.early_stop_min_delta:
                    patience_reference_loss = smoothed_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                stopped_early = True
        should_log_epoch = args.log_freq > 0 and (
            (epoch + 1) % args.log_freq == 0 or epoch + 1 == args.num_epochs
        )
        if should_log_epoch or stopped_early:
            elapsed_so_far = time.time() - start_time
            early_stop_status = ""
            if smoothed_loss_available:
                early_stop_status = (
                    f", smoothed={smoothed_loss:.6e}, best={best_smoothed_loss:.6e}"
                )
                if early_stopping_active:
                    improvement_text = (
                        "initialized" if improvement is None else f"{improvement:.6e}"
                    )
                    early_stop_status += (
                        f", improvement={improvement_text}, "
                        f"min_delta={args.early_stop_min_delta:.3e}, "
                        f"no_improve={epochs_without_improvement}/"
                        f"{args.early_stop_patience}"
                    )
                else:
                    early_stop_status += ", patience=warmup"
            print(
                f"Epoch {epoch + 1}/{args.num_epochs}: "
                f"mean MSE={loss_history[-1]:.6e}{early_stop_status}, "
                f"elapsed={elapsed_so_far:.1f}s",
                flush=True,
            )
        if stopped_early:
            print(
                f"Early stopping at epoch {epoch + 1}: smoothed MSE has not "
                f"improved by {args.early_stop_min_delta:.3e} for "
                f"{args.early_stop_patience} epochs. Best epoch: {best_epoch}, "
                f"best smoothed MSE: {best_smoothed_loss:.6e}.",
                flush=True,
            )
            break
    elapsed = time.time() - start_time
    print(f"Training took {elapsed:.2f} seconds.")

    if (
        args.early_stop_patience > 0
        and best_image_state is not None
        and best_phase_state is not None
    ):
        network.g_im.load_state_dict(best_image_state)
        network.g_g.load_state_dict(best_phase_state)
        print(f"Restored the best smoothed-loss state from epoch {best_epoch}.")

    output_errors = []
    output_aberrations = []
    output_images = []
    output_images_network_units = []
    final_field = None
    final_phase = None
    network.eval()
    with torch.no_grad():
        for frame in range(len(dataset)):
            current_time = _time_coordinate(
                torch.tensor([frame], dtype=torch.long, device=device), len(dataset)
            )
            image_estimate, sim_g, sim_phs = network.get_estimates(current_time)
            image_network_np = torch.clamp(image_estimate, min=0).squeeze().cpu().numpy()
            image_np = np.clip(image_network_np, 0, 1)
            output_images_network_units.append(image_network_np)
            output_images.append(image_np)
            # Materialize the phase first, then derive the complex field from it.
            # Keeping two independent NumPy views of temporary CUDA outputs can
            # leave the saved field inconsistent with the saved phase after the
            # allocator reuses the temporary host storage.
            phase_np = sim_phs.squeeze().cpu().numpy().copy()
            field_np = np.exp(1j * phase_np).astype(np.complex64)
            output_errors.append(np.uint8(np.clip(ang_to_unit(np.angle(field_np)), 0, 1) * 255))
            output_aberrations.append(np.uint8(_normalize_for_display(sim_phs.squeeze()) * 255))
            final_field, final_phase = field_np, phase_np
            if args.save_per_frame and not args.static_phase:
                sio.savemat(per_frame_dir / f"sim_phase_{frame}.mat", {"angle": phase_np})

    if args.dynamic_scene:
        imageio.mimsave(
            final_dir / "final_I.gif",
            [np.uint8(np.clip(image, 0, 1) * 255) for image in output_images],
            duration=1.0 / 30,
        )
    else:
        imageio.imwrite(final_dir / "final_I_est.png", np.uint8(output_images[-1] * 255))
    sio.savemat(final_dir / "final_I_est.mat", {"image": output_images[-1]})
    sio.savemat(
        final_dir / "final_I_est_network_units.mat",
        {"image": output_images_network_units[-1]},
    )
    sio.savemat(
        final_dir / "final_aberration.mat",
        {"field": final_field, "phase": final_phase},
    )

    if args.static_phase:
        imageio.imwrite(final_dir / "final_aberrations_angle.png", output_errors[0])
        imageio.imwrite(final_dir / "final_aberrations.png", output_aberrations[0])
    else:
        imageio.mimsave(final_dir / "final_aberrations_angle_grey.gif", output_errors, duration=1.0 / 30)
        imageio.mimsave(final_dir / "final_aberrations.gif", output_aberrations, duration=1.0 / 30)

    cmap = plt.get_cmap("rainbow")
    colored_errors = [np.uint8(cmap(error / 255.0)[..., :3] * 255) for error in output_errors]
    if args.save_per_frame:
        for frame, colored in enumerate(colored_errors):
            imageio.imwrite(per_frame_dir / f"{frame:03d}.jpg", colored)
    imageio.mimsave(final_dir / "final_aberrations_angle.gif", colored_errors, duration=1.0 / 30)
    summary = {
        "data_dir": str(data_dir),
        "device": str(device),
        "width": width,
        "num_frames": len(dataset),
        "num_epochs": len(loss_history),
        "requested_num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "phase_layers": args.phs_layers,
        "network_zernike_features": args.zernike_features,
        "static_phase": args.static_phase,
        "measurement_normalization_max": dataset.max_intensity,
        "measurement_normalization": dataset.normalization,
        "loss_history": loss_history,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "early_stopping": {
            "enabled": args.early_stop_patience > 0,
            "stopped_early": stopped_early,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "warmup": args.early_stop_warmup,
            "window": args.early_stop_window,
            "best_epoch": best_epoch if best_epoch else None,
            "best_smoothed_loss": best_smoothed_loss if np.isfinite(best_smoothed_loss) else None,
            "patience_reference_loss": (
                patience_reference_loss
                if np.isfinite(patience_reference_loss)
                else None
            ),
        },
    }
    (final_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Training concludes.")


if __name__ == "__main__":
    main()
