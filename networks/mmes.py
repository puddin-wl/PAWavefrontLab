"""MMES representations integrated with the current NeuWS optical model.

The implementations in this module intentionally reuse :mod:`networks.neuws`
for the optical forward model.  Only the requested object and aberration
representations are replaced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .neuws import MovingDiffuse, StaticDiffuseNet


class PatchEmbedding(nn.Module):
    """Extract row-major ``tau x tau`` patches with TensorFlow REFLECT edges."""

    def __init__(self, tau: int):
        super().__init__()
        if type(tau) is not int or tau < 1:
            raise ValueError("tau must be a positive integer.")
        self.tau = tau
        self.register_buffer(
            "filters", torch.eye(tau * tau).reshape(tau * tau, 1, tau, tau)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected a single-channel NCHW image [B,1,H,W].")
        padding = self.tau - 1
        if min(image.shape[-2:]) <= padding:
            raise ValueError("REFLECT padding requires image dimensions >= tau.")
        padded = (
            F.pad(image, (padding,) * 4, mode="reflect") if padding else image
        )
        return F.conv2d(padded, self.filters)

    def inverse(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1] != self.tau**2:
            raise ValueError(
                "Expected patches [B,tau**2,H+tau-1,W+tau-1]."
            )
        padding = self.tau - 1
        height, width = patches.shape[-2:]
        if min(height, width) <= padding:
            raise ValueError("Embedded spatial dimensions are too small.")
        accumulated = F.conv_transpose2d(patches, self.filters)
        return accumulated[..., padding:height, padding:width] / (self.tau**2)


class MMESObject(nn.Module):
    """Trainable latent field decoded by the shared MMES patch autoencoder."""

    def __init__(
        self,
        width: int,
        tau: int = 4,
        ranks: list[int] | tuple[int, int, int] | None = None,
        noise_std: float = 0.01,
        seed: int = 0,
        init_scale: float = 0.1,
        chunk_size: int = 4096,
        checkpoint_chunks: bool = True,
        channels: int = 1,
        num_frames: int = 1,
    ):
        super().__init__()
        if type(width) is not int or width < tau:
            raise ValueError("width must be an integer >= tau.")
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer.")
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("num_frames must be a positive integer.")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative.")
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("init_scale must be finite and positive.")
        if type(chunk_size) is not int or chunk_size < 0:
            raise ValueError("chunk_size must be non-negative; 0 disables chunking.")

        self.embedding = PatchEmbedding(tau)
        vector_width = channels * tau * tau
        ranks = (
            [32 * vector_width, vector_width, 32 * vector_width]
            if ranks is None
            else list(ranks)
        )
        if len(ranks) != 3 or any(
            type(rank) is not int or rank < 1 for rank in ranks
        ):
            raise ValueError("ranks must contain exactly three positive integers.")

        self.width = width
        self.channels = channels
        self.num_frames = num_frames
        self.noise_std = float(noise_std)
        self.chunk_size = chunk_size
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.last_ae_loss: torch.Tensor | None = None

        # A local RNG makes object and aberration seeds independent and avoids
        # perturbing initialization of the unchanged NeuWS branch.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.z = nn.Parameter(
                torch.rand(num_frames, channels, width, width) * init_scale
            )
            dimensions = [vector_width, *ranks, vector_width]
            layers: list[nn.Module] = []
            for index in range(4):
                linear = nn.Linear(dimensions[index], dimensions[index + 1])
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if index < 3:
                    layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.autoencoder = nn.Sequential(*layers)

    def decode_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        chunk_size = self.chunk_size or len(vectors)
        decoded = []
        for chunk in vectors.split(chunk_size):
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                value = checkpoint(self.autoencoder, chunk, use_reentrant=False)
            else:
                value = self.autoencoder(chunk)
            decoded.append(value)
        return torch.cat(decoded, dim=0)

    def forward_with_loss(
        self,
        *,
        add_noise: bool | None = None,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if add_noise is None:
            add_noise = self.training
        if self.num_frames == 1:
            if frame_indices is not None:
                raise ValueError("Single-frame fields do not take frame_indices.")
            latent = self.z
        else:
            if (
                frame_indices is None
                or frame_indices.ndim != 1
                or not frame_indices.numel()
            ):
                raise ValueError(
                    "Time-varying MMES requires non-empty 1-D frame indices."
                )
            latent = self.z.index_select(
                0, frame_indices.to(device=self.z.device, dtype=torch.long)
            )

        batch, channels, height, width = latent.shape
        embedded = self.embedding(latent.reshape(batch * channels, 1, height, width))
        _, patch_channels, embedded_height, embedded_width = embedded.shape
        clean = embedded.reshape(
            batch, channels * patch_channels, embedded_height, embedded_width
        )
        noisy = clean
        if add_noise and self.noise_std:
            noisy = clean + self.noise_std * torch.randn_like(clean)
        vector_width = channels * patch_channels
        vectors = noisy.permute(0, 2, 3, 1).reshape(-1, vector_width)
        decoded = self.decode_vectors(vectors)
        decoded = decoded.reshape(
            batch, embedded_height, embedded_width, vector_width
        ).permute(0, 3, 1, 2).contiguous()
        ae_loss = F.mse_loss(decoded, clean)
        field = self.embedding.inverse(
            decoded.reshape(
                batch * channels, patch_channels, embedded_height, embedded_width
            )
        )
        return field.reshape(batch, channels, height, width), ae_loss

    def forward(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        field, self.last_ae_loss = self.forward_with_loss(frame_indices=frame_indices)
        return field

    def pop_ae_loss(self) -> torch.Tensor:
        if self.last_ae_loss is None:
            raise RuntimeError("Call the optical forward before collecting its AE loss.")
        loss = self.last_ae_loss
        self.last_ae_loss = None
        return loss


class StaticDiffuseMMES(StaticDiffuseNet):
    """NeuWS with only the static object representation replaced by MMES."""

    def __init__(
        self,
        width: int,
        PSF_size: int,
        phs_layers: int = 4,
        use_FFT: bool = True,
        bsize: int = 8,
        use_pe: bool = False,
        static_phase: bool = True,
        zernike_features: int = 28,
        mmes_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            width,
            PSF_size,
            phs_layers=phs_layers,
            use_FFT=use_FFT,
            bsize=bsize,
            use_pe=use_pe,
            static_phase=static_phase,
            zernike_features=zernike_features,
        )
        self.g_im = MMESObject(width=width, **dict(mmes_kwargs or {}))

    def get_estimates(self, t: torch.Tensor):
        if not self.static_phase:
            return super().get_estimates(t)
        # The static basis slices are identical.  Evaluating the original MLP
        # once and expanding its result is mathematically equivalent, preserves
        # summed gradients, and avoids B copies of all per-pixel activations.
        obj = self.g_im()
        parameters = self.g_g(self.basis[:1]).permute(0, 3, 1, 2)
        phase = parameters[:, 1:2]
        amplitude = parameters[:, 0:1]
        field = amplitude * torch.exp(1j * phase)
        return (
            obj,
            field.expand(t.numel(), -1, -1, -1),
            phase.expand(t.numel(), -1, -1, -1),
        )

    def forward(self, x_batch: torch.Tensor, t: torch.Tensor):
        prediction, kernel, field, phase, obj = super().forward(x_batch, t)
        return (
            prediction.reshape(x_batch.shape[0], *obj.shape[-2:]),
            kernel,
            field,
            phase,
            obj,
        )


class DualMMES(StaticDiffuseNet):
    """Independent MMES object and signed-amplitude/phase aberration branches."""

    def __init__(
        self,
        width: int,
        PSF_size: int,
        bsize: int = 8,
        static_phase: bool = True,
        num_frames: int = 100,
        object_kwargs: Mapping[str, Any] | None = None,
        aberration_kwargs: Mapping[str, Any] | None = None,
        amplitude_offset: float = 1.0,
        phs_layers: int = 4,
        use_FFT: bool = True,
        zernike_features: int = 28,
    ):
        if width != PSF_size:
            raise ValueError("DualMMES requires PSF_size == width.")
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("num_frames must be a positive integer.")
        if not math.isfinite(amplitude_offset):
            raise ValueError("amplitude_offset must be finite.")
        super().__init__(
            width,
            PSF_size,
            phs_layers=phs_layers,
            use_FFT=use_FFT,
            # DualMMES immediately replaces both branches and removes the base
            # coordinate buffers.  Avoid transiently allocating B copies of a
            # large Zernike basis that can exceed 1 GiB at 900 x 900.
            bsize=1,
            static_phase=static_phase,
            zernike_features=zernike_features,
        )
        object_config = dict(object_kwargs or {})
        aberration_config = dict(aberration_kwargs or {})
        if any(
            key in config
            for config in (object_config, aberration_config)
            for key in ("channels", "num_frames")
        ):
            raise ValueError("channels and num_frames are managed by DualMMES.")
        object_config.setdefault("seed", 0)
        aberration_config.setdefault("seed", 1)
        self.g_im = MMESObject(width, channels=1, num_frames=1, **object_config)
        self.g_g = MMESObject(
            width,
            channels=2,
            num_frames=1 if static_phase else num_frames,
            **aberration_config,
        )
        self.num_frames = num_frames
        self.amplitude_offset = float(amplitude_offset)
        del self.basis
        del self.t_grid

    def frame_indices(self, t: torch.Tensor) -> torch.Tensor | None:
        if t.ndim != 1 or not t.numel() or not torch.isfinite(t).all():
            raise ValueError("t must be a non-empty finite 1-D vector.")
        if self.num_frames == 1:
            if not torch.allclose(t, torch.full_like(t, -0.5), atol=1e-6, rtol=0):
                raise ValueError("A single-frame dynamic model expects t=-0.5.")
            return None
        positions = (t + 0.5) * (self.num_frames - 1)
        indices = positions.round().long()
        if ((indices < 0) | (indices >= self.num_frames)).any() or not torch.allclose(
            positions, indices.to(positions.dtype), atol=1e-4, rtol=0
        ):
            raise ValueError(
                "Dynamic MMES supports only the discrete acquired frame times."
            )
        return indices

    def get_aberration_components(self, t: torch.Tensor):
        if t.ndim != 1 or not t.numel():
            raise ValueError("t must be a non-empty 1-D tensor.")
        obj = self.g_im()
        if self.static_phase:
            parameters = self.g_g().expand(t.numel(), -1, -1, -1)
        else:
            parameters = self.g_g(frame_indices=self.frame_indices(t))
        amplitude_residual = parameters[:, 0:1]
        phase_parameter = parameters[:, 1:2]
        amplitude_parameter = self.amplitude_offset + amplitude_residual
        field = amplitude_parameter * torch.exp(1j * phase_parameter)
        return (
            obj,
            field,
            phase_parameter,
            amplitude_parameter,
            amplitude_residual,
        )

    def get_estimates(self, t: torch.Tensor):
        obj, field, phase, _, _ = self.get_aberration_components(t)
        return obj, field, phase

    def forward(self, x_batch: torch.Tensor, t: torch.Tensor):
        if x_batch.shape[0] != t.numel():
            raise ValueError("SLM batch and t must have equal lengths.")
        prediction, kernel, field, phase, obj = super().forward(x_batch, t)
        return (
            prediction.reshape(x_batch.shape[0], *obj.shape[-2:]),
            kernel,
            field,
            phase,
            obj,
        )


def build_model(config: Mapping[str, Any]) -> StaticDiffuseNet:
    """Build any supported backend from a checkpoint-friendly configuration."""

    mode = str(config["model_mode"])
    shared = {
        "width": int(config["width"]),
        "PSF_size": int(config.get("psf_size", config["width"])),
        "phs_layers": int(config.get("phs_layers", 2)),
        "use_FFT": bool(config.get("use_fft", True)),
        "bsize": int(config.get("batch_size", 8)),
        "static_phase": bool(config.get("static_phase", True)),
        "zernike_features": int(config.get("zernike_features", 28)),
    }
    if mode == "original":
        network_class = (
            MovingDiffuse if config.get("dynamic_scene", False) else StaticDiffuseNet
        )
        return network_class(**shared)
    if mode == "object_mmes":
        return StaticDiffuseMMES(
            **shared, mmes_kwargs=config.get("object_mmes_config", {})
        )
    if mode == "dual_mmes":
        return DualMMES(
            **shared,
            num_frames=int(config["num_frames"]),
            object_kwargs=config.get("object_mmes_config", {}),
            aberration_kwargs=config.get("aberration_mmes_config", {}),
            amplitude_offset=float(config.get("amplitude_offset", 1.0)),
        )
    raise ValueError("model_mode must be original, object_mmes, or dual_mmes.")


def model_from_checkpoint(
    saved: Mapping[str, Any], device: str | torch.device = "cpu"
) -> StaticDiffuseNet:
    """Restore a model saved by ``reconstruct_neuws.py``."""

    config = saved.get("model_config", saved.get("config"))
    if config is None:
        raise KeyError("Checkpoint does not contain model_config.")
    model = build_model(config)
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()
