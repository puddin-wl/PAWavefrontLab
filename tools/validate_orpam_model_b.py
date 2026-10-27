#!/usr/bin/env python3
"""Reproducible physical validation and 50-frame model-A/B comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib caches out of the repository and support headless runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pawavefrontlab-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from dataio.orpam_slm import HardwareSLMPhaseLibrary  # noqa: E402
from optics.legacy import fft_2xPad_Conv2D  # noqa: E402
from optics.orpam_model_b import (  # noqa: E402
    ORPAMForwardModelB,
    ORPAMModelBConfig,
    integrate_equivalent_kernel,
    scan_correlation,
)
from tools.run_orpam_model_b import make_fibre_phantom  # noqa: E402


CONDITION_ZERO = "zero_aberration"
CONDITION_NONZERO = "polynomial_aberration_v1"


def artificial_aberration(size: int, *, device: torch.device) -> torch.Tensor:
    """A grid-independent, explicitly recorded non-zero system phase."""

    coordinate = (
        torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    ) / (size / 2.0)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    radius2 = x.square() + y.square()
    # Radians.  Terms are intentionally defined over the full square pupil.
    return 0.45 * radius2 + 0.35 * (x.square() - y.square()) + 0.30 * x * radius2


def aberration_record() -> dict[str, object]:
    return {
        "name": CONDITION_NONZERO,
        "coordinate_definition": "x,y=(index-(N-1)/2)/(N/2)",
        "phase_radians": "0.45*(x^2+y^2) + 0.35*(x^2-y^2) + 0.30*x*(x^2+y^2)",
        "terms": {
            "radial_quadratic_rad": 0.45,
            "square_astigmatism_rad": 0.35,
            "x_cubic_coma_like_rad": 0.30,
        },
    }


def load_model_phase(library: HardwareSLMPhaseLibrary, index: int) -> np.ndarray:
    specification = str(library.manifest["model_output"]["mat"])
    path_text, variable = specification.rsplit(":", 1)
    path = library.root / path_text.replace("N", str(index))
    values = sio.loadmat(path)
    if variable not in values:
        raise KeyError(f"{path} does not contain {variable!r}.")
    phase = np.asarray(values[variable], dtype=np.float32).squeeze()
    expected = tuple(int(value) for value in library.manifest["model_shape"])
    if phase.shape != expected or not np.isfinite(phase).all():
        raise ValueError(f"Frame {index} model phase is invalid: {phase.shape}.")
    return phase


def model_a_forward(
    object_image: torch.Tensor,
    phase: torch.Tensor,
    aberration: torch.Tensor,
    phase_sign: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the legacy NeuWS array-level optical path with unit amplitude."""

    pupil = torch.exp(1j * (aberration + phase_sign * phase))[None, None]
    psf = torch.fft.fftshift(
        torch.fft.fft2(pupil, norm="forward"), dim=(-2, -1)
    ).abs().square()
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True)
    # This explicit flip followed by legacy FFT convolution is the current
    # model-A implementation of the physical point-scan correlation.
    image = fft_2xPad_Conv2D(object_image, psf.flip(-2, -1))
    return psf, image


def normalized_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    """||a/||a||_2 - b/||b||_2||_2, without fitting gain or offset."""

    first = first.to(torch.float64)
    second = second.to(torch.float64)
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    if first_norm <= 0 or second_norm <= 0:
        raise ValueError("Normalized L2 inputs must have non-zero norm.")
    return float(torch.linalg.vector_norm(first / first_norm - second / second_norm))


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.to(torch.float64))
    if denominator <= 0:
        raise ValueError("Relative L2 reference must have non-zero norm.")
    return float(
        torch.linalg.vector_norm(value.to(torch.float64) - reference.to(torch.float64))
        / denominator
    )


def psf_metrics(psf: torch.Tensor, pixel_pitch: float) -> dict[str, float]:
    """Peak, centroid, and RMS radius without materializing a 2-D coordinate grid."""

    values = psf[0, 0].to(torch.float64)
    total = values.sum()
    coordinate_y = (
        torch.arange(values.shape[0], device=values.device, dtype=torch.float64)
        - values.shape[0] // 2
    ) * pixel_pitch
    coordinate_x = (
        torch.arange(values.shape[1], device=values.device, dtype=torch.float64)
        - values.shape[1] // 2
    ) * pixel_pitch
    marginal_y = values.sum(dim=1) / total
    marginal_x = values.sum(dim=0) / total
    centroid_y = (marginal_y * coordinate_y).sum()
    centroid_x = (marginal_x * coordinate_x).sum()
    variance_y = (marginal_y * (coordinate_y - centroid_y).square()).sum()
    variance_x = (marginal_x * (coordinate_x - centroid_x).square()).sum()
    peak_index = torch.nonzero(values == values.max())[0]
    return {
        "peak": float(values.max()),
        "centroid_row": float(centroid_y),
        "centroid_column": float(centroid_x),
        "rms_radius": float(torch.sqrt(variance_y + variance_x)),
        "peak_row": float(coordinate_y[int(peak_index[0])]),
        "peak_column": float(coordinate_x[int(peak_index[1])]),
    }


def describe(rows: list[dict], key: str) -> dict[str, float | int]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    minimum = int(np.argmin(values))
    maximum = int(np.argmax(values))
    ordered = np.argsort(values)
    middle = int(ordered[len(ordered) // 2])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values[minimum]),
        "minimum_frame": int(rows[minimum]["frame"]),
        "middle_frame": int(rows[middle]["frame"]),
        "middle_value": float(values[middle]),
        "maximum": float(values[maximum]),
        "maximum_frame": int(rows[maximum]["frame"]),
    }


def correlation(rows: list[dict], x_key: str, y_key: str) -> dict[str, float]:
    x = np.asarray([float(row[x_key]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _crop_tensor(psf: torch.Tensor, pixel_pitch_m: float, extent_um: float) -> np.ndarray:
    half = int(math.ceil(extent_um * 1e-6 / pixel_pitch_m))
    centre_y = psf.shape[-2] // 2
    centre_x = psf.shape[-1] // 2
    return (
        psf[0, 0, centre_y - half : centre_y + half + 1, centre_x - half : centre_x + half + 1]
        .detach()
        .cpu()
        .numpy()
    )


def _fwhm(profile: np.ndarray, coordinate_m: np.ndarray) -> float:
    centre = int(np.argmax(profile))
    target = profile[centre] / 2.0
    right_candidates = np.nonzero(profile[centre:] <= target)[0]
    left_candidates = np.nonzero(profile[: centre + 1] <= target)[0]
    if not len(right_candidates) or not len(left_candidates):
        return float("nan")
    right = centre + int(right_candidates[0])
    left = int(left_candidates[-1])

    def crossing(a: int, b: int) -> float:
        fraction = (target - profile[a]) / (profile[b] - profile[a])
        return float(coordinate_m[a] + fraction * (coordinate_m[b] - coordinate_m[a]))

    return crossing(right - 1, right) - crossing(left, left + 1)


def _enclosed_energy(
    psf: torch.Tensor, pixel_pitch_m: float, radii_um: np.ndarray
) -> np.ndarray:
    extent_um = float(radii_um.max())
    crop = _crop_tensor(psf, pixel_pitch_m, extent_um)
    half_y, half_x = crop.shape[0] // 2, crop.shape[1] // 2
    y = (np.arange(crop.shape[0]) - half_y) * pixel_pitch_m * 1e6
    x = (np.arange(crop.shape[1]) - half_x) * pixel_pitch_m * 1e6
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    pixel_energy = crop * pixel_pitch_m**2
    return np.asarray([pixel_energy[radius <= value].sum() for value in radii_um])


def _figure_focus(
    path: Path,
    ideal_psf: torch.Tensor,
    aberrated_psf: torch.Tensor,
    config: ORPAMModelBConfig,
) -> dict[str, float]:
    dx = config.focus_pixel_pitch_m
    extent_um = 25.0
    ideal_crop = _crop_tensor(ideal_psf, dx, extent_um)
    aberrated_crop = _crop_tensor(aberrated_psf, dx, extent_um)
    ideal_peak = float(ideal_psf.max())
    half = ideal_crop.shape[0] // 2
    coordinate = (np.arange(ideal_crop.shape[0]) - half) * dx * 1e6
    ideal_profile = ideal_crop[half] / ideal_peak
    aberrated_profile = aberrated_crop[half] / ideal_peak
    fwhm_um = _fwhm(ideal_profile, coordinate * 1e-6) * 1e6
    radii = np.linspace(0.0, 200.0, 160)
    ideal_energy = _enclosed_energy(ideal_psf, dx, radii)
    aberrated_energy = _enclosed_energy(aberrated_psf, dx, radii)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    common = LogNorm(vmin=1e-6, vmax=1.0)
    extent = [coordinate[0], coordinate[-1], coordinate[-1], coordinate[0]]
    for axis, image, title in (
        (axes[0, 0], ideal_crop / ideal_peak, "Ideal square pupil"),
        (axes[0, 1], aberrated_crop / ideal_peak, "Recorded polynomial aberration"),
    ):
        shown = axis.imshow(np.maximum(image, 1e-8), extent=extent, norm=common, cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("x at focus (µm)")
        axis.set_ylabel("y at focus (µm)")
    fig.colorbar(shown, ax=axes[0], label="Intensity / ideal peak")
    axes[1, 0].plot(coordinate, ideal_profile, label=f"ideal, FWHM={fwhm_um:.3f} µm")
    axes[1, 0].plot(coordinate, aberrated_profile, label="aberrated")
    axes[1, 0].set(xlabel="x at focus (µm)", ylabel="Intensity / ideal peak", ylim=(-0.02, 1.03))
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(radii, ideal_energy, label="ideal")
    axes[1, 1].plot(radii, aberrated_energy, label="aberrated")
    axes[1, 1].set(xlabel="Circular radius (µm)", ylabel="Enclosed normalized energy", ylim=(0, 1.01))
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(
        f"Model B focus: λ=532 nm, f=50 mm, 1080² SLM, NFFT={config.fft_size}, dx={dx*1e6:.3f} µm"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "measured_ideal_fwhm_um": fwhm_um,
        "ideal_peak_inverse_m2": ideal_peak,
        "aberrated_peak_inverse_m2": float(aberrated_psf.max()),
        "aberrated_strehl_ratio": float(aberrated_psf.max() / ideal_psf.max()),
    }


def _figure_psf_kernel(
    path: Path,
    psf: torch.Tensor,
    area_kernel: torch.Tensor,
    point_kernel: torch.Tensor,
    config: ORPAMModelBConfig,
    frame: int,
) -> None:
    dx = config.focus_pixel_pitch_m
    psf_crop = _crop_tensor(psf, dx, 40.0)
    half = psf_crop.shape[0] // 2
    coordinate = (np.arange(psf_crop.shape[0]) - half) * dx * 1e6
    area = area_kernel[0, 0].detach().cpu().numpy()
    point = point_kernel[0, 0].detach().cpu().numpy()
    offsets = np.arange(-config.kernel_radius, config.kernel_radius + 1) * config.scan_pitch_m * 1e6
    difference = (area - point) / max(float(area.max()), np.finfo(float).eps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    shown = axes[0, 0].imshow(
        np.maximum(psf_crop / psf_crop.max(), 1e-7),
        extent=[coordinate[0], coordinate[-1], coordinate[-1], coordinate[0]],
        cmap="magma",
        norm=LogNorm(vmin=1e-6, vmax=1),
    )
    for value in np.arange(-40, 41, 5):
        axes[0, 0].axvline(value, color="white", alpha=0.12, lw=0.5)
        axes[0, 0].axhline(value, color="white", alpha=0.12, lw=0.5)
    axes[0, 0].set(title="Continuous-grid optical PSF with 5 µm cell boundaries", xlabel="x (µm)", ylabel="y (µm)")
    fig.colorbar(shown, ax=axes[0, 0], label="PSF / peak")
    shown = axes[0, 1].imshow(
        area,
        extent=[offsets[0]-2.5, offsets[-1]+2.5, offsets[-1]+2.5, offsets[0]-2.5],
        cmap="magma",
    )
    axes[0, 1].set(title=f"Area-integrated K, sum={area.sum():.6f}", xlabel="column offset (µm)", ylabel="row offset (µm)")
    fig.colorbar(shown, ax=axes[0, 1], label="Cell response")
    centre = config.kernel_radius
    axes[1, 0].plot(offsets, area[centre], "o-", ms=3, label="5 µm area integral")
    axes[1, 0].plot(offsets, point[centre], ".--", label="centre sample × area")
    axes[1, 0].set(xlabel="column offset (µm)", ylabel="Response", title="Not a resize: integral versus point sample")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    limit = max(abs(float(difference.min())), abs(float(difference.max())))
    shown = axes[1, 1].imshow(
        difference,
        extent=[offsets[0]-2.5, offsets[-1]+2.5, offsets[-1]+2.5, offsets[0]-2.5],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[1, 1].set(title="(area integral − point sample) / max(K)", xlabel="column offset (µm)", ylabel="row offset (µm)")
    fig.colorbar(shown, ax=axes[1, 1])
    fig.suptitle(
        f"Frame {frame}: physical PSF → piecewise-constant-object kernel, NFFT={config.fft_size}, Q={config.quadrature_order}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_images(
    path: Path,
    object_image: np.ndarray,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    frame: int,
    difference_value: float,
) -> None:
    a = image_a[0, 0].detach().cpu().numpy()
    b = image_b[0, 0].detach().cpu().numpy()
    difference = b - a
    display_max = float(np.percentile(np.concatenate((a.ravel(), b.ravel())), 99.8))
    difference_limit = float(np.percentile(np.abs(difference), 99.8))
    extent = [0, 3000, 3000, 0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    axes[0].imshow(object_image, cmap="gray", vmin=0, vmax=float(object_image.max()), extent=extent)
    axes[0].set_title("Common fibre phantom")
    for axis, image, title in (
        (axes[1], a, "Model A"),
        (axes[2], b, "Model B"),
    ):
        shown = axis.imshow(image, cmap="magma", vmin=0, vmax=display_max, extent=extent)
        axis.set_title(title)
    fig.colorbar(shown, ax=axes[1:3], label="Response (common display scale)")
    shown = axes[3].imshow(
        difference,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
        extent=extent,
    )
    axes[3].set_title("Model B − Model A")
    fig.colorbar(shown, ax=axes[3], label="Response difference")
    for axis in axes:
        axis.set_xlabel("scan x (µm)")
        axis.set_ylabel("scan y (µm)")
    fig.suptitle(
        f"Frame {frame}, zero system aberration; normalized L2={difference_value:.6f}. A/B share one display range."
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_statistics(path: Path, rows: list[dict], correlations: dict) -> None:
    zero = [row for row in rows if row["condition"] == CONDITION_ZERO]
    nonzero = [row for row in rows if row["condition"] == CONDITION_NONZERO]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for subset, label in ((zero, "zero aberration"), (nonzero, "non-zero aberration")):
        frame = [row["frame"] for row in subset]
        axes[0, 0].plot(frame, [row["ab_normalized_l2"] for row in subset], "o-", ms=3, label=label)
        axes[0, 1].plot(frame, [100 * row["kernel_truncation_error"] for row in subset], "o-", ms=3, label=label)
    axes[0, 0].set(title="Model A/B difference", xlabel="real SLM frame", ylabel="normalized L2")
    axes[0, 1].set(title="Model-B finite-kernel loss", xlabel="real SLM frame", ylabel="truncation error (%)")
    for axis in axes[0]:
        axis.legend()
        axis.grid(alpha=0.25)

    scatter = axes[1, 0].scatter(
        [row["phase_rms_rad"] for row in zero],
        [row["ab_normalized_l2"] for row in zero],
        c=[row["model_b_rms_radius_um"] for row in zero],
        cmap="viridis",
    )
    axes[1, 0].set(title="Difference versus known-phase strength", xlabel="hardware phase RMS (rad)", ylabel="normalized L2")
    fig.colorbar(scatter, ax=axes[1, 0], label="Model-B RMS radius (µm)")
    scatter = axes[1, 1].scatter(
        [row["model_b_rms_radius_um"] for row in zero],
        [row["ab_normalized_l2"] for row in zero],
        c=[row["coefficient_l2_rad"] for row in zero],
        cmap="plasma",
    )
    corr = correlations[CONDITION_ZERO]["difference_vs_psf_rms"]
    axes[1, 1].set(
        title=f"Difference versus PSF spread (Pearson r={corr['pearson_r']:.3f})",
        xlabel="Model-B RMS radius (µm)",
        ylabel="normalized L2",
    )
    fig.colorbar(scatter, ax=axes[1, 1], label="Zernike coefficient L2 (rad)")
    for axis in axes[1]:
        axis.grid(alpha=0.25)
    fig.suptitle("50 real SLM phases; common fibre object; primary NFFT=2160")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_convergence(path: Path, rows: list[dict], representatives: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label, frame in representatives.items():
        subset = sorted(
            [row for row in rows if row["frame"] == frame], key=lambda row: row["fft_size"]
        )
        dx = [row["focus_pitch_um"] for row in subset]
        axes[0].plot(dx, [row["kernel_relative_l2_vs_4320"] for row in subset], "o-", label=f"{label}: frame {frame}")
        axes[1].plot(dx, [row["image_normalized_l2_vs_4320"] for row in subset], "o-")
        axes[2].plot(dx, [row["ab_normalized_l2"] for row in subset], "o-")
    axes[0].set(title="Kernel convergence", ylabel="relative L2 vs 4320")
    axes[1].set(title="Model-B image convergence", ylabel="normalized L2 vs 4320")
    axes[2].set(title="A/B metric convergence", ylabel="A/B normalized L2")
    for axis in axes:
        axis.set_xlabel("focus-plane pitch (µm; finer ←)")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Representative minimum / middle / maximum frames, zero aberration")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, summary: dict) -> None:
    zero_condition = summary["conditions"][CONDITION_ZERO]
    nonzero_condition = summary["conditions"][CONDITION_NONZERO]
    zero = zero_condition["ab_normalized_l2"]
    nonzero = nonzero_condition["ab_normalized_l2"]
    corr = summary["correlations"][CONDITION_ZERO]
    factor = summary["factor_decomposition"]
    focus = summary["ideal_focus"]
    representatives = summary["representatives"]
    text = f"""# OR-PAM 模型 B 验证与模型 A/B 系统性对比

本报告由 `tools/validate_orpam_model_b.py` 使用实验对应的 50 张真实 SLM 相位生成。原始实验
数据、大尺寸 PSF 和 NPZ 未纳入 Git；仓库只保存本报告、小体积 PNG、CSV 和 JSON 摘要。

## 1. 模型、物理参数和比较约定

模型 B 使用 1080×1080 硬件相位、6.4 µm SLM 像元、532 nm 波长、50 mm 焦距和方形均匀
瞳。复瞳为

`P_i(u,v) = exp(+i Phi(u,v)) exp(-i Gamma_i(u,v))`。

焦平面采样为 `dx = lambda*f/(N_fft*p)`，PSF 按 `sum(h)*dx^2=1` 做物理面积归一化。
5 µm 分块常数物体产生

`K[k,l] = integral_[-Delta/2,Delta/2]^2 h(k*Delta+u,l*Delta+v) du dv`。

最终 `I[p,q] = sum O[p+k,q+l]K[k,l]` 由 PyTorch `conv2d` 直接作互相关，不翻转模型 B
核。有限核不重新归一化，物体外为零延拓。

主 50 帧统计使用 NFFT={summary['settings']['primary_fft_size']}、焦平面步长
{summary['settings']['primary_focus_pitch_um']:.4f} µm、{summary['settings']['kernel_size']} 核和
{summary['settings']['quadrature_order']} 阶 Gauss–Legendre 求积。模型 A/B 都固定为均匀瞳振幅，
使用同一人工纤维物体；没有把 `StaticDiffuseNet` 的自由振幅表示混入比较。

非零像差明确设为：

`Phi = 0.45(x^2+y^2) + 0.35(x^2-y^2) + 0.30*x*(x^2+y^2)` rad，

其中 `x,y=(index-(N-1)/2)/(N/2)`，分别在 600 和 1080 网格求值。

归一化 L2 指标严格定义为

`D(A,B) = ||A/||A||_2 - B/||B||_2||_2`。

不拟合增益或偏置。它衡量两张预测图的形状差异，**不能解释为模型 B 准确度提升的百分比**。

## 2. 物理与数值正确性

![理想焦斑与畸变焦斑](figures/orpam_model_b_validation/figure_1_focus.png)

4320 FFT 的理想方孔焦斑实测 FWHM 为 {focus['measured_ideal_fwhm_um']:.4f} µm，理论 sinc²
FWHM 为 {summary['settings']['ideal_sinc2_fwhm_um']:.4f} µm。非零像差 Strehl 比为
{focus['aberrated_strehl_ratio']:.4f}。专项测试还验证了面积归一化、补零收敛、区域积分和
600×600 输出。端到端梯度测试贯穿“相位→PSF→等效核→扫描图像”，float64 中央有限差分
步长为 `1e-5`，判据为 `rtol=3e-4, atol=2e-7`。

人为正向相位倾斜测试得到：`exp(+i Phi)` 使 PSF 和 K 向 +x 移动，固定单点物体的扫描峰向
-x 移动；`exp(-i Gamma)` 的已知相位产生相反移动。这与点扫描式 `S(p)=h(-p)` 一致，证明
模型 B 不应额外翻核。

模型 A 和 B 的张量轴均按 row=`y`、column=`x` 比较；模型 A 保留其原实现的“PSF 先翻转、
再做 legacy FFT convolution”，模型 B 则把 `K` 直接交给 `conv2d`。因此下列 A/B 图像处于
同一数组方向，而不是通过事后翻转配准。

![PSF 与等效核](figures/orpam_model_b_validation/figure_2_psf_kernel.png)

上图直接比较 5 µm 面积积分与“中心点采样×面积”。二者并不等价，等效核不是 PSF resize。

## 3. 50 帧 A/B 统计

| 条件 | 平均值 | 中位数 | 最小值（帧） | 最大值（帧） |
| --- | ---: | ---: | ---: | ---: |
| 零像差 | {zero['mean']:.6f} | {zero['median']:.6f} | {zero['minimum']:.6f}（{zero['minimum_frame']}） | {zero['maximum']:.6f}（{zero['maximum_frame']}） |
| 固定非零像差 | {nonzero['mean']:.6f} | {nonzero['median']:.6f} | {nonzero['minimum']:.6f}（{nonzero['minimum_frame']}） | {nonzero['maximum']:.6f}（{nonzero['maximum_frame']}） |

模型 B 的逐帧光学与核统计如下；“尺寸”采用整个物理焦平面上的强度 RMS 半径，逐帧原始值和
二维质心均保存在 CSV 中。

| 条件 | PSF 峰值均值 (m^-2) | RMS 半径均值 (µm) | row 质心范围 (µm) | column 质心范围 (µm) | 核保留能量均值 | 截断误差均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 零像差 | {zero_condition['model_b_psf_peak_inverse_m2']['mean']:.6g} | {zero_condition['model_b_rms_radius_um']['mean']:.4f} | {zero_condition['model_b_centroid_row_um']['minimum']:.3f}…{zero_condition['model_b_centroid_row_um']['maximum']:.3f} | {zero_condition['model_b_centroid_column_um']['minimum']:.3f}…{zero_condition['model_b_centroid_column_um']['maximum']:.3f} | {zero_condition['kernel_retained_energy']['mean']:.6f} | {100*zero_condition['kernel_truncation_error']['mean']:.4f}% |
| 固定非零像差 | {nonzero_condition['model_b_psf_peak_inverse_m2']['mean']:.6g} | {nonzero_condition['model_b_rms_radius_um']['mean']:.4f} | {nonzero_condition['model_b_centroid_row_um']['minimum']:.3f}…{nonzero_condition['model_b_centroid_row_um']['maximum']:.3f} | {nonzero_condition['model_b_centroid_column_um']['minimum']:.3f}…{nonzero_condition['model_b_centroid_column_um']['maximum']:.3f} | {nonzero_condition['kernel_retained_energy']['mean']:.6f} | {100*nonzero_condition['kernel_truncation_error']['mean']:.4f}% |

![50 帧统计](figures/orpam_model_b_validation/figure_4_frame_statistics.png)

零像差条件下，差异与硬件相位 RMS 的 Pearson 相关为
{corr['difference_vs_phase_rms']['pearson_r']:.4f}，与模型 B PSF RMS 半径的 Pearson 相关为
{corr['difference_vs_psf_rms']['pearson_r']:.4f}。相关不等于因果；相位强度、焦斑扩展、旧模型
无物理坐标以及区域积分共同变化。

## 4. 受控因素分解

代表帧为最小差异帧 {representatives['minimum']}、中间帧 {representatives['middle']}、最大差异帧
{representatives['maximum']}。中间帧的受控阶梯结果：

| 改变 | 归一化 L2 | 可归因范围 |
| --- | ---: | --- |
| 原模型 A → 600 网格物理 FFT + 点采样 | {factor['legacy_a_vs_physical600_point']:.6f} | 物理坐标映射与重新采样共同影响，不能再拆开 |
| 600 网格点采样 → 600 网格面积积分 | {factor['physical600_point_vs_area']:.6f} | 5 µm 物体基函数/区域积分 |
| 600 网格面积积分 → 1080 网格面积积分 | {factor['physical600_area_vs_physical1080_area']:.6f} | 同系数相位的 SLM 网格采样；仍含有限采样/视场差异 |
| 1080 网格点采样 → 面积积分 | {factor['physical1080_point_vs_area']:.6f} | 5 µm 区域积分 |
| 正确核 → 错误翻核 | {factor['correct_vs_wrong_flipped_kernel']:.6f} | 纯方向错误 |

有限核重新归一化后，归一化图像指标变化仅
{factor['truncated_vs_renormalized_normalized_l2']:.3e}，但原始幅度相对变化为
{factor['truncated_vs_renormalized_raw_relative_l2']:.6f}；因此逐图归一化会掩盖真实截断能量。
全图 A/B 差异为 {factor['legacy_a_vs_b_global']:.6f}，去除宽度等于核半径的边缘后为
{factor['legacy_a_vs_b_interior']:.6f}，量化了零延拓边界的影响。

![A/B 图像](figures/orpam_model_b_validation/figure_3_model_ab_images.png)

A/B 图使用完全相同的显示范围，差异图使用对称色标，没有逐图独立拉伸。

## 5. 代表帧采样收敛

![采样收敛](figures/orpam_model_b_validation/figure_5_convergence.png)

三帧均比较 1080、2160 和 4320 FFT；4320 作为本报告的代表帧参考，而不是宣称绝对真值。
没有对 50 帧强行执行 8640 FFT。既有单帧检查显示 4320 与 8640 等效核仍有约 0.7% 相对
差异，因此训练接入前应继续对少量代表帧抽查。

## 6. 已确认结论

- 模型 B 的物理坐标、面积归一化、5 µm 区域积分和互相关方向自洽且可微。
- 真实 50 帧硬件相位、模型相位和实验图像编号逐项匹配；未 resize 或重生成历史相位。
- 50 张硬件 MAT/PNG 的最大圆周相位误差为
  {summary['maximum_mat_png_circular_phase_error_rad']:.3e} rad，符合 uint16 包裹相位量化。
- A/B 差异不是单一因素造成。旧模型缺少物理焦面坐标，因此“旧离散 FFT”和“物理尺度”
  无法在输出层完全正交分离。
- 统一振幅后差异仍显著，说明不能把差异归因于 `StaticDiffuseNet` 的自由振幅能力。
- 核重归一化会隐藏截断幅度误差，故模型 B 保留实际核和。

## 7. 尚未确认与下一阶段建议

- MAT 表示设计相位，硬件 PNG 未应用器件 LUT；实际调制相位仍需实验标定。
- 绝对相位符号和实验图像朝向已与软件约定一致，但仍建议用已知倾斜靶做硬件闭环确认。
- 方形均匀瞳尚未包含实际近圆形光束、振幅非均匀、SLM 像元填充因子、1 m 传播和声学响应。
- 亚 5 µm 碳纤维与扫描区外吸收会造成分块常数和零边界失配。
- 接入 NeuWS 时应新增可选 forward backend，不改对象网络；逐帧/小 batch 生成 K，记录每帧
  截断能量，并先做冻结物体或冻结像差的短实验。不要用训练 Loss 单独判定物理正确性。

## 8. 可复现数据

- [50 帧逐项指标](data/orpam_model_b_validation/frame_metrics.csv)
- [代表帧收敛](data/orpam_model_b_validation/convergence_metrics.csv)
- [因素分解](data/orpam_model_b_validation/factor_decomposition.csv)
- [机器可读摘要](data/orpam_model_b_validation/summary.json)
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=PROJECT_ROOT / "data/slm50_noll4_28_decay4x_600_1080",
    )
    parser.add_argument(
        "--measurement-dir",
        type=Path,
        default=PROJECT_ROOT / "data/2026-10-13_object_mmes",
    )
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--primary-fft-size", type=int, default=2160)
    parser.add_argument("--reference-fft-size", type=int, default=4320)
    parser.add_argument("--kernel-radius", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/orpam_model_b_validation",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/orpam_model_b_validation",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/data/orpam_model_b_validation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs/orpam_model_b_validation.md",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.num_frames <= 0 or args.num_frames > 50:
        raise ValueError("num_frames must be in [1,50].")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    library = HardwareSLMPhaseLibrary(args.phase_root)
    mapping = library.validate_measurement_mapping(args.measurement_dir)
    png_phase_errors = [
        library.validate_png_correspondence(frame)
        for frame in range(1, args.num_frames + 1)
    ]
    coefficients = np.load(library.root / library.manifest["coefficients"]["npy"])
    object_np = make_fibre_phantom(600)
    object_tensor = torch.as_tensor(object_np, device=device)[None, None]
    model_size = int(library.manifest["model_shape"][0])
    aberrations_a = {
        CONDITION_ZERO: torch.zeros((model_size, model_size), device=device),
        CONDITION_NONZERO: artificial_aberration(model_size, device=device),
    }
    aberrations_b = {
        CONDITION_ZERO: torch.zeros(library.shape, device=device),
        CONDITION_NONZERO: artificial_aberration(library.shape[0], device=device),
    }
    config = ORPAMModelBConfig(
        slm_size=library.shape[0],
        fft_size=args.primary_fft_size,
        phase_sign=library.phase_sign,
        kernel_radius=args.kernel_radius,
        quadrature_order=args.quadrature_order,
    )
    model_b = ORPAMForwardModelB(config)

    rows: list[dict] = []
    print(f"Running {args.num_frames} real SLM frames at NFFT={args.primary_fft_size}...", flush=True)
    with torch.inference_mode():
        for frame in range(1, args.num_frames + 1):
            hardware_phase = library.load(frame, device=device)
            model_phase = torch.as_tensor(load_model_phase(library, frame), device=device)
            phase_rms = float((hardware_phase - hardware_phase.mean()).std())
            coefficient_l2 = float(np.linalg.norm(coefficients[frame - 1]))
            for condition in (CONDITION_ZERO, CONDITION_NONZERO):
                psf_a, image_a = model_a_forward(
                    object_tensor,
                    model_phase,
                    aberrations_a[condition],
                    library.phase_sign,
                )
                optical = model_b.optical_psf(hardware_phase, aberrations_b[condition])
                kernel = model_b.equivalent_kernel(optical.intensity)
                image_b = model_b.scan_forward(object_tensor, kernel)
                metrics_a = psf_metrics(psf_a, 1.0)
                metrics_b = psf_metrics(optical.intensity, config.focus_pixel_pitch_m)
                radius = args.kernel_radius
                interior_a = image_a[..., radius:-radius, radius:-radius]
                interior_b = image_b[..., radius:-radius, radius:-radius]
                retained = float(kernel.sum())
                rows.append(
                    {
                        "frame": frame,
                        "condition": condition,
                        "phase_rms_rad": phase_rms,
                        "coefficient_l2_rad": coefficient_l2,
                        "model_a_psf_peak": metrics_a["peak"],
                        "model_a_centroid_row_pixels": metrics_a["centroid_row"],
                        "model_a_centroid_column_pixels": metrics_a["centroid_column"],
                        "model_a_rms_radius_pixels": metrics_a["rms_radius"],
                        "model_b_psf_peak_inverse_m2": metrics_b["peak"],
                        "model_b_centroid_row_um": metrics_b["centroid_row"] * 1e6,
                        "model_b_centroid_column_um": metrics_b["centroid_column"] * 1e6,
                        "model_b_rms_radius_um": metrics_b["rms_radius"] * 1e6,
                        "model_b_peak_row_um": metrics_b["peak_row"] * 1e6,
                        "model_b_peak_column_um": metrics_b["peak_column"] * 1e6,
                        "kernel_retained_energy": retained,
                        "kernel_truncation_error": 1.0 - retained,
                        "ab_normalized_l2": normalized_l2(image_a, image_b),
                        "ab_interior_normalized_l2": normalized_l2(interior_a, interior_b),
                    }
                )
            if frame % 10 == 0 or frame == args.num_frames:
                print(f"  completed frame {frame}/{args.num_frames}", flush=True)

    _save_csv(args.data_dir / "frame_metrics.csv", rows)
    condition_rows = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in (CONDITION_ZERO, CONDITION_NONZERO)
    }
    condition_summary = {
        condition: {
            "ab_normalized_l2": describe(subset, "ab_normalized_l2"),
            "kernel_retained_energy": describe(subset, "kernel_retained_energy"),
            "kernel_truncation_error": describe(subset, "kernel_truncation_error"),
            "model_b_psf_peak_inverse_m2": describe(
                subset, "model_b_psf_peak_inverse_m2"
            ),
            "model_b_rms_radius_um": describe(subset, "model_b_rms_radius_um"),
            "model_b_centroid_row_um": describe(
                subset, "model_b_centroid_row_um"
            ),
            "model_b_centroid_column_um": describe(
                subset, "model_b_centroid_column_um"
            ),
        }
        for condition, subset in condition_rows.items()
    }
    correlations = {
        condition: {
            "difference_vs_phase_rms": correlation(
                subset, "phase_rms_rad", "ab_normalized_l2"
            ),
            "difference_vs_coefficient_l2": correlation(
                subset, "coefficient_l2_rad", "ab_normalized_l2"
            ),
            "difference_vs_psf_rms": correlation(
                subset, "model_b_rms_radius_um", "ab_normalized_l2"
            ),
        }
        for condition, subset in condition_rows.items()
    }
    zero_description = condition_summary[CONDITION_ZERO]["ab_normalized_l2"]
    representatives = {
        "minimum": int(zero_description["minimum_frame"]),
        "middle": int(zero_description["middle_frame"]),
        "maximum": int(zero_description["maximum_frame"]),
    }

    convergence_rows: list[dict] = []
    convergence_cache: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    median_psf: torch.Tensor | None = None
    print(f"Running representative convergence frames {representatives}...", flush=True)
    with torch.inference_mode():
        for frame in representatives.values():
            hardware_phase = library.load(frame, device=device)
            model_phase = torch.as_tensor(load_model_phase(library, frame), device=device)
            _, image_a = model_a_forward(
                object_tensor,
                model_phase,
                aberrations_a[CONDITION_ZERO],
                library.phase_sign,
            )
            outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            metrics_by_fft: dict[int, dict[str, float]] = {}
            for fft_size in sorted({1080, args.primary_fft_size, args.reference_fft_size}):
                current = ORPAMModelBConfig(
                    fft_size=fft_size,
                    kernel_radius=args.kernel_radius,
                    quadrature_order=args.quadrature_order,
                    phase_sign=library.phase_sign,
                )
                current_model = ORPAMForwardModelB(current)
                optical = current_model.optical_psf(
                    hardware_phase, aberrations_b[CONDITION_ZERO]
                )
                kernel = current_model.equivalent_kernel(optical.intensity)
                image = current_model.scan_forward(object_tensor, kernel)
                outputs[fft_size] = (kernel.clone(), image.clone())
                metrics_by_fft[fft_size] = {
                    "focus_pitch_um": current.focus_pixel_pitch_m * 1e6,
                    "kernel_retained_energy": float(kernel.sum()),
                    "ab_normalized_l2": normalized_l2(image_a, image),
                }
                if (
                    frame == representatives["middle"]
                    and fft_size == args.reference_fft_size
                ):
                    median_psf = optical.intensity.clone()
            reference_kernel, reference_image = outputs[args.reference_fft_size]
            for fft_size, (kernel, image) in outputs.items():
                convergence_rows.append(
                    {
                        "frame": frame,
                        "fft_size": fft_size,
                        **metrics_by_fft[fft_size],
                        "kernel_relative_l2_vs_4320": relative_l2(
                            kernel, reference_kernel
                        ),
                        "image_normalized_l2_vs_4320": normalized_l2(
                            image, reference_image
                        ),
                    }
                )
            convergence_cache[frame] = outputs
    if median_psf is None:
        raise RuntimeError("Median representative PSF was not retained.")
    _save_csv(args.data_dir / "convergence_metrics.csv", convergence_rows)

    median_frame = representatives["middle"]
    median_phase_hw = library.load(median_frame, device=device)
    median_phase_600 = torch.as_tensor(load_model_phase(library, median_frame), device=device)
    _, median_image_a = model_a_forward(
        object_tensor,
        median_phase_600,
        aberrations_a[CONDITION_ZERO],
        library.phase_sign,
    )
    kernel_1080_area, median_image_b = convergence_cache[median_frame][
        args.reference_fft_size
    ]
    reference_config = ORPAMModelBConfig(
        fft_size=args.reference_fft_size,
        kernel_radius=args.kernel_radius,
        quadrature_order=args.quadrature_order,
        phase_sign=library.phase_sign,
    )
    kernel_1080_point = integrate_equivalent_kernel(
        median_psf,
        focus_pixel_pitch_m=reference_config.focus_pixel_pitch_m,
        scan_pitch_m=reference_config.scan_pitch_m,
        kernel_radius=args.kernel_radius,
        quadrature_order=1,
    )
    image_1080_point = scan_correlation(object_tensor, kernel_1080_point)

    physical_600_pitch = reference_config.slm_width_m / model_size
    physical_600_fft = int(args.reference_fft_size * model_size / library.shape[0])
    config_600 = ORPAMModelBConfig(
        slm_size=model_size,
        slm_pixel_pitch_m=physical_600_pitch,
        fft_size=physical_600_fft,
        kernel_radius=args.kernel_radius,
        quadrature_order=args.quadrature_order,
        phase_sign=library.phase_sign,
    )
    model_600 = ORPAMForwardModelB(config_600)
    with torch.inference_mode():
        optical_600 = model_600.optical_psf(
            median_phase_600, aberrations_a[CONDITION_ZERO]
        )
        kernel_600_area = model_600.equivalent_kernel(optical_600.intensity)
        kernel_600_point = integrate_equivalent_kernel(
            optical_600.intensity,
            focus_pixel_pitch_m=config_600.focus_pixel_pitch_m,
            scan_pitch_m=config_600.scan_pitch_m,
            kernel_radius=args.kernel_radius,
            quadrature_order=1,
        )
        image_600_area = model_600.scan_forward(object_tensor, kernel_600_area)
        image_600_point = model_600.scan_forward(object_tensor, kernel_600_point)
        wrong_flipped = scan_correlation(
            object_tensor, kernel_1080_area.flip(-2, -1)
        )
        renormalized = scan_correlation(
            object_tensor, kernel_1080_area / kernel_1080_area.sum()
        )

    radius = args.kernel_radius
    factor = {
        "frame": median_frame,
        "legacy_a_vs_physical600_point": normalized_l2(
            median_image_a, image_600_point
        ),
        "physical600_point_vs_area": normalized_l2(
            image_600_point, image_600_area
        ),
        "physical600_area_vs_physical1080_area": normalized_l2(
            image_600_area, median_image_b
        ),
        "physical1080_point_vs_area": normalized_l2(
            image_1080_point, median_image_b
        ),
        "correct_vs_wrong_flipped_kernel": normalized_l2(
            median_image_b, wrong_flipped
        ),
        "truncated_vs_renormalized_normalized_l2": normalized_l2(
            median_image_b, renormalized
        ),
        "truncated_vs_renormalized_raw_relative_l2": relative_l2(
            renormalized, median_image_b
        ),
        "legacy_a_vs_b_global": normalized_l2(median_image_a, median_image_b),
        "legacy_a_vs_b_interior": normalized_l2(
            median_image_a[..., radius:-radius, radius:-radius],
            median_image_b[..., radius:-radius, radius:-radius],
        ),
        "physical600_focus_pitch_um": config_600.focus_pixel_pitch_m * 1e6,
        "physical1080_focus_pitch_um": reference_config.focus_pixel_pitch_m * 1e6,
        "physical600_kernel_energy": float(kernel_600_area.sum()),
        "physical1080_kernel_energy": float(kernel_1080_area.sum()),
    }
    _save_csv(args.data_dir / "factor_decomposition.csv", [factor])

    # Figures use only cropped arrays; full PSFs remain local/in-memory.
    with torch.inference_mode():
        zero_slm = torch.zeros(library.shape, device=device)
        ideal = ORPAMForwardModelB(reference_config).optical_psf(
            zero_slm, aberrations_b[CONDITION_ZERO]
        ).intensity
        aberrated = ORPAMForwardModelB(reference_config).optical_psf(
            zero_slm, aberrations_b[CONDITION_NONZERO]
        ).intensity
    focus_summary = _figure_focus(
        args.figure_dir / "figure_1_focus.png", ideal, aberrated, reference_config
    )
    _figure_psf_kernel(
        args.figure_dir / "figure_2_psf_kernel.png",
        median_psf,
        kernel_1080_area,
        kernel_1080_point,
        reference_config,
        median_frame,
    )
    _figure_images(
        args.figure_dir / "figure_3_model_ab_images.png",
        object_np,
        median_image_a,
        median_image_b,
        median_frame,
        factor["legacy_a_vs_b_global"],
    )
    _figure_statistics(
        args.figure_dir / "figure_4_frame_statistics.png", rows, correlations
    )
    _figure_convergence(
        args.figure_dir / "figure_5_convergence.png",
        convergence_rows,
        representatives,
    )

    summary = {
        "schema_version": 1,
        "phase_library": "data/slm50_noll4_28_decay4x_600_1080",
        "measurement_dataset": "data/2026-10-13_object_mmes",
        "validated_mapping_count": len(mapping),
        "validated_png_count": len(png_phase_errors),
        "maximum_mat_png_circular_phase_error_rad": max(png_phase_errors),
        "object": {
            "name": "deterministic three-line fibre phantom",
            "shape": [600, 600],
            "scan_pitch_um": 5.0,
        },
        "aberration": aberration_record(),
        "normalized_l2_definition": "||A/||A||_2 - B/||B||_2||_2",
        "settings": {
            "primary_fft_size": args.primary_fft_size,
            "reference_fft_size": args.reference_fft_size,
            "primary_focus_pitch_um": config.focus_pixel_pitch_m * 1e6,
            "reference_focus_pitch_um": reference_config.focus_pixel_pitch_m * 1e6,
            "kernel_radius": args.kernel_radius,
            "kernel_size": 2 * args.kernel_radius + 1,
            "quadrature_order": args.quadrature_order,
            "ideal_sinc2_fwhm_um": (
                0.8859
                * reference_config.wavelength_m
                * reference_config.focal_length_m
                / reference_config.slm_width_m
                * 1e6
            ),
        },
        "conditions": condition_summary,
        "correlations": correlations,
        "representatives": representatives,
        "factor_decomposition": factor,
        "ideal_focus": focus_summary,
        "convergence_rows": convergence_rows,
        "limitations": [
            "Model A has no physical focal-plane coordinate, so its discrete FFT and physical scaling cannot be completely orthogonalized.",
            "The controlled A/B comparison fixes pupil amplitude to one and does not measure the trained StaticDiffuseNet amplitude branch.",
            "The 65x65 Model-B kernel is not renormalized after truncation.",
            "No 50-frame 8640 FFT sweep was attempted.",
        ],
    }
    (args.data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(args.report, summary)
    print(f"Validation report written to {args.report}")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
