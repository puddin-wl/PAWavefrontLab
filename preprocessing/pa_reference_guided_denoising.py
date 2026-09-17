"""Reference-guided experiment-adaptive denoising for photoacoustic A-lines.

This module combines:
1) the validated fixed-template decorrelation core from preprocessing.pa_denoising;
2) adaptive signal/noise depth-window calibration from preprocessing.pa_adaptive_denoising.

The historical motor-crosstalk template acts only as a teacher. High-confidence
matches from the current experiment are aligned and robustly averaged to form a
new experiment-specific template. Final subtraction then reuses the old
normalized-xcorr + least-squares shifted-template algorithm. The final image is
formed inside the automatically detected PA signal window using excess RMS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal.windows import tukey

from preprocessing.pa_adaptive_denoising import (
    AdaptiveProjectionResult,
    CalibrationError,
    adaptive_excess_rms_projection,
    choose_noise_windows,
    detect_signal_window,
    robust_linear_detrend,
    sample_packed12_alines,
)
from preprocessing.pa_denoising import clean_traces
from preprocessing.packed12 import decode_packed12, packed12_byte_count

ALGORITHM_VERSION = "reference-guided-adaptive-aline-v1"


@dataclass(frozen=True)
class ReferenceGuidedTemplate:
    values: np.ndarray = field(repr=False)
    correlation_threshold: float = 0.70
    minimum_fitted_peak_adc: float = 80.0
    source_event_count: int = 0
    source_frame_count: int = 0
    median_teacher_ncc: float = 0.0
    teacher_to_refined_ncc: float = 0.0
    signal_window_match_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 1 or values.size < 3 or values.size % 2 != 1:
            raise ValueError("参考引导模板必须是一维、奇数长度且至少包含 3 点。")
        if not np.isfinite(values).all():
            raise ValueError("参考引导模板必须全部为有限值。")
        peak = float(np.max(np.abs(values)))
        if peak <= 0:
            raise ValueError("参考引导模板不能全为零。")
        object.__setattr__(self, "values", values / np.float32(peak))
        if not 0 < float(self.correlation_threshold) <= 1:
            raise ValueError("最终相关阈值必须位于 (0, 1]。")
        if float(self.minimum_fitted_peak_adc) < 0:
            raise ValueError("最小拟合峰值必须为非负数。")

    def metadata(self) -> dict:
        values = np.asarray(self.values, dtype="<f4")
        return {
            "role": "experiment-specific template learned from historical teacher",
            "length_samples": int(values.size),
            "relative_samples_inclusive": [
                -int(values.size // 2), int(values.size // 2)
            ],
            "correlation_threshold_absolute": float(self.correlation_threshold),
            "minimum_fitted_peak_adc": float(self.minimum_fitted_peak_adc),
            "source_event_count": int(self.source_event_count),
            "source_frame_count": int(self.source_frame_count),
            "median_teacher_ncc": float(self.median_teacher_ncc),
            "teacher_to_refined_ncc": float(self.teacher_to_refined_ncc),
            "signal_window_match_fraction": float(self.signal_window_match_fraction),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }


@dataclass(frozen=True)
class ReferenceGuidedCalibration:
    height: int
    width: int
    depth: int
    signal_window: tuple[int, int]
    noise_windows: tuple[tuple[int, int], tuple[int, int]]
    template: ReferenceGuidedTemplate
    teacher_sha256: str
    teacher_correlation_threshold: float
    teacher_refit_threshold: float
    seed: int = 0
    grid_size: int = 20
    qc: dict = field(default_factory=dict)
    source_records: tuple[dict, ...] = ()

    @property
    def templates(self) -> tuple[ReferenceGuidedTemplate, ...]:
        return (self.template,)

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "source_shape": [self.height, self.width, self.depth],
            "signal_window_start_inclusive_stop_exclusive": list(self.signal_window),
            "noise_windows_start_inclusive_stop_exclusive": [
                list(window) for window in self.noise_windows
            ],
            "seed": int(self.seed),
            "grid_size": int(self.grid_size),
            "spatial_filter": None,
            "projection": "signal-window depth-adaptive noise-power-subtracted RMS",
            "decorrelation": {
                "method": "historical teacher -> experiment template -> old xcorr+LS subtraction",
                "teacher_sha256": self.teacher_sha256,
                "teacher_correlation_threshold_absolute": float(
                    self.teacher_correlation_threshold
                ),
                "teacher_refit_threshold_absolute": float(self.teacher_refit_threshold),
                "final_match_threshold_absolute": float(
                    self.template.correlation_threshold
                ),
                "minimum_fitted_peak_adc": float(
                    self.template.minimum_fitted_peak_adc
                ),
                "maximum_subtractions_per_aline": 1,
            },
            "templates": [self.template.metadata()],
            "quality_control": self.qc,
            "sources": list(self.source_records),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.metadata(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(payload)
        digest.update(np.asarray(self.template.values, dtype="<f4").tobytes())
        return digest.hexdigest()


def _window_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return float(intersection / max(union, 1))


def _normalize_template(values: np.ndarray) -> np.ndarray:
    template = np.asarray(values, dtype=np.float32)
    if template.ndim != 1 or template.size < 3 or template.size % 2 != 1:
        raise ValueError("teacher_template 必须是一维奇数长度模板。")
    if not np.isfinite(template).all():
        raise ValueError("teacher_template 包含 NaN 或无穷值。")
    peak = float(np.max(np.abs(template)))
    if peak <= 0:
        raise ValueError("teacher_template 不能全为零。")
    return (template / np.float32(peak)).astype(np.float32, copy=False)


def _template_ncc(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("模板 NCC 比较要求长度相同。")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 0 else float(np.dot(a, b) / denom)


def fit_experiment_template_from_teacher(
    traces: np.ndarray,
    frame_ids: np.ndarray,
    signal_window: tuple[int, int],
    teacher_template: np.ndarray,
    *,
    teacher_correlation_threshold: float = 0.70,
    teacher_refit_threshold: float = 0.80,
    final_correlation_threshold: float = 0.70,
    minimum_fitted_peak_adc: float = 80.0,
    minimum_events: int = 50,
    minimum_frames: int = 5,
    maximum_events: int = 5000,
    minimum_teacher_to_refined_ncc: float = 0.75,
) -> tuple[ReferenceGuidedTemplate, dict]:
    """Identify current crosstalk with the old teacher and refit a new template."""
    values = np.asarray(traces, dtype=np.float32)
    ids = np.asarray(frame_ids, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("traces 必须是二维 A-line 数组。")
    if ids.shape != (values.shape[0],):
        raise ValueError("frame_ids 必须与 A-line 数量一致。")
    if teacher_refit_threshold < teacher_correlation_threshold:
        raise ValueError("teacher_refit_threshold 不应低于 teacher_correlation_threshold。")

    teacher = _normalize_template(teacher_template)

    # First pass: historical validated detector. The lower teacher threshold is
    # kept for diagnostics; only stronger matches are used to refit the template.
    _, best_corr, best_centers, coefficients, candidate_mask = clean_traces(
        values,
        teacher,
        correlation_threshold=teacher_correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
    )
    candidate_rows = np.flatnonzero(candidate_mask)
    strong = np.abs(best_corr[candidate_rows]) >= teacher_refit_threshold
    rows = candidate_rows[strong]

    if rows.size < minimum_events:
        raise CalibrationError(
            "旧模板在本实验中找到的高可信串扰事件不足，不能安全重拟合模板。",
            diagnostics={
                "teacher_candidate_threshold": float(teacher_correlation_threshold),
                "teacher_refit_threshold": float(teacher_refit_threshold),
                "candidate_event_count": int(candidate_rows.size),
                "refit_event_count": int(rows.size),
                "minimum_events": int(minimum_events),
            },
        )

    if rows.size > maximum_events:
        score = np.abs(best_corr[rows])
        keep = np.argpartition(score, -maximum_events)[-maximum_events:]
        rows = rows[keep]

    frame_count = int(np.unique(ids[rows]).size)
    if frame_count < minimum_frames:
        raise CalibrationError(
            "高可信串扰事件集中在过少测量帧，不能代表整次实验。",
            diagnostics={
                "refit_event_count": int(rows.size),
                "matched_frame_count": frame_count,
                "minimum_frames": int(minimum_frames),
            },
        )

    centered = values - np.median(values, axis=1, keepdims=True)
    length = teacher.size
    half = length // 2
    relative = np.arange(-half, half + 1, dtype=np.int32)
    aligned = np.full((rows.size, length), np.nan, dtype=np.float32)

    # Scale all events to unit fitted teacher peak and unify polarity. This is
    # the automated counterpart of the old N1-N6 positive-peak alignment.
    fitted_peaks = np.abs(coefficients[rows]) * float(np.max(np.abs(teacher)))
    polarities = np.sign(coefficients[rows]).astype(np.float32)
    polarities[polarities == 0] = 1.0
    for out_row, (source_row, center, fitted_peak, polarity) in enumerate(
        zip(rows, best_centers[rows], fitted_peaks, polarities)
    ):
        source_indices = relative + int(center)
        valid = (source_indices >= 0) & (source_indices < values.shape[1])
        aligned[out_row, valid] = (
            centered[source_row, source_indices[valid]]
            * np.float32(polarity / max(float(fitted_peak), 1e-6))
        )

    support = np.sum(np.isfinite(aligned), axis=0)
    with np.errstate(invalid="ignore"):
        refined = np.nanmedian(aligned, axis=0).astype(np.float32)
    min_support = max(5, int(np.ceil(rows.size * 0.05)))
    fallback = (support < min_support) | ~np.isfinite(refined)
    refined[fallback] = teacher[fallback]

    # Preserve the successful historical template cleanup steps.
    edge_count = min(15, max(3, length // 20))
    left = float(np.median(refined[:edge_count]))
    right = float(np.median(refined[-edge_count:]))
    refined -= np.linspace(left, right, length, dtype=np.float32)
    refined *= tukey(length, alpha=0.15).astype(np.float32)
    peak = float(np.max(np.abs(refined)))
    if peak <= 0:
        raise CalibrationError("重拟合后的实验串扰模板没有有效幅值。")
    refined /= np.float32(peak)

    teacher_to_refined = _template_ncc(teacher, refined)
    if teacher_to_refined < 0:
        refined *= np.float32(-1.0)
        teacher_to_refined = -teacher_to_refined
    if teacher_to_refined < minimum_teacher_to_refined_ncc:
        raise CalibrationError(
            "本实验重拟合模板与历史电机串扰模板不一致。",
            diagnostics={
                "teacher_to_refined_ncc": float(teacher_to_refined),
                "minimum_required_ncc": float(minimum_teacher_to_refined_ncc),
            },
        )

    centers = best_centers[rows].astype(np.int32)
    in_signal = (centers >= signal_window[0]) & (centers < signal_window[1])
    model = ReferenceGuidedTemplate(
        values=refined,
        correlation_threshold=final_correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
        source_event_count=int(rows.size),
        source_frame_count=frame_count,
        median_teacher_ncc=float(np.median(np.abs(best_corr[rows]))),
        teacher_to_refined_ncc=float(teacher_to_refined),
        signal_window_match_fraction=float(np.mean(in_signal)),
    )
    diagnostics = {
        "method": "historical teacher template identifies current-experiment events",
        "teacher_candidate_threshold_absolute": float(teacher_correlation_threshold),
        "teacher_refit_threshold_absolute": float(teacher_refit_threshold),
        "final_correlation_threshold_absolute": float(final_correlation_threshold),
        "minimum_fitted_peak_adc": float(minimum_fitted_peak_adc),
        "candidate_event_count": int(candidate_rows.size),
        "refit_event_count": int(rows.size),
        "matched_frame_count": frame_count,
        "teacher_ncc_p10_p50_p90": [
            float(v) for v in np.percentile(np.abs(best_corr[rows]), (10, 50, 90))
        ],
        "teacher_to_refined_ncc": float(teacher_to_refined),
        "signal_window_match_fraction": float(np.mean(in_signal)),
        "template_support_fallback_points": int(np.sum(fallback)),
    }
    return model, diagnostics


def calibrate_from_samples_reference_guided(
    traces_by_frame: Sequence[np.ndarray],
    *,
    teacher_template: np.ndarray,
    height: int,
    width: int,
    depth: int,
    seed: int = 0,
    grid_size: int = 20,
    teacher_correlation_threshold: float = 0.70,
    teacher_refit_threshold: float = 0.80,
    final_correlation_threshold: float = 0.70,
    minimum_fitted_peak_adc: float = 80.0,
    source_records: Sequence[dict] = (),
) -> ReferenceGuidedCalibration:
    if len(traces_by_frame) < 5:
        raise CalibrationError("五折自动标定至少需要 5 个测量帧。")
    frames = [np.asarray(frame, dtype=np.float32) for frame in traces_by_frame]
    if any(
        frame.ndim != 2 or frame.shape[1] != depth or not np.isfinite(frame).all()
        for frame in frames
    ):
        raise ValueError("每个标定帧必须是有限的 (A-line 数, depth) 数组。")

    fold_windows = []
    for fold in range(5):
        training = np.concatenate(
            [frame for i, frame in enumerate(frames) if i % 5 != fold], axis=0
        )
        window, _ = detect_signal_window(training)
        fold_windows.append(window)
    minimum_iou = min(
        _window_iou(first, second)
        for i, first in enumerate(fold_windows)
        for second in fold_windows[i + 1 :]
    )
    if minimum_iou < 0.85:
        raise CalibrationError(
            f"五折信号窗口不稳定（最小 IoU={minimum_iou:.3f}）。",
            diagnostics={"fold_signal_windows": [list(w) for w in fold_windows]},
        )

    all_traces = np.concatenate(frames, axis=0)
    frame_ids = np.concatenate(
        [np.full(frame.shape[0], i, dtype=np.int32) for i, frame in enumerate(frames)]
    )
    signal_window, signal_diag = detect_signal_window(all_traces)
    if float(signal_diag["significant_energy_coverage"]) < 0.995:
        raise CalibrationError("信号窗口显著能量覆盖率不足。")
    noise_windows, noise_diag = choose_noise_windows(all_traces, signal_window)

    teacher = _normalize_template(teacher_template)
    model, template_diag = fit_experiment_template_from_teacher(
        all_traces,
        frame_ids,
        signal_window,
        teacher,
        teacher_correlation_threshold=teacher_correlation_threshold,
        teacher_refit_threshold=teacher_refit_threshold,
        final_correlation_threshold=final_correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
    )

    per_frame_windows = []
    per_frame_ious = []
    for frame in frames:
        try:
            frame_window, _ = detect_signal_window(frame)
        except CalibrationError:
            per_frame_windows.append(None)
            continue
        per_frame_windows.append(list(frame_window))
        per_frame_ious.append(_window_iou(frame_window, signal_window))
    overlapping = int(sum(value >= 0.50 for value in per_frame_ious))
    required = int(np.ceil(len(frames) * 0.60))
    if overlapping < required:
        raise CalibrationError("多数测量帧的信号深度无法复现实验级窗口。")

    qc = {
        "passed": True,
        "calibration_strategy": "reference_guided",
        "fold_signal_windows": [list(w) for w in fold_windows],
        "minimum_fold_window_iou": float(minimum_iou),
        "per_frame_signal_windows": per_frame_windows,
        "frames_overlapping_experiment_gate": overlapping,
        "required_overlapping_frames": required,
        "signal": {k: v for k, v in signal_diag.items() if k not in {"profile", "evidence"}},
        "noise": noise_diag,
        "reference_guided_template": template_diag,
        "no_spatial_filtering": True,
    }
    teacher_sha256 = hashlib.sha256(
        np.asarray(teacher, dtype="<f4").tobytes()
    ).hexdigest()
    calibration = ReferenceGuidedCalibration(
        height=height,
        width=width,
        depth=depth,
        signal_window=signal_window,
        noise_windows=noise_windows,
        template=model,
        teacher_sha256=teacher_sha256,
        teacher_correlation_threshold=teacher_correlation_threshold,
        teacher_refit_threshold=teacher_refit_threshold,
        seed=seed,
        grid_size=grid_size,
        qc=qc,
        source_records=tuple(source_records),
    )
    object.__setattr__(calibration, "_signal_profile", signal_diag["profile"])
    object.__setattr__(calibration, "_signal_evidence", signal_diag["evidence"])
    return calibration


def calibrate_experiment_reference_guided(
    sources: Sequence[str | Path],
    *,
    teacher_template: np.ndarray,
    height: int,
    width: int,
    depth: int,
    grid_size: int = 20,
    seed: int = 0,
    teacher_correlation_threshold: float = 0.70,
    teacher_refit_threshold: float = 0.80,
    final_correlation_threshold: float = 0.70,
    minimum_fitted_peak_adc: float = 80.0,
) -> ReferenceGuidedCalibration:
    traces_by_frame = []
    records = []
    for source_value in sources:
        path = Path(source_value).expanduser().resolve()
        stat = path.stat()
        traces_by_frame.append(
            sample_packed12_alines(
                path,
                height=height,
                width=width,
                depth=depth,
                grid_size=grid_size,
            )
        )
        records.append(
            {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        )
    return calibrate_from_samples_reference_guided(
        traces_by_frame,
        teacher_template=teacher_template,
        height=height,
        width=width,
        depth=depth,
        seed=seed,
        grid_size=grid_size,
        teacher_correlation_threshold=teacher_correlation_threshold,
        teacher_refit_threshold=teacher_refit_threshold,
        final_correlation_threshold=final_correlation_threshold,
        minimum_fitted_peak_adc=minimum_fitted_peak_adc,
        source_records=records,
    )


def load_packed12_reference_guided_projection(
    source: str | Path,
    *,
    calibration: ReferenceGuidedCalibration,
    backend: str = "cpu",
    chunk_rows: int = 600,
    gpu_pipeline=None,
) -> AdaptiveProjectionResult:
    """Project one packed-12 volume using CPU or the pipelined CUDA path.

    CUDA mode uses two pinned packed-byte buffers.  While a dedicated GPU
    worker processes chunk N, the main thread reads chunk N+1 into the other
    pinned buffer and enqueues its asynchronous H2D copy.  Compute itself stays
    serial, so a 16-GB GPU does not need two simultaneous FFT workspaces.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows 必须是正整数。")
    path = Path(source).expanduser().resolve()
    expected = packed12_byte_count(
        calibration.height * calibration.width * calibration.depth
    )
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path} 大小错误：应为 {expected} 字节，实际为 {path.stat().st_size}。"
        )
    samples_per_row = calibration.width * calibration.depth
    if samples_per_row % 2:
        raise ValueError("流式 packed-12 解码要求 width×depth 为偶数。")

    projection = np.empty((calibration.height, calibration.width), dtype=np.float32)
    total_matches = 0
    total_before = 0.0
    total_after = 0.0
    signal_slice = slice(*calibration.signal_window)
    row_starts = list(range(0, calibration.height, chunk_rows))

    if backend == "cuda":
        from preprocessing.pa_reference_guided_gpu import ReferenceGuidedGpuPipeline

        owns_gpu_pipeline = gpu_pipeline is None
        if gpu_pipeline is None:
            gpu_pipeline = ReferenceGuidedGpuPipeline(calibration)
        bytes_per_scan_row = samples_per_row * 3 // 2
        maximum_rows = min(chunk_rows, calibration.height)
        slots = gpu_pipeline.prepare_streaming(
            max_packed_bytes=maximum_rows * bytes_per_scan_row,
            max_alines=maximum_rows * calibration.width,
        )

        def read_exact_into(stream, destination: np.ndarray, nbytes: int) -> None:
            """Fill a pinned uint8 view without allocating a temporary ndarray."""
            view = memoryview(destination).cast("B")
            offset = 0
            while offset < nbytes:
                read = stream.readinto(view[offset:nbytes])
                if read is None:
                    continue
                if read == 0:
                    raise EOFError(f"读取 {path} 时意外到达文件末尾。")
                offset += int(read)

        def fill_slot(stream, slot, row_start: int):
            row_stop = min(row_start + chunk_rows, calibration.height)
            row_count = row_stop - row_start
            nbytes = row_count * bytes_per_scan_row
            read_exact_into(stream, slot.host, nbytes)
            gpu_pipeline.enqueue_h2d(slot, nbytes=nbytes)
            return row_start, row_count, slot

        def submit(executor, item):
            _row_start, row_count, slot = item
            return executor.submit(
                gpu_pipeline.process_preloaded_slot,
                slot,
                aline_count=row_count * calibration.width,
            )

        def store(item, result) -> None:
            nonlocal total_matches, total_before, total_after
            row_start, row_count, _slot = item
            projection[row_start : row_start + row_count] = result.projection.reshape(
                row_count, calibration.width
            )
            total_matches += int(result.matched_count)
            total_before += float(result.signal_power_before)
            total_after += float(result.signal_power_after)

        try:
            # With chunk_rows >= image height (the verified 600-row production
            # path), one BIN is one GPU batch.  Avoid a ThreadPoolExecutor and
            # condition-variable round trip because there is no next chunk to
            # overlap inside the file.  The multi-chunk path keeps ping-pong H2D.
            if len(row_starts) == 1:
                with path.open("rb", buffering=0) as stream:
                    current = fill_slot(stream, slots[0], row_starts[0])
                    try:
                        result = gpu_pipeline.process_preloaded_slot(
                            current[2],
                            aline_count=current[1] * calibration.width,
                        )
                    except gpu_pipeline.out_of_memory_error as exc:
                        raise RuntimeError(
                            "CUDA 整帧处理显存不足；请降低 --chunk-rows。"
                        ) from exc
                    store(current, result)
            else:
                # Unbuffered file I/O writes directly into page-locked buffers.
                # One GPU worker keeps compute launches ordered while the main
                # thread prepares the next pinned/H2D input slot.
                with path.open("rb", buffering=0) as stream, ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="pa-cuda"
                ) as executor:
                    current = fill_slot(stream, slots[0], row_starts[0])
                    current_future = submit(executor, current)

                    for position in range(1, len(row_starts)):
                        next_item = fill_slot(
                            stream, slots[position % 2], row_starts[position]
                        )
                        next_future = submit(executor, next_item)
                        try:
                            current_result = current_future.result()
                        except gpu_pipeline.out_of_memory_error as exc:
                            raise RuntimeError(
                                "CUDA 流水线显存不足；请降低 --chunk-rows。"
                            ) from exc
                        store(current, current_result)
                        current = next_item
                        current_future = next_future

                    try:
                        final_result = current_future.result()
                    except gpu_pipeline.out_of_memory_error as exc:
                        raise RuntimeError(
                            "CUDA 流水线显存不足；请降低 --chunk-rows。"
                        ) from exc
                    store(current, final_result)
        finally:
            # _process_values() already performs the single required frame-end
            # stream synchronization before returning CPU results.  Do not add a
            # redundant per-volume synchronize for a shared persistent pipeline.
            if owns_gpu_pipeline:
                gpu_pipeline.shutdown()

    elif backend == "cpu":
        with path.open("rb") as stream:
            for row_start in row_starts:
                row_stop = min(row_start + chunk_rows, calibration.height)
                row_count = row_stop - row_start
                pair_count = row_count * samples_per_row // 2
                packed = np.fromfile(stream, dtype=np.uint8, count=pair_count * 3)
                if packed.size != pair_count * 3:
                    raise EOFError(f"读取 {path} 时意外到达文件末尾。")
                raw = decode_packed12(packed).reshape(
                    row_count * calibration.width, calibration.depth
                )
                cleaned, _corr, _centers, _coeff, matched = clean_traces(
                    raw,
                    calibration.template.values,
                    correlation_threshold=calibration.template.correlation_threshold,
                    minimum_fitted_peak_adc=calibration.template.minimum_fitted_peak_adc,
                )
                before_residual, _ = robust_linear_detrend(raw)
                after_residual, _ = robust_linear_detrend(cleaned)
                total_before += float(
                    np.sum(before_residual[:, signal_slice] ** 2, dtype=np.float64)
                )
                total_after += float(
                    np.sum(after_residual[:, signal_slice] ** 2, dtype=np.float64)
                )
                total_matches += int(np.sum(matched))
                projected = adaptive_excess_rms_projection(cleaned, calibration)
                projection[row_start:row_stop] = projected.reshape(
                    row_count, calibration.width
                )
    else:
        raise ValueError(f"未知后端：{backend}")

    if not np.isfinite(projection).all() or float(np.min(projection)) < 0:
        raise RuntimeError("reference-guided A-line 投影产生了无效值。")
    aline_count = calibration.height * calibration.width
    return AdaptiveProjectionResult(
        projection=projection,
        template_match_counts=(int(total_matches),),
        template_match_fractions=(float(total_matches / aline_count),),
        signal_power_before=float(total_before),
        signal_power_after=float(total_after),
    )

def verify_source_records(records: Iterable[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        stat = path.stat()
        current = (int(stat.st_size), int(stat.st_mtime_ns))
        expected = (int(record["size"]), int(record["mtime_ns"]))
        if current != expected:
            raise RuntimeError(f"标定后原始 BIN 发生变化：{path}")
