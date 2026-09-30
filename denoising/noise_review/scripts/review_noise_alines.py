#!/usr/bin/env python3
"""Build a human-review packet for isolated PA A-line transients.

The script reads packed unsigned-12 PA volumes without modifying them.  It
uses spatial isolation and frame-to-frame instability only to *rank*
candidates; the exported plots and traces are intended for human labeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.packed12 import decode_packed12, packed12_byte_count


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _read_projection(
    path: Path, *, height: int, width: int, depth: int, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    expected = packed12_byte_count(height * width * depth)
    if path.stat().st_size != expected:
        raise ValueError(f"Unexpected file size for {path}: {path.stat().st_size} != {expected}")
    projection = np.empty((height, width), dtype=np.float32)
    peak_index = np.empty((height, width), dtype=np.uint16)
    samples_per_row = width * depth
    with path.open("rb") as stream:
        for row_start in range(0, height, chunk_rows):
            row_stop = min(height, row_start + chunk_rows)
            row_count = row_stop - row_start
            packed_count = row_count * samples_per_row * 3 // 2
            packed = np.fromfile(stream, dtype=np.uint8, count=packed_count)
            if packed.size != packed_count:
                raise EOFError(f"Unexpected end of file: {path}")
            traces = decode_packed12(packed).reshape(row_count * width, depth)
            values = traces.astype(np.float32)
            values -= np.median(values, axis=1, keepdims=True)
            indices = np.argmax(values, axis=1)
            rows = np.arange(values.shape[0])
            projection[row_start:row_stop] = np.maximum(values[rows, indices], 0).reshape(
                row_count, width
            )
            peak_index[row_start:row_stop] = indices.reshape(row_count, width)
    return projection, peak_index


def _read_aline(path: Path, *, y: int, x: int, width: int, depth: int) -> np.ndarray:
    byte_offset = (y * width + x) * depth * 3 // 2
    byte_count = depth * 3 // 2
    with path.open("rb") as stream:
        stream.seek(byte_offset)
        packed = np.fromfile(stream, dtype=np.uint8, count=byte_count)
    if packed.size != byte_count:
        raise EOFError(f"Could not read A-line y={y}, x={x} from {path}")
    trace = decode_packed12(packed).astype(np.float32)
    trace -= np.median(trace)
    return trace


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    absolute_deviation = np.abs(values - median)
    scale = float(np.median(absolute_deviation)) * 1.4826
    if scale <= 1e-6:
        # Isolation maps are sparse, so their ordinary MAD is often exactly
        # zero.  Use the typical non-zero deviation instead of an epsilon;
        # otherwise harmless ADC values turn into billion-sized display scores.
        nonzero = absolute_deviation[absolute_deviation > 0]
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
    return ((values - median) / scale).astype(np.float32)


def _pick_candidates(
    projections: np.ndarray,
    peak_indices: np.ndarray,
    *,
    signal_window: tuple[int, int],
    count: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    consensus = np.median(projections, axis=0).astype(np.float32)
    local_residuals = np.empty_like(projections)
    temporal_residuals = projections - consensus[None, :, :]
    scores = np.empty_like(projections)
    for frame in range(projections.shape[0]):
        local_background = median_filter(projections[frame], size=5, mode="reflect")
        local = np.maximum(projections[frame] - local_background, 0)
        temporal = np.maximum(temporal_residuals[frame], 0)
        local_residuals[frame] = local
        score = np.maximum(_robust_z(local), 0) + np.maximum(_robust_z(temporal), 0)
        # Out-of-window events receive a mild ranking preference, but in-window
        # isolated transients remain eligible for human review.
        start, stop = signal_window
        outside = (peak_indices[frame] < start) | (peak_indices[frame] >= stop)
        score += outside.astype(np.float32) * 1.5
        scores[frame] = score

    flat_order = np.argsort(scores.ravel())[::-1]
    selected: list[dict] = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for flat in flat_order:
        frame, y, x = np.unravel_index(int(flat), scores.shape)
        if x < 8 or y < 8 or x >= scores.shape[2] - 8 or y >= scores.shape[1] - 8:
            continue
        # Avoid near-duplicates within a frame while retaining events at the
        # same coordinate in different acquisitions.
        if any((y - yy) ** 2 + (x - xx) ** 2 < 12**2 for yy, xx in occupied.get(frame, [])):
            continue
        selected.append(
            {
                "candidate_id": f"C{len(selected) + 1:03d}",
                "frame_index": int(frame),
                "y": int(y),
                "x": int(x),
                "peak_index": int(peak_indices[frame, y, x]),
                "peak_adc": float(projections[frame, y, x]),
                "local_residual_adc": float(local_residuals[frame, y, x]),
                "temporal_residual_adc": float(temporal_residuals[frame, y, x]),
                "ranking_score": float(scores[frame, y, x]),
                "peak_inside_signal_window": bool(
                    signal_window[0] <= peak_indices[frame, y, x] < signal_window[1]
                ),
            }
        )
        occupied.setdefault(int(frame), []).append((int(y), int(x)))
        if len(selected) >= count:
            break
    return selected, consensus, local_residuals


def _save_overview(
    output: Path,
    *,
    candidates: list[dict],
    paths: list[Path],
    projections: np.ndarray,
    consensus: np.ndarray,
    width: int,
    depth: int,
    signal_window: tuple[int, int],
) -> None:
    samples = np.arange(depth)
    page_size = 8
    for page_start in range(0, len(candidates), page_size):
        subset = candidates[page_start : page_start + page_size]
        fig, axes = plt.subplots(
            len(subset), 3, figsize=(17, 2.65 * len(subset)), constrained_layout=True
        )
        if len(subset) == 1:
            axes = axes[None, :]
        for row, item in enumerate(subset):
            frame = item["frame_index"]
            y, x = item["y"], item["x"]
            projection = projections[frame]
            crop_radius = 18
            ys = slice(y - crop_radius, y + crop_radius + 1)
            xs = slice(x - crop_radius, x + crop_radius + 1)
            crop = projection[ys, xs]
            low, high = np.percentile(projection[projection > 0], (1, 99.7))
            axes[row, 0].imshow(crop, cmap="gray", origin="lower", vmin=low, vmax=high)
            axes[row, 0].scatter([crop_radius], [crop_radius], s=45, facecolors="none", edgecolors="#ff3030")
            axes[row, 0].set_title(
                f"{item['candidate_id']} {paths[frame].name.split('_')[0]} crop\n"
                f"(y={y}, x={x}), peak={item['peak_index']}"
            )
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            consensus_crop = consensus[ys, xs]
            c_low, c_high = np.percentile(consensus[consensus > 0], (1, 99.7))
            axes[row, 1].imshow(consensus_crop, cmap="gray", origin="lower", vmin=c_low, vmax=c_high)
            axes[row, 1].scatter([crop_radius], [crop_radius], s=45, facecolors="none", edgecolors="#00bfff")
            axes[row, 1].set_title("Cross-frame median at same location")
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

            trace = _read_aline(paths[frame], y=y, x=x, width=width, depth=depth)
            neighbors = []
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                neighbors.append(_read_aline(paths[frame], y=yy, x=xx, width=width, depth=depth))
            neighbor_median = np.median(np.asarray(neighbors), axis=0)
            axes[row, 2].plot(samples, trace, color="#d62728", linewidth=1.0, label="candidate")
            axes[row, 2].plot(samples, neighbor_median, color="#1f77b4", linewidth=0.9, alpha=0.9, label="4-neighbor median")
            axes[row, 2].axvspan(*signal_window, color="#68b36b", alpha=0.16)
            axes[row, 2].axvline(item["peak_index"], color="#d62728", linewidth=0.7, linestyle="--")
            axes[row, 2].axhline(0, color="#777777", linewidth=0.6)
            axes[row, 2].set_xlim(0, depth - 1)
            axes[row, 2].set_ylabel("relative ADC")
            axes[row, 2].grid(True, linewidth=0.45, alpha=0.45)
            axes[row, 2].legend(loc="upper right", fontsize=8, frameon=False)
            axes[row, 2].set_title(
                f"local +{item['local_residual_adc']:.0f}, temporal +{item['temporal_residual_adc']:.0f} ADC"
            )
        axes[-1, 2].set_xlabel("A-line sample")
        fig.suptitle(
            "PA transient candidates for human review — green band is provisional signal depth",
            fontsize=14,
        )
        fig.savefig(output / f"candidate_review_page_{page_start // page_size + 1:02d}.png", dpi=170)
        plt.close(fig)


def _save_trace_csv(
    destination: Path,
    *,
    candidates: list[dict],
    paths: list[Path],
    width: int,
    depth: int,
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "source_file", "y", "x", "sample_index", "relative_adc"])
        for item in candidates:
            path = paths[item["frame_index"]]
            trace = _read_aline(path, y=item["y"], x=item["x"], width=width, depth=depth)
            for index, value in enumerate(trace):
                writer.writerow(
                    [item["candidate_id"], path.name, item["y"], item["x"], index, f"{float(value):.6f}"]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--chunk-rows", type=int, default=20)
    parser.add_argument("--signal-start", type=int, default=222)
    parser.add_argument("--signal-stop", type=int, default=313)
    parser.add_argument("--candidate-count", type=int, default=40)
    args = parser.parse_args()

    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob(args.glob), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"No files matching {args.glob!r} in {source}")

    projections = np.empty((len(paths), args.height, args.width), dtype=np.float32)
    peak_indices = np.empty((len(paths), args.height, args.width), dtype=np.uint16)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        projections[index - 1], peak_indices[index - 1] = _read_projection(
            path,
            height=args.height,
            width=args.width,
            depth=args.depth,
            chunk_rows=args.chunk_rows,
        )

    signal_window = (args.signal_start, args.signal_stop)
    candidates, consensus, _ = _pick_candidates(
        projections,
        peak_indices,
        signal_window=signal_window,
        count=args.candidate_count,
    )
    for item in candidates:
        item["source_file"] = paths[item.pop("frame_index")].name
    # Restore frame indices for plotting while keeping source names in the JSON.
    path_index = {path.name: index for index, path in enumerate(paths)}
    for item in candidates:
        item["frame_index"] = path_index[item["source_file"]]

    _save_overview(
        output,
        candidates=candidates,
        paths=paths,
        projections=projections,
        consensus=consensus,
        width=args.width,
        depth=args.depth,
        signal_window=signal_window,
    )
    _save_trace_csv(
        output / "candidate_alines.csv",
        candidates=candidates,
        paths=paths,
        width=args.width,
        depth=args.depth,
    )
    report = {
        "purpose": "human review; ranking is not a final noise label",
        "source_dir": str(source),
        "source_files": [path.name for path in paths],
        "shape": [args.height, args.width, args.depth],
        "provisional_signal_window": list(signal_window),
        "ranking": "positive 5x5 local-median residual + positive residual from cross-frame median, mild preference for out-of-window peaks",
        "candidates": [{key: value for key, value in item.items() if key != "frame_index"} for item in candidates],
    }
    (output / "candidates.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output / "cross_frame_median_projection.npy", consensus)
    print(f"Review packet written to {output}")


if __name__ == "__main__":
    main()
