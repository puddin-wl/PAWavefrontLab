#!/usr/bin/env python3
"""Randomly select one saved noise point and one continuous-signal point.

The corresponding 512-sample PA and PD A-lines are read directly from the
packed-12 acquisition files.  Noise is shown in red and continuous signal in
blue.  The random seed and selected coordinates are saved with the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = WORK_DIR / "aline_analysis" / "analysis.json"
DEFAULT_OUTPUT = WORK_DIR / "results" / "random_pa_pd_alines_v1"
WIDTH = 600
DEPTH = 512


def read_packed12_aline(source: Path, *, y: int, x: int, width: int, depth: int) -> np.ndarray:
    """Read one contiguous A-line from packed little-endian unsigned-12 data."""
    if depth % 2:
        raise ValueError("depth must be even for packed-12 pair decoding")
    byte_offset = (y * width + x) * depth * 3 // 2
    byte_count = depth * 3 // 2
    with source.open("rb") as stream:
        stream.seek(byte_offset)
        packed = np.fromfile(stream, dtype=np.uint8, count=byte_count)
    if packed.size != byte_count:
        raise EOFError(f"Could not read A-line ({y}, {x}) from {source}")

    triples = packed.reshape(-1, 3)
    decoded = np.empty(depth, dtype=np.uint16)
    decoded[0::2] = triples[:, 0].astype(np.uint16) | (
        (triples[:, 1] & 0x0F).astype(np.uint16) << 8
    )
    decoded[1::2] = (triples[:, 1] >> 4).astype(np.uint16) | (
        triples[:, 2].astype(np.uint16) << 4
    )
    return decoded


def baseline_correct(trace: np.ndarray, stop: int = 220) -> tuple[np.ndarray, float]:
    baseline = float(np.median(trace[:stop]))
    return trace.astype(np.float32) - baseline, baseline


def choose_points(analysis: dict, seed: int) -> tuple[dict, dict]:
    candidates = analysis["selected_points"]
    noise = sorted((point for point in candidates if point["category"] == "noise"), key=lambda p: p["label"])
    signal = sorted((point for point in candidates if point["category"] == "signal"), key=lambda p: p["label"])
    if not noise or not signal:
        raise ValueError("analysis.json does not contain both noise and signal candidates")
    generator = random.Random(seed)
    return generator.choice(noise), generator.choice(signal)


def save_csv(
    destination: Path,
    *,
    time_us: np.ndarray,
    noise_pa: np.ndarray,
    signal_pa: np.ndarray,
    noise_pd: np.ndarray,
    signal_pd: np.ndarray,
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_index",
                "absolute_time_us",
                "noise_pa_relative_adc",
                "continuous_signal_pa_relative_adc",
                "noise_pd_relative_adc",
                "continuous_signal_pd_relative_adc",
            ]
        )
        for index in range(time_us.size):
            writer.writerow(
                [
                    index,
                    f"{time_us[index]:.6f}",
                    f"{noise_pa[index]:.3f}",
                    f"{signal_pa[index]:.3f}",
                    f"{noise_pd[index]:.3f}",
                    f"{signal_pd[index]:.3f}",
                ]
            )


def save_figure(
    destination: Path,
    *,
    samples: np.ndarray,
    noise_pa: np.ndarray,
    signal_pa: np.ndarray,
    noise_pd: np.ndarray,
    signal_pd: np.ndarray,
    noise_point: dict,
    signal_point: dict,
    gate_start: int,
    gate_stop: int,
) -> None:
    noise_color = "#d62728"
    signal_color = "#1f77b4"
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True, constrained_layout=True)

    for axis in axes:
        axis.axvspan(gate_start, gate_stop, color="#b8b8b8", alpha=0.22, label="PA gate 250:320")
        axis.grid(True, color="#d7d7d7", linewidth=0.6, alpha=0.65)
        axis.set_xlim(0, samples[-1])

    axes[0].plot(
        samples,
        noise_pa,
        color=noise_color,
        linewidth=1.25,
        label=f"Noise {noise_point['label']}  (y={noise_point['y']}, x={noise_point['x']})",
    )
    axes[0].plot(
        samples,
        signal_pa,
        color=signal_color,
        linewidth=1.25,
        label=f"Continuous signal {signal_point['label']}  (y={signal_point['y']}, x={signal_point['x']})",
    )
    noise_pa_peak = int(np.argmax(noise_pa))
    signal_pa_peak = int(np.argmax(signal_pa))
    axes[0].scatter([noise_pa_peak], [noise_pa[noise_pa_peak]], color=noise_color, s=34, zorder=5)
    axes[0].scatter([signal_pa_peak], [signal_pa[signal_pa_peak]], color=signal_color, s=34, zorder=5)
    axes[0].annotate(
        f"noise peak: {noise_pa_peak}",
        (noise_pa_peak, noise_pa[noise_pa_peak]),
        xytext=(-82, 18),
        textcoords="offset points",
        color=noise_color,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": noise_color, "linewidth": 0.8},
    )
    axes[0].annotate(
        f"signal peak: {signal_pa_peak}",
        (signal_pa_peak, signal_pa[signal_pa_peak]),
        xytext=(18, 18),
        textcoords="offset points",
        color=signal_color,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": signal_color, "linewidth": 0.8},
    )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("PA amplitude (relative ADC)")
    axes[0].set_title("PA A-lines")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)

    axes[1].plot(
        samples,
        noise_pd,
        color=noise_color,
        linewidth=1.7,
        label=f"Noise-point PD {noise_point['label']}",
    )
    axes[1].plot(
        samples,
        signal_pd,
        color=signal_color,
        linewidth=1.7,
        linestyle="--",
        label=f"Signal-point PD {signal_point['label']}",
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("A-line sample index (2 ns/sample)")
    axes[1].set_ylabel("PD amplitude (relative ADC)")
    axes[1].set_title("PD A-lines at the same spatial positions")
    axes[1].legend(loc="upper left", frameon=False, ncol=2)

    fig.suptitle(
        "Random noise point versus continuous PA signal\n"
        "red = noise point, blue = continuous signal",
        fontsize=15,
    )
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--sampling-rate-msps", type=float, default=500.0)
    parser.add_argument("--trigger-delay-us", type=float, default=12.3)
    parser.add_argument("--gate-start", type=int, default=250)
    parser.add_argument("--gate-stop", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_path = args.analysis.expanduser().resolve()
    output = args.output.expanduser().resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    pa_source = Path(analysis["source_pa"])
    pd_source = Path(analysis["source_pd"])
    noise_point, signal_point = choose_points(analysis, args.seed)

    traces: dict[str, np.ndarray] = {}
    baselines: dict[str, float] = {}
    for category, point in (("noise", noise_point), ("signal", signal_point)):
        for channel, source in (("pa", pa_source), ("pd", pd_source)):
            raw = read_packed12_aline(
                source,
                y=int(point["y"]),
                x=int(point["x"]),
                width=args.width,
                depth=args.depth,
            )
            traces[f"{category}_{channel}"], baselines[f"{category}_{channel}"] = baseline_correct(raw)

    samples = np.arange(args.depth)
    time_us = args.trigger_delay_us + samples / args.sampling_rate_msps
    output.mkdir(parents=True, exist_ok=True)
    save_csv(
        output / "random_pa_pd_alines.csv",
        time_us=time_us,
        noise_pa=traces["noise_pa"],
        signal_pa=traces["signal_pa"],
        noise_pd=traces["noise_pd"],
        signal_pd=traces["signal_pd"],
    )
    save_figure(
        output / "random_pa_pd_alines.png",
        samples=samples,
        noise_pa=traces["noise_pa"],
        signal_pa=traces["signal_pa"],
        noise_pd=traces["noise_pd"],
        signal_pd=traces["signal_pd"],
        noise_point=noise_point,
        signal_point=signal_point,
        gate_start=args.gate_start,
        gate_stop=args.gate_stop,
    )

    noise_pa_peak = int(np.argmax(traces["noise_pa"]))
    signal_pa_peak = int(np.argmax(traces["signal_pa"]))
    noise_pd_peak = int(np.argmax(traces["noise_pd"]))
    signal_pd_peak = int(np.argmax(traces["signal_pd"]))
    result = {
        "random_seed": args.seed,
        "selection_pool": "six saved representative candidates per category in analysis.json",
        "noise_point": {k: noise_point[k] for k in ("label", "y", "x")},
        "continuous_signal_point": {k: signal_point[k] for k in ("label", "y", "x")},
        "pa_source": str(pa_source),
        "pd_source": str(pd_source),
        "baseline_method": "subtract median of samples 0:220 independently for each trace",
        "sampling_rate_msps": args.sampling_rate_msps,
        "trigger_delay_us": args.trigger_delay_us,
        "gate_samples_start_stop": [args.gate_start, args.gate_stop],
        "noise_pa_peak": {
            "sample": noise_pa_peak,
            "time_us": float(time_us[noise_pa_peak]),
            "amplitude_relative_adc": float(traces["noise_pa"][noise_pa_peak]),
        },
        "continuous_signal_pa_peak": {
            "sample": signal_pa_peak,
            "time_us": float(time_us[signal_pa_peak]),
            "amplitude_relative_adc": float(traces["signal_pa"][signal_pa_peak]),
        },
        "noise_pd_peak": {
            "sample": noise_pd_peak,
            "time_us": float(time_us[noise_pd_peak]),
            "amplitude_relative_adc": float(traces["noise_pd"][noise_pd_peak]),
        },
        "continuous_signal_pd_peak": {
            "sample": signal_pd_peak,
            "time_us": float(time_us[signal_pd_peak]),
            "amplitude_relative_adc": float(traces["signal_pd"][signal_pd_peak]),
        },
        "raw_baselines_adc": baselines,
        "colors": {"noise": "red", "continuous_signal": "blue"},
    }
    (output / "selection_and_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
