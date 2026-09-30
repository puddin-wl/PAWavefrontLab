#!/usr/bin/env python3
"""Plot saved noise candidates as separate top-to-bottom PA and PD lanes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_random_pa_pd_alines import baseline_correct, read_packed12_aline


WORK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = WORK_DIR / "aline_analysis" / "analysis.json"
DEFAULT_OUTPUT = WORK_DIR / "results" / "multiple_noise_alines_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--sampling-rate-msps", type=float, default=500.0)
    parser.add_argument("--trigger-delay-us", type=float, default=12.3)
    parser.add_argument("--gate-start", type=int, default=250)
    parser.add_argument("--gate-stop", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.expanduser().resolve().read_text(encoding="utf-8"))
    points = sorted(
        (point for point in analysis["selected_points"] if point["category"] == "noise"),
        key=lambda point: point["label"],
    )
    if not points:
        raise ValueError("No noise points were found in analysis.json")

    pa_source = Path(analysis["source_pa"])
    pd_source = Path(analysis["source_pd"])
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = np.arange(args.depth)
    time_us = args.trigger_delay_us + samples / args.sampling_rate_msps

    records: list[dict] = []
    pa_traces: list[np.ndarray] = []
    pd_traces: list[np.ndarray] = []
    for point in points:
        pa_raw = read_packed12_aline(
            pa_source,
            y=int(point["y"]),
            x=int(point["x"]),
            width=args.width,
            depth=args.depth,
        )
        pd_raw = read_packed12_aline(
            pd_source,
            y=int(point["y"]),
            x=int(point["x"]),
            width=args.width,
            depth=args.depth,
        )
        pa, pa_baseline = baseline_correct(pa_raw)
        pd, pd_baseline = baseline_correct(pd_raw)
        pa_traces.append(pa)
        pd_traces.append(pd)
        pa_peak = int(np.argmax(pa))
        pd_peak = int(np.argmax(pd))
        records.append(
            {
                "label": point["label"],
                "y": int(point["y"]),
                "x": int(point["x"]),
                "pa_peak_sample": pa_peak,
                "pa_peak_time_us": float(time_us[pa_peak]),
                "pa_peak_relative_adc": float(pa[pa_peak]),
                "pa_peak_inside_gate": bool(args.gate_start <= pa_peak < args.gate_stop),
                "pd_peak_sample": pd_peak,
                "pd_peak_time_us": float(time_us[pd_peak]),
                "pd_peak_relative_adc": float(pd[pd_peak]),
                "pa_baseline_adc": pa_baseline,
                "pd_baseline_adc": pd_baseline,
            }
        )

    with (output / "multiple_noise_pa_pd_alines.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "y",
                "x",
                "sample_index",
                "absolute_time_us",
                "pa_relative_adc",
                "pd_relative_adc",
            ]
        )
        for point, pa, pd in zip(points, pa_traces, pd_traces):
            for index in range(args.depth):
                writer.writerow(
                    [
                        point["label"],
                        point["y"],
                        point["x"],
                        index,
                        f"{time_us[index]:.6f}",
                        f"{pa[index]:.3f}",
                        f"{pd[index]:.3f}",
                    ]
                )

    pa_low = float(min(np.min(trace) for trace in pa_traces))
    pa_high = float(max(np.max(trace) for trace in pa_traces))
    pd_low = float(min(np.min(trace) for trace in pd_traces))
    pd_high = float(max(np.max(trace) for trace in pd_traces))
    pa_pad = max(10.0, (pa_high - pa_low) * 0.06)
    pd_pad = max(10.0, (pd_high - pd_low) * 0.06)

    noise_red = "#d62728"
    pd_blue = "#1f77b4"
    figure, axes = plt.subplots(
        len(points),
        2,
        figsize=(15.5, 3.0 * len(points)),
        sharex=True,
        constrained_layout=True,
    )
    if len(points) == 1:
        axes = np.asarray([axes])

    for row, (point, record, pa, pd) in enumerate(zip(points, records, pa_traces, pd_traces)):
        pa_axis, pd_axis = axes[row]
        for axis in (pa_axis, pd_axis):
            axis.axvspan(
                args.gate_start,
                args.gate_stop,
                color="#b8b8b8",
                alpha=0.20,
            )
            axis.axhline(0, color="#555555", linewidth=0.7)
            axis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.60)
            axis.set_xlim(0, args.depth - 1)

        pa_axis.plot(samples, pa, color=noise_red, linewidth=1.15)
        pa_axis.scatter(
            [record["pa_peak_sample"]],
            [record["pa_peak_relative_adc"]],
            color=noise_red,
            s=24,
            zorder=4,
        )
        pa_axis.set_ylim(pa_low - pa_pad, pa_high + pa_pad)
        pa_axis.set_ylabel("PA relative ADC")
        pa_axis.set_title(
            f"{point['label']} PA — (y={point['y']}, x={point['x']}), "
            f"peak={record['pa_peak_sample']}"
        )

        pd_axis.plot(samples, pd, color=pd_blue, linewidth=1.15)
        pd_axis.scatter(
            [record["pd_peak_sample"]],
            [record["pd_peak_relative_adc"]],
            color=pd_blue,
            s=24,
            zorder=4,
        )
        pd_axis.set_ylim(pd_low - pd_pad, pd_high + pd_pad)
        pd_axis.set_ylabel("PD relative ADC")
        pd_axis.set_title(
            f"{point['label']} PD — same position, peak={record['pd_peak_sample']}"
        )

    axes[-1, 0].set_xlabel("A-line sample index (2 ns/sample)")
    axes[-1, 1].set_xlabel("A-line sample index (2 ns/sample)")
    figure.suptitle(
        "Six noise candidates in separate top-to-bottom lanes\n"
        "red = PA, blue = PD; gray = PA response gate 250:320",
        fontsize=15,
    )
    figure.savefig(output / "multiple_noise_pa_pd_alines.png", dpi=180)
    plt.close(figure)

    summary = {
        "pa_source": str(pa_source),
        "pd_source": str(pd_source),
        "baseline_method": "subtract median of samples 0:220 independently for each trace",
        "sampling_rate_msps": args.sampling_rate_msps,
        "trigger_delay_us": args.trigger_delay_us,
        "gate_samples_start_stop": [args.gate_start, args.gate_stop],
        "layout": "one noise point per row; PA and PD are separate non-overlapping panels",
        "colors": {"pa": "red", "pd": "blue"},
        "points": records,
    }
    (output / "multiple_noise_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
