#!/usr/bin/env python3
"""Fit and validate a PA transient template from manually labeled A-lines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey


def load_traces(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(row["candidate_id"], []).append(float(row["relative_adc"]))
    traces = {key: np.asarray(values, dtype=np.float32) for key, values in rows.items()}
    if not traces or len({trace.size for trace in traces.values()}) != 1:
        raise ValueError("Candidate trace CSV is empty or has inconsistent trace lengths")
    return traces


def align_trace(trace: np.ndarray, *, half_width: int) -> tuple[np.ndarray, int, float]:
    centered = trace.astype(np.float32) - np.median(trace)
    peak = int(np.argmax(centered))
    amplitude = float(centered[peak])
    if amplitude <= 0:
        raise ValueError("Trace has no positive peak")
    relative = np.arange(-half_width, half_width + 1)
    source = relative + peak
    valid = (source >= 0) & (source < centered.size)
    aligned = np.full(relative.size, np.nan, dtype=np.float32)
    aligned[valid] = centered[source[valid]] / amplitude
    return aligned, peak, amplitude


def fit_template(traces: list[np.ndarray], *, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.asarray([align_trace(trace, half_width=half_width)[0] for trace in traces])
    with np.errstate(invalid="ignore"):
        template = np.nanmedian(aligned, axis=0).astype(np.float32)
    if not np.isfinite(template).all():
        raise RuntimeError("Template has unsupported relative samples")
    edge_count = min(15, max(3, template.size // 20))
    left = float(np.median(template[:edge_count]))
    right = float(np.median(template[-edge_count:]))
    template -= np.linspace(left, right, template.size, dtype=np.float32)
    template *= tukey(template.size, alpha=0.15).astype(np.float32)
    template /= float(np.max(np.abs(template)))
    if int(np.argmax(template)) != half_width:
        raise RuntimeError("Fitted template peak moved away from center")
    return template, aligned


def best_abs_ncc(trace: np.ndarray, template: np.ndarray) -> float:
    trace = trace.astype(np.float32) - np.median(trace)
    half_width = template.size // 2
    best = 0.0
    for center in range(trace.size):
        left = max(0, center - half_width)
        right = min(trace.size, center + half_width + 1)
        t_left = half_width - (center - left)
        t_right = t_left + (right - left)
        x = trace[left:right]
        t = template[t_left:t_right]
        denominator = float(np.linalg.norm(x) * np.linalg.norm(t))
        if denominator > 0:
            best = max(best, abs(float(np.dot(x, t)) / denominator))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--half-width", type=int, default=180)
    args = parser.parse_args()
    review = args.review_dir.expanduser().resolve()
    labels = json.loads((review / "manual_labels.json").read_text(encoding="utf-8"))
    traces = load_traces(review / "candidate_alines.csv")
    template_ids = labels["motor_crosstalk"]
    template, aligned = fit_template(
        [traces[candidate_id] for candidate_id in template_ids],
        half_width=args.half_width,
    )

    template_path = review / "manual_motor_crosstalk_template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_sample", "template_relative_adc"])
        for offset, value in zip(range(-args.half_width, args.half_width + 1), template):
            writer.writerow([offset, f"{float(value):.9f}"])

    groups = {
        "motor_crosstalk": labels["motor_crosstalk"],
        "other_noise": labels["other_noise_not_used_for_template"],
        "true_signal": labels["true_signal_excluded"],
        "ambiguous_or_signal": labels["conservative_signal_or_ambiguous_excluded"],
    }
    scores = {
        group: {candidate_id: best_abs_ncc(traces[candidate_id], template) for candidate_id in ids}
        for group, ids in groups.items()
    }
    report = {
        "template_source_ids": template_ids,
        "template_length": int(template.size),
        "template_peak_normalization": float(np.max(np.abs(template))),
        "best_absolute_ncc": scores,
        "group_ncc_summary": {
            group: {
                "min": float(np.min(list(values.values()))),
                "median": float(np.median(list(values.values()))),
                "max": float(np.max(list(values.values()))),
            }
            for group, values in scores.items()
        },
    }
    (review / "manual_template_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    relative = np.arange(-args.half_width, args.half_width + 1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for trace in aligned:
        axes[0].plot(relative, trace, color="#999999", alpha=0.28, linewidth=0.65)
    axes[0].plot(relative, template, color="#d62728", linewidth=2.0, label="robust median template")
    axes[0].axvline(0, color="#555555", linewidth=0.7)
    axes[0].set_title(f"Manually selected motor-crosstalk A-lines (n={len(template_ids)})")
    axes[0].set_ylabel("peak-normalized relative ADC")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.35)

    colors = {
        "motor_crosstalk": "#d62728",
        "other_noise": "#ff7f0e",
        "true_signal": "#1f77b4",
        "ambiguous_or_signal": "#2ca02c",
    }
    position = 0
    ticks = []
    tick_labels = []
    for group, values in scores.items():
        ordered = sorted(values.items())
        xs = np.arange(position, position + len(ordered))
        axes[1].scatter(xs, [value for _, value in ordered], s=28, color=colors[group], label=group)
        ticks.extend(xs.tolist())
        tick_labels.extend([key for key, _ in ordered])
        position += len(ordered) + 1
    axes[1].axhline(0.70, color="#555555", linestyle="--", linewidth=0.9, label="|NCC|=0.70")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("best absolute NCC")
    axes[1].set_xticks(ticks, tick_labels, rotation=90, fontsize=7)
    axes[1].set_title("Template similarity for all manually reviewed candidates")
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].legend(frameon=False, ncol=3)
    fig.savefig(review / "manual_template_and_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["group_ncc_summary"], indent=2))


if __name__ == "__main__":
    main()
