"""Plot the cross-size timing audit with log-log axes and uncertainty bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scaling-audit.json")
    parser.add_argument("--pdf", default="figures-publication/scaling-audit.pdf")
    parser.add_argument("--png", default="figures-publication/scaling-audit.png")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data["rows"]
    kernels = [("fft", "FFT"), ("spmv", "CSR SpMV"), ("stencil", "3-D stencil"), ("monte_carlo", "Monte Carlo")]
    colors = {"reference-f64": "#1f77b4", "approx-f32": "#d95f02"}
    labels = {"reference-f64": "reference f64", "approx-f32": "approx. f32"}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.3), constrained_layout=True)
    for axis, (kernel, title) in zip(axes.flat, kernels):
        for variant in colors:
            selected = sorted((row for row in rows if row["kernel"] == kernel and row["variant"] == variant), key=lambda row: row["work_units"])
            if not selected:
                continue
            x = [row["work_units"] for row in selected]
            y = [row["median_time_s"] for row in selected]
            lower = [row["median_time_s"] - row["p10_time_s"] for row in selected]
            upper = [row["p90_time_s"] - row["median_time_s"] for row in selected]
            axis.errorbar(x, y, yerr=[lower, upper], marker="o" if variant == "reference-f64" else "s", linewidth=1.6, capsize=2.5,
                          color=colors[variant], label=labels[variant])
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("dominant work units", fontsize=9)
        axis.set_ylabel("median time (s)", fontsize=9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
        axis.tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Cross-size CPU timing audit", fontsize=12)
    for output in (args.pdf, args.png):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
