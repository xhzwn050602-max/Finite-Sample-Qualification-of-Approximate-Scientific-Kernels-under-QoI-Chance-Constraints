"""Render the heat iteration and data-quantization audit figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(input_path: Path, output_dir: Path) -> None:
    rows = list(csv.DictReader(input_path.open(encoding="utf-8")))
    distributions = ("light", "heavy", "correlated")
    precision_labels = {16: "diag/RHS: float16", 32: "diag/RHS: float32", 64: "diag/RHS: float64"}
    colors = {16: "#D55E00", 32: "#0072B2", 64: "#009E73"}
    linestyles = {16: "--", 32: ":", 64: "-"}
    markers = {16: "s", 32: "^", 64: "o"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.55,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True, constrained_layout=True)
    for column, distribution in enumerate(distributions):
        subset = [r for r in rows if r["distribution"] == distribution]
        # Draw the reference first so distinct styles remain visible when
        # data-quantization curves overlap it numerically.
        for precision in (64, 32, 16):
            values = sorted((r for r in subset if int(r["precision"]) == precision), key=lambda r: int(r["iterations"]))
            x = [int(r["iterations"]) for r in values]
            style = dict(marker=markers[precision], ms=3.2, linewidth=1.2,
                         linestyle=linestyles[precision], color=colors[precision],
                         label=precision_labels[precision])
            axes[0, column].plot(x, [float(r["field_error_p95"]) for r in values], **style)
            axes[1, column].plot(x, [float(r["residual_p95"]) for r in values], **style)
        axes[0, column].set_title(distribution.capitalize())
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks([1, 5, 10, 25, 50, 100, 200, 400, 800], ["1", "5", "10", "25", "50", "100", "200", "400", "800"])
        axes[1, column].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("p95 relative field error")
    axes[1, 0].set_ylabel("p95 relative residual")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("64x64 heat: diagonal/RHS quantization; float64 iteration arithmetic", fontsize=11)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stability-audit.pdf")
    fig.savefig(output_dir / "stability-audit.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("stability-audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures-publication"))
    args = parser.parse_args()
    plot(args.input, args.output_dir)
    print(f"wrote {args.output_dir / 'stability-audit.pdf'}")


if __name__ == "__main__":
    main()
