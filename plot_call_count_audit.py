"""Plot the post hoc deployment call-count audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


CASE_LABELS = {
    "fft": "FFT",
    "spmv": "CSR SpMV",
    "stencil": "3-D stencil",
    "monte_carlo": "Monte Carlo",
    "2d_poisson": "Poisson",
}
COLORS = {
    "fft": "#0072B2",
    "spmv": "#D55E00",
    "stencil": "#009E73",
    "monte_carlo": "#CC79A7",
    "2d_poisson": "#6A3D9A",
}


def plot(input_path: Path, output_dir: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    data = json.loads(input_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in data["summaries"]:
        grouped[row["distribution"]].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    for ax, distribution in zip(axes, ("light", "heavy", "correlated")):
        for case in CASE_LABELS:
            rows = sorted(
                (row for row in grouped[distribution] if row["case"] == case),
                key=lambda row: row["n_calls"],
            )
            if not rows:
                continue
            x = [row["n_calls"] for row in rows]
            y = [100.0 * row["mean_u_savings_vs_o"] for row in rows]
            low = [100.0 * row["ci95_low"] for row in rows]
            high = [100.0 * row["ci95_high"] for row in rows]
            ax.plot(x, y, marker="o", linewidth=1.4, markersize=3.5,
                    color=COLORS[case], label=CASE_LABELS[case])
            ax.fill_between(x, low, high, color=COLORS[case], alpha=0.10, linewidth=0)
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(distribution.capitalize())
        ax.set_xlabel("Deployment calls N")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[0].set_ylabel("Mean U savings vs O* (%)")
    axes[-1].legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle("Call-count sensitivity of matched qualification cost", fontsize=11, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "call-count-audit.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "call-count-audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("call-count-audit-fair.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures-publication"))
    args = parser.parse_args()
    plot(args.input, args.output_dir)
    print(json.dumps({"output": str(args.output_dir / "call-count-audit.pdf")}))


if __name__ == "__main__":
    main()
