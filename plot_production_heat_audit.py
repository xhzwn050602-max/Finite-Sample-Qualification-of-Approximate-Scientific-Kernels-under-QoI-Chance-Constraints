"""Plot the larger-grid heat-conduction fallback audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("heat-production-audit-summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures-publication"))
    args = parser.parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))["summary"]
    order = ["light", "heavy", "correlated"]
    by_dist = {row["distribution"]: row for row in summary}
    colors = {"U": "#0072B2", "O": "#D55E00"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#D9D9D9", "grid.linewidth": 0.6,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    width = 0.34
    for i, policy in enumerate(("U", "O")):
        values = [by_dist[d]["u_mean_total_cost" if policy == "U" else "oblivious_mean_total_cost"] for d in order]
        axes[0].bar(x + (i - 0.5) * width, values, width, color=colors[policy],
                    label="O*" if policy == "O" else policy, hatch="//" if policy == "O" else None)
    axes[0].set_title("Accounted total cost")
    axes[0].set_ylabel("Mean accounted time (s)")
    axes[0].set_ylim(0, max(max(by_dist[d][key] for key in
                              ("u_mean_total_cost", "oblivious_mean_total_cost")) for d in order) * 1.25)
    axes[0].set_xticks(x, [d.capitalize() for d in order])
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x, [100.0 * by_dist[d]["u_fallback_rate"] for d in order], width=0.55, color="#666666")
    axes[1].set_title("U direct-reference fallback")
    axes[1].set_ylabel("Seed rows (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x, [d.capitalize() for d in order])
    fig.suptitle("64x64 non-manufactured heat-conduction audit", fontsize=11)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "production-heat-audit.pdf")
    fig.savefig(args.output_dir / "production-heat-audit.png", dpi=300)
    plt.close(fig)
    print(f"wrote {args.output_dir / 'production-heat-audit.pdf'}")


if __name__ == "__main__":
    main()
