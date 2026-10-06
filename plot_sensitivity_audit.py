"""Plot the Monte Carlo tolerance-sensitivity trace-replay audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sensitivity-audit-delivery.json")
    parser.add_argument("--output-dir", default="figures-publication")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["summary"]
        if row["case"] == "monte_carlo" and abs(row["delta"] - 0.10) < 1e-12
    ]
    if not rows:
        raise ValueError("no Monte Carlo sensitivity rows found")
    by_scale = {}
    for row in rows:
        by_scale.setdefault(float(row["epsilon_scale"]), []).append(row)
    scales = np.array(sorted(by_scale))
    u_cov = np.array([np.mean([r["u_mean_test_coverage"] for r in by_scale[s]]) for s in scales])
    o_cov = np.array([np.mean([r["oblivious_mean_test_coverage"] for r in by_scale[s]]) for s in scales])
    fallback = np.array([np.mean([r["u_fallback_rate"] for r in by_scale[s]]) for s in scales])
    savings = 100.0 * np.array([np.mean([r["mean_relative_savings"] for r in by_scale[s]]) for s in scales])

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), constrained_layout=True)
    blue, orange, green = "#1f77b4", "#d95f02", "#238b45"

    axes[0].semilogx(scales, u_cov, marker="o", color=blue, label="U")
    axes[0].semilogx(scales, o_cov, marker="s", color=orange, label="O*")
    axes[0].axhline(0.90, color="#555555", linestyle="--", linewidth=0.9, label="target")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Mean test coverage")
    axes[0].set_title("Coverage")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7, handlelength=1.5)

    axes[1].semilogx(scales, fallback, marker="o", color=green)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("Fraction with no accepted candidate")
    axes[1].set_title("Qualification failure")

    axes[2].semilogx(scales, savings, marker="o", color="#6a3d9a")
    axes[2].axhline(0.0, color="#555555", linewidth=0.9)
    axes[2].set_ylim(min(-1.0, float(savings.min()) - 3.0), max(4.0, float(savings.max()) + 3.0))
    axes[2].set_ylabel("U savings vs O* (%)")
    axes[2].set_title("Cost trade-off")

    # Keep every observed point but label a readable subset of the log-scale
    # ticks; the full scale remains recoverable from the plotted markers and
    # the JSON audit.
    label_indices = list(range(0, len(scales), 2))
    if label_indices[-1] != len(scales) - 1:
        label_indices.append(len(scales) - 1)
    for axis in axes:
        axis.set_xlabel(r"Tolerance scale $\varepsilon/\varepsilon_0$")
        axis.set_xticks(scales)
        axis.set_xticklabels(
            [f"{value:g}" if i in label_indices else "" for i, value in enumerate(scales)],
            rotation=30,
            ha="right",
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "sensitivity-audit.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sensitivity-audit.png", dpi=300, bbox_inches="tight")
    print(json.dumps({"scales": scales.tolist(), "output": str(output_dir / "sensitivity-audit.pdf")}))


if __name__ == "__main__":
    main()
