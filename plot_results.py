"""Create publication-style cost, coverage, and Poisson frontier figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_LABELS = {
    "fft": "FFT",
    "spmv": "CSR SpMV",
    "stencil": "3-D stencil",
    "monte_carlo": "Monte Carlo",
    "2d_poisson": "2-D Poisson",
}
DIST_ORDER = ("light", "heavy", "correlated")
DIST_LABELS = {"light": "Light", "heavy": "Heavy", "correlated": "Correlated"}
COLORS = {"U": "#0072B2", "O": "#D55E00", "target": "#555555"}
PRECISION_COLORS = {16: "#CC79A7", 32: "#E69F00", 64: "#56B4E9", "exact": "#009E73"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _percent1(value: float) -> str:
    return "0.0" if abs(value) < 0.05 else f"{value:.1f}"


def load_summaries(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"summary must be a JSON list: {path}")
        rows.extend(payload)
    return rows


def plot_cost(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), sharex="col",
                             gridspec_kw={"height_ratios": [1.0, 1.25]})
    for col, distribution in enumerate(DIST_ORDER):
        ax_top, ax_bottom = axes[:, col]
        subset = [r for r in rows if r["distribution"] == distribution]
        subset.sort(key=lambda r: list(CASE_LABELS).index(r["case"]))
        labels = [CASE_LABELS[r["case"]] for r in subset]
        x = np.arange(len(subset))
        top_savings = np.array([100.0 * r["u_relative_savings_mean"] for r in subset])
        ax_top.axhline(0.0, color="#222222", linewidth=0.8)
        top_colors = ["#0072B2" if value > 0.005 else "#C44E52" if value < -0.005 else "#9A9A9A"
                      for value in top_savings]
        top_bars = ax_top.bar(x, top_savings, color=top_colors, alpha=0.9, width=0.68)
        for bar, value in zip(top_bars, top_savings):
            va = "bottom" if value >= 0 else "top"
            offset = 0.08 if value >= 0 else -0.08
            ax_top.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}%",
                        ha="center", va=va, fontsize=7)

        # The second row exposes the substantive comparison with the exact
        # reference, which is hidden when U and O* select the same candidate.
        u_exact = np.array([
            100.0 * (r["exact_mean_total_cost"] - r["u_mean_total_cost"])
            / r["exact_mean_total_cost"] if r["exact_mean_total_cost"] else 0.0
            for r in subset
        ])
        o_exact = np.array([
            100.0 * (r["exact_mean_total_cost"] - r["oblivious_mean_total_cost"])
            / r["exact_mean_total_cost"] if r["exact_mean_total_cost"] else 0.0
            for r in subset
        ])
        width = 0.34
        ax_bottom.axhline(0.0, color="#222222", linewidth=0.8)
        u_bars = ax_bottom.bar(x - width / 2, u_exact, width, color="#0072B2", label="U")
        o_bars = ax_bottom.bar(x + width / 2, o_exact, width, color="#D55E00", label="O*")
        # Combine nearly identical labels so the high-budget Monte Carlo bars
        # remain legible at journal-column scale (order is U / O*).
        for i, (u_value, o_value) in enumerate(zip(u_exact, o_exact)):
            if abs(u_value - o_value) <= 0.25:
                value = max(u_value, o_value)
                va = "bottom" if value >= 0 else "top"
                offset = 0.45 if value >= 0 else -0.45
                ax_bottom.text(x[i], value + offset, f"{_percent1(u_value)} / {_percent1(o_value)}%",
                               ha="center", va=va, fontsize=6.5)
            else:
                for bar, value in ((u_bars[i], u_value), (o_bars[i], o_value)):
                    va = "bottom" if value >= 0 else "top"
                    offset = 0.45 if value >= 0 else -0.45
                    ax_bottom.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{_percent1(value)}%",
                                   ha="center", va=va, fontsize=6.5)

        ax_top.set_title(DIST_LABELS[distribution])
        ax_top.set_xticks(x, labels, rotation=30, ha="right")
        ax_bottom.set_xticks(x, labels, rotation=30, ha="right")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("U vs O* (%)" if col == 0 else "")
        ax_bottom.set_ylabel("vs exact reference (%)" if col == 0 else "")
        ax_top.set_ylim(-1.0, 4.7)
        ax_bottom.set_ylim(-1.5, 35.0)
    axes[1, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Matched-qualification savings: policy comparison and exact reference", y=0.995, fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "cost-savings")


def plot_coverage(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for ax, distribution in zip(axes, DIST_ORDER):
        subset = [r for r in rows if r["distribution"] == distribution]
        subset.sort(key=lambda r: list(CASE_LABELS).index(r["case"]))
        x = np.arange(len(subset))
        u = np.array([r["u_cp_lower"] for r in subset])
        o = np.array([r["oblivious_cp_lower"] for r in subset])
        ax.axhline(0.90, color=COLORS["target"], linestyle="--", linewidth=1.0, label="target")
        ax.plot(x - 0.07, u, "o", color=COLORS["U"], markersize=5, label="U")
        ax.plot(x + 0.07, o, "s", color=COLORS["O"], markersize=4.5, label="O*")
        ax.set_title(DIST_LABELS[distribution])
        ax.set_xticks(x, [CASE_LABELS[r["case"]] for r in subset], rotation=35, ha="right")
        ax.set_ylabel("One-sided CP lower bound" if ax is axes[0] else "")
        ax.set_ylim(0.75, 1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Held-out coverage audit: U versus O*", y=1.13, fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "coverage-bounds")


def load_poisson_trace_stats(paths: list[str]) -> dict:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    residuals: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("record_type") == "metadata" or item.get("case") != "2d_poisson" or item.get("split") != "test":
                    continue
                cfg = item["config"]
                solver = "reference" if cfg.get("solver") == "direct" else str(cfg.get("precision"))
                key = (item["distribution"], int(cfg.get("iterations", 0)), solver)
                grouped[key].append(float(item["error"]))
                if "residual_rel" in item:
                    residuals[key].append(float(item["residual_rel"]))
    return {"error": grouped, "residual": residuals}


def plot_poisson_frontier(trace_paths: list[str], output_dir: Path) -> None:
    stats = load_poisson_trace_stats(trace_paths)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for ax, distribution in zip(axes, DIST_ORDER):
        keys = [key for key in stats["error"] if key[0] == distribution]
        offsets = {16: -1.5, 32: 0.0, 64: 1.5}
        markers = {16: "o", 32: "s", 64: "^"}
        for precision in (16, 32, 64):
            points = sorted((key[1], np.mean(stats["error"][key])) for key in keys if key[2] == str(precision))
            if points:
                ax.plot(
                    [p[0] + offsets[precision] for p in points],
                    [p[1] for p in points],
                    marker=markers[precision], linestyle="-", color=PRECISION_COLORS[precision],
                    label=f"{precision}-bit",
                )
        reference = [key for key in keys if key[2] == "reference"]
        if reference:
            value = np.mean(stats["error"][reference[0]])
            ax.axhline(value, color=PRECISION_COLORS["exact"], linestyle=":", linewidth=1.2, label="direct reference")
        ax.set_title(DIST_LABELS[distribution])
        ax.set_xlabel("Jacobi iterations")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
    axes[0].set_ylabel("Relative field QoI error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Poisson approximation frontier", y=1.13, fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "poisson-frontier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--poisson-traces", nargs="+", required=True)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    _style()
    output_dir = Path(args.output_dir)
    rows = load_summaries(args.summary)
    plot_cost(rows, output_dir)
    plot_coverage(rows, output_dir)
    plot_poisson_frontier(args.poisson_traces, output_dir)
    print(f"wrote figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
