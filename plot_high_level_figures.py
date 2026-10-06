"""Render manuscript figures from locked CSVs and the explicit heat replay."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures-publication"
COLORS = ["#0072B2", "#D55E00", "#505050"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.titlesize": 9,
         "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None}
PROVENANCE = {}


def save(fig, name, inputs, description):
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=350, facecolor="white")
    PROVENANCE[name] = {"inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                   for p in inputs}, "description": description,
                        "size_inches": list(fig.get_size_inches()), "png_dpi": 350}
    plt.close(fig)


def method():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#78828A", lw=.8))
        ax.text(x+2, y+h-4, title, fontsize=8.2, weight="bold", va="top")
        ax.text(x+2, y+h-12, body, fontsize=7.4, va="top", linespacing=1.6)

    def arrow(a, b, color="#505050"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                     lw=1, color=color, shrinkA=1, shrinkB=1))

    ax.text(1, 97, "A  Fixed workload and disjoint trial streams", weight="bold", fontsize=9)
    box(1, 55, 26, 32, "Declare the target",
        "$m_a(X),\\;\\varepsilon,\\;1-\\delta,\\;\\alpha$\nFinite menu $\\mathcal{A}_K$; reference E\nPaired trials for each candidate", "#F3F3F3")
    box(34, 55, 28, 32, "Calibration: profile",
        "Mean errors and timings\nO*: minimum calibration cost\nsubject to mean-error threshold", "#FFF1E6")
    box(69, 55, 30, 32, "Validation: qualify U",
        "$L_a=\\mathrm{CP}_{\\alpha/M}(S_a,n)$\n$\\mathcal{F}=\\{a:L_a\\geq 1-\\delta\\}$\nRank by validation timing", "#E9F3FA")
    arrow((27, 72), (34, 72))
    arrow((62, 72), (69, 72))
    ax.text(1, 43, "B  Fixed-policy evaluation", weight="bold", fontsize=9)
    box(1, 3, 29, 29, "Independent test audit",
        "Coverage and CP lower bound\nMeasured per-call cost\nTwo qualification-cost views", "#F3F3F3")
    box(37, 3, 29, 29, "Accepted configuration",
        "$\\mathcal{F}\\ne\\varnothing$: lowest timed cost\nFixed implementation for all calls\nAcceptance-error bound applies", "#E7F3EF")
    box(73, 3, 26, 29, "Reference execution",
        "$\\mathcal{F}=\\varnothing$: execute E\nReport qualification failure\nNo gate-derived certificate", "#FFF1E6")
    arrow((80, 55), (51.5, 32), "#0072B2")
    arrow((92, 55), (86, 32), "#D55E00")
    arrow((37, 17), (30, 17))
    save(fig, "method-schematic", ["benchmark.py"],
         "Conceptual protocol. Test outcomes never flow into qualification; O* is calibrated independently.")


def results():
    with (ROOT / "confirmatory-mc.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")
    ax = axes[0]
    for i, prefix in enumerate(("", "oblivious_", "reference_")):
        coverage = np.array([float(r[prefix+"test_coverage"]) for r in rows])
        lows = [float(r[prefix+"test_lower_bound"]) for r in rows]
        ax.scatter(i+np.linspace(-.13, .13, len(rows)), coverage,
                   s=13, color=COLORS[i], alpha=.65, zorder=3)
        ax.plot([i, i], [min(lows), coverage.mean()], color=COLORS[i], lw=1.4)
        ax.scatter([i], [min(lows)], marker="_", s=110, color=COLORS[i], zorder=4)
        ax.scatter([i], [coverage.mean()], marker="D", s=28, edgecolor="white", lw=.5,
                   color=COLORS[i], zorder=5)
        ax.text(i, coverage.max()+.019, f"{coverage.mean():.3f}", ha="center", fontsize=7.5)
    ax.axhline(.9, color="#555555", linestyle="--", lw=.8)
    ax.text(2.33, .907, "target", ha="right", fontsize=7)
    ax.set(xticks=range(3), xticklabels=["U", "O*", "E"], ylim=(.65, 1.055),
           xlim=(-.45, 2.45), ylabel="Held-out coverage")
    ax.set_title("a  Coverage and lower bounds", loc="left", fontsize=8)
    ax.grid(axis="y", color="#E6E6E6", lw=.5)
    for ax, title, keys, labels in [
        (axes[1], "b  Equal qualification budgets",
         ["u_matched_total_cost", "o_matched_total_cost", "reference_matched_total_cost"], ["U", "O*", "E"]),
        (axes[2], "c  Policy-specific budgets",
         ["u_policy_specific_total_cost", "o_policy_specific_total_cost"], ["U", "O*"])]:
        for i, key in enumerate(keys):
            vals = np.array([float(r[key]) for r in rows])
            half = t.ppf(.975, len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
            ax.bar(i, vals.mean(), color=COLORS[i], width=.58, alpha=.8,
                   yerr=half, capsize=3, error_kw={"elinewidth": .9})
            ax.scatter(i+np.linspace(-.12, .12, len(vals)), vals, color="#242424", s=6, zorder=4)
            ax.text(i, vals.max()+(.55 if len(keys)==3 else .22), f"{vals.mean():.2f}",
                    ha="center", fontsize=7.5)
        ax.set(xticks=range(len(keys)), xticklabels=labels, ylabel="Accounted cost (s)")
        ax.set_ylim(0, 26 if len(keys)==3 else 6.3)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", color="#E6E6E6", lw=.5)
        ax.set_axisbelow(True)
    save(fig, "confirmatory-summary", ["confirmatory-mc.csv"],
         "Ten per-group observations. Coverage diamonds are means; whisker ends are minimum individual one-sided 95% CP bounds, not simultaneous intervals. Cost bars are means with descriptive 95% t intervals; points are group costs. All models use 1000 extrapolated calls.")


def heat():
    report = json.loads((ROOT / "evidence/numerical-audit.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), layout="constrained")
    ns = np.arange(5, 101)
    ax = axes[0]
    ax.plot(ns, (.05/13)**(1/ns), color=COLORS[0], lw=1.4)
    ax.axhline(.9, color="#555555", ls="--", lw=.8)
    for n, label in [(16, "64 x 64"), (32, "24 x 24"), (53, "floor")]:
        bound = (.05/13)**(1/n)
        ax.scatter(n, bound, color=COLORS[1], s=24, zorder=4)
        positions = {16: (22, .56), 32: (44, .72), 53: (65, .83)}
        ax.annotate(f"n = {n}\n{label}", (n, bound), xytext=positions[n],
                    textcoords="data", fontsize=7, va="center",
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .6})
    ax.set(xlabel="Validation trials per candidate", ylabel="Maximum CP lower bound",
           ylim=(.3, 1.02), xlim=(5, 102))
    ax.set_title("a  Original budget limitation", loc="left", fontsize=8)
    for ax, audit in zip(axes[1:], report["heat"]):
        for cell in audit["cells"]:
            for candidate in cell["candidates"]:
                name = candidate["candidate"]
                color, marker = (COLORS[0], "D") if name=="heat-exact" else (COLORS[1], "o")
                ax.scatter(candidate["validation_mean_cost_s"]*1000, candidate["pooled_validation_lower"],
                           color=color, marker=marker, s=18, alpha=.7, linewidths=.3, edgecolor="white")
        ax.axhline(.9, color="#555555", ls="--", lw=.8)
        ax.set(xlabel="Validation solve time (ms)", ylabel="Pooled CP lower bound",
               ylim=(-.05, 1.1))
        ax.set_title(f"{'b' if audit['grid']==24 else 'c'}  {audit['grid']} x {audit['grid']}: pooled validation", loc="left", fontsize=8)
        ax.plot([], [], marker="D", color=COLORS[0], ls="", label="Direct")
        ax.plot([], [], marker="o", color=COLORS[1], ls="", label="Jacobi-like")
        ax.legend(frameon=False, loc="center right", fontsize=7, handletextpad=.4)
    for ax in axes:
        ax.grid(color="#E6E6E6", lw=.5)
        ax.set_axisbelow(True)
    save(fig, "heat-qualification", ["evidence/numerical-audit.json"],
         "All 13 candidates and three regimes plotted per grid. Overlapping points retain equal coordinates. Timings exclude assembly. Post hoc validation pooling is not a new independent experiment.")


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        method()
        results()
        heat()
    (OUT / "headline-figure-provenance.json").write_text(json.dumps(PROVENANCE, indent=2)+"\n", encoding="utf-8")
