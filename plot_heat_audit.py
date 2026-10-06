"""Plot the non-manufactured heat-conduction extension audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    csv_path = Path("heat-audit-full-fair.csv")
    trace_path = Path("heat-audit-full.jsonl")
    out_dir = Path("figures-publication")
    out_dir.mkdir(exist_ok=True)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    traces = []
    for line in trace_path.open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("record_type") != "metadata" and item.get("split") == "test":
            traces.append(item)
    order = ["light", "heavy", "correlated"]
    labels = ["Light", "Heavy", "Correlated"]
    colors = {"U": "#0072B2", "O": "#D55E00"}
    assert len(rows) == 30
    assert all(r["chosen"] == r["oblivious_chosen"] == "heat-exact" for r in rows)
    assert all(float(r["total_cost"]) == float(r["oblivious_total_cost"]) for r in rows)
    means = {p: [] for p in ("U", "O")}
    q95 = {p: [] for p in ("U", "O")}
    for dist in order:
        subset = [r for r in rows if r["distribution"] == dist]
        means["U"].append(float(np.mean([float(r["total_cost"]) for r in subset])))
        means["O"].append(float(np.mean([float(r["oblivious_total_cost"]) for r in subset])))
        for policy, key in (("U", "chosen"), ("O", "oblivious_chosen")):
            values = []
            for row in subset:
                values.append(np.quantile([
                    float(t["error"]) for t in traces
                    if t["distribution"] == dist and int(t["seed_group"]) == int(row["seed"])
                    and t["config"]["name"] == row[key]
                ], 0.95))
            q95[policy].append(float(np.max(values)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(3)
    width = 0.34
    for i, policy in enumerate(("U", "O")):
        label = "O*" if policy == "O" else policy
        axes[0].bar(x + (i - 0.5) * width, means[policy], width, color=colors[policy], label=label,
                    hatch="//" if policy == "O" else None)
        axes[1].plot(x + (i - 0.5) * width, q95[policy], "s" if policy == "O" else "o",
                     color=colors[policy], label=label, clip_on=False)
    axes[0].set_title("(a) Matched cost, 1,000 calls")
    axes[0].set_ylabel("Mean accounted time (s)")
    axes[0].set_ylim(0, max(max(values) for values in means.values()) * 1.25)
    axes[1].set_title("(b) Selected QoI tail")
    axes[1].set_ylabel("95th-percentile relative QoI error")
    assert all(value == 0 for values in q95.values() for value in values)
    axes[1].set_ylim(-0.005, 0.055)
    axes[1].set_yticks([0, 0.025, 0.05])
    axes[1].axhline(0.05, color="#666666", linestyle="--", linewidth=1)
    axes[1].text(1, 0.044, "Tolerance = 0.05", ha="center", fontsize=8)
    axes[1].text(1, 0.016, "Both compare direct with direct", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(True, axis="y")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("24x24 heat: reference execution in all 10 groups", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "heat-audit.pdf")
    fig.savefig(out_dir / "heat-audit.png", dpi=300)
    plt.close(fig)
    provenance = {"inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (csv_path, trace_path)},
                  "mean_matched_cost_s": means, "max_group_p95_relative_qoi_error": q95,
                  "uncertainty": "Descriptive means only; legacy test seeds overlap across groups."}
    (out_dir / "heat-audit-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(f"wrote {out_dir / 'heat-audit.pdf'}")


if __name__ == "__main__":
    main()
