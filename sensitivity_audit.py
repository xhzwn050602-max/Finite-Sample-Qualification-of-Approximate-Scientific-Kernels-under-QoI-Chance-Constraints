"""Replay controller sensitivity to QoI tolerance and chance level.

The audit reads the immutable publication traces and replays only the policy
decision. It never changes the primary configuration, timings, or reported
main-run values. For each case, ``epsilon_scale`` multiplies the declared
case-specific epsilon and the oblivious policy uses the same scaled mean-error
threshold, matching the prototype default.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

from benchmark import add_cost_accounting, select_oblivious, selection_key


EXACT = {
    "fft": "fft-64-1.0",
    "spmv": "spmv-64-0.0",
    "stencil": "stencil-64-0.0",
    "monte_carlo": "mc-64-1000000",
    "2d_poisson": "poisson-exact",
}


def cp_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def load_traces(paths: list[Path]):
    grouped: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("record_type") == "metadata":
                    metadata.append(item.get("metadata", {}))
                    continue
                config = item.get("config", {})
                key = (
                    str(item["case"]),
                    str(item["distribution"]),
                    int(item["seed_group"]),
                    str(config["name"]),
                )
                split = str(item["split"])
                grouped[key][f"{split}_error"].append(float(item["error"]))
                grouped[key][f"{split}_time"].append(float(item["wall_time_s"]))
    if not grouped or not metadata:
        raise ValueError("missing traces or metadata")
    digests = {entry.get("config_sha256") for entry in metadata}
    if len(digests) != 1:
        raise ValueError(f"mixed configuration digests: {sorted(digests)!r}")
    return grouped, metadata[0]


def replay(
    grouped,
    config: dict,
    epsilon_scales: list[float],
    deltas: list[float],
    alpha: float,
    n_calls: int,
):
    by_cell: dict[tuple[str, str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for (case, distribution, seed, name), values in grouped.items():
        by_cell[(case, distribution, seed)][name] = {
            "calibration_cost": mean(values["calibration_time"]),
            "validation_cost": mean(values["validation_time"]),
            "test_cost": mean(values["test_time"]),
            "calibration_mean_error": mean(values["calibration_error"]),
            "validation_error": values["validation_error"],
            "test_error": values["test_error"],
        }

    output = []
    base_epsilon = {
        "2d_poisson": float(config["poisson_epsilon"]),
        "default": float(config["kernel_epsilon"]),
    }
    for (case, distribution, seed), records in sorted(by_cell.items()):
        exact_name = EXACT[case]
        # The calibration and validation sizes are equal in the publication run;
        # use the explicit calibration split when charging overhead.
        n_cal = len(grouped[(case, distribution, seed, exact_name)]["calibration_error"])
        n_val = len(grouped[(case, distribution, seed, exact_name)]["validation_error"])
        n_configurations = len(records)
        selection_alpha = alpha / n_configurations
        for scale in epsilon_scales:
            epsilon = scale * base_epsilon.get(case, base_epsilon["default"])
            for delta in deltas:
                metrics = {}
                for name, record in records.items():
                    val_errors = record["validation_error"]
                    test_errors = record["test_error"]
                    val_successes = sum(error <= epsilon for error in val_errors)
                    test_successes = sum(error <= epsilon for error in test_errors)
                    metrics[name] = {
                        # Match the live selector's deterministic name
                        # tie-break during post hoc replay.
                        "config": {"name": name},
                        **record,
                        "validation_lower_bound": cp_lower(
                            val_successes, len(val_errors), selection_alpha
                        ),
                        "test_coverage": test_successes / len(test_errors),
                        "test_lower_bound": cp_lower(test_successes, len(test_errors), alpha),
                    }
                costs = add_cost_accounting(metrics.values(), n_cal, n_val, n_calls)
                all_feasible = [
                    name
                    for name in metrics
                    if metrics[name]["validation_lower_bound"] >= 1.0 - delta
                ]
                chosen = min(
                    all_feasible,
                    key=lambda name: selection_key(metrics[name]),
                ) if all_feasible else exact_name
                oblivious_record = select_oblivious(metrics.values(), epsilon, metrics[exact_name])
                oblivious = str(oblivious_record["config"]["name"])
                u = metrics[chosen]
                o = metrics[oblivious]
                output.append(
                    {
                        "case": case,
                        "distribution": distribution,
                        "seed": seed,
                        "epsilon_scale": scale,
                        "epsilon": epsilon,
                        "delta": delta,
                        "selection_alpha": selection_alpha,
                        "chosen": chosen,
                        "oblivious_chosen": oblivious,
                        "fallback_to_exact": not bool(all_feasible),
                        "u_test_coverage": u["test_coverage"],
                        "u_test_lower_bound": u["test_lower_bound"],
                        "oblivious_test_coverage": o["test_coverage"],
                        "oblivious_test_lower_bound": o["test_lower_bound"],
                        **costs,
                        "u_post_qualification_cost": u["post_qualification_cost"],
                        "oblivious_post_qualification_cost": o["post_qualification_cost"],
                        "u_total_cost": u["matched_total_cost"],
                        "oblivious_total_cost": o["matched_total_cost"],
                        "u_policy_specific_total_cost": u["policy_specific_u_total_cost"],
                        "oblivious_policy_specific_total_cost": o["policy_specific_o_total_cost"],
                        "relative_savings": (
                            (o["matched_total_cost"] - u["matched_total_cost"]) / o["matched_total_cost"]
                            if o["matched_total_cost"] > 0
                            else 0.0
                        ),
                    }
                )
    return output


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["distribution"], row["epsilon_scale"], row["delta"])].append(row)
    summary = []
    for (case, distribution, scale, delta), cell_rows in sorted(grouped.items()):
        summary.append(
            {
                "case": case,
                "distribution": distribution,
                "epsilon_scale": scale,
                "delta": delta,
                "n_seed_rows": len(cell_rows),
                "u_fallback_rate": float(np.mean([r["fallback_to_exact"] for r in cell_rows])),
                "u_mean_test_coverage": float(np.mean([r["u_test_coverage"] for r in cell_rows])),
                "u_min_test_lower_bound": min(r["u_test_lower_bound"] for r in cell_rows),
                "oblivious_mean_test_coverage": float(np.mean([r["oblivious_test_coverage"] for r in cell_rows])),
                "oblivious_min_test_lower_bound": min(r["oblivious_test_lower_bound"] for r in cell_rows),
                "mean_relative_savings": float(np.mean([r["relative_savings"] for r in cell_rows])),
                "u_selection_counts": dict(Counter(r["chosen"] for r in cell_rows)),
                "oblivious_selection_counts": dict(Counter(r["oblivious_chosen"] for r in cell_rows)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="publication-config.json")
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output-json", default="sensitivity-audit.json")
    parser.add_argument("--output-csv", default="sensitivity-audit.csv")
    parser.add_argument("--epsilon-scales", default="0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2")
    parser.add_argument("--deltas", default="0.01,0.05,0.10,0.20")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    grouped, metadata = load_traces([Path(path) for path in args.traces])
    scales = [float(value) for value in args.epsilon_scales.split(",")]
    deltas = [float(value) for value in args.deltas.split(",")]
    rows = replay(
        grouped,
        config,
        scales,
        deltas,
        float(config["alpha"]),
        int(config["n_calls"]),
    )
    summary = summarize(rows)
    payload = {
        "config_file": Path(args.config).name,
        "config_sha256": metadata["config_sha256"],
        "epsilon_scales": scales,
        "deltas": deltas,
        "interpretation": "trace replay sensitivity audit; no new timings",
        "rows": rows,
        "summary": summary,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = list(rows[0])
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "config_sha256": metadata["config_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
