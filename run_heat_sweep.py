"""Run the locked non-manufactured heat-conduction physics audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark import environment_metadata
from heat_case import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="heat-experiment-config.json")
    parser.add_argument("--output", default="heat-audit.csv")
    parser.add_argument("--raw-output", default="heat-audit-traces.jsonl")
    parser.add_argument("--distribution", choices=["light", "heavy", "correlated", "all"], default="all")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    distributions = cfg["distributions"] if args.distribution == "all" else [args.distribution]
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metadata = {**environment_metadata(), "config_file": config_path.name, "config_sha256": digest}
    rows = []
    traces = []
    for distribution in distributions:
        for seed in cfg["seeds"]:
            result = run(seed, cfg["calibration"], cfg["validation"], cfg["test"], cfg["epsilon"], cfg["delta"], cfg["alpha"], cfg["n_calls"], distribution, include_traces=True, oblivious_mean_threshold=cfg["oblivious_mean_threshold"])
            chosen = result["chosen"]["name"]
            selected = next(row for row in result["records"] if row["config"]["name"] == chosen)
            oblivious = result["oblivious_chosen"]["name"]
            selected_o = next(row for row in result["records"] if row["config"]["name"] == oblivious)
            rows.append({
                "case": result["case"], "distribution": distribution, "seed": seed,
                "chosen": chosen, "oblivious_chosen": oblivious,
                "selection_lower_bound": selected["validation_lower_bound"],
                "n_test": result["n_test"], "n_calls": result["n_calls"],
                "selection_cost_basis": result["selection_cost_basis"],
                "test_coverage": selected["test_coverage"], "test_successes": selected["test_successes"],
                "test_lower_bound": selected["test_lower_bound"], "oblivious_test_coverage": selected_o["test_coverage"],
                "oblivious_test_successes": selected_o["test_successes"], "oblivious_test_lower_bound": selected_o["test_lower_bound"],
                "test_p95_error": selected["test_p95_error"], "test_p99_error": selected["test_p99_error"],
                "test_cost": selected["test_cost"], "total_cost": selected["total_cost"],
                "calibration_overhead": result["calibration_overhead"],
                "oblivious_total_cost": result["oblivious_total_cost"], "oblivious_test_cost": selected_o["test_cost"],
                "oblivious_mean_threshold": result["oblivious_mean_threshold"], "selection_alpha": result["selection_alpha"],
                "n_configurations": result["n_configurations"], "energy_measurement": result["energy_measurement"],
                "fallback_to_exact": result["fallback_to_exact"],
                "qoi_center_mean": None, "qoi_center_p95": None,
                "residual_mean": selected["test_p95_residual_rel"], "residual_p95": selected["test_p95_residual_rel"],
                "finite_rate": selected["test_finite_rate"],
                "selected_p95_field_error": selected["test_p95_field_error"],
                **metadata,
            })
            traces.extend({"case": result["case"], "distribution": distribution, "seed_group": seed, **trace} for trace in result["traces"])
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with Path(args.raw_output).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": metadata}, sort_keys=True) + "\n")
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "traces": len(traces), "config_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
