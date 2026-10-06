"""Replay the heat audit with calibration overhead charged per trial."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

from benchmark import add_cost_accounting, select_oblivious, selection_key


def lower_bound(successes: int, total: int, alpha: float) -> float:
    if total == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def mean(values):
    return float(np.mean(values))


def load_traces(path: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    metadata = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata = item.get("metadata")
                continue
            cfg = item.get("config", {})
            key = (str(item["case"]), str(item["distribution"]), int(item["seed_group"]), str(cfg["name"]))
            split = str(item["split"])
            grouped[key][f"{split}_error"].append(float(item["error"]))
            grouped[key][f"{split}_time"].append(float(item["wall_time_s"]))
            for name in ("residual_rel", "finite", "field_error"):
                if name in item:
                    grouped[key][f"{split}_{name}"].append(item[name])
    if metadata is None:
        raise ValueError(f"missing metadata in {path}")
    return grouped, metadata


def replay(csv_path: Path, trace_path: Path, output_path: Path, config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grouped, metadata = load_traces(trace_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    if not old_rows:
        raise ValueError(f"empty CSV: {csv_path}")
    epsilon = float(config["epsilon"])
    threshold = float(config["oblivious_mean_threshold"])
    delta = float(config["delta"])
    alpha = float(config["alpha"])
    new_rows = []
    for old in old_rows:
        case, distribution, seed = old["case"], old["distribution"], int(old["seed"])
        keys = [key for key in grouped if key[:3] == (case, distribution, seed)]
        names = sorted({key[3] for key in keys})
        if not names:
            raise ValueError(f"no traces for {case}/{distribution}/{seed}")
        selection_alpha = float(old["selection_alpha"])
        records = {}
        for name in names:
            values = grouped[(case, distribution, seed, name)]
            cal_err = values["calibration_error"]
            val_err = values["validation_error"]
            test_err = values["test_error"]
            records[name] = {
                # Keep replay tie-breaking identical to the live selector.
                "config": {"name": name},
                "calibration_cost": mean(values["calibration_time"]),
                "validation_cost": mean(values["validation_time"]),
                "test_cost": mean(values["test_time"]),
                "calibration_mean_error": mean(cal_err),
                "validation_lower_bound": lower_bound(sum(e <= epsilon for e in val_err), len(val_err), selection_alpha),
                "test_errors": test_err,
                "test_successes": sum(e <= epsilon for e in test_err),
                "test_lower_bound": lower_bound(sum(e <= epsilon for e in test_err), len(test_err), alpha),
                "trace_values": values,
            }
        n_cal = len(grouped[(case, distribution, seed, names[0])]["calibration_error"])
        n_val = len(grouped[(case, distribution, seed, names[0])]["validation_error"])
        n_calls = int(old["n_calls"])
        costs = add_cost_accounting(records.values(), n_cal, n_val, n_calls)
        exact = "heat-exact"
        feasible = [name for name, record in records.items() if record["validation_lower_bound"] >= 1.0 - delta]
        chosen = min(feasible, key=lambda name: selection_key(records[name])) if feasible else exact
        non_exact = [name for name in names if name != exact]
        oblivious_record = select_oblivious(records.values(), threshold, records[exact])
        oblivious = str(oblivious_record["config"]["name"])
        selected, selected_o = records[chosen], records[oblivious]
        selected_values = selected["trace_values"]
        row = dict(old)
        row.update({
            "chosen": chosen,
            "oblivious_chosen": oblivious,
            "selection_lower_bound": selected["validation_lower_bound"],
            "selection_cost_basis": "validation_timing",
            "test_coverage": selected["test_successes"] / len(selected["test_errors"]),
            "test_successes": selected["test_successes"],
            "test_lower_bound": selected["test_lower_bound"],
            "oblivious_test_coverage": selected_o["test_successes"] / len(selected_o["test_errors"]),
            "oblivious_test_successes": selected_o["test_successes"],
            "oblivious_test_lower_bound": selected_o["test_lower_bound"],
            "test_p95_error": float(np.quantile(selected["test_errors"], 0.95)),
            "test_p99_error": float(np.quantile(selected["test_errors"], 0.99)),
            "test_cost": selected["test_cost"],
            **costs,
            "post_qualification_cost": selected["post_qualification_cost"],
            "matched_total_cost": selected["matched_total_cost"],
            "policy_specific_u_total_cost": selected["policy_specific_u_total_cost"],
            "total_cost": selected["matched_total_cost"],
            "oblivious_post_qualification_cost": selected_o["post_qualification_cost"],
            "oblivious_matched_total_cost": selected_o["matched_total_cost"],
            "oblivious_policy_specific_total_cost": selected_o["policy_specific_o_total_cost"],
            "oblivious_total_cost": selected_o["matched_total_cost"],
            "oblivious_test_cost": selected_o["test_cost"],
            "fallback_to_exact": not bool([name for name in non_exact if records[name]["validation_lower_bound"] >= 1.0 - delta]),
            "residual_mean": mean(selected_values.get("test_residual_rel", [])) if selected_values.get("test_residual_rel") else "",
            "residual_p95": float(np.quantile(selected_values["test_residual_rel"], 0.95)) if selected_values.get("test_residual_rel") else "",
            "finite_rate": mean(selected_values.get("test_finite", [])) if selected_values.get("test_finite") else "",
            "selected_p95_field_error": float(np.quantile(selected_values["test_field_error"], 0.95)) if selected_values.get("test_field_error") else "",
        })
        new_rows.append(row)
    fieldnames = [
        "case", "distribution", "seed", "chosen", "oblivious_chosen", "selection_lower_bound", "n_test", "n_calls",
        "selection_cost_basis", "test_coverage", "test_successes", "test_lower_bound", "oblivious_test_coverage",
        "oblivious_test_successes", "oblivious_test_lower_bound", "test_p95_error", "test_p99_error", "test_cost",
        "calibration_overhead", "validation_overhead", "qualification_overhead", "post_qualification_cost",
        "matched_total_cost", "policy_specific_u_total_cost", "total_cost", "oblivious_post_qualification_cost",
        "oblivious_matched_total_cost", "oblivious_policy_specific_total_cost", "oblivious_total_cost",
        "oblivious_test_cost", "oblivious_mean_threshold",
        "selection_alpha", "n_configurations", "energy_measurement", "fallback_to_exact", "qoi_center_mean", "qoi_center_p95",
        "residual_mean", "residual_p95", "finite_rate", "selected_p95_field_error", "python_version", "numpy_version",
        "scipy_version", "platform", "processor", "cpu_count", "thread_env", "byteorder", "config_file", "config_sha256",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    return metadata, new_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="heat-experiment-config-full.json")
    parser.add_argument("--input-csv", default="heat-audit-full.csv")
    parser.add_argument("--traces", default="heat-audit-full.jsonl")
    parser.add_argument("--output-csv", default="heat-audit-full-corrected.csv")
    args = parser.parse_args()
    _, rows = replay(Path(args.input_csv), Path(args.traces), Path(args.output_csv), Path(args.config))
    print(json.dumps({"rows": len(rows), "output": args.output_csv}, indent=2))


if __name__ == "__main__":
    main()
