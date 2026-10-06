"""Run the locked, independent tight-tolerance Monte Carlo confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t

from benchmark import (
    Config,
    InputRNG,
    add_cost_accounting,
    environment_metadata,
    lower_bound,
    monte_carlo_case,
    select_oblivious,
    selection_key,
    trial_seed_for,
)


def load_locked_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    required = {
        "seeds", "candidates", "reference", "calibration", "validation",
        "test", "epsilon", "delta", "alpha", "oblivious_mean_threshold", "n_calls",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"confirmatory config missing fields: {missing}")
    if config.get("status") != "locked before confirmatory execution":
        raise ValueError("confirmatory config is not marked as locked")
    names = [candidate["name"] for candidate in config["candidates"]]
    if len(names) != len(set(names)) or config["reference"] not in names:
        raise ValueError("candidate names must be unique and include the reference")
    return config, hashlib.sha256(raw).hexdigest()


def _configs(config: dict) -> list[Config]:
    return [
        Config(str(item["name"]), int(item["precision"]), float(item["samples"]))
        for item in config["candidates"]
    ]


def run_seed(config: dict, seed: int) -> tuple[dict, list[dict]]:
    cfgs = _configs(config)
    n_cal = int(config["calibration"])
    n_val = int(config["validation"])
    n_test = int(config["test"])
    epsilon = float(config["epsilon"])
    alpha = float(config["alpha"])
    delta = float(config["delta"])
    n_calls = int(config["n_calls"])
    selection_alpha = alpha / len(cfgs)
    buffers = {
        cfg.name: {"cal_error": [], "cal_time": [], "val_error": [], "val_time": [], "test_error": [], "test_time": []}
        for cfg in cfgs
    }
    traces: list[dict] = []
    for split, count, prefix in (
        ("calibration", n_cal, "cal"),
        ("validation", n_val, "val"),
        ("test", n_test, "test"),
    ):
        for index in range(count):
            trial_seed = trial_seed_for(seed, split, index)
            for cfg in cfgs:
                error, elapsed = monte_carlo_case(InputRNG(trial_seed, "light"), cfg)
                buffers[cfg.name][f"{prefix}_error"].append(error)
                buffers[cfg.name][f"{prefix}_time"].append(elapsed)
                traces.append({
                    "case": "monte_carlo_tight", "seed_group": seed, "split": split,
                    "index": index, "seed": trial_seed,
                    "config": {"name": cfg.name, "precision": cfg.precision, "samples": int(cfg.knob)},
                    "error": error, "wall_time_s": elapsed,
                })
    records = []
    for cfg in cfgs:
        values = buffers[cfg.name]
        val_successes = sum(error <= epsilon for error in values["val_error"])
        test_successes = sum(error <= epsilon for error in values["test_error"])
        records.append({
            "config": {"name": cfg.name, "precision": cfg.precision, "samples": int(cfg.knob)},
            "calibration_cost": fmean(values["cal_time"]),
            "validation_cost": fmean(values["val_time"]),
            "test_cost": fmean(values["test_time"]),
            "calibration_mean_error": fmean(values["cal_error"]),
            "validation_successes": val_successes,
            "validation_coverage": val_successes / n_val,
            "validation_lower_bound": lower_bound(val_successes, n_val, selection_alpha),
            "test_successes": test_successes,
            "test_coverage": test_successes / n_test,
            "test_lower_bound": lower_bound(test_successes, n_test, alpha),
            "test_p95_error": float(np.quantile(values["test_error"], 0.95)),
            "test_p99_error": float(np.quantile(values["test_error"], 0.99)),
        })
    costs = add_cost_accounting(records, n_cal, n_val, n_calls)
    reference = next(record for record in records if record["config"]["name"] == config["reference"])
    feasible = [record for record in records if record["validation_lower_bound"] >= 1.0 - delta]
    chosen = min(feasible, key=selection_key) if feasible else reference
    oblivious = select_oblivious(records, float(config["oblivious_mean_threshold"]), reference)
    row = {
        "case": "monte_carlo_tight", "seed": seed,
        "chosen": chosen["config"]["name"], "oblivious_chosen": oblivious["config"]["name"],
        "reference": reference["config"]["name"], "n_calibration": n_cal,
        "n_validation": n_val, "n_test": n_test, "n_calls": n_calls,
        "epsilon": epsilon, "delta": delta, "alpha": alpha,
        "selection_alpha": selection_alpha, "selection_lower_bound": chosen["validation_lower_bound"],
        "test_successes": chosen["test_successes"], "test_coverage": chosen["test_coverage"],
        "test_lower_bound": chosen["test_lower_bound"], "test_p95_error": chosen["test_p95_error"],
        "test_p99_error": chosen["test_p99_error"],
        "oblivious_test_successes": oblivious["test_successes"],
        "oblivious_test_coverage": oblivious["test_coverage"],
        "oblivious_test_lower_bound": oblivious["test_lower_bound"],
        "reference_test_successes": reference["test_successes"],
        "reference_test_coverage": reference["test_coverage"],
        "reference_test_lower_bound": reference["test_lower_bound"],
        **costs,
        "u_post_qualification_cost": chosen["post_qualification_cost"],
        "o_post_qualification_cost": oblivious["post_qualification_cost"],
        "reference_post_qualification_cost": reference["post_qualification_cost"],
        "u_matched_total_cost": chosen["matched_total_cost"],
        "o_matched_total_cost": oblivious["matched_total_cost"],
        "reference_matched_total_cost": reference["matched_total_cost"],
        "u_policy_specific_total_cost": chosen["policy_specific_u_total_cost"],
        "o_policy_specific_total_cost": oblivious["policy_specific_o_total_cost"],
    }
    return row, traces


def summarize(rows: list[dict], config_sha256: str) -> dict:
    total_test = sum(int(row["n_test"]) for row in rows)
    u_successes = sum(int(row["test_successes"]) for row in rows)
    o_successes = sum(int(row["oblivious_test_successes"]) for row in rows)
    reference_successes = sum(int(row["reference_test_successes"]) for row in rows)

    def mean_savings(numerator_key: str, denominator_key: str) -> float:
        return fmean(
            (float(row[denominator_key]) - float(row[numerator_key])) / float(row[denominator_key])
            for row in rows
        )

    def savings_interval(numerator_key: str, denominator_key: str) -> dict:
        values = [
            (float(row[denominator_key]) - float(row[numerator_key])) / float(row[denominator_key])
            for row in rows
        ]
        mean = fmean(values)
        if len(values) < 2:
            return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
        half = float(t.ppf(0.975, len(values) - 1) * stdev(values) / np.sqrt(len(values)))
        return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}

    return {
        "config_sha256": config_sha256,
        "n_seed_groups": len(rows), "n_test_observations": total_test,
        "u_selection_counts": dict(Counter(row["chosen"] for row in rows)),
        "o_star_selection_counts": dict(Counter(row["oblivious_chosen"] for row in rows)),
        "u_test_coverage": u_successes / total_test,
        "o_star_test_coverage": o_successes / total_test,
        "reference_test_coverage": reference_successes / total_test,
        "u_min_per_seed_cp_lower": min(float(row["test_lower_bound"]) for row in rows),
        "o_star_min_per_seed_cp_lower": min(float(row["oblivious_test_lower_bound"]) for row in rows),
        "reference_min_per_seed_cp_lower": min(float(row["reference_test_lower_bound"]) for row in rows),
        "mean_costs": {
            key: fmean(float(row[key]) for row in rows)
            for key in (
                "u_post_qualification_cost", "o_post_qualification_cost", "reference_post_qualification_cost",
                "u_matched_total_cost", "o_matched_total_cost", "reference_matched_total_cost",
                "u_policy_specific_total_cost", "o_policy_specific_total_cost",
            )
        },
        "u_matched_savings_vs_o_star": mean_savings("u_matched_total_cost", "o_matched_total_cost"),
        "u_matched_savings_vs_o_star_ci95": savings_interval("u_matched_total_cost", "o_matched_total_cost"),
        "u_post_qualification_savings_vs_o_star": mean_savings("u_post_qualification_cost", "o_post_qualification_cost"),
        "u_post_qualification_savings_vs_o_star_ci95": savings_interval("u_post_qualification_cost", "o_post_qualification_cost"),
        "u_policy_specific_savings_vs_o_star": mean_savings("u_policy_specific_total_cost", "o_policy_specific_total_cost"),
        "u_policy_specific_savings_vs_o_star_ci95": savings_interval("u_policy_specific_total_cost", "o_policy_specific_total_cost"),
        "u_matched_savings_vs_reference": mean_savings("u_matched_total_cost", "reference_matched_total_cost"),
        "u_matched_savings_vs_reference_ci95": savings_interval("u_matched_total_cost", "reference_matched_total_cost"),
        "u_post_qualification_savings_vs_reference": mean_savings("u_post_qualification_cost", "reference_post_qualification_cost"),
        "u_post_qualification_savings_vs_reference_ci95": savings_interval("u_post_qualification_cost", "reference_post_qualification_cost"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("confirmatory-mc-config.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("confirmatory-mc.csv"))
    parser.add_argument("--output-traces", type=Path, default=Path("confirmatory-mc-traces.jsonl"))
    parser.add_argument("--output-summary", type=Path, default=Path("confirmatory-mc-summary.json"))
    parser.add_argument("--summary-only", action="store_true", help="recompute the summary from an existing CSV without rerunning measurements")
    args = parser.parse_args()
    config, digest = load_locked_config(args.config)
    if args.summary_only:
        with args.output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or {row.get("config_sha256") for row in rows} != {digest}:
            raise ValueError("existing confirmatory CSV is empty or has the wrong config digest")
        summary = summarize(rows, digest)
        args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    rows: list[dict] = []
    all_traces: list[dict] = []
    for seed in config["seeds"]:
        row, traces = run_seed(config, int(seed))
        row.update({"config_file": args.config.name, "config_sha256": digest})
        rows.append(row)
        all_traces.extend(traces)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"record_type": "metadata", "metadata": {**environment_metadata(), "config_file": args.config.name, "config_sha256": digest}}
    with args.output_traces.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata) + "\n")
        for trace in all_traces:
            handle.write(json.dumps(trace) + "\n")
    summary = summarize(rows, digest)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
