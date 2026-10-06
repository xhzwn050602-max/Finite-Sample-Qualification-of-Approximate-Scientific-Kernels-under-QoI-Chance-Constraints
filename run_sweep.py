"""Run a small multi-seed sweep and emit a flat, auditable CSV summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark import KERNELS, environment_metadata, run_kernel
from pde_case import run as run_poisson


def load_config(path: str) -> dict:
    """Load and validate a JSON experiment configuration."""
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid experiment config {path!r}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("experiment config 'seeds' must be a non-empty list of integers")
    distributions = config.get("distributions")
    allowed = {"light", "heavy", "correlated"}
    if not isinstance(distributions, list) or not distributions or any(item not in allowed for item in distributions):
        raise ValueError("experiment config 'distributions' must contain light, heavy, or correlated")
    positive_integer_keys = (
        "kernel_calibration", "kernel_validation", "kernel_test",
        "poisson_calibration", "poisson_validation", "poisson_test", "n_calls",
    )
    for key in positive_integer_keys:
        if key in config and (isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0):
            raise ValueError(f"experiment config {key!r} must be a positive integer")
    for key in ("kernel_epsilon", "poisson_epsilon", "oblivious_mean_threshold", "poisson_oblivious_mean_threshold", "delta", "alpha"):
        if key in config and (isinstance(config[key], bool) or not isinstance(config[key], (int, float))):
            raise ValueError(f"experiment config {key!r} must be numeric")
    return config


def config_provenance(path: str | None) -> dict:
    """Return portable config identity fields for CSV and JSONL metadata."""
    if path is None:
        return {"config_file": "cli-defaults", "config_sha256": None}
    config_path = Path(path)
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {"config_file": config_path.name, "config_sha256": digest}


def _parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON experiment configuration")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n-cal", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--poisson-epsilon", type=float, default=None)
    parser.add_argument("--oblivious-mean-threshold", type=float, default=None)
    parser.add_argument("--poisson-oblivious-mean-threshold", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--n-calls", type=int, default=None)
    parser.add_argument("--distribution", choices=["light", "heavy", "correlated", "all"], default=None)
    parser.add_argument("--skip-kernels", action="store_true", help="Only run the Poisson case")
    parser.add_argument("--skip-poisson", action="store_true", help="Only run reusable kernels")
    parser.add_argument("--output", default="sweep-summary.csv")
    parser.add_argument("--raw-output", default=None, help="Optional JSONL path for paired per-trial traces")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    if args.seeds is not None:
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = list(config.get("seeds", [7, 11, 19]))
    if args.distribution is None:
        distributions = list(config.get("distributions", ["light"]))
    elif args.distribution == "all":
        distributions = list(config.get("distributions", ["light", "heavy", "correlated"]))
    else:
        distributions = [args.distribution]
    n_cal = args.n_cal if args.n_cal is not None else config.get("kernel_calibration", 8)
    n_val = args.n_val if args.n_val is not None else config.get("kernel_validation", 8)
    n_test = args.n_test if args.n_test is not None else config.get("kernel_test", 16)
    poisson_n_cal = args.n_cal if args.n_cal is not None else config.get("poisson_calibration", n_cal)
    poisson_n_val = args.n_val if args.n_val is not None else config.get("poisson_validation", n_val)
    poisson_n_test = args.n_test if args.n_test is not None else config.get("poisson_test", n_test)
    epsilon = args.epsilon if args.epsilon is not None else config.get("kernel_epsilon", 0.05)
    poisson_epsilon = args.poisson_epsilon if args.poisson_epsilon is not None else config.get("poisson_epsilon", 0.20)
    oblivious_threshold = args.oblivious_mean_threshold if args.oblivious_mean_threshold is not None else config.get("oblivious_mean_threshold")
    poisson_oblivious_threshold = args.poisson_oblivious_mean_threshold if args.poisson_oblivious_mean_threshold is not None else config.get("poisson_oblivious_mean_threshold")
    delta = args.delta if args.delta is not None else config.get("delta", 0.10)
    alpha = args.alpha if args.alpha is not None else config.get("alpha", 0.05)
    n_calls = args.n_calls if args.n_calls is not None else config.get("n_calls", 100)
    if any(value <= 0 for value in (n_cal, n_val, n_test, poisson_n_cal, poisson_n_val, poisson_n_test, n_calls)):
        parser.error("sample sizes and --n-calls must be positive")
    provenance = config_provenance(args.config)
    rows = []
    trace_rows = []
    metadata = {**environment_metadata(), **provenance}
    if args.skip_kernels and args.skip_poisson:
        parser.error("--skip-kernels and --skip-poisson cannot both be used")
    for distribution in distributions:
        for seed in seeds:
            for kernel in ([] if args.skip_kernels else KERNELS):
                result = run_kernel(
                    kernel, seed, n_cal, n_val, n_test,
                    delta, alpha, epsilon, n_calls,
                    distribution, include_traces=bool(args.raw_output),
                    oblivious_mean_threshold=oblivious_threshold,
                )
                chosen = result["chosen"]["name"]
                record = next(item for item in result["records"] if item["config"]["name"] == chosen)
                oblivious = result["oblivious_chosen"]["name"]
                oblivious_record = next(item for item in result["records"] if item["config"]["name"] == oblivious)
                rows.append({
                    "case": kernel,
                    "distribution": distribution,
                    "seed": seed,
                    "chosen": chosen,
                    "selection_lower_bound": record["validation_lower_bound"],
                    "n_test": result["n_test"],
                    "n_calls": result["n_calls"],
                    "selection_cost_basis": result["selection_cost_basis"],
                    "test_coverage": record["test_coverage"],
                    "test_successes": record["test_successes"],
                    "test_lower_bound": record["test_lower_bound"],
                    "test_p95_error": record["test_p95_error"],
                    "test_p99_error": record["test_p99_error"],
                    "test_cost": record["test_cost"],
                    "calibration_overhead": result["calibration_overhead"],
                    "validation_overhead": result["validation_overhead"],
                    "qualification_overhead": result["qualification_overhead"],
                    "post_qualification_cost": record["post_qualification_cost"],
                    "matched_total_cost": record["matched_total_cost"],
                    "policy_specific_u_total_cost": record["policy_specific_u_total_cost"],
                    "total_cost": record["matched_total_cost"],
                    "oblivious_post_qualification_cost": oblivious_record["post_qualification_cost"],
                    "oblivious_matched_total_cost": oblivious_record["matched_total_cost"],
                    "oblivious_policy_specific_total_cost": oblivious_record["policy_specific_o_total_cost"],
                    "oblivious_total_cost": result["oblivious_total_cost"],
                    "oblivious_mean_threshold": result["oblivious_mean_threshold"],
                    "selection_alpha": result["selection_alpha"],
                    "n_configurations": result["n_configurations"],
                    "energy_measurement": result["energy_measurement"],
                    "fallback_to_exact": result["fallback_to_exact"],
                    "oblivious_chosen": oblivious,
                    "oblivious_test_coverage": oblivious_record["test_coverage"],
                    "oblivious_test_successes": oblivious_record["test_successes"],
                    "oblivious_test_lower_bound": oblivious_record["test_lower_bound"],
                    "oblivious_test_cost": oblivious_record["test_cost"],
                    "qoi_center_mean": None,
                    "qoi_center_p95": None,
                    "residual_mean": None,
                    "residual_p95": None,
                    "finite_rate": None,
                    **metadata,
                })
                if args.raw_output:
                    trace_rows.extend({"case": kernel, "distribution": distribution, "seed_group": seed, **trace} for trace in result["traces"])
            if args.skip_poisson:
                continue
            result = run_poisson(
                seed, poisson_n_cal, poisson_n_val, poisson_n_test, poisson_epsilon,
                delta, alpha, n_calls, distribution,
                include_traces=bool(args.raw_output),
                oblivious_mean_threshold=poisson_oblivious_threshold,
            )
            chosen = result["chosen"]["name"]
            record = next(item for item in result["records"] if item["config"]["name"] == chosen)
            oblivious = result["oblivious_chosen"]["name"]
            oblivious_record = next(item for item in result["records"] if item["config"]["name"] == oblivious)
            rows.append({
                "case": "2d_poisson",
                "distribution": distribution,
                "seed": seed,
                "chosen": chosen,
                "selection_lower_bound": record["validation_lower_bound"],
                    "n_test": result["n_test"],
                    "n_calls": result["n_calls"],
                    "selection_cost_basis": result["selection_cost_basis"],
                "test_coverage": record["test_coverage"],
                "test_successes": record["test_successes"],
                "test_lower_bound": record["test_lower_bound"],
                "test_p95_error": record["test_p95_error"],
                "test_p99_error": record["test_p99_error"],
                "test_cost": record["test_cost"],
                "calibration_overhead": result["calibration_overhead"],
                "validation_overhead": result["validation_overhead"],
                "qualification_overhead": result["qualification_overhead"],
                "post_qualification_cost": record["post_qualification_cost"],
                "matched_total_cost": record["matched_total_cost"],
                "policy_specific_u_total_cost": record["policy_specific_u_total_cost"],
                "total_cost": record["matched_total_cost"],
                "oblivious_post_qualification_cost": oblivious_record["post_qualification_cost"],
                "oblivious_matched_total_cost": oblivious_record["matched_total_cost"],
                "oblivious_policy_specific_total_cost": oblivious_record["policy_specific_o_total_cost"],
                "oblivious_total_cost": result["oblivious_total_cost"],
                "oblivious_mean_threshold": result["oblivious_mean_threshold"],
                "selection_alpha": result["selection_alpha"],
                "n_configurations": result["n_configurations"],
                "energy_measurement": result["energy_measurement"],
                "fallback_to_exact": result["fallback_to_exact"],
                "oblivious_chosen": oblivious,
                "oblivious_test_coverage": oblivious_record["test_coverage"],
                "oblivious_test_successes": oblivious_record["test_successes"],
                "oblivious_test_lower_bound": oblivious_record["test_lower_bound"],
                "oblivious_test_cost": oblivious_record["test_cost"],
                "qoi_center_mean": record.get("test_mean_center_error"),
                "qoi_center_p95": record.get("test_p95_center_error"),
                "residual_mean": record.get("test_mean_residual_rel"),
                "residual_p95": record.get("test_p95_residual_rel"),
                "finite_rate": record.get("test_finite_rate"),
                **metadata,
            })
            if args.raw_output:
                trace_rows.extend({"case": "2d_poisson", "distribution": distribution, "seed_group": seed, **trace} for trace in result["traces"])
    if not rows:
        parser.error("no cases selected")
    fieldnames = list(rows[0])
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.raw_output:
        raw_path = Path(args.raw_output)
        with raw_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": "metadata", "metadata": metadata}, sort_keys=True) + "\n")
            for trace in trace_rows:
                handle.write(json.dumps(trace, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
