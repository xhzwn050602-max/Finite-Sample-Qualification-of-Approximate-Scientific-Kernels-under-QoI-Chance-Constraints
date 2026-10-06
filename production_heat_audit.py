"""Larger-grid non-manufactured heat-conduction audit.

This supplemental experiment keeps the primary 24-by-24 heat audit untouched
and repeats the same paired controller protocol on a 64-by-64 grid.  Matrix
assembly is shared setup and is excluded from the solver timing, as in the
primary heat extension.  Raw JSONL records retain every split/configuration
observation so the CSV decisions can be replayed independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import beta

from benchmark import add_cost_accounting, InputRNG, environment_metadata, lower_bound, select_oblivious, selection_key, trial_seed_for
from heat_case import Config, _assemble, _conductivity, _jacobi, _unflatten


def _json_safe(value):
    """Convert non-finite numeric values to JSON null without hiding failures."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _configs() -> List[Config]:
    return [
        *[Config(f"heat-{p}-{it}", p, it) for p in (16, 32, 64) for it in (25, 50, 100, 400)],
        Config("heat-exact", 64, 0, "direct"),
    ]


def _prepare_trial(seed: int, distribution: str, n: int) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, float, float]:
    shared: Dict = {}
    rng = InputRNG(seed, distribution, shared)
    k = _conductivity(rng, n, shared)
    matrix, rhs, _ = _assemble(k)
    t_ref = time.perf_counter()
    reference_interior = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    reference_elapsed = time.perf_counter() - t_ref
    reference_field = _unflatten(reference_interior, n)
    reference_qoi = float(np.mean(reference_interior))
    return matrix, rhs, reference_interior, reference_field, reference_qoi, reference_elapsed


def _solve_trial(seed: int, distribution: str, cfg: Config, n: int, prepared=None) -> Dict[str, float | bool]:
    if prepared is None:
        prepared = _prepare_trial(seed, distribution, n)
    matrix, rhs, reference_interior, reference_field, reference_qoi, reference_elapsed = prepared
    if cfg.solver == "direct":
        t0 = time.perf_counter()
        interior = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
        elapsed = time.perf_counter() - t0
    else:
        dtype = np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64
        t0 = time.perf_counter()
        interior = _jacobi(matrix, rhs, cfg.iterations, dtype)
        elapsed = time.perf_counter() - t0
    field = _unflatten(interior, n)
    qoi = float(np.mean(interior))
    qoi_error = abs(qoi - reference_qoi) / (abs(reference_qoi) + 1.0e-12)
    field_error = float(np.linalg.norm(field - reference_field) / (np.linalg.norm(reference_field) + 1.0e-12))
    residual = np.asarray(matrix.dot(interior) - rhs, dtype=np.float64)
    residual_rel = float(np.linalg.norm(residual) / (np.linalg.norm(rhs) + 1.0e-12))
    finite = bool(np.isfinite(interior).all() and np.isfinite(qoi_error) and np.isfinite(field_error) and np.isfinite(residual_rel))
    return {
        "error": float(qoi_error),
        "field_error": field_error,
        "residual_rel": residual_rel,
        "finite": finite,
        "wall_time_s": float(elapsed),
        "reference_wall_time_s": float(reference_elapsed),
        "mean_temperature": qoi,
    }


def _aggregate(values: Dict[str, List[float]], cfg: Config, n_cal: int, n_val: int, n_test: int,
               epsilon: float, delta: float, alpha: float, n_calls: int) -> Dict:
    selection_alpha = alpha / len(_configs())
    cal_err, val_err, test_err = values["cal_error"], values["validation_error"], values["test_error"]
    record = {
        "config": {"name": cfg.name, "precision": cfg.precision, "iterations": cfg.iterations, "solver": cfg.solver},
        "calibration_cost": float(np.mean(values["cal_time"])),
        "validation_cost": float(np.mean(values["validation_time"])),
        "test_cost": float(np.mean(values["test_time"])),
        "calibration_mean_error": float(np.mean(cal_err)),
        "validation_mean_error": float(np.mean(val_err)),
        "test_mean_error": float(np.mean(test_err)),
        "test_p95_error": float(np.quantile(test_err, 0.95)),
        "test_p99_error": float(np.quantile(test_err, 0.99)),
        "test_p95_field_error": float(np.quantile(values["test_field_error"], 0.95)),
        "test_p95_residual_rel": float(np.quantile(values["test_residual_rel"], 0.95)),
        "calibration_finite_rate": float(np.mean(values["cal_finite"])),
        "validation_finite_rate": float(np.mean(values["validation_finite"])),
        "test_finite_rate": float(np.mean(values["test_finite"])),
    }
    record.update({
        "calibration_successes": sum(e <= epsilon for e in cal_err),
        "validation_successes": sum(e <= epsilon for e in val_err),
        "test_successes": sum(e <= epsilon for e in test_err),
    })
    record["calibration_coverage"] = record["calibration_successes"] / n_cal
    record["validation_coverage"] = record["validation_successes"] / n_val
    record["test_coverage"] = record["test_successes"] / n_test
    record["calibration_lower_bound"] = lower_bound(record["calibration_successes"], n_cal, selection_alpha)
    record["validation_lower_bound"] = lower_bound(record["validation_successes"], n_val, selection_alpha)
    record["test_lower_bound"] = lower_bound(record["test_successes"], n_test, alpha)
    return record


def run_cell(seed: int, distribution: str, n: int, n_cal: int, n_val: int, n_test: int,
             epsilon: float, delta: float, alpha: float, n_calls: int,
             oblivious_mean_threshold: float, include_traces: bool = False) -> tuple[Dict, List[Dict]]:
    cfgs = _configs()
    buffers = {
        cfg.name: {key: [] for key in (
            "cal_error", "validation_error", "test_error", "cal_time", "validation_time", "test_time",
            "cal_field_error", "validation_field_error", "test_field_error", "cal_residual_rel",
            "validation_residual_rel", "test_residual_rel", "cal_finite", "validation_finite", "test_finite",
        )} for cfg in cfgs
    }
    traces: List[Dict] = []
    for split, count in (("calibration", n_cal), ("validation", n_val), ("test", n_test)):
        for index in range(count):
            trial_seed = trial_seed_for(seed, split, index)
            prepared = _prepare_trial(trial_seed, distribution, n)
            for cfg in cfgs:
                result = _solve_trial(trial_seed, distribution, cfg, n, prepared=prepared)
                prefix = {"calibration": "cal", "validation": "validation", "test": "test"}[split]
                for field, key in (("error", f"{prefix}_error"), ("wall_time_s", f"{prefix}_time"),
                                   ("field_error", f"{prefix}_field_error"), ("residual_rel", f"{prefix}_residual_rel"),
                                   ("finite", f"{prefix}_finite")):
                    buffers[cfg.name][key].append(result[field])
                if include_traces:
                    traces.append({"case": "2d_heat_conduction_64", "distribution": distribution,
                                   "seed_group": seed, "split": split, "index": index, "seed": trial_seed,
                                   "config": cfg.__dict__, **result})
    records = [_aggregate(buffers[cfg.name], cfg, n_cal, n_val, n_test, epsilon, delta, alpha, n_calls) for cfg in cfgs]
    costs = add_cost_accounting(records, n_cal, n_val, n_calls)
    exact = next(r for r in records if r["config"]["name"] == "heat-exact")
    feasible = [r for r in records if r["validation_lower_bound"] >= 1.0 - delta]
    chosen = min(feasible, key=selection_key) if feasible else exact
    non_exact = [r for r in records if r is not exact]
    approximate_feasible = [r for r in non_exact if r["validation_lower_bound"] >= 1.0 - delta]
    oblivious = select_oblivious(records, oblivious_mean_threshold, exact)
    validation_nonfinite = sum(not bool(value) for values in buffers.values() for value in values["validation_finite"])
    row = {
        "case": "2d_heat_conduction_64", "distribution": distribution, "seed": seed, "grid": n,
        "chosen": chosen["config"]["name"], "oblivious_chosen": oblivious["config"]["name"],
        "selection_lower_bound": chosen["validation_lower_bound"], "n_calibration": n_cal,
        "n_validation": n_val, "n_test": n_test, "n_calls": n_calls,
        "selection_cost_basis": "validation_timing", "test_coverage": chosen["test_coverage"],
        "test_successes": chosen["test_successes"], "test_lower_bound": chosen["test_lower_bound"],
        "oblivious_test_coverage": oblivious["test_coverage"], "oblivious_test_successes": oblivious["test_successes"],
        "oblivious_test_lower_bound": oblivious["test_lower_bound"], "test_p95_error": chosen["test_p95_error"],
        "test_p99_error": chosen["test_p99_error"], "test_cost": chosen["test_cost"],
        "total_cost": chosen["matched_total_cost"], **costs,
        "post_qualification_cost": chosen["post_qualification_cost"],
        "matched_total_cost": chosen["matched_total_cost"],
        "policy_specific_u_total_cost": chosen["policy_specific_u_total_cost"],
        "oblivious_total_cost": oblivious["matched_total_cost"], "oblivious_test_cost": oblivious["test_cost"],
        "oblivious_post_qualification_cost": oblivious["post_qualification_cost"],
        "oblivious_matched_total_cost": oblivious["matched_total_cost"],
        "oblivious_policy_specific_total_cost": oblivious["policy_specific_o_total_cost"],
        "oblivious_mean_threshold": oblivious_mean_threshold, "selection_alpha": alpha / len(cfgs),
        "n_configurations": len(cfgs), "energy_measurement": "unavailable",
        "fallback_to_exact": not bool(approximate_feasible), "selected_p95_field_error": chosen["test_p95_field_error"],
        "selected_p95_residual_rel": chosen["test_p95_residual_rel"], "selected_finite_rate": chosen["test_finite_rate"],
        "validation_nonfinite_candidates": int(validation_nonfinite),
        "selected_test_nonfinite": int(round((1.0 - float(chosen["test_finite_rate"])) * n_test)),
    }
    return row, traces


def summarize(rows: List[Dict], alpha: float = 0.05) -> List[Dict]:
    grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["distribution"])].append(row)
    output = []
    for (case, distribution), cell in sorted(grouped.items()):
        u_successes = sum(int(row["test_successes"]) for row in cell)
        o_successes = sum(int(row["oblivious_test_successes"]) for row in cell)
        n_test = sum(int(row["n_test"]) for row in cell)
        u_cost = np.asarray([float(row["total_cost"]) for row in cell])
        o_cost = np.asarray([float(row["oblivious_total_cost"]) for row in cell])
        savings = (o_cost - u_cost) / np.maximum(o_cost, 1.0e-300)
        output.append({
            "case": case, "distribution": distribution, "grid": int(cell[0]["grid"]),
            "n_seed_rows": len(cell), "n_test_observations": n_test,
            "u_selection_counts": dict(__import__("collections").Counter(r["chosen"] for r in cell)),
            "oblivious_selection_counts": dict(__import__("collections").Counter(r["oblivious_chosen"] for r in cell)),
            "u_coverage": u_successes / n_test, "u_cp_lower": float(beta.ppf(alpha, u_successes, n_test - u_successes + 1)),
            "oblivious_coverage": o_successes / n_test, "oblivious_cp_lower": float(beta.ppf(alpha, o_successes, n_test - o_successes + 1)),
            "u_mean_total_cost": float(np.mean(u_cost)), "oblivious_mean_total_cost": float(np.mean(o_cost)),
            "u_relative_savings_mean": float(np.mean(savings)), "u_fallback_rate": float(np.mean([r["fallback_to_exact"] for r in cell])),
            "selected_p95_field_error_max": float(max(r["selected_p95_field_error"] for r in cell)),
            "selected_p95_residual_rel_max": float(max(r["selected_p95_residual_rel"] for r in cell)),
            "selected_finite_rate_min": float(min(r["selected_finite_rate"] for r in cell)),
            "validation_nonfinite_candidates": int(sum(r["validation_nonfinite_candidates"] for r in cell)),
            "selected_test_nonfinite": int(sum(r["selected_test_nonfinite"] for r in cell)),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("heat-production-config.json"))
    parser.add_argument("--output", type=Path, default=Path("heat-production-audit.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("heat-production-audit.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("heat-production-audit-summary.json"))
    parser.add_argument("--distribution", choices=["light", "heavy", "correlated", "all"], default="all")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    distributions = cfg["distributions"] if args.distribution == "all" else [args.distribution]
    rows: List[Dict] = []
    traces: List[Dict] = []
    for distribution in distributions:
        for seed in cfg["seeds"]:
            row, cell_traces = run_cell(seed, distribution, int(cfg["grid"]), int(cfg["calibration"]), int(cfg["validation"]), int(cfg["test"]), float(cfg["epsilon"]), float(cfg["delta"]), float(cfg["alpha"]), int(cfg["n_calls"]), float(cfg["oblivious_mean_threshold"]), include_traces=True)
            rows.append({**row, "config_file": args.config.name, "config_sha256": digest, **environment_metadata()})
            traces.extend(cell_traces)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {"record_type": "metadata", "metadata": {"case": "2d_heat_conduction_64", "config_file": args.config.name, "config_sha256": digest, "grid": cfg["grid"], "config": cfg, **environment_metadata()}}
    with args.raw_output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for trace in traces:
            handle.write(json.dumps(_json_safe(trace), sort_keys=True, allow_nan=False) + "\n")
    payload = {"config_file": args.config.name, "config_sha256": digest, "interpretation": "larger-grid non-manufactured heat audit; solver timing excludes shared assembly", "rows": rows, "summary": summarize(rows, float(cfg["alpha"]))}
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "traces": len(traces), "summary_rows": len(payload["summary"]), "config_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
