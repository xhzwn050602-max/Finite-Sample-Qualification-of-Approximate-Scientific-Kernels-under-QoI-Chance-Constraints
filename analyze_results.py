"""Aggregate sweep rows into auditable coverage and cost summaries."""

from __future__ import annotations

import argparse
import csv
import json
from statistics import fmean, stdev
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, t


def cp_lower(successes: int, total: int, alpha: float) -> float:
    if total == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def mean_ci(values):
    """Descriptive two-sided 95% t interval over seed-group rows.

    The canonical publication rerun uses collision-free seed-group streams, so
    these intervals summarize independent group rows under the declared
    pseudo-random design.  They remain descriptive summaries of this fixed CPU
    experiment, not cross-hardware uncertainty claims.
    """
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "sd": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = fmean(values)
    if len(values) < 2:
        return {"mean": mean, "sd": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = stdev(values)
    half_width = float(t.ppf(0.975, len(values) - 1) * sd / (len(values) ** 0.5))
    return {"mean": mean, "sd": sd, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def _trace_tail_stats(path: str | None) -> dict[tuple[str, str, int, str], tuple[float, float]]:
    """Read optional JSONL traces and index test-tail errors by seed/config.

    Older publication runs predate the CSV p95/p99 columns, but their raw
    traces still contain the per-trial errors.  Reconstructing these values
    here keeps the aggregate auditable without rerunning an expensive sweep.
    """
    if path is None:
        return {}
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata" or item.get("split") != "test":
                continue
            config = item.get("config", {})
            name = config.get("name")
            if name is None:
                continue
            key = (str(item["case"]), str(item["distribution"]), int(item["seed_group"]), str(name))
            grouped[key].append(float(item["error"]))
    return {
        key: (float(np.quantile(values, 0.95)), float(np.quantile(values, 0.99)))
        for key, values in grouped.items()
    }


def _trace_seed_overlap(path: str | None) -> bool | None:
    """Return whether trial seeds are reused across seed groups.

    A seed is expected to repeat across configurations within one group, so
    only reuse by two or more distinct groups is considered overlap.  ``None``
    means that no raw trace was supplied and the pooled diagnostic remains
    conservatively unqualified.
    """
    if path is None:
        return None
    groups: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                continue
            if item.get("split") not in {"calibration", "validation", "test"}:
                continue
            if item.get("seed") is None or item.get("seed_group") is None:
                continue
            key = (str(item.get("case")), str(item.get("split")), int(item["seed"]))
            groups[key].add(int(item["seed_group"]))
    return any(len(seed_groups) > 1 for seed_groups in groups.values())


EXACT_CONFIGS = {
    "fft": "fft-64-1.0",
    "spmv": "spmv-64-0.0",
    "stencil": "stencil-64-0.0",
    "monte_carlo": "mc-64-1000000",
    "2d_poisson": "poisson-exact",
    "2d_heat_conduction": "heat-exact",
}


def _exact_total_costs(path: str | None, n_calls: int) -> dict[tuple[str, str, int], float]:
    """Reconstruct matched-budget exact-policy costs from paired traces."""
    if path is None:
        return {}
    sums: dict[tuple[str, str, int, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, int, str, str], int] = defaultdict(int)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata" or item.get("split") not in {"calibration", "validation", "test"}:
                continue
            if "wall_time_s" not in item or "seed_group" not in item:
                continue
            config_name = item.get("config", {}).get("name")
            if config_name is None:
                continue
            key = (str(item["case"]), str(item["distribution"]), int(item["seed_group"]), str(config_name), str(item["split"]))
            sums[key] += float(item["wall_time_s"])
            counts[key] += 1
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    split_counts: dict[tuple[str, str, int, str, str], int] = defaultdict(int)
    for (case, distribution, seed, config, split), total in sums.items():
        if case not in EXACT_CONFIGS:
            continue
        mean = total / counts[(case, distribution, seed, config, split)]
        if split in {"calibration", "validation"}:
            prefix = "cal" if split == "calibration" else "val"
            grouped[(case, distribution, seed)][f"{prefix}:{config}"] = mean
            split_counts[(case, distribution, seed, config, split)] = counts[(case, distribution, seed, config, split)]
        elif config == EXACT_CONFIGS[case]:
            grouped[(case, distribution, seed)]["exact_test"] = mean
    output: dict[tuple[str, str, int], float] = {}
    for key, values in grouped.items():
        if "exact_test" not in values:
            continue
        calibration_n = max(
            (count for (case, distribution, seed, _, split), count in split_counts.items()
             if (case, distribution, seed) == key and split == "calibration"),
            default=0,
        )
        validation_n = max(
            (count for (case, distribution, seed, _, split), count in split_counts.items()
             if (case, distribution, seed) == key and split == "validation"),
            default=0,
        )
        output[key] = (
            calibration_n * sum(value for name, value in values.items() if name.startswith("cal:"))
            + validation_n * sum(value for name, value in values.items() if name.startswith("val:"))
            + n_calls * values["exact_test"]
        )
    return output


def summarize(path: str, alpha: float = 0.05, raw_traces: str | None = None):
    grouped = defaultdict(list)
    trace_tails = _trace_tail_stats(raw_traces)
    n_calls_by_group = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["case"], row["distribution"])].append(row)
            n_calls_by_group[(row["case"], row["distribution"])] = int(row.get("n_calls") or 0)
    output = []
    seed_overlap = _trace_seed_overlap(raw_traces)
    if seed_overlap is None:
        coverage_aggregation = (
            "minimum per seed-group CP bound; pooled CP unqualified without raw traces"
        )
    elif seed_overlap:
        coverage_aggregation = (
            "minimum per seed-group CP bound; pooled CP is invalid because trial seeds overlap"
        )
    else:
        coverage_aggregation = (
            "minimum per seed-group CP bound; pooled CP is valid only for a homogeneous selected configuration"
        )
    for (case, distribution), rows in sorted(grouped.items()):
        provenance = {(row.get("config_file", ""), row.get("config_sha256", "")) for row in rows}
        if len(provenance) > 1:
            raise ValueError(
                f"mixed experiment configurations for {case}/{distribution}: "
                f"{sorted(provenance)!r}"
            )
        config_file, config_sha256 = next(iter(provenance))
        selection_alphas = {row.get("selection_alpha", "") for row in rows}
        configuration_counts = {row.get("n_configurations", "") for row in rows}
        if len(selection_alphas) > 1 or len(configuration_counts) > 1:
            raise ValueError(f"inconsistent selection metadata for {case}/{distribution}")
        selection_alpha = next(iter(selection_alphas))
        n_configurations = next(iter(configuration_counts))
        n_test = sum(int(row["n_test"]) for row in rows)
        per_row_u_lower = [float(row["test_lower_bound"]) for row in rows]
        per_row_o_lower = [float(row["oblivious_test_lower_bound"]) for row in rows]
        u_successes = sum(int(row["test_successes"]) for row in rows)
        o_successes = sum(int(row["oblivious_test_successes"]) for row in rows)
        explicit_costs = all(
            row.get("matched_total_cost") not in (None, "")
            and row.get("oblivious_matched_total_cost") not in (None, "")
            for row in rows
        )
        u_costs = [float(row["matched_total_cost"] if explicit_costs else row["total_cost"]) for row in rows]
        o_costs = [float(row["oblivious_matched_total_cost"] if explicit_costs else row["oblivious_total_cost"]) for row in rows]
        u_post_costs = [float(row["post_qualification_cost"]) for row in rows] if explicit_costs else []
        o_post_costs = [float(row["oblivious_post_qualification_cost"]) for row in rows] if explicit_costs else []
        u_policy_costs = [float(row["policy_specific_u_total_cost"]) for row in rows] if explicit_costs else []
        o_policy_costs = [float(row["oblivious_policy_specific_total_cost"]) for row in rows] if explicit_costs else []
        paired_deltas = [u - o for u, o in zip(u_costs, o_costs)]
        paired_savings = [(o - u) / o if o > 0.0 else 0.0 for u, o in zip(u_costs, o_costs)]
        exact_cost_map = _exact_total_costs(raw_traces, n_calls_by_group[(case, distribution)])
        exact_costs = [exact_cost_map.get((case, distribution, int(row["seed"]))) for row in rows]
        exact_costs = [value for value in exact_costs if value is not None]
        exact_stats = mean_ci(exact_costs)
        exact_savings = [
            (exact - u) / exact
            for exact, u in zip(
                [exact_cost_map.get((case, distribution, int(row["seed"]))) for row in rows], u_costs
            )
            if exact is not None and exact > 0.0
        ]
        u_stats = mean_ci(u_costs)
        o_stats = mean_ci(o_costs)
        u_post_stats = mean_ci(u_post_costs)
        o_post_stats = mean_ci(o_post_costs)
        delta_stats = mean_ci(paired_deltas)
        center_values = [float(row["qoi_center_mean"]) for row in rows if row.get("qoi_center_mean") not in (None, "")]
        residual_values = [float(row["residual_mean"]) for row in rows if row.get("residual_mean") not in (None, "")]
        residual_p95_values = [float(row["residual_p95"]) for row in rows if row.get("residual_p95") not in (None, "")]
        finite_values = [float(row["finite_rate"]) for row in rows if row.get("finite_rate") not in (None, "")]
        p95_values = []
        p99_values = []
        for row in rows:
            if row.get("test_p95_error") not in (None, ""):
                p95, p99 = float(row["test_p95_error"]), float(row["test_p99_error"])
            else:
                p95, p99 = trace_tails.get(
                    (row["case"], row["distribution"], int(row["seed"]), row["chosen"]),
                    (None, None),
                )
            if p95 is not None:
                p95_values.append(p95)
            if p99 is not None:
                p99_values.append(p99)
        u_selected_configs = {row["chosen"] for row in rows}
        o_selected_configs = {row["oblivious_chosen"] for row in rows}
        pooled_cp_valid_u = seed_overlap is False and len(u_selected_configs) == 1
        pooled_cp_valid_o = seed_overlap is False and len(o_selected_configs) == 1
        output.append({
            "case": case,
            "distribution": distribution,
            "config_file": config_file or None,
            "config_sha256": config_sha256 or None,
            "selection_alpha": float(selection_alpha) if selection_alpha else None,
            "n_configurations": int(n_configurations) if n_configurations else None,
            "u_selection_counts": dict(Counter(row["chosen"] for row in rows)),
            "oblivious_selection_counts": dict(Counter(row["oblivious_chosen"] for row in rows)),
            "n_seed_rows": len(rows),
            "n_test_observations": n_test,
            "n_test_observations_per_seed_group": int(rows[0]["n_test"]) if rows else 0,
            "u_coverage": u_successes / n_test if n_test else 0.0,
            # The plotted and headline bound is the minimum per-group CP bound.
            # Pooled counts are retained only as descriptive diagnostics unless
            # all rows use one configuration and the trial streams are disjoint.
            "u_cp_lower": min(per_row_u_lower) if per_row_u_lower else 0.0,
            "u_cp_lower_pooled_descriptive": cp_lower(u_successes, n_test, alpha),
            "u_selected_config_counts": dict(Counter(row["chosen"] for row in rows)),
            "u_pooled_cp_valid": pooled_cp_valid_u,
            "oblivious_coverage": o_successes / n_test if n_test else 0.0,
            "oblivious_cp_lower": min(per_row_o_lower) if per_row_o_lower else 0.0,
            "oblivious_cp_lower_pooled_descriptive": cp_lower(o_successes, n_test, alpha),
            "oblivious_selected_config_counts": dict(Counter(row["oblivious_chosen"] for row in rows)),
            "oblivious_pooled_cp_valid": pooled_cp_valid_o,
            "coverage_aggregation": coverage_aggregation,
            "pooled_cp_valid": pooled_cp_valid_u and pooled_cp_valid_o,
            "u_mean_total_cost": u_stats["mean"],
            "u_total_cost_sd": u_stats["sd"],
            "u_total_cost_ci95_low": u_stats["ci95_low"],
            "u_total_cost_ci95_high": u_stats["ci95_high"],
            "oblivious_mean_total_cost": o_stats["mean"],
            "oblivious_total_cost_sd": o_stats["sd"],
            "oblivious_total_cost_ci95_low": o_stats["ci95_low"],
            "oblivious_total_cost_ci95_high": o_stats["ci95_high"],
            "u_minus_oblivious_mean_cost": delta_stats["mean"],
            "u_minus_oblivious_ci95_low": delta_stats["ci95_low"],
            "u_minus_oblivious_ci95_high": delta_stats["ci95_high"],
            "u_relative_savings_mean": fmean(paired_savings) if paired_savings else 0.0,
            "cost_accounting": "matched qualification" if explicit_costs else "legacy",
            "u_mean_post_qualification_cost": u_post_stats["mean"] if u_post_costs else None,
            "oblivious_mean_post_qualification_cost": o_post_stats["mean"] if o_post_costs else None,
            "u_post_qualification_savings_mean": (
                fmean((o - u) / o if o > 0.0 else 0.0 for u, o in zip(u_post_costs, o_post_costs))
                if u_post_costs else None
            ),
            "u_policy_specific_savings_vs_o_mean": (
                fmean((o - u) / o if o > 0.0 else 0.0 for u, o in zip(u_policy_costs, o_policy_costs))
                if u_policy_costs else None
            ),
            "exact_mean_total_cost": exact_stats["mean"] if exact_costs else None,
            "exact_total_cost_sd": exact_stats["sd"] if exact_costs else None,
            "u_relative_savings_vs_exact_mean": fmean(exact_savings) if exact_savings else None,
            "u_fallback_rate": fmean(row["fallback_to_exact"].lower() == "true" for row in rows) if rows else 0.0,
            "u_min_test_lower_bound": min(float(row["test_lower_bound"]) for row in rows),
            "oblivious_min_test_lower_bound": min(float(row["oblivious_test_lower_bound"]) for row in rows),
            "selected_qoi_center_mean": fmean(center_values) if center_values else None,
            "selected_residual_mean": fmean(residual_values) if residual_values else None,
            "selected_residual_p95_max": max(residual_p95_values) if residual_p95_values else None,
            "selected_finite_rate_min": min(finite_values) if finite_values else None,
            "selected_test_p95_error_max": max(p95_values) if p95_values else None,
            "selected_test_p99_error_max": max(p99_values) if p99_values else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="results-summary.json")
    parser.add_argument("--raw-traces", default=None, help="Optional JSONL traces for reconstructing p95/p99 tails")
    args = parser.parse_args()
    summary = summarize(args.input_csv, args.alpha, args.raw_traces)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
