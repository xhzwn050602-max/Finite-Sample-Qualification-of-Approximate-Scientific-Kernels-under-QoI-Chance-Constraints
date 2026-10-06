"""Replay policy choices over deployment call counts from immutable traces.

This is a post hoc accounting diagnostic. It reuses the publication errors and
timings, changes only the declared production call count, and never measures a
new kernel execution. U uses the validation CP gate and deterministic
cost/confidence tie-break; O uses its calibration mean-error gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import beta, t

from benchmark import add_cost_accounting, select_oblivious, selection_key


EXACT = {
    "fft": "fft-64-1.0",
    "spmv": "spmv-64-0.0",
    "stencil": "stencil-64-0.0",
    "monte_carlo": "mc-64-1000000",
    "2d_poisson": "poisson-exact",
}
DEFAULT_CALL_COUNTS = (1, 10, 100, 1_000, 10_000, 100_000)


def cp_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(values) - 1) * stdev(values) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def load_traces(paths: list[Path]):
    grouped = defaultdict(lambda: defaultdict(list))
    metadata_by_path = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            metadata = None
            for line in handle:
                item = json.loads(line)
                if item.get("record_type") == "metadata":
                    metadata = item.get("metadata")
                    continue
                key = (
                    str(item["case"]),
                    str(item["distribution"]),
                    int(item["seed_group"]),
                    str(item["config"]["name"]),
                )
                split = str(item["split"])
                grouped[key][f"{split}_error"].append(float(item["error"]))
                grouped[key][f"{split}_time"].append(float(item["wall_time_s"]))
            if metadata is None:
                raise ValueError(f"missing metadata in {path}")
            metadata_by_path[str(path)] = metadata
    digests = {m.get("config_sha256") for m in metadata_by_path.values()}
    if len(digests) != 1:
        raise ValueError(f"trace configuration digests differ: {sorted(digests)}")
    return grouped, metadata_by_path, next(iter(digests))


def audit(
    paths: list[Path],
    output_json: Path,
    output_csv: Path,
    call_counts: tuple[int, ...] = DEFAULT_CALL_COUNTS,
    alpha: float = 0.05,
    delta: float = 0.10,
    config_path: Path = Path("publication-config.json"),
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grouped, metadata_by_path, digest = load_traces(paths)
    rows = []
    for case, exact_name in EXACT.items():
        keys = [key for key in grouped if key[0] == case]
        names = sorted({key[3] for key in keys})
        if not names or exact_name not in names:
            raise ValueError(f"missing candidate set for {case}")
        epsilon = float(config["poisson_epsilon"] if case == "2d_poisson" else config["kernel_epsilon"])
        threshold = float(
            config["poisson_oblivious_mean_threshold"]
            if case == "2d_poisson"
            else config["oblivious_mean_threshold"]
        )
        selection_alpha = alpha / len(names)
        distributions = sorted({key[1] for key in keys})
        for distribution in distributions:
            seeds = sorted({key[2] for key in keys if key[1] == distribution})
            for seed in seeds:
                records = {}
                for name in names:
                    values = grouped[(case, distribution, seed, name)]
                    cal_errors = values["calibration_error"]
                    val_errors = values["validation_error"]
                    test_errors = values["test_error"]
                    val_successes = sum(error <= epsilon for error in val_errors)
                    records[name] = {
                        "config": {"name": name},
                        "calibration_cost": fmean(values["calibration_time"]),
                        "validation_cost": fmean(values["validation_time"]),
                        "test_cost": fmean(values["test_time"]),
                        "calibration_mean_error": fmean(cal_errors),
                        "validation_lower_bound": cp_lower(
                            val_successes, len(val_errors), selection_alpha
                        ),
                        "test_coverage": sum(error <= epsilon for error in test_errors) / len(test_errors),
                    }
                n_cal = len(grouped[(case, distribution, seed, names[0])]["calibration_error"])
                n_val = len(grouped[(case, distribution, seed, names[0])]["validation_error"])
                for n_calls in call_counts:
                    costs = add_cost_accounting(records.values(), n_cal, n_val, n_calls)
                    feasible = [
                        name
                        for name, record in records.items()
                        if record["validation_lower_bound"] >= 1.0 - delta
                    ]
                    chosen = (
                        min(feasible, key=lambda name: selection_key(records[name]))
                        if feasible
                        else exact_name
                    )
                    oblivious_record = select_oblivious(records.values(), threshold, records[exact_name])
                    oblivious = str(oblivious_record["config"]["name"])
                    u = records[chosen]
                    o = records[oblivious]
                    u_cost = u["matched_total_cost"]
                    o_cost = o["matched_total_cost"]
                    rows.append(
                        {
                            "case": case,
                            "distribution": distribution,
                            "seed": seed,
                            "n_calls": n_calls,
                            "u_chosen": chosen,
                            "o_chosen": oblivious,
                            "u_total_cost": u_cost,
                            "o_total_cost": o_cost,
                            **costs,
                            "u_post_qualification_cost": u["post_qualification_cost"],
                            "o_post_qualification_cost": o["post_qualification_cost"],
                            "u_policy_specific_total_cost": u["policy_specific_u_total_cost"],
                            "o_policy_specific_total_cost": o["policy_specific_o_total_cost"],
                            "u_savings_vs_o": (o_cost - u_cost) / o_cost if o_cost else 0.0,
                            "u_test_coverage": u["test_coverage"],
                            "o_test_coverage": o["test_coverage"],
                            "u_fallback_to_exact": chosen == exact_name,
                        }
                    )
    summary_groups = defaultdict(list)
    for row in rows:
        summary_groups[(row["case"], row["distribution"], row["n_calls"])].append(row)
    summaries = []
    for (case, distribution, n_calls), items in sorted(summary_groups.items()):
        savings = [float(item["u_savings_vs_o"]) for item in items]
        mean, ci_low, ci_high = _mean_ci(savings)
        summaries.append(
            {
                "case": case,
                "distribution": distribution,
                "n_calls": n_calls,
                "n_seed_rows": len(items),
                "mean_u_savings_vs_o": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_u_test_coverage": fmean(item["u_test_coverage"] for item in items),
                "mean_o_test_coverage": fmean(item["o_test_coverage"] for item in items),
                "u_selection_counts": dict(Counter(item["u_chosen"] for item in items)),
                "o_selection_counts": dict(Counter(item["o_chosen"] for item in items)),
            }
        )
    result = {
        "trace_files": [str(path) for path in paths],
        "trace_metadata": metadata_by_path,
        "config_sha256": digest,
        "call_counts": list(call_counts),
        "alpha": alpha,
        "delta": delta,
        "rows": rows,
        "summaries": summaries,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = [
        "case", "distribution", "seed", "n_calls", "u_chosen", "o_chosen",
        "u_total_cost", "o_total_cost", "calibration_overhead", "validation_overhead",
        "qualification_overhead", "u_post_qualification_cost", "o_post_qualification_cost",
        "u_policy_specific_total_cost", "o_policy_specific_total_cost", "u_savings_vs_o", "u_test_coverage",
        "o_test_coverage", "u_fallback_to_exact",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "summary_rows": len(summaries), "config_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("call-count-audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("call-count-audit.csv"))
    parser.add_argument("--call-counts", nargs="+", type=int, default=list(DEFAULT_CALL_COUNTS))
    args = parser.parse_args()
    print(json.dumps(audit(args.traces, args.output_json, args.output_csv, tuple(args.call_counts)), indent=2))


if __name__ == "__main__":
    main()
