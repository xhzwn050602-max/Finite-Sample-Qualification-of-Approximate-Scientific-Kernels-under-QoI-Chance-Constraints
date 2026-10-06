"""Numerical-stability stress audit for the larger heat-conduction problem.

The audit varies Jacobi precision and iteration count on fixed 64-by-64
conductivity fields.  It measures finite-value rate, QoI/field error, and
discrete residual against the sparse direct reference.  It is a diagnostic
stress test, not an additional policy-selection or coverage claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark import environment_metadata
from heat_case import Config
from production_heat_audit import _prepare_trial, _solve_trial


def _configs(iterations: list[int]) -> list[Config]:
    return [Config(f"heat-{precision}-{count}", precision, count) for precision in (16, 32, 64) for count in iterations]


def run(config: dict, config_path: Path) -> tuple[list[dict], list[dict]]:
    n = int(config["grid"])
    cfgs = _configs([int(value) for value in config["iterations"]])
    rows: list[dict] = []
    traces: list[dict] = []
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for distribution in config["distributions"]:
        for seed in config["seeds"]:
            trial_seed = int(seed) + int(config.get("seed_offset", 30000))
            prepared = _prepare_trial(trial_seed, distribution, n)
            for cfg in cfgs:
                result = _solve_trial(trial_seed, distribution, cfg, n, prepared=prepared)
                traces.append({"case": "2d_heat_conduction_64", "distribution": distribution, "seed_group": int(seed), "seed": trial_seed, "config": cfg.__dict__, **result})
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for trace in traces:
        grouped[(trace["distribution"], int(trace["config"]["precision"]), int(trace["config"]["iterations"]))].append(trace)
    for (distribution, precision, iterations), values in sorted(grouped.items()):
        rows.append({
            "case": "2d_heat_conduction_64", "distribution": distribution, "grid": n,
            "precision": precision, "iterations": iterations, "n_trials": len(values),
            "finite_rate": float(np.mean([bool(v["finite"]) for v in values])),
            "qoi_error_p95": float(np.quantile([float(v["error"]) for v in values], 0.95)),
            "field_error_p95": float(np.quantile([float(v["field_error"]) for v in values], 0.95)),
            "residual_p95": float(np.quantile([float(v["residual_rel"]) for v in values], 0.95)),
            "qoi_error_max": float(np.max([float(v["error"]) for v in values])),
            "field_error_max": float(np.max([float(v["field_error"]) for v in values])),
            "residual_max": float(np.max([float(v["residual_rel"]) for v in values])),
            "config_file": config_path.name, "config_sha256": digest, **environment_metadata(),
        })
    return rows, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("stability-audit-config.json"))
    parser.add_argument("--output", type=Path, default=Path("stability-audit.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("stability-audit.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("stability-audit-summary.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, traces = run(config, args.config)
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {"record_type": "metadata", "metadata": {"config_file": args.config.name, "config_sha256": digest, "interpretation": "diagnostic stability stress audit; no policy selection", **config, **environment_metadata()}}
    with args.raw_output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    payload = {"config_file": args.config.name, "config_sha256": digest, "interpretation": "diagnostic stability stress audit; no policy selection or new coverage claim", "rows": rows}
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "traces": len(traces), "config_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
