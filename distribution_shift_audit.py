"""Post hoc distribution-shift audit for the locked publication traces.

The primary controller guarantee is conditional on the declared input law.  This
audit deliberately selects U/O from one recorded distribution and evaluates the
same per-seed configurations on another distribution's held-out traces.  It is
descriptive only: no target-distribution data are used for selection and no
cross-distribution guarantee is claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import beta


DEFAULT_DISTRIBUTIONS = ("light", "heavy", "correlated")


def cp_lower(successes: int, total: int, alpha: float) -> float:
    """One-sided exact lower bound for a Bernoulli success probability."""
    if total <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _read_source(path: Path, expected_distribution: str) -> tuple[dict[tuple[str, int], dict], float, str]:
    selections: dict[tuple[str, int], dict] = {}
    alpha_values: set[float] = set()
    digest_values: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("distribution") != expected_distribution:
                raise ValueError(f"source CSV {path} contains an unexpected distribution")
            key = (row["case"], int(row["seed"]))
            if key in selections:
                raise ValueError(f"duplicate source selection for {expected_distribution}/{key}")
            selections[key] = {
                "u": row["chosen"],
                "o": row["oblivious_chosen"],
            }
            alpha_values.add(float(row["selection_alpha"]) * int(row["n_configurations"]))
            digest_values.add(row.get("config_sha256", ""))
    if not selections:
        raise ValueError(f"empty source CSV: {path}")
    if len(alpha_values) != 1:
        raise ValueError(f"inconsistent alpha values in {path}: {sorted(alpha_values)}")
    if len(digest_values) != 1 or not next(iter(digest_values)):
        raise ValueError(f"missing or inconsistent configuration digest in {path}")
    return selections, next(iter(alpha_values)), next(iter(digest_values))


def _target_errors(
    path: Path,
    selections: dict[tuple[str, int], dict],
    expected_distribution: str,
    expected_digest: str,
) -> dict[tuple[str, int, str], list[float]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    metadata: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata.append(item.get("metadata", {}))
                continue
            if item.get("distribution") != expected_distribution:
                raise ValueError(f"target trace {path} contains an unexpected distribution")
            if item.get("split") != "test":
                continue
            case = str(item["case"])
            seed = int(item["seed_group"])
            selected = selections.get((case, seed))
            if selected is None:
                continue
            config_name = str(item.get("config", {}).get("name", ""))
            for policy in ("u", "o"):
                if config_name == selected[policy]:
                    grouped[(case, seed, policy)].append(float(item["error"]))
    if len(metadata) != 1 or metadata[0].get("config_sha256") != expected_digest:
        raise ValueError(f"target trace {path} configuration digest does not match source CSV")
    return grouped


def _target_seed_overlap(path: Path) -> bool:
    """Check reuse of trial seeds across target seed groups."""
    groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata" or item.get("split") != "test":
                continue
            if item.get("seed") is None or item.get("seed_group") is None:
                continue
            groups[(str(item["case"]), int(item["seed"]))].add(int(item["seed_group"]))
    return any(len(seed_groups) > 1 for seed_groups in groups.values())


def audit_pair(source_distribution: str, target_distribution: str, source_csv: Path, target_traces: Path) -> list[dict]:
    selections, alpha, digest = _read_source(source_csv, source_distribution)
    grouped = _target_errors(target_traces, selections, target_distribution, digest)
    seed_overlap = _target_seed_overlap(target_traces)
    coverage_aggregation = (
        "minimum per seed-group CP bound; pooled CP is invalid because target trial seeds overlap"
        if seed_overlap else
        "minimum per seed-group CP bound; pooled CP is descriptive only because selected configurations may differ"
    )
    cases = sorted({case for case, _ in selections})
    rows: list[dict] = []
    for case in cases:
        for policy in ("u", "o"):
            per_seed = {
                seed: errors
                for (row_case, seed, row_policy), errors in grouped.items()
                if row_case == case and row_policy == policy
            }
            values = [value for errors in per_seed.values() for value in errors]
            selected_names = [selected[policy] for (row_case, _seed), selected in selections.items()
                              if row_case == case]
            if not values:
                raise ValueError(f"no target test traces for {source_distribution}->{target_distribution}/{case}/{policy}")
            tolerance = 0.05 if case != "2d_poisson" else 0.20
            successes = sum(value <= tolerance for value in values)
            per_seed_bounds = [
                cp_lower(sum(value <= tolerance for value in errors), len(errors), alpha)
                for errors in per_seed.values()
            ]
            pooled_homogeneous = len(set(selected_names)) == 1
            rows.append({
                "source_distribution": source_distribution,
                "target_distribution": target_distribution,
                "case": case,
                "policy": policy.upper(),
                "source_seed_rows": len(selected_names),
                "selected_config_counts": dict(Counter(selected_names)),
                "n_target_test_observations": len(values),
                "n_target_test_observations_per_seed_group": min((len(errors) for errors in per_seed.values()), default=0),
                "tolerance": tolerance,
                "target_coverage": successes / len(values),
                "target_successes": successes,
                "target_cp_lower": min(per_seed_bounds) if per_seed_bounds else 0.0,
                "target_cp_lower_pooled_descriptive": cp_lower(successes, len(values), alpha),
                "coverage_aggregation": coverage_aggregation,
                "pooled_cp_valid": (not seed_overlap) and pooled_homogeneous,
                "selected_configuration_homogeneous": pooled_homogeneous,
                "target_p95_error": float(np.quantile(values, 0.95)),
                "target_p99_error": float(np.quantile(values, 0.99)),
                "selection_data_used": "source calibration/validation only",
                "guarantee_status": "not_applicable_under_distribution_shift",
            })
    return rows


def run_audit(root: Path) -> list[dict]:
    sources = {d: root / f"publication-{d}.csv" for d in DEFAULT_DISTRIBUTIONS}
    traces = {d: root / f"publication-{d}-traces.jsonl" for d in DEFAULT_DISTRIBUTIONS}
    for path in (*sources.values(), *traces.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows: list[dict] = []
    for source in DEFAULT_DISTRIBUTIONS:
        for target in DEFAULT_DISTRIBUTIONS:
            if source == target:
                continue
            rows.extend(audit_pair(source, target, sources[source], traces[target]))
    return rows


def write_outputs(rows: Iterable[dict], output_json: Path, output_csv: Path) -> None:
    rows = list(rows)
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fields = [
        "source_distribution", "target_distribution", "case", "policy",
        "source_seed_rows", "selected_config_counts", "n_target_test_observations",
        "n_target_test_observations_per_seed_group",
        "tolerance", "target_coverage", "target_successes", "target_cp_lower",
        "target_cp_lower_pooled_descriptive", "coverage_aggregation", "pooled_cp_valid",
        "selected_configuration_homogeneous",
        "target_p95_error", "target_p99_error", "selection_data_used", "guarantee_status",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["selected_config_counts"] = json.dumps(item["selected_config_counts"], sort_keys=True)
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=Path("distribution-shift-audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("distribution-shift-audit.csv"))
    args = parser.parse_args()
    rows = run_audit(args.root)
    write_outputs(rows, args.output_json, args.output_csv)
    print(json.dumps({"rows": len(rows), "output_json": str(args.output_json), "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
