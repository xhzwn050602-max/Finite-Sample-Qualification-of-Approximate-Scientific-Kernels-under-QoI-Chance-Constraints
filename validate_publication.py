"""Validate the completeness and provenance of publication sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_CASES = {"fft", "spmv", "stencil", "monte_carlo", "2d_poisson"}
EXPECTED_SPLITS = {"calibration": 256, "validation": 256, "test": 1024}
EXPECTED_CONFIGS = {"fft": 9, "spmv": 9, "stencil": 9, "monte_carlo": 9, "2d_poisson": 13}


def validate(csv_path: Path, trace_path: Path, config_path: Path, policy_path: Path) -> dict:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("source_config") != config_path.name or policy.get("source_config_sha256") != digest:
        raise ValueError(f"{policy_path}: source configuration identity mismatch")
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if len(rows) != 50:
        raise ValueError(f"{csv_path}: expected 50 data rows, found {len(rows)}")
    distributions = {row["distribution"] for row in rows}
    if len(distributions) != 1:
        raise ValueError(f"{csv_path}: expected one distribution, found {distributions}")
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 10:
        raise ValueError(f"{csv_path}: expected 10 seeds, found {len(seeds)}")
    if {row.get("config_sha256") for row in rows} != {digest}:
        raise ValueError(f"{csv_path}: configuration digest mismatch")
    if {row.get("config_file") for row in rows} != {config_path.name}:
        raise ValueError(f"{csv_path}: configuration filename mismatch")
    if {row.get("selection_cost_basis") for row in rows} != {"validation_timing"}:
        raise ValueError(f"{csv_path}: selection cost basis must be validation_timing")
    if {row.get("policy_spec_file") for row in rows} != {policy_path.name}:
        raise ValueError(f"{csv_path}: policy specification filename mismatch")
    if {row.get("policy_spec_sha256") for row in rows} != {policy_digest}:
        raise ValueError(f"{csv_path}: policy specification digest mismatch")
    required_cost_fields = {
        "post_qualification_cost", "matched_total_cost", "policy_specific_u_total_cost",
        "oblivious_post_qualification_cost", "oblivious_matched_total_cost",
        "oblivious_policy_specific_total_cost", "validation_overhead", "qualification_overhead",
    }
    if not rows or not required_cost_fields.issubset(rows[0]):
        raise ValueError(f"{csv_path}: canonical fair-accounting fields are incomplete")
    if {row["case"] for row in rows} != EXPECTED_CASES:
        raise ValueError(f"{csv_path}: case set mismatch")
    trace_counts = Counter()
    pair_configs: dict[tuple[str, str, int, int], set[str]] = {}
    split_seeds: dict[tuple[str, str, int], set[int]] = {}
    seed_owners: dict[tuple[str, str, int], set[int]] = {}
    calibration_errors: dict[tuple[str, int, str], list[float]] = {}
    calibration_times: dict[tuple[str, int, str], list[float]] = {}
    metadata = None
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata = item.get("metadata")
                continue
            case = item.get("case")
            split = item.get("split")
            candidate_config = item.get("config", {})
            seed_group = item.get("seed_group")
            trial_seed = item.get("seed")
            index = item.get("index")
            config_name = candidate_config.get("name") if isinstance(candidate_config, dict) else None
            if (
                case not in EXPECTED_CASES
                or split not in EXPECTED_SPLITS
                or "error" not in item
                or not isinstance(seed_group, int)
                or not isinstance(trial_seed, int)
                or not isinstance(index, int)
                or not isinstance(config_name, str)
                or not config_name
            ):
                raise ValueError(f"{trace_path}: malformed trace record")
            trace_counts[(case, split)] += 1
            pair_key = (case, split, seed_group, index)
            pair_configs.setdefault(pair_key, set()).add(config_name)
            split_seeds.setdefault((case, split, seed_group), set()).add(trial_seed)
            seed_owners.setdefault((case, split, trial_seed), set()).add(seed_group)
            if split == "calibration":
                calibration_errors.setdefault((case, seed_group, config_name), []).append(float(item["error"]))
                calibration_times.setdefault((case, seed_group, config_name), []).append(float(item["wall_time_s"]))
    if metadata is None or metadata.get("config_sha256") != digest:
        raise ValueError(f"{trace_path}: missing or inconsistent metadata")
    expected = {
        (case, split): 10 * EXPECTED_SPLITS[split] * EXPECTED_CONFIGS[case]
        for case in EXPECTED_CASES
        for split in EXPECTED_SPLITS
    }
    if trace_counts != Counter(expected):
        raise ValueError(f"{trace_path}: trace counts differ from expected")
    expected_pair_keys = {
        (case, split): 10 * EXPECTED_SPLITS[split]
        for case in EXPECTED_CASES
        for split in EXPECTED_SPLITS
    }
    observed_pair_keys = Counter((case, split) for case, split, _, _ in pair_configs)
    if observed_pair_keys != Counter(expected_pair_keys):
        raise ValueError(f"{trace_path}: candidate trials are not uniquely paired")
    for (case, split, _, _), configs_seen in pair_configs.items():
        if len(configs_seen) != EXPECTED_CONFIGS[case]:
            raise ValueError(f"{trace_path}: incomplete candidate pair for {case}/{split}")
    for case in EXPECTED_CASES:
        for seed_group in seeds:
            calibration = split_seeds.get((case, "calibration", seed_group), set())
            validation = split_seeds.get((case, "validation", seed_group), set())
            test = split_seeds.get((case, "test", seed_group), set())
            if calibration & validation or calibration & test or validation & test:
                raise ValueError(f"{trace_path}: overlapping split seeds for {case}, seed {seed_group}")
    for row in rows:
        case = row["case"]
        seed = int(row["seed"])
        threshold = float(
            source_config["poisson_oblivious_mean_threshold"]
            if case == "2d_poisson" else source_config["oblivious_mean_threshold"]
        )
        candidates = []
        for (candidate_case, candidate_seed, name), errors in calibration_errors.items():
            if (candidate_case, candidate_seed) != (case, seed):
                continue
            mean_error = sum(errors) / len(errors)
            times = calibration_times[(candidate_case, candidate_seed, name)]
            mean_time = sum(times) / len(times)
            if mean_error <= threshold:
                candidates.append((mean_time, name))
        if not candidates:
            raise ValueError(f"{trace_path}: no feasible fair O* candidate for {case}/{seed}")
        expected_o = min(candidates)[1]
        if row.get("oblivious_chosen") != expected_o:
            raise ValueError(
                f"{csv_path}: O* mismatch for {case}/{seed}: "
                f"expected {expected_o}, found {row.get('oblivious_chosen')}"
            )
    cross_group_overlap = sum(1 for owners in seed_owners.values() if len(owners) > 1)
    cross_group_overlap_by_case_split = {
        f"{case}/{split}": sum(
            1 for (c, s, _trial_seed), owners in seed_owners.items()
            if c == case and s == split and len(owners) > 1
        )
        for case in EXPECTED_CASES
        for split in EXPECTED_SPLITS
    }
    unique_trial_seeds = {
        f"{case}/{split}": len({trial_seed for (c, s, trial_seed) in seed_owners if c == case and s == split})
        for case in EXPECTED_CASES
        for split in EXPECTED_SPLITS
    }
    selected_config_homogeneity = {}
    for case in EXPECTED_CASES:
        case_rows = [row for row in rows if row.get("case") == case]
        u_names = sorted({row.get("chosen", "") for row in case_rows})
        o_names = sorted({row.get("oblivious_chosen", "") for row in case_rows})
        selected_config_homogeneity[case] = {
            "u": len(u_names) == 1,
            "o": len(o_names) == 1,
            "u_configurations": u_names,
            "o_configurations": o_names,
        }
    pooled_cp_valid = (
        cross_group_overlap == 0
        and all(item["u"] and item["o"] for item in selected_config_homogeneity.values())
    )
    coverage_aggregation = (
        "per-seed-group bounds; pooled CP valid only for homogeneous selected configurations"
        if cross_group_overlap == 0
        else "per-seed-group bounds; pooled CP is invalid because trial seeds overlap"
    )
    return {
        "csv": str(csv_path),
        "traces": str(trace_path),
        "distribution": next(iter(distributions)),
        "csv_rows": len(rows),
        "seeds": sorted(seeds),
        "trace_records": sum(trace_counts.values()),
        "config_sha256": digest,
        "selection_cost_basis": "validation_timing",
        "policy_spec_file": policy_path.name,
        "policy_spec_sha256": policy_digest,
        "fair_oblivious_replay_verified": True,
        "metadata_fields": sorted(metadata),
        "cross_group_trial_seed_overlap": cross_group_overlap,
        "cross_group_trial_seed_overlap_by_case_split": cross_group_overlap_by_case_split,
        "unique_trial_seeds_by_case_split": unique_trial_seeds,
        "coverage_aggregation": coverage_aggregation,
        "selected_config_homogeneity": selected_config_homogeneity,
        "pooled_cp_valid": pooled_cp_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="publication-config.json")
    parser.add_argument("--policy-spec", default="publication-policy-replay.json")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("CSV", "JSONL"), required=True)
    args = parser.parse_args()
    config = Path(args.config)
    policy = Path(args.policy_spec)
    reports = [validate(Path(csv_path), Path(trace_path), config, policy) for csv_path, trace_path in args.pair]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
