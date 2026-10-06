"""Validate the larger-grid and stability supplemental audit artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_production(config: Path, csv_path: Path, trace_path: Path) -> dict:
    digest = _digest(config)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if len(rows) != 30:
        raise ValueError(f"{csv_path}: expected 30 rows, found {len(rows)}")
    expected = {(d, int(seed)) for d in ("light", "heavy", "correlated") for seed in (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)}
    observed = {(row["distribution"], int(row["seed"])) for row in rows}
    if observed != expected:
        raise ValueError("production CSV has incorrect seed/distribution combinations")
    if {row["config_sha256"] for row in rows} != {digest}:
        raise ValueError("production CSV digest mismatch")
    trace_count = 0
    metadata = []
    split_counts: Counter[tuple[str, str, int, str]] = Counter()
    nonfinite = 0
    selected_nonfinite = 0
    selected = {(row["distribution"], int(row["seed"])): {row["chosen"], row["oblivious_chosen"]} for row in rows}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata.append(item.get("metadata", {}))
                continue
            trace_count += 1
            key = (str(item["distribution"]), int(item["seed_group"]), str(item["config"]["name"]), str(item["split"]))
            split_counts[key] += 1
            if not bool(item.get("finite", False)):
                nonfinite += 1
                if item.get("split") == "test" and str(item["config"]["name"]) in selected.get((str(item["distribution"]), int(item["seed_group"])), set()):
                    selected_nonfinite += 1
    if len(metadata) != 1 or metadata[0].get("config_sha256") != digest:
        raise ValueError("production trace metadata digest mismatch")
    if trace_count != 37_440:
        raise ValueError(f"{trace_path}: expected 37440 records, found {trace_count}")
    if len(split_counts) != 30 * 13 * 3:
        raise ValueError("production trace has missing cell/config/split groups")
    if set(split_counts.values()) != {16, 64}:
        raise ValueError("production trace has unexpected split counts")
    return {"csv_rows": len(rows), "trace_records": trace_count, "nonfinite_records": nonfinite, "selected_test_nonfinite": selected_nonfinite, "config_sha256": digest}


def validate_stability(config: Path, csv_path: Path, trace_path: Path) -> dict:
    digest = _digest(config)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if len(rows) != 81:
        raise ValueError(f"{csv_path}: expected 81 rows, found {len(rows)}")
    if {row["config_sha256"] for row in rows} != {digest}:
        raise ValueError("stability CSV digest mismatch")
    trace_count = 0
    metadata = []
    finite = True
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata.append(item.get("metadata", {}))
                continue
            trace_count += 1
            finite = finite and bool(item.get("finite", False))
    if len(metadata) != 1 or metadata[0].get("config_sha256") != digest:
        raise ValueError("stability trace metadata digest mismatch")
    if trace_count != 810:
        raise ValueError(f"{trace_path}: expected 810 records, found {trace_count}")
    return {"csv_rows": len(rows), "trace_records": trace_count, "finite": finite, "config_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-config", type=Path, default=Path("heat-production-config.json"))
    parser.add_argument("--production-csv", type=Path, default=Path("heat-production-audit.csv"))
    parser.add_argument("--production-traces", type=Path, default=Path("heat-production-audit.jsonl"))
    parser.add_argument("--stability-config", type=Path, default=Path("stability-audit-config.json"))
    parser.add_argument("--stability-csv", type=Path, default=Path("stability-audit.csv"))
    parser.add_argument("--stability-traces", type=Path, default=Path("stability-audit.jsonl"))
    args = parser.parse_args()
    result = {"production": validate_production(args.production_config, args.production_csv, args.production_traces), "stability": validate_stability(args.stability_config, args.stability_csv, args.stability_traces)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
