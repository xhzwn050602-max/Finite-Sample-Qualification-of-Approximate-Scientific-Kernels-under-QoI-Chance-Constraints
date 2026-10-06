"""Audit the configured trial-seed schedule without running kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import trial_seed_for


def audit(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    split_counts = {
        "calibration": int(config.get("kernel_calibration", config.get("calibration", 1))),
        "validation": int(config.get("kernel_validation", config.get("validation", 1))),
        "test": int(config.get("kernel_test", config.get("test", 1))),
    }
    streams = {
        (seed, split): {trial_seed_for(seed, split, i) for i in range(count)}
        for seed in seeds for split, count in split_counts.items()
    }
    owners: dict[int, list[tuple[int, str]]] = {}
    for key, values in streams.items():
        for value in values:
            owners.setdefault(value, []).append(key)
    overlap = {str(seed): owners_list for seed, owners_list in owners.items() if len(owners_list) > 1}
    return {
        "config_file": Path(config_path).name,
        "seeds": seeds,
        "split_counts": split_counts,
        "unique_trial_seeds": len(owners),
        "overlap_count": len(overlap),
        "collision_free": not overlap,
        "schedule": "seed_group * 1_000_000 + {calibration:0, validation:100_000, test:200_000} + index",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="publication-config.json")
    parser.add_argument("--output", default="seed-schedule-audit.json")
    args = parser.parse_args()
    result = audit(args.config)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["collision_free"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
