"""Recompute numerical claim checks from the recorded experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from benchmark import lower_bound
from confirmatory_mc import summarize


ROOT = Path(__file__).resolve().parent


def audit_heat(config_name: str, trace_name: str) -> dict:
    config_path = ROOT / config_name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    grouped = defaultdict(list)
    identities = defaultdict(lambda: defaultdict(set))
    with (ROOT / trace_name).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                assert item["metadata"]["config_sha256"] == digest
                continue
            name = item["config"]["name"]
            key = (item["distribution"], name, item["split"])
            grouped[key].append(item)
            identities[(item["distribution"], name)][item["split"]].add(item["seed"])
    names = sorted({key[1] for key in grouped})
    alpha = config["alpha"] / len(names)
    ceiling = lower_bound(config["validation"], config["validation"], alpha)
    cells = []
    for regime in config["distributions"]:
        records = []
        for name in names:
            validation = grouped[(regime, name, "validation")]
            test = grouped[(regime, name, "test")]
            expected_val = len(config["seeds"]) * config["validation"]
            expected_test = len(config["seeds"]) * config["test"]
            assert len(validation) == expected_val and len(test) == expected_test
            split_ids = identities[(regime, name)]
            overlap = len(split_ids["validation"] & split_ids["test"])
            assert overlap == 0, "validation/test overlap invalidates the diagnostic"
            successes = sum(math.isfinite(r["error"]) and r["error"] <= config["epsilon"] for r in validation)
            unique_val = len(split_ids["validation"])
            assert unique_val == len(validation), "pooled validation requires distinct trials"
            records.append({
                "candidate": name,
                "validation_successes": successes,
                "validation_trials": len(validation),
                "pooled_validation_lower": lower_bound(successes, len(validation), alpha),
                "validation_mean_cost_s": sum(r["wall_time_s"] for r in validation) / len(validation),
                "test_trials": len(test),
                "unique_test_seeds": len(split_ids["test"]),
                "test_duplicate_seed_records": len(test) - len(split_ids["test"]),
                "validation_test_overlap": overlap,
            })
        feasible = [r for r in records if r["pooled_validation_lower"] >= 1 - config["delta"]]
        selected = min(feasible, key=lambda r: (r["validation_mean_cost_s"], -r["pooled_validation_lower"], r["candidate"])) if feasible else None
        cells.append({
            "regime": regime,
            "pooled_validation_feasible": [r["candidate"] for r in feasible],
            "pooled_validation_selected": selected["candidate"] if selected else None,
            "candidates": records,
        })
    return {
        "config": config_name,
        "trace": trace_name,
        "config_sha256": digest,
        "grid": config["grid"],
        "candidate_count": len(names),
        "original_validation_n": config["validation"],
        "all_success_lower_ceiling": ceiling,
        "minimum_all_success_n": math.ceil(math.log(alpha) / math.log(1 - config["delta"])),
        "original_gate_can_qualify": ceiling >= 1 - config["delta"],
        "interpretation": "Post hoc pooling of existing validation groups diagnoses sample-budget effects; it is not a new independent experiment. No pooled test confidence is asserted.",
        "cells": cells,
    }


def main() -> None:
    with (ROOT / "confirmatory-mc.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = json.loads((ROOT / "confirmatory-mc-summary.json").read_text(encoding="utf-8"))
    confirmation = summarize(rows, expected["config_sha256"])
    exact_match = confirmation == expected
    for key, value in expected.items():
        if key.endswith("_ci95"):
            assert confirmation[key].keys() == value.keys()
            assert confirmation[key]["mean"] == value["mean"]
            for bound in ("ci95_low", "ci95_high"):
                assert math.isclose(confirmation[key][bound], value[bound], rel_tol=1e-10, abs_tol=1e-12), (key, bound)
        else:
            assert confirmation[key] == value, key
    assert sum(int(row["test_successes"]) for row in rows) == 2560
    heat = [audit_heat("heat-experiment-config-full.json", "heat-audit-full.jsonl"),
            audit_heat("heat-production-config.json", "heat-production-audit.jsonl")]
    report = {"confirmation_summary_exact_match": exact_match, "confirmation_summary_within_tolerance": True, "confirmation": confirmation, "heat": heat}
    destination = ROOT / "evidence" / "numerical-audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"confirmation_summary_exact_match": exact_match, "confirmation_summary_within_tolerance": True, "heat": [
        {key: value for key, value in item.items() if key != "cells"} | {"regime_results": [
            {"regime": cell["regime"], "feasible": cell["pooled_validation_feasible"],
             "selected": cell["pooled_validation_selected"],
             "test_duplicate_seed_records": cell["candidates"][0]["test_duplicate_seed_records"]}
            for cell in item["cells"]]} for item in heat], "scope": "Numerical evidence only"}, indent=2))


if __name__ == "__main__":
    main()
