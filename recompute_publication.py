"""Replay publication policy selection from paired raw traces.

The original sweep measured all split timings and errors correctly, but an
earlier selector used held-out test timing. This utility preserves the raw
observations and regenerates the publication CSV using validation timing only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

from benchmark import add_cost_accounting, select_oblivious, selection_key


EXACT = {
    "fft": "fft-64-1.0",
    "spmv": "spmv-64-0.0",
    "stencil": "stencil-64-0.0",
    "monte_carlo": "mc-64-1000000",
    "2d_poisson": "poisson-exact",
}


def lower_bound(successes: int, total: int, alpha: float) -> float:
    if total == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def load_traces(path: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    metadata = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("record_type") == "metadata":
                metadata = item.get("metadata")
                continue
            config = item.get("config", {})
            key = (str(item["case"]), str(item["distribution"]), int(item["seed_group"]), str(config["name"]))
            split = str(item["split"])
            grouped[key][f"{split}_error"].append(float(item["error"]))
            grouped[key][f"{split}_time"].append(float(item["wall_time_s"]))
            for name in ("center_error", "residual_rel", "finite"):
                if name in item:
                    grouped[key][f"{split}_{name}"].append(item[name])
    if metadata is None:
        raise ValueError(f"missing metadata in {path}")
    return grouped, metadata


def mean(values):
    # Match the sweep runner's NumPy reduction so deterministic replay is
    # byte-identical, including floating-point QoI diagnostics.
    return float(np.mean(values))


def load_policy_spec(path: Path, config_path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    spec = json.loads(raw)
    source_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if spec.get("source_config") != config_path.name:
        raise ValueError("policy sidecar source configuration filename mismatch")
    if spec.get("source_config_sha256") != source_digest:
        raise ValueError("policy sidecar source configuration digest mismatch")
    if "including the declared exact/reference candidate" not in str(spec.get("policy", "")):
        raise ValueError("policy sidecar does not declare reference-inclusive O* selection")
    return spec, hashlib.sha256(raw).hexdigest()


def replay(
    csv_path: Path,
    trace_path: Path,
    output_path: Path,
    config_path: Path,
    policy_path: Path = Path("publication-policy-replay.json"),
):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, policy_digest = load_policy_spec(policy_path, config_path)
    grouped, metadata = load_traces(trace_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    if not old_rows:
        raise ValueError(f"empty CSV: {csv_path}")
    new_rows = []
    for old in old_rows:
        case, distribution, seed = old["case"], old["distribution"], int(old["seed"])
        keys = [key for key in grouped if key[:3] == (case, distribution, seed)]
        if not keys:
            raise ValueError(f"no traces for {case}/{distribution}/{seed}")
        names = sorted({key[3] for key in keys})
        epsilon = float(config["poisson_epsilon"] if case == "2d_poisson" else config["kernel_epsilon"])
        threshold = float(config["poisson_oblivious_mean_threshold"] if case == "2d_poisson" else config["oblivious_mean_threshold"])
        alpha = float(config["alpha"])
        selection_alpha = float(old["selection_alpha"])
        delta = float(config["delta"])
        records = {}
        for name in names:
            values = grouped[(case, distribution, seed, name)]
            cal_err = values["calibration_error"]
            val_err = values["validation_error"]
            test_err = values["test_error"]
            records[name] = {
                # Preserve the same deterministic name tie-break used by the
                # live selector when replaying validation records.
                "config": {"name": name},
                "calibration_cost": mean(values["calibration_time"]),
                "validation_cost": mean(values["validation_time"]),
                "test_cost": mean(values["test_time"]),
                "calibration_mean_error": mean(cal_err),
                "validation_lower_bound": lower_bound(sum(e <= epsilon for e in val_err), len(val_err), selection_alpha),
                "test_successes": sum(e <= epsilon for e in test_err),
                "test_errors": test_err,
                "test_lower_bound": lower_bound(sum(e <= epsilon for e in test_err), len(test_err), alpha),
                "trace_values": values,
            }
        n_calibration = len(grouped[(case, distribution, seed, names[0])]["calibration_error"])
        n_validation = len(grouped[(case, distribution, seed, names[0])]["validation_error"])
        n_calls = int(old["n_calls"])
        costs = add_cost_accounting(records.values(), n_calibration, n_validation, n_calls)
        feasible = [name for name, record in records.items() if record["validation_lower_bound"] >= 1.0 - delta]
        exact = EXACT[case]
        chosen = min(feasible, key=lambda name: selection_key(records[name])) if feasible else exact
        non_exact = [name for name in names if name != exact]
        oblivious_record = select_oblivious(records.values(), threshold, records[exact])
        oblivious = str(oblivious_record["config"]["name"])
        selected, selected_o = records[chosen], records[oblivious]
        row = dict(old)
        row.update({
            "chosen": chosen,
            "selection_lower_bound": selected["validation_lower_bound"],
            "selection_cost_basis": "validation_timing",
            "test_coverage": selected["test_successes"] / len(selected["test_errors"]),
            "test_successes": selected["test_successes"],
            "test_lower_bound": selected["test_lower_bound"],
            "test_p95_error": float(np.quantile(selected["test_errors"], 0.95)),
            "test_p99_error": float(np.quantile(selected["test_errors"], 0.99)),
            "test_cost": selected["test_cost"],
            **costs,
            "post_qualification_cost": selected["post_qualification_cost"],
            "matched_total_cost": selected["matched_total_cost"],
            "policy_specific_u_total_cost": selected["policy_specific_u_total_cost"],
            "total_cost": selected["matched_total_cost"],
            "oblivious_post_qualification_cost": selected_o["post_qualification_cost"],
            "oblivious_matched_total_cost": selected_o["matched_total_cost"],
            "oblivious_policy_specific_total_cost": selected_o["policy_specific_o_total_cost"],
            "oblivious_total_cost": selected_o["matched_total_cost"],
            "fallback_to_exact": not bool([name for name in non_exact if records[name]["validation_lower_bound"] >= 1.0 - delta]),
            "oblivious_chosen": oblivious,
            "oblivious_test_coverage": selected_o["test_successes"] / len(selected_o["test_errors"]),
            "oblivious_test_successes": selected_o["test_successes"],
            "oblivious_test_lower_bound": selected_o["test_lower_bound"],
            "oblivious_test_cost": selected_o["test_cost"],
            "policy_spec_file": policy_path.name,
            "policy_spec_sha256": policy_digest,
        })
        for metric, field in (("center_error", "qoi_center_mean"), ("residual_rel", "residual_mean")):
            values = grouped[(case, distribution, seed, chosen)].get(f"test_{metric}")
            if values:
                row[field] = mean([float(v) for v in values])
        for metric, field in (("center_error", "qoi_center_p95"), ("residual_rel", "residual_p95")):
            values = grouped[(case, distribution, seed, chosen)].get(f"test_{metric}")
            if values:
                row[field] = float(np.quantile([float(v) for v in values], 0.95))
        finite = grouped[(case, distribution, seed, chosen)].get("test_finite")
        if finite:
            row["finite_rate"] = mean([float(v) for v in finite])
        new_rows.append(row)
    # Keep the replayed CSV schema identical to the live sweep runner so that
    # downstream tools cannot silently depend on historical column ordering.
    fieldnames = [
        "case", "distribution", "seed", "chosen", "selection_lower_bound",
        "n_test", "n_calls", "selection_cost_basis", "test_coverage",
        "test_successes", "test_lower_bound", "test_p95_error",
        "test_p99_error", "test_cost", "post_qualification_cost", "matched_total_cost",
        "policy_specific_u_total_cost", "total_cost", "oblivious_post_qualification_cost",
        "oblivious_matched_total_cost", "oblivious_policy_specific_total_cost", "oblivious_total_cost",
        "oblivious_mean_threshold", "selection_alpha", "n_configurations",
        "energy_measurement", "fallback_to_exact", "oblivious_chosen",
        "oblivious_test_coverage", "oblivious_test_successes",
        "oblivious_test_lower_bound", "oblivious_test_cost", "qoi_center_mean",
        "calibration_overhead", "validation_overhead", "qualification_overhead", "qoi_center_p95", "residual_mean",
        "residual_p95", "finite_rate",
        "python_version", "numpy_version", "scipy_version", "platform",
        "processor", "cpu_count", "thread_env", "byteorder", "config_file",
        "config_sha256", "policy_spec_file", "policy_spec_sha256",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    return metadata, new_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="publication-config.json")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--policy-spec", default="publication-policy-replay.json")
    args = parser.parse_args()
    _, rows = replay(
        Path(args.input_csv), Path(args.traces), Path(args.output_csv),
        Path(args.config), Path(args.policy_spec),
    )
    print(json.dumps({"rows": len(rows), "output": args.output_csv}, indent=2))


if __name__ == "__main__":
    main()
