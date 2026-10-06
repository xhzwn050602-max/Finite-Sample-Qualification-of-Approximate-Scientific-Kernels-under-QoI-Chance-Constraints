"""Numerically audit the finite-sample CP/Bonferroni formulas used in the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import beta

from benchmark import lower_bound


def audit(alpha: float = 0.05, delta: float = 0.10, candidate_counts=(9, 13)) -> dict:
    rows = []
    for candidates in candidate_counts:
        per_candidate_alpha = alpha / candidates
        n = 1
        while per_candidate_alpha ** (1.0 / n) < 1.0 - delta:
            n += 1
        closed_form_n = math.ceil(math.log(per_candidate_alpha) / math.log(1.0 - delta))
        cp = float(beta.ppf(per_candidate_alpha, n, 1))
        implementation = lower_bound(n, n, per_candidate_alpha)
        rows.append({
            "candidates": candidates,
            "alpha_per_candidate": per_candidate_alpha,
            "minimal_all_success_n": n,
            "closed_form_n": closed_form_n,
            "cp_beta_ppf": cp,
            "implementation_lower_bound": implementation,
            "meets_target": cp >= 1.0 - delta,
            "previous_n_fails": (n == 1 or per_candidate_alpha ** (1.0 / (n - 1)) < 1.0 - delta),
        })
    return {
        "alpha": alpha,
        "delta": delta,
        "rows": rows,
        "all_checks_pass": all(
            row["minimal_all_success_n"] == row["closed_form_n"]
            and abs(row["cp_beta_ppf"] - row["implementation_lower_bound"]) < 1e-15
            and row["meets_target"]
            and row["previous_n_fails"]
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="theory-audit.json")
    args = parser.parse_args()
    result = audit()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit("theory audit failed")


if __name__ == "__main__":
    main()
