"""CPU-only 2-D Poisson case for the uncertainty-aware controller.

The source amplitude is uncertain and the QoI is the relative L2 error of the
computed field against the analytic solution. This is a small physics case
for protocol validation, not a publication-grade solver benchmark. The direct
path is exact for the assembled finite-difference system; its field error
against the analytic solution therefore includes grid discretization error.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from benchmark import add_cost_accounting, InputRNG, lower_bound, select_oblivious, selection_key, trial_seed_for


_STATIC_FIELDS: Dict[int, Tuple[np.ndarray, float]] = {}


def _static_field(n: int) -> Tuple[np.ndarray, float]:
    """Cache geometry that is independent of the uncertain amplitude."""
    if n not in _STATIC_FIELDS:
        grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        _STATIC_FIELDS[n] = (np.sin(np.pi * xx) * np.sin(np.pi * yy), (1.0 / (n - 1)) ** 2)
    return _STATIC_FIELDS[n]


@dataclass(frozen=True)
class Config:
    name: str
    precision: int
    iterations: int
    solver: str = "jacobi"


def solve_case_details(rng: InputRNG, cfg: Config) -> Dict[str, float | bool]:
    """Solve one paired trial and return QoI plus numerical-health diagnostics."""
    n = 48
    dtype = np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64
    mode, h2 = _static_field(n)
    amplitude = 1.0 + 0.15 * rng.normal()
    # The analytic manufactured solution is the QoI reference. The direct path
    # below solves the discrete system, so its residual can be near machine
    # precision while its field error retains finite-grid discretization error.
    exact = amplitude * mode / (2.0 * np.pi**2)
    # For u = amplitude * sin(pi x) sin(pi y) / (2 pi^2),
    # -Delta u = amplitude * sin(pi x) sin(pi y).
    rhs = amplitude * mode
    t0 = time.perf_counter()
    if cfg.solver == "direct":
        m = n - 2
        diagonal = np.full(m * m, 4.0 / h2)
        off_x = np.full(m * m - 1, -1.0 / h2)
        off_y = np.full(m * m - m, -1.0 / h2)
        for row in range(1, m):
            off_x[row * m - 1] = 0.0
        matrix = diags((off_y, off_x, diagonal, off_x, off_y), (-m, -1, 0, 1, m), shape=(m * m, m * m), format="csc")
        interior_solution = spsolve(matrix, rhs[1:-1, 1:-1].reshape(-1)).reshape((m, m))
        u = np.zeros_like(rhs, dtype=np.float64)
        u[1:-1, 1:-1] = interior_solution
    elif cfg.solver == "jacobi":
        u = np.zeros_like(rhs, dtype=dtype)
        for _ in range(cfg.iterations):
            nxt = u.copy()
            nxt[1:-1, 1:-1] = 0.25 * (
                u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] + h2 * rhs[1:-1, 1:-1]
            )
            u = nxt
    else:
        raise ValueError(f"unknown Poisson solver: {cfg.solver}")
    elapsed = time.perf_counter() - t0
    rel_l2 = float(np.linalg.norm(u.astype(np.float64) - exact) / (np.linalg.norm(exact) + 1e-12))
    field = u.astype(np.float64)
    center = n // 2
    center_error = float(abs(field[center, center] - exact[center, center]) / (abs(exact[center, center]) + 1e-12))
    interior = field[1:-1, 1:-1]
    discrete_laplacian = (4.0 * interior - field[:-2, 1:-1] - field[2:, 1:-1] - field[1:-1, :-2] - field[1:-1, 2:]) / h2
    residual_rel = float(np.linalg.norm(discrete_laplacian - rhs[1:-1, 1:-1]) / (np.linalg.norm(rhs[1:-1, 1:-1]) + 1e-12))
    finite = bool(np.isfinite(field).all() and np.isfinite(rel_l2) and np.isfinite(center_error) and np.isfinite(residual_rel))
    return {
        "field_error": rel_l2,
        "center_error": center_error,
        "residual_rel": residual_rel,
        "max_abs": float(np.max(np.abs(field))),
        "finite": finite,
        "wall_time_s": elapsed,
    }


def solve_case(rng: InputRNG, cfg: Config) -> Tuple[float, float]:
    """Compatibility wrapper returning the historical field-error/time pair."""
    details = solve_case_details(rng, cfg)
    return float(details["field_error"]), float(details["wall_time_s"])


def configs() -> Iterable[Config]:
    jacobi = [Config(f"poisson-{p}-{it}", p, it) for p in (16, 32, 64) for it in (25, 50, 100, 400)]
    return [*jacobi, Config("poisson-exact", 64, 0, "direct")]


def run(seed: int, n_cal: int, n_val: int, n_test: int, epsilon: float, delta: float, alpha: float, n_calls: int = 100, distribution: str = "light", include_traces: bool = False, oblivious_mean_threshold: Optional[float] = None) -> Dict:
    if min(n_cal, n_val, n_test) <= 0:
        raise ValueError("calibration, validation, and test sizes must be positive")
    if n_calls < 0:
        raise ValueError("n_calls must be non-negative")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    if oblivious_mean_threshold is None:
        oblivious_mean_threshold = epsilon
    if oblivious_mean_threshold < 0.0:
        raise ValueError("oblivious_mean_threshold must be non-negative")
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must lie in [0, 1)")
    cfgs = list(configs())
    selection_alpha = alpha / len(cfgs)
    buffers = {
        cfg.name: {
            "cal_err": [], "cal_time": [], "val_err": [], "val_time": [],
            "test_err": [], "test_time": [], "cal_center": [],
            "val_center": [], "test_center": [], "cal_residual": [],
            "val_residual": [], "test_residual": [], "cal_finite": [],
            "val_finite": [], "test_finite": [],
        }
        for cfg in cfgs
    }
    traces = []
    # Process one trial across all configurations so the uncertain amplitude is
    # generated once and reused for an exactly paired comparison.
    for split, count in (("calibration", n_cal), ("validation", n_val), ("test", n_test)):
        for i in range(count):
            trial_seed = trial_seed_for(seed, split, i)
            shared_cache = {}
            for cfg in cfgs:
                details = solve_case_details(InputRNG(trial_seed, distribution, shared_cache), cfg)
                e, t = float(details["field_error"]), float(details["wall_time_s"])
                buffer = buffers[cfg.name]
                prefix = {"calibration": "cal", "validation": "val", "test": "test"}[split]
                buffer[f"{prefix}_err"].append(e)
                buffer[f"{prefix}_time"].append(t)
                buffer[f"{prefix}_center"].append(float(details["center_error"]))
                buffer[f"{prefix}_residual"].append(float(details["residual_rel"]))
                buffer[f"{prefix}_finite"].append(bool(details["finite"]))
                if include_traces:
                    traces.append({"config": asdict(cfg), "split": split, "index": i, "seed": trial_seed, "error": e, "center_error": details["center_error"], "residual_rel": details["residual_rel"], "max_abs": details["max_abs"], "finite": details["finite"], "wall_time_s": t})
    records: List[Dict] = []
    for cfg in cfgs:
        buffer = buffers[cfg.name]
        cal_err, cal_time = buffer["cal_err"], buffer["cal_time"]
        val_err, val_time = buffer["val_err"], buffer["val_time"]
        test_err, test_time = buffer["test_err"], buffer["test_time"]
        cal_center, val_center, test_center = buffer["cal_center"], buffer["val_center"], buffer["test_center"]
        cal_residual, val_residual, test_residual = buffer["cal_residual"], buffer["val_residual"], buffer["test_residual"]
        cal_finite, val_finite, test_finite = buffer["cal_finite"], buffer["val_finite"], buffer["test_finite"]
        records.append({
            "config": asdict(cfg),
            "calibration_cost": float(np.mean(cal_time)),
            "test_cost": float(np.mean(test_time)),
            "energy_j": None,
            "calibration_mean_error": float(np.mean(cal_err)),
            "validation_cost": float(np.mean(val_time)),
            "validation_mean_error": float(np.mean(val_err)),
            "test_mean_error": float(np.mean(test_err)),
            "test_p95_error": float(np.quantile(test_err, 0.95)),
            "test_p99_error": float(np.quantile(test_err, 0.99)),
            "calibration_mean_center_error": float(np.mean(cal_center)),
            "validation_mean_center_error": float(np.mean(val_center)),
            "test_mean_center_error": float(np.mean(test_center)),
            "test_p95_center_error": float(np.quantile(test_center, 0.95)),
            "calibration_mean_residual_rel": float(np.mean(cal_residual)),
            "validation_mean_residual_rel": float(np.mean(val_residual)),
            "test_mean_residual_rel": float(np.mean(test_residual)),
            "test_p95_residual_rel": float(np.quantile(test_residual, 0.95)),
            "calibration_finite_rate": float(np.mean(cal_finite)),
            "validation_finite_rate": float(np.mean(val_finite)),
            "test_finite_rate": float(np.mean(test_finite)),
            "calibration_errors": cal_err,
            "validation_errors": val_err,
            "test_errors": test_err,
        })
    exact = next(r for r in records if r["config"]["name"] == "poisson-exact")
    for r in records:
        r["calibration_successes"] = sum(e <= epsilon for e in r["calibration_errors"])
        r["test_successes"] = sum(e <= epsilon for e in r["test_errors"])
        r["calibration_coverage"] = r["calibration_successes"] / n_cal
        r["calibration_lower_bound"] = lower_bound(r["calibration_successes"], n_cal, selection_alpha)
        r["validation_successes"] = sum(e <= epsilon for e in r["validation_errors"])
        r["validation_coverage"] = r["validation_successes"] / n_val
        r["validation_lower_bound"] = lower_bound(r["validation_successes"], n_val, selection_alpha)
        r["test_coverage"] = r["test_successes"] / n_test
        r["test_lower_bound"] = lower_bound(r["test_successes"], n_test, alpha)
    costs = add_cost_accounting(records, n_cal, n_val, n_calls)
    feasible = [r for r in records if r["validation_lower_bound"] >= 1.0 - delta]
    chosen = min(feasible, key=selection_key) if feasible else exact
    non_exact = [r for r in records if r is not exact]
    approximate_feasible = [r for r in non_exact if r["validation_lower_bound"] >= 1.0 - delta]
    oblivious_threshold = float(oblivious_mean_threshold)
    oblivious_chosen = select_oblivious(records, oblivious_threshold, exact)
    for r in records:
        r.pop("calibration_errors")
        r.pop("validation_errors")
        r.pop("test_errors")
    output = {
        "case": "2d_poisson",
        "seed": seed,
        "n_calibration": n_cal,
        "n_validation": n_val,
        "n_test": n_test,
        "n_calls": n_calls,
        **costs,
        "selection_cost_basis": "validation_timing",
        "epsilon": epsilon,
        "delta": delta,
        "alpha": alpha,
        "selection_alpha": selection_alpha,
        "n_configurations": len(cfgs),
        "distribution": distribution,
        "chosen": chosen["config"],
        "oblivious_threshold": oblivious_threshold,
        "oblivious_mean_threshold": oblivious_threshold,
        "oblivious_chosen": oblivious_chosen["config"],
        "post_qualification_cost": chosen["post_qualification_cost"],
        "matched_total_cost": chosen["matched_total_cost"],
        "policy_specific_u_total_cost": chosen["policy_specific_u_total_cost"],
        "oblivious_post_qualification_cost": oblivious_chosen["post_qualification_cost"],
        "oblivious_matched_total_cost": oblivious_chosen["matched_total_cost"],
        "oblivious_policy_specific_total_cost": oblivious_chosen["policy_specific_o_total_cost"],
        "oblivious_total_cost": oblivious_chosen["matched_total_cost"],
        # Exact is always eligible for U when its validation gate passes. This
        # flag records whether no non-exact configuration passed that gate.
        "fallback_to_exact": not bool(approximate_feasible),
        "energy_measurement": "unavailable",
        "records": records,
    }
    if include_traces:
        output["traces"] = traces
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-cal", type=int, default=32)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-calls", type=int, default=100)
    parser.add_argument("--distribution", choices=["light", "heavy", "correlated"], default="light")
    parser.add_argument("--oblivious-mean-threshold", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.n_cal, args.n_val, args.n_test, args.epsilon, args.delta, args.alpha, args.n_calls, args.distribution, oblivious_mean_threshold=args.oblivious_mean_threshold), indent=2))


if __name__ == "__main__":
    main()
