"""Non-manufactured 2-D steady heat-conduction case.

The conductivity is an exponentiated clipped random field sampled from the same
declared input regimes as the kernel benchmark.  Fixed-temperature boundaries
define a steady diffusion problem with a uniform volumetric source.  The
64-bit sparse direct solution is the discrete reference; Jacobi iterations are
the approximate configurations with diagonal/right-hand-side quantization and
float64 iteration arithmetic.  The primary QoI is the mean interior
temperature, while field error and the discrete residual are diagnostics.

This module is an independent physics application and does not replace the
manufactured Poisson protocol already used for the publication artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from benchmark import add_cost_accounting, InputRNG, lower_bound, select_oblivious, selection_key, trial_seed_for


@dataclass(frozen=True)
class Config:
    name: str
    precision: int
    iterations: int
    solver: str = "jacobi"


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b + 1.0e-15)


def _conductivity(rng: InputRNG, n: int, cache: Dict) -> np.ndarray:
    key = ("heat_conductivity", n)
    if key in cache:
        return cache[key].copy()
    z = np.asarray(rng.normal(size=(n, n)), dtype=np.float64)
    # The bounded positive field keeps the matrix uniformly elliptic while
    # retaining heterogeneous, uncertain material properties.
    conductivity = np.exp(np.clip(0.35 * z, -2.0, 2.0))
    cache[key] = conductivity.copy()
    return conductivity


def _assemble(k: np.ndarray) -> Tuple[csc_matrix, np.ndarray, np.ndarray]:
    n = k.shape[0]
    m = n - 2
    h2 = (1.0 / (n - 1)) ** 2
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs = np.full(m * m, 0.05, dtype=np.float64)

    def index(i: int, j: int) -> int:
        return (i - 1) * m + (j - 1)

    # Dirichlet temperature: left=1, right=0, bottom=top=0.
    boundaries = {(0, 1.0), (1, 0.0)}
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = index(i, j)
            coeffs = (
                (i - 1, j, _harmonic(k[i, j], k[i - 1, j]), "left"),
                (i + 1, j, _harmonic(k[i, j], k[i + 1, j]), "right"),
                (i, j - 1, _harmonic(k[i, j], k[i, j - 1]), "bottom"),
                (i, j + 1, _harmonic(k[i, j], k[i, j + 1]), "top"),
            )
            diagonal = 0.0
            for ni, nj, coeff, side in coeffs:
                scaled = float(coeff / h2)
                diagonal += scaled
                if 0 < ni < n - 1 and 0 < nj < n - 1:
                    rows.append(row)
                    cols.append(index(ni, nj))
                    data.append(-scaled)
                else:
                    if side == "left":
                        rhs[row] += scaled
                    # Other boundary temperatures are zero.
            rows.append(row)
            cols.append(row)
            data.append(diagonal)
    matrix = csc_matrix((data, (rows, cols)), shape=(m * m, m * m))
    return matrix, rhs, np.asarray([h2, float(m)], dtype=np.float64)


def _unflatten(interior: np.ndarray, n: int) -> np.ndarray:
    m = n - 2
    field = np.zeros((n, n), dtype=np.float64)
    field[0, :] = 1.0
    field[1:-1, 1:-1] = interior.reshape(m, m)
    return field


def _jacobi(matrix: csc_matrix, rhs: np.ndarray, iterations: int, dtype: np.dtype) -> np.ndarray:
    diag = matrix.diagonal().astype(dtype)
    x = np.zeros(rhs.size, dtype=dtype)
    # Quantize the diagonal and RHS, retaining the float64 sparse operator.
    # off retains the diagonal quantization discrepancy; the resulting
    # Jacobi-like updates promote x to float64 on the first iteration.
    # dtype is therefore a data-quantization label, not iteration precision.
    off = matrix - csc_matrix((diag.astype(np.float64), (np.arange(rhs.size), np.arange(rhs.size))), shape=matrix.shape)
    rhs_dtype = rhs.astype(dtype)
    for _ in range(iterations):
        x = (rhs_dtype - off.dot(x)) / diag
    return x.astype(np.float64)


def solve_case_details(rng: InputRNG, cfg: Config, cache: Optional[Dict] = None) -> Dict[str, float | bool]:
    cache = cache if cache is not None else {}
    n = 24
    k = _conductivity(rng, n, cache)
    key = ("heat_reference", n)
    if key not in cache:
        matrix, rhs, meta = _assemble(k)
        t_ref = time.perf_counter()
        reference_interior = spsolve(matrix, rhs)
        reference_elapsed = time.perf_counter() - t_ref
        cache[key] = (matrix, rhs, meta, reference_interior, reference_elapsed)
    matrix, rhs, meta, reference_interior, reference_elapsed = cache[key]
    n_h = n - 2
    reference_field = _unflatten(reference_interior, n)
    reference_qoi = float(np.mean(reference_interior))
    if cfg.solver == "direct":
        t0 = time.perf_counter()
        interior = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
        elapsed = time.perf_counter() - t0
    elif cfg.solver == "jacobi":
        dtype = np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64
        t0 = time.perf_counter()
        interior = _jacobi(matrix, rhs, cfg.iterations, dtype)
        elapsed = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown heat solver: {cfg.solver}")
    field = _unflatten(interior, n)
    qoi = float(np.mean(interior))
    qoi_error = abs(qoi - reference_qoi) / (abs(reference_qoi) + 1.0e-12)
    field_error = float(np.linalg.norm(field - reference_field) / (np.linalg.norm(reference_field) + 1.0e-12))
    residual = np.asarray(matrix.dot(interior) - rhs, dtype=np.float64)
    residual_rel = float(np.linalg.norm(residual) / (np.linalg.norm(rhs) + 1.0e-12))
    finite = bool(np.isfinite(field).all() and np.isfinite(qoi_error) and np.isfinite(field_error) and np.isfinite(residual_rel))
    return {
        "qoi_error": float(qoi_error),
        "field_error": field_error,
        "residual_rel": residual_rel,
        "mean_temperature": qoi,
        "reference_mean_temperature": reference_qoi,
        "finite": finite,
        "wall_time_s": float(elapsed),
        "reference_wall_time_s": float(reference_elapsed),
        "n_interior": float(n_h * n_h),
    }


def configs() -> Iterable[Config]:
    jacobi = [Config(f"heat-{p}-{it}", p, it) for p in (16, 32, 64) for it in (25, 50, 100, 400)]
    return [*jacobi, Config("heat-exact", 64, 0, "direct")]


def run(seed: int, n_cal: int, n_val: int, n_test: int, epsilon: float, delta: float, alpha: float, n_calls: int = 100, distribution: str = "light", include_traces: bool = False, oblivious_mean_threshold: Optional[float] = None) -> Dict:
    if min(n_cal, n_val, n_test) <= 0:
        raise ValueError("calibration, validation, and test sizes must be positive")
    if n_calls < 0 or epsilon < 0.0 or not 0.0 <= delta < 1.0:
        raise ValueError("invalid cost, tolerance, or delta")
    if oblivious_mean_threshold is None:
        oblivious_mean_threshold = epsilon
    if oblivious_mean_threshold < 0.0:
        raise ValueError("oblivious_mean_threshold must be non-negative")
    cfgs = list(configs())
    selection_alpha = alpha / len(cfgs)
    buffers = {c.name: {"cal_err": [], "val_err": [], "test_err": [], "cal_time": [], "val_time": [], "test_time": [], "cal_field": [], "val_field": [], "test_field": [], "cal_res": [], "val_res": [], "test_res": [], "cal_finite": [], "val_finite": [], "test_finite": []} for c in cfgs}
    traces: List[Dict] = []
    for split, count in (("calibration", n_cal), ("validation", n_val), ("test", n_test)):
        for i in range(count):
            trial_seed = trial_seed_for(seed, split, i)
            shared_cache: Dict = {}
            for cfg in cfgs:
                details = solve_case_details(InputRNG(trial_seed, distribution, shared_cache), cfg, shared_cache)
                prefix = {"calibration": "cal", "validation": "val", "test": "test"}[split]
                b = buffers[cfg.name]
                b[f"{prefix}_err"].append(float(details["qoi_error"]))
                b[f"{prefix}_time"].append(float(details["wall_time_s"]))
                b[f"{prefix}_field"].append(float(details["field_error"]))
                b[f"{prefix}_res"].append(float(details["residual_rel"]))
                b[f"{prefix}_finite"].append(bool(details["finite"]))
                if include_traces:
                    traces.append({"config": asdict(cfg), "split": split, "index": i, "seed": trial_seed, "error": details["qoi_error"], "field_error": details["field_error"], "residual_rel": details["residual_rel"], "finite": details["finite"], "mean_temperature": details["mean_temperature"], "wall_time_s": details["wall_time_s"]})
    records: List[Dict] = []
    for cfg in cfgs:
        b = buffers[cfg.name]
        records.append({
            "config": asdict(cfg),
            "calibration_cost": float(np.mean(b["cal_time"])),
            "validation_cost": float(np.mean(b["val_time"])),
            "test_cost": float(np.mean(b["test_time"])),
            "energy_j": None,
            "calibration_mean_error": float(np.mean(b["cal_err"])),
            "validation_mean_error": float(np.mean(b["val_err"])),
            "test_mean_error": float(np.mean(b["test_err"])),
            "test_p95_error": float(np.quantile(b["test_err"], 0.95)),
            "test_p99_error": float(np.quantile(b["test_err"], 0.99)),
            "test_p95_field_error": float(np.quantile(b["test_field"], 0.95)),
            "test_p95_residual_rel": float(np.quantile(b["test_res"], 0.95)),
            "calibration_finite_rate": float(np.mean(b["cal_finite"])),
            "validation_finite_rate": float(np.mean(b["val_finite"])),
            "test_finite_rate": float(np.mean(b["test_finite"])),
            "calibration_errors": b["cal_err"], "validation_errors": b["val_err"], "test_errors": b["test_err"],
        })
    exact = next(r for r in records if r["config"]["name"] == "heat-exact")
    for r in records:
        r["calibration_successes"] = sum(e <= epsilon for e in r["calibration_errors"])
        r["validation_successes"] = sum(e <= epsilon for e in r["validation_errors"])
        r["test_successes"] = sum(e <= epsilon for e in r["test_errors"])
        r["calibration_coverage"] = r["calibration_successes"] / n_cal
        r["validation_coverage"] = r["validation_successes"] / n_val
        r["test_coverage"] = r["test_successes"] / n_test
        r["calibration_lower_bound"] = lower_bound(r["calibration_successes"], n_cal, selection_alpha)
        r["validation_lower_bound"] = lower_bound(r["validation_successes"], n_val, selection_alpha)
        r["test_lower_bound"] = lower_bound(r["test_successes"], n_test, alpha)
    costs = add_cost_accounting(records, n_cal, n_val, n_calls)
    feasible = [r for r in records if r["validation_lower_bound"] >= 1.0 - delta]
    chosen = min(feasible, key=selection_key) if feasible else exact
    non_exact = [r for r in records if r is not exact]
    approximate_feasible = [r for r in non_exact if r["validation_lower_bound"] >= 1.0 - delta]
    oblivious_chosen = select_oblivious(records, float(oblivious_mean_threshold), exact)
    for r in records:
        r.pop("calibration_errors"); r.pop("validation_errors"); r.pop("test_errors")
    return {
        "case": "2d_heat_conduction", "seed": seed, "n_calibration": n_cal, "n_validation": n_val, "n_test": n_test, "n_calls": n_calls,
        **costs, "selection_cost_basis": "validation_timing", "epsilon": epsilon, "delta": delta, "alpha": alpha,
        "selection_alpha": selection_alpha, "n_configurations": len(cfgs), "distribution": distribution, "chosen": chosen["config"],
        "oblivious_threshold": float(oblivious_mean_threshold), "oblivious_mean_threshold": float(oblivious_mean_threshold),
        "oblivious_chosen": oblivious_chosen["config"],
        "post_qualification_cost": chosen["post_qualification_cost"], "matched_total_cost": chosen["matched_total_cost"],
        "policy_specific_u_total_cost": chosen["policy_specific_u_total_cost"],
        "oblivious_post_qualification_cost": oblivious_chosen["post_qualification_cost"],
        "oblivious_matched_total_cost": oblivious_chosen["matched_total_cost"],
        "oblivious_policy_specific_total_cost": oblivious_chosen["policy_specific_o_total_cost"],
        "oblivious_total_cost": oblivious_chosen["matched_total_cost"],
        "fallback_to_exact": not bool(approximate_feasible), "energy_measurement": "unavailable", "records": records,
        **({"traces": traces} if include_traces else {}),
    }


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
