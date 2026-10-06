"""Small reproducible prototype for the uncertainty-aware kernel controller.

This prototype is intentionally CPU-only and is not a substitute for the
planned multi-backend study. It produces paired calibration/validation/test traces and
selects the cheapest configuration whose one-sided coverage bound meets 1-delta.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Config:
    name: str
    precision: int
    knob: float


class InputRNG:
    """Deterministic input sampler for the declared distribution regimes."""

    def __init__(self, seed: int, distribution: str = "light", shared_cache: Optional[Dict] = None) -> None:
        if distribution not in {"light", "heavy", "correlated"}:
            raise ValueError(f"unknown input distribution: {distribution}")
        self._base = np.random.default_rng(seed)
        self.seed = seed
        self.distribution = distribution
        self._shared_cache = shared_cache


    @staticmethod
    def _shape_key(size):
        if isinstance(size, list):
            return tuple(size)
        return size

    def normal(self, size=None, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
        key = ("normal", self._shape_key(size), float(loc), float(scale))
        if self._shared_cache is not None and key in self._shared_cache:
            value = self._shared_cache[key]
            return value.copy() if isinstance(value, np.ndarray) else value
        if self.distribution == "heavy":
            # Student-t with three degrees of freedom, standardized to unit
            # scale for a comparable central spread.
            values = self._base.standard_t(3, size=size) / np.sqrt(3.0)
        else:
            values = self._base.normal(size=size)
            if self.distribution == "correlated" and np.size(values) > 1:
                flat = np.asarray(values, dtype=float).reshape(-1)
                rho = 0.8
                for i in range(1, flat.size):
                    flat[i] = rho * flat[i - 1] + np.sqrt(1.0 - rho * rho) * flat[i]
                flat /= np.std(flat) + 1e-12
                values = flat.reshape(np.shape(values))
        result = loc + scale * values
        if self._shared_cache is not None:
            self._shared_cache[key] = result.copy() if isinstance(result, np.ndarray) else result
        return result

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None) -> np.ndarray:
        key = ("uniform", self._shape_key(size), float(low), float(high))
        if self._shared_cache is not None and key in self._shared_cache:
            value = self._shared_cache[key]
            return value.copy() if isinstance(value, np.ndarray) else value
        values = self._base.uniform(low, high, size=size)
        if self.distribution == "heavy":
            unit = (values - low) / (high - low)
            centered = 2.0 * unit - 1.0
            values = low + (high - low) * (0.5 + 0.5 * np.sign(centered) * np.sqrt(np.abs(centered)))
        elif self.distribution == "correlated" and np.size(values) > 1:
            flat = np.asarray(values, dtype=float).reshape(-1)
            flat[1:] = 0.8 * flat[:-1] + 0.2 * flat[1:]
            values = flat.reshape(np.shape(values))
        if self._shared_cache is not None:
            self._shared_cache[key] = values.copy() if isinstance(values, np.ndarray) else values
        return values

    def algorithm_uniform(self, low: float = 0.0, high: float = 1.0, size=None) -> np.ndarray:
        """Draw algorithm-internal samples without transforming the regime."""
        return self._base.uniform(low, high, size=size)


def selection_key(record: Dict) -> Tuple[float, float, str]:
    """Order feasible U candidates by cost, then confidence, then name."""
    config = record.get("config", {})
    return (
        float(record["selection_total_cost"]),
        -float(record["validation_lower_bound"]),
        str(config.get("name", "")),
    )


def add_cost_accounting(
    records: Iterable[Dict], n_cal: int, n_val: int, n_calls: int
) -> Dict[str, float]:
    """Attach explicit qualification and deployment costs to each candidate.

    U profiles every candidate on both calibration and validation splits.  O*
    uses calibration only.  ``matched_total_cost`` charges both policies the
    same full qualification budget; the policy-specific fields expose the
    lean-O alternative instead of hiding it in an ambiguous total.
    """
    records = list(records)
    calibration_overhead = n_cal * sum(float(r["calibration_cost"]) for r in records)
    validation_overhead = n_val * sum(float(r["validation_cost"]) for r in records)
    qualification_overhead = calibration_overhead + validation_overhead
    for record in records:
        record["selection_execution_cost"] = n_calls * float(record["validation_cost"])
        record["selection_total_cost"] = qualification_overhead + record["selection_execution_cost"]
        record["oblivious_selection_execution_cost"] = n_calls * float(record["calibration_cost"])
        record["oblivious_selection_total_cost"] = (
            calibration_overhead + record["oblivious_selection_execution_cost"]
        )
        record["post_qualification_cost"] = n_calls * float(record["test_cost"])
        record["matched_total_cost"] = qualification_overhead + record["post_qualification_cost"]
        record["policy_specific_u_total_cost"] = record["matched_total_cost"]
        record["policy_specific_o_total_cost"] = (
            calibration_overhead + record["post_qualification_cost"]
        )
        # Backward-compatible alias. New analyses must name the accounting view.
        record["total_cost"] = record["matched_total_cost"]
    return {
        "calibration_overhead": calibration_overhead,
        "validation_overhead": validation_overhead,
        "qualification_overhead": qualification_overhead,
    }


def select_oblivious(records: Iterable[Dict], mean_threshold: float, exact: Dict) -> Dict:
    """Select O* from the full candidate set using calibration-only evidence."""
    feasible = [
        record
        for record in records
        if float(record["calibration_mean_error"]) <= float(mean_threshold)
    ]
    if not feasible:
        return exact
    return min(
        feasible,
        key=lambda record: (
            float(record["oblivious_selection_total_cost"]),
            str(record.get("config", {}).get("name", "")),
        ),
    )


def lower_bound(successes: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower bound.

    The controller treats this bound as a hard gate, so silently replacing it
    with a different interval would change the statistical guarantee.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("successes and n must satisfy 0 <= successes <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if n == 0:
        return 0.0
    try:
        from scipy.stats import beta
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the Clopper-Pearson gate") from exc
    return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, n - successes + 1))


def fft_case(rng: np.random.Generator, cfg: Config) -> Tuple[float, float]:
    n = 256
    x = rng.normal(size=n)
    ref = np.fft.fft(x)
    t0 = time.perf_counter()
    # NumPy promotes float16 FFT inputs to complex64 internally. The 16-bit
    # candidate therefore quantizes the input but does not claim a native
    # float16 FFT implementation; this backend behavior is disclosed in the
    # manuscript and README.
    y = np.fft.fft(x.astype(np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64))
    if cfg.knob < 1.0:
        keep = max(1, int(cfg.knob * n / 2))
        y[keep:-keep] = 0
    elapsed = time.perf_counter() - t0
    return float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-12)), elapsed


def spmv_case(rng: np.random.Generator, cfg: Config) -> Tuple[float, float]:
    n = 256
    dense = rng.normal(size=(n, n))
    dense[np.abs(dense) < 1.2] = 0.0
    x = rng.normal(size=n)
    # Build one CSR matrix per paired trial so exact and approximate paths
    # operate on identical nonzero structure and input values.
    mask = np.abs(dense) >= 1.2
    indptr = np.concatenate(([0], np.cumsum(mask.sum(axis=1), dtype=np.int64)))
    indices = np.tile(np.arange(n, dtype=np.int32), n)[mask.ravel()]
    values = dense[mask].astype(np.float64, copy=False)

    def csr_matvec(data: np.ndarray, vector: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=np.result_type(data.dtype, vector.dtype))
        for i in range(n):
            start, stop = indptr[i], indptr[i + 1]
            out[i] = np.dot(data[start:stop], vector[indices[start:stop]])
        return out

    ref = csr_matrix((values, indices, indptr), shape=(n, n)).dot(x)
    t0 = time.perf_counter()
    dtype = np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64
    work = values.astype(dtype)
    if cfg.knob > 0:
        # Zero values in place; preserve CSR indices and stored-entry count.
        work[np.abs(work) < cfg.knob] = 0
    # SciPy's CSR kernels provide a representative implementation for the
    # supported float32/float64 paths. SciPy does not accept float16 sparse
    # storage, so the half-precision configuration retains the explicit paired
    # fallback above rather than silently changing its numerical semantics.
    if dtype == np.float16:
        y = csr_matvec(work, x.astype(dtype))
    else:
        y = csr_matrix((work, indices, indptr), shape=(n, n)).dot(x.astype(dtype))
    elapsed = time.perf_counter() - t0
    return float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-12)), elapsed


def stencil_case(rng: np.random.Generator, cfg: Config) -> Tuple[float, float]:
    shape = (16, 16, 16)
    x = rng.normal(size=shape)
    ref = x.copy()
    for _ in range(10):
        ref = (np.roll(ref, 1, 0) + np.roll(ref, -1, 0) + np.roll(ref, 1, 1) + np.roll(ref, -1, 1) + np.roll(ref, 1, 2) + np.roll(ref, -1, 2)) / 6
    t0 = time.perf_counter()
    y = x.astype(np.float16 if cfg.precision == 16 else np.float32 if cfg.precision == 32 else np.float64)
    for i in range(10):
        if cfg.knob and i % int(cfg.knob + 1) == 0:
            continue
        y = (np.roll(y, 1, 0) + np.roll(y, -1, 0) + np.roll(y, 1, 1) + np.roll(y, -1, 1) + np.roll(y, 1, 2) + np.roll(y, -1, 2)) / 6
    elapsed = time.perf_counter() - t0
    return float(np.linalg.norm(y.astype(float) - ref) / (np.linalg.norm(ref) + 1e-12)), elapsed


def monte_carlo_case(rng: InputRNG, cfg: Config) -> Tuple[float, float]:
    n = int(cfg.knob)
    t0 = time.perf_counter()
    # The uniform law is the integration measure, not an exogenous input.
    u = rng.algorithm_uniform(-1.0, 1.0, size=n)
    if cfg.precision == 16:
        u = u.astype(np.float16)
    elif cfg.precision == 32:
        u = u.astype(np.float32)
    if cfg.precision == 99:
        # The reference uses a larger sample budget in this prototype.
        n = 1_000_000
        u = rng.algorithm_uniform(-1.0, 1.0, size=n)
    y = np.mean(np.exp(-(u * u)))
    ref = 0.7468241328124271
    elapsed = time.perf_counter() - t0
    return float(abs(y - ref)), elapsed


Kernel = Callable[[InputRNG, Config], Tuple[float, float]]


def configs(kernel: str) -> Iterable[Config]:
    if kernel == "fft":
        return [Config(f"fft-{p}-{r}", p, r) for p in (16, 32, 64) for r in (0.5, 0.75, 1.0)]
    if kernel == "spmv":
        # The matrix keeps entries with |value| >= 1.2, so these thresholds
        # are deliberately on the same scale and produce distinct sparsity
        # patterns rather than silently becoming no-ops.
        return [Config(f"spmv-{p}-{d}", p, d) for p in (16, 32, 64) for d in (0.0, 1.25, 1.5)]
    if kernel == "stencil":
        return [Config(f"stencil-{p}-{s}", p, s) for p in (16, 32, 64) for s in (0.0, 1.0, 2.0)]
    if kernel == "monte_carlo":
        return [Config(f"mc-{p}-{n}", p, float(n)) for p in (16, 32, 64) for n in (10_000, 100_000, 1_000_000)]
    raise ValueError(kernel)


KERNELS: Dict[str, Kernel] = {
    "fft": fft_case,
    "spmv": spmv_case,
    "stencil": stencil_case,
    "monte_carlo": monte_carlo_case,
}

# Keep every split and seed group on a disjoint, deterministic PRNG stream.
# The former ``seed + offset + index`` schedule overlapped windows from nearby
# configured seed groups (for example seeds 101 and 211).
SPLIT_SEED_OFFSETS = {"calibration": 0, "validation": 100_000, "test": 200_000}
SEED_GROUP_STRIDE = 1_000_000


def trial_seed_for(seed: int, split: str, index: int) -> int:
    """Return a collision-free trial seed for a configured seed group."""
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"unknown split {split!r}")
    if index < 0:
        raise ValueError("trial index must be non-negative")
    return int(seed) * SEED_GROUP_STRIDE + SPLIT_SEED_OFFSETS[split] + int(index)


def run_kernel(
    kernel: str,
    seed: int,
    n_cal: int,
    n_val: int,
    n_test: int,
    delta: float,
    alpha: float,
    epsilon: float,
    n_calls: int = 100,
    distribution: str = "light",
    include_traces: bool = False,
    oblivious_mean_threshold: Optional[float] = None,
) -> Dict:
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
    fn = KERNELS[kernel]
    cfgs = list(configs(kernel))
    selection_alpha = alpha / len(cfgs)
    buffers = {
        cfg.name: {
            "cal_err": [], "cal_time": [], "val_err": [], "val_time": [],
            "test_err": [], "test_time": [],
        }
        for cfg in cfgs
    }
    traces = []
    # Process one trial across all configurations so exogenous inputs (matrix,
    # vector, or field) are generated once and paired exactly across variants.
    for split, count in (("calibration", n_cal), ("validation", n_val), ("test", n_test)):
        for i in range(count):
            trial_seed = trial_seed_for(seed, split, i)
            shared_cache = {}
            for cfg in cfgs:
                err, elapsed = fn(InputRNG(trial_seed, distribution, shared_cache), cfg)
                buffer = buffers[cfg.name]
                prefix = {"calibration": "cal", "validation": "val", "test": "test"}[split]
                buffer[f"{prefix}_err"].append(err)
                buffer[f"{prefix}_time"].append(elapsed)
                if include_traces:
                    traces.append({"config": asdict(cfg), "split": split, "index": i, "seed": trial_seed, "error": err, "wall_time_s": elapsed})
    records = []
    for cfg in cfgs:
        buffer = buffers[cfg.name]
        cal_err, cal_time = buffer["cal_err"], buffer["cal_time"]
        val_err, val_time = buffer["val_err"], buffer["val_time"]
        test_err, test_time = buffer["test_err"], buffer["test_time"]
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
            "calibration_errors": cal_err,
            "validation_errors": val_err,
            "test_errors": test_err,
        })
    if kernel == "fft":
        exact = next(r for r in records if r["config"]["precision"] == 64 and r["config"]["knob"] == 1.0)
    elif kernel == "monte_carlo":
        exact = next(r for r in records if r["config"]["precision"] == 64 and r["config"]["knob"] == 1_000_000.0)
    else:
        exact = next(r for r in records if r["config"]["precision"] == 64 and r["config"]["knob"] == 0.0)
    tolerance = epsilon
    for r in records:
        r["calibration_successes"] = sum(e <= tolerance for e in r["calibration_errors"])
        r["test_successes"] = sum(e <= tolerance for e in r["test_errors"])
        r["calibration_coverage"] = r["calibration_successes"] / n_cal
        r["calibration_lower_bound"] = lower_bound(r["calibration_successes"], n_cal, selection_alpha)
        r["validation_successes"] = sum(e <= tolerance for e in r["validation_errors"])
        r["validation_coverage"] = r["validation_successes"] / n_val
        r["validation_lower_bound"] = lower_bound(r["validation_successes"], n_val, selection_alpha)
        r["test_coverage"] = r["test_successes"] / n_test
        r["test_lower_bound"] = lower_bound(r["test_successes"], n_test, alpha)
    costs = add_cost_accounting(records, n_cal, n_val, n_calls)
    feasible = [r for r in records if r["validation_lower_bound"] >= 1.0 - delta]
    chosen = min(feasible, key=selection_key) if feasible else exact

    # O* applies a fixed mean-error threshold to the full candidate set.  Its
    # gate and timing objective use calibration data only; held-out test data
    # are reserved for evaluation.
    non_exact = [r for r in records if r is not exact]
    approximate_feasible = [r for r in non_exact if r["validation_lower_bound"] >= 1.0 - delta]
    oblivious_threshold = float(oblivious_mean_threshold)
    oblivious_chosen = select_oblivious(records, oblivious_threshold, exact)
    for r in records:
        r.pop("calibration_errors")
        r.pop("validation_errors")
        r.pop("test_errors")
    output = {
        "kernel": kernel,
        "seed": seed,
        "n_calibration": n_cal,
        "n_validation": n_val,
        "n_test": n_test,
        "n_calls": n_calls,
        **costs,
        "selection_cost_basis": "validation_timing",
        "delta": delta,
        "alpha": alpha,
        "selection_alpha": selection_alpha,
        "n_configurations": len(cfgs),
        "tolerance": tolerance,
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
        "records": records,
        # Exact is always eligible for U when its validation gate passes. This
        # flag records whether no non-exact configuration passed that gate.
        "fallback_to_exact": not bool(approximate_feasible),
        "energy_measurement": "unavailable",
    }
    if include_traces:
        output["traces"] = traces
    return output


def environment_metadata() -> Dict[str, object]:
    """Return compact, JSON-serializable runtime metadata for audit files."""
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": _scipy_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "byteorder": sys.byteorder,
    }


def _scipy_version() -> str:
    try:
        import scipy
    except ImportError:
        return "unavailable"
    return scipy.__version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=[*KERNELS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-cal", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--n-calls", type=int, default=100)
    parser.add_argument("--distribution", choices=["light", "heavy", "correlated"], default="light")
    parser.add_argument("--oblivious-mean-threshold", type=float, default=None)
    args = parser.parse_args()
    selected = KERNELS if args.kernel == "all" else {args.kernel: KERNELS[args.kernel]}
    output = [run_kernel(k, args.seed, args.n_cal, args.n_val, args.n_test, args.delta, args.alpha, args.epsilon, args.n_calls, args.distribution, oblivious_mean_threshold=args.oblivious_mean_threshold) for k in selected]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
