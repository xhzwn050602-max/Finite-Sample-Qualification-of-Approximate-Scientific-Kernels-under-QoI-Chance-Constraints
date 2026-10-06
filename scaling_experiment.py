"""Cross-size CPU timing audit for the reusable scientific kernels.

The audit measures the same numerical paths used by ``benchmark.py`` at
several sizes.  It reports robust timing summaries and log-log slopes against
the dominant work count; it is a complexity sanity check, not a speedup claim
for another machine or backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from benchmark import environment_metadata


def _timings(fn: Callable[[], object], repeats: int, batch: int) -> list[float]:
    if batch < 1:
        raise ValueError("batch must be positive")
    fn()  # warm up dispatch and lazy backend initialization
    values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = None
        for _ in range(batch):
            result = fn()
        elapsed = time.perf_counter() - start
        if isinstance(result, np.ndarray) and not np.all(np.isfinite(result)):
            raise ValueError("timed kernel returned non-finite values")
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("invalid timing")
        values.append(elapsed)
    return values


def _summary(values: Iterable[float]) -> tuple[float, float, float]:
    ordered = sorted(float(value) for value in values)
    return (
        float(statistics.median(ordered)),
        float(np.quantile(ordered, 0.10)),
        float(np.quantile(ordered, 0.90)),
    )


def _fft_case(size: int, seed: int, repeats: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    variants = {
        "reference-f64": lambda: np.fft.fft(x.astype(np.float64, copy=False)),
        "approx-f32": lambda: np.fft.fft(x.astype(np.float32, copy=False)),
    }
    work = float(size * math.log2(size))
    return _rows("fft", size, work, variants, repeats, batch=10)


def _spmv_case(size: int, seed: int, repeats: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(size, size))
    dense[np.abs(dense) < 1.2] = 0.0
    mask = np.abs(dense) >= 1.2
    indptr = np.concatenate(([0], np.cumsum(mask.sum(axis=1), dtype=np.int64)))
    indices = np.tile(np.arange(size, dtype=np.int32), size)[mask.ravel()]
    values = dense[mask].astype(np.float64, copy=False)
    vector = rng.normal(size=size)
    reference = csr_matrix((values, indices, indptr), shape=(size, size))
    approx = csr_matrix((values.astype(np.float32), indices, indptr), shape=(size, size))
    variants = {
        "reference-f64": lambda: reference.dot(vector),
        "approx-f32": lambda: approx.dot(vector.astype(np.float32, copy=False)),
    }
    return _rows("spmv", size, float(values.size), variants, repeats, batch=20)


def _stencil_case(size: int, seed: int, repeats: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, size, size))

    def update(field: np.ndarray) -> np.ndarray:
        result = field
        for _ in range(10):
            result = (
                np.roll(result, 1, 0)
                + np.roll(result, -1, 0)
                + np.roll(result, 1, 1)
                + np.roll(result, -1, 1)
                + np.roll(result, 1, 2)
                + np.roll(result, -1, 2)
            ) / 6
        return result

    variants = {
        "reference-f64": lambda: update(x.astype(np.float64, copy=False)),
        "approx-f32": lambda: update(x.astype(np.float32, copy=False)),
    }
    return _rows("stencil", size, float(10 * size**3), variants, repeats, batch=3)


def _monte_carlo_case(size: int, seed: int, repeats: int) -> list[dict]:
    variants = {}
    for label, dtype in (("reference-f64", np.float64), ("approx-f32", np.float32)):
        generator = np.random.default_rng(seed)

        def draw(dtype=dtype, generator=generator):
            samples = generator.uniform(-1.0, 1.0, size=size).astype(dtype, copy=False)
            return np.mean(np.exp(-(samples * samples)))

        variants[label] = draw
    return _rows("monte_carlo", size, float(size), variants, repeats, batch=1)


def _rows(kernel: str, size: int, work: float, variants: dict[str, Callable[[], object]], repeats: int, batch: int) -> list[dict]:
    rows = []
    for variant, fn in variants.items():
        values = _timings(fn, repeats, batch)
        median, p10, p90 = _summary(values)
        rows.append({
            "kernel": kernel,
            "size": int(size),
            "work_units": work,
            "variant": variant,
            "median_time_s": median,
            "p10_time_s": p10,
            "p90_time_s": p90,
            "repeats": repeats,
            "batch": batch,
        })
    return rows


def _fit(rows: list[dict]) -> list[dict]:
    output = []
    for kernel in sorted({row["kernel"] for row in rows}):
        for variant in sorted({row["variant"] for row in rows if row["kernel"] == kernel}):
            selected = [row for row in rows if row["kernel"] == kernel and row["variant"] == variant]
            x = np.log(np.asarray([row["work_units"] for row in selected], dtype=float))
            y = np.log(np.asarray([row["median_time_s"] for row in selected], dtype=float))
            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            output.append({
                "kernel": kernel,
                "variant": variant,
                "n_sizes": len(selected),
                "slope_vs_work": float(slope),
                "r2": 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
                "min_size": min(row["size"] for row in selected),
                "max_size": max(row["size"] for row in selected),
            })
    return output


def run(config_path: str) -> dict:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    repeats = int(config["repeats"])
    if repeats < 3:
        raise ValueError("repeats must be at least 3")
    seed = int(config["seed"])
    rows: list[dict] = []
    rows.extend(row for size in config["fft_sizes"] for row in _fft_case(int(size), seed, repeats))
    rows.extend(row for size in config["spmv_sizes"] for row in _spmv_case(int(size), seed, repeats))
    rows.extend(row for size in config["stencil_sizes"] for row in _stencil_case(int(size), seed, repeats))
    rows.extend(row for size in config["monte_carlo_sizes"] for row in _monte_carlo_case(int(size), seed, repeats))
    digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
    return {
        "config_file": config_file.name,
        "config_sha256": digest,
        "seed": seed,
        "environment": environment_metadata(),
        "rows": rows,
        "fits": _fit(rows),
        "interpretation": "CPU timing audit; slopes are descriptive and not cross-hardware speedup claims",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scaling-experiment-config.json")
    parser.add_argument("--output", default="scaling-audit.json")
    parser.add_argument("--csv", default="scaling-audit.csv")
    args = parser.parse_args()
    result = run(args.config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(result["rows"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"config_sha256": result["config_sha256"], "rows": len(result["rows"]), "fits": result["fits"]}, indent=2))


if __name__ == "__main__":
    main()
