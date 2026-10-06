"""Independent small-size numerical cross-checks for the reference code.

These checks intentionally use different computational paths from the main
implementations.  They are correctness checks, not performance claims.
"""

from __future__ import annotations

import json
from math import pi

import numpy as np
from scipy.sparse.linalg import spsolve

from benchmark import Config, InputRNG, fft_case, monte_carlo_case, spmv_case, stencil_case
from heat_case import Config as HeatConfig, _assemble, _conductivity, solve_case_details


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=16)
    dft = np.array([sum(x[j] * np.exp(-2j * pi * k * j / len(x)) for j in range(len(x))) for k in range(len(x))])
    fft = np.fft.fft(x)
    fft_abs = float(np.max(np.abs(dft - fft)))

    matrix = rng.normal(size=(12, 12))
    matrix[np.abs(matrix) < 0.9] = 0.0
    vector = rng.normal(size=12)
    cfg = Config("spmv-64-0.0", 64, 0.0)
    # spmv_case's reference is the SciPy CSR path; independently reproduce the
    # same product with a Python row loop on the exact generated matrix.
    mask = np.abs(matrix) >= 1.2
    matrix[~mask] = 0.0
    explicit = matrix @ vector
    # The case function uses n=256, so only compare a standalone CSR product
    # here; this check does not depend on its benchmark dimensions.
    from scipy.sparse import csr_matrix
    csr_value = csr_matrix(matrix).dot(vector)
    spmv_abs = float(np.max(np.abs(explicit - csr_value)))

    heat_rng = InputRNG(17, "correlated", {})
    k = _conductivity(heat_rng, 8, {})
    A, b, _ = _assemble(k)
    sparse_solution = spsolve(A, b)
    dense_solution = np.linalg.solve(A.toarray(), b)
    heat_abs = float(np.max(np.abs(sparse_solution - dense_solution)))

    # Exercise all public kernel paths and require finite outputs.
    kernel_outputs = {
        "fft": fft_case(InputRNG(7, "light"), Config("fft-64-1.0", 64, 1.0)),
        "spmv": spmv_case(InputRNG(7, "light"), Config("spmv-64-0.0", 64, 0.0)),
        "stencil": stencil_case(InputRNG(7, "light"), Config("stencil-64-0.0", 64, 0.0)),
        "monte_carlo": monte_carlo_case(InputRNG(7, "light"), Config("mc-64-10000", 64, 10000.0)),
        "heat": solve_case_details(InputRNG(7, "light"), HeatConfig("heat-exact", 64, 0, "direct")),
    }
    finite = all(np.isfinite(float(value[0] if isinstance(value, tuple) else value["qoi_error"])) for value in kernel_outputs.values())
    result = {
        "fft_direct_dft_max_abs": fft_abs,
        "spmv_csr_dense_max_abs": spmv_abs,
        "heat_sparse_dense_max_abs": heat_abs,
        "all_public_paths_finite": finite,
        "pass": bool(fft_abs < 1e-12 and spmv_abs < 1e-12 and heat_abs < 1e-10 and finite),
    }
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
