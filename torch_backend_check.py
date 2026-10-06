"""Cross-backend CPU correctness audit using PyTorch.

The publication implementation remains NumPy/SciPy.  This optional audit
recomputes the four reusable-kernel outputs and the heat Jacobi path through
PyTorch CPU operators on the same fixed arrays. It is a numerical portability
check, not a timing, energy, or GPU claim; the heat direct reference is
deliberately excluded because this release has no equivalent portable PyTorch
sparse factorization.
"""

from __future__ import annotations

import json
import warnings
from math import pi

import numpy as np


def run_audit() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError("PyTorch is required for the optional backend audit") from exc

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    dtype = torch.float64

    # FFT: compare the same real signal through NumPy and torch.fft.
    x = rng.normal(size=32).astype(np.float64)
    np_fft = np.fft.fft(x)
    torch_fft = torch.fft.fft(torch.from_numpy(x)).numpy()
    fft_abs = float(np.max(np.abs(np_fft - torch_fft)))

    # CSR SpMV: build a true torch sparse CSR tensor from the same thresholded
    # matrix and compare it with the NumPy dense product.
    matrix = rng.normal(size=(24, 24))
    matrix[np.abs(matrix) < 1.2] = 0.0
    vector = rng.normal(size=24)
    rows, cols = np.nonzero(matrix)
    counts = np.bincount(rows, minlength=matrix.shape[0])
    crow = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    np_spmv = matrix @ vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch_csr = torch.sparse_csr_tensor(
            torch.from_numpy(crow),
            torch.from_numpy(cols.astype(np.int64)),
            torch.from_numpy(matrix[rows, cols]),
            size=matrix.shape,
            dtype=dtype,
        )
    torch_spmv = torch.sparse.mm(torch_csr, torch.from_numpy(vector[:, None])).squeeze(1).numpy()
    spmv_abs = float(np.max(np.abs(np_spmv - torch_spmv)))

    # Seven-point periodic stencil: use torch.roll with the same six-neighbor
    # update and ten updates as the reference implementation.
    field = rng.normal(size=(8, 8, 8)).astype(np.float64)
    np_stencil = field.copy()
    torch_stencil = torch.from_numpy(field.copy())
    for _ in range(10):
        np_stencil = (
            np.roll(np_stencil, 1, 0) + np.roll(np_stencil, -1, 0)
            + np.roll(np_stencil, 1, 1) + np.roll(np_stencil, -1, 1)
            + np.roll(np_stencil, 1, 2) + np.roll(np_stencil, -1, 2)
        ) / 6.0
        torch_stencil = (
            torch.roll(torch_stencil, 1, 0) + torch.roll(torch_stencil, -1, 0)
            + torch.roll(torch_stencil, 1, 1) + torch.roll(torch_stencil, -1, 1)
            + torch.roll(torch_stencil, 1, 2) + torch.roll(torch_stencil, -1, 2)
        ) / 6.0
    stencil_abs = float(np.max(np.abs(np_stencil - torch_stencil.numpy())))

    # Monte Carlo: use one fixed uniform sample stream and compare the same
    # pointwise integrand and reduction.
    samples = rng.uniform(-1.0, 1.0, size=100_000).astype(np.float64)
    np_mc = float(np.mean(np.exp(-(samples * samples))))
    torch_samples = torch.from_numpy(samples)
    torch_mc = float(torch.mean(torch.exp(-(torch_samples * torch_samples))).item())
    mc_abs = abs(np_mc - torch_mc)

    # Heat Jacobi: compare the approximate iterative path on a small assembled
    # operator. The SciPy sparse direct factorization remains the reference and
    # is intentionally not reimplemented here.
    from benchmark import InputRNG
    from heat_case import _assemble, _conductivity, _jacobi

    heat_rng = InputRNG(321, "light", {})
    conductivity = _conductivity(heat_rng, 8, {})
    matrix, rhs, _ = _assemble(conductivity)
    np_heat = _jacobi(matrix, rhs, 25, np.float64)
    dense_heat = torch.from_numpy(matrix.toarray())
    torch_rhs = torch.from_numpy(rhs)
    heat_diag = torch.diag(dense_heat)
    heat_off = dense_heat - torch.diag(heat_diag)
    torch_heat = torch.zeros_like(torch_rhs)
    for _ in range(25):
        torch_heat = (torch_rhs - heat_off @ torch_heat) / heat_diag
    heat_abs = float(np.max(np.abs(np_heat - torch_heat.numpy())))

    errors = {
        "fft_max_abs": fft_abs,
        "spmv_max_abs": spmv_abs,
        "stencil_max_abs": stencil_abs,
        "monte_carlo_abs": mc_abs,
        "heat_jacobi_max_abs": heat_abs,
    }
    return {
        "backend": "torch-cpu",
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "checked_kernels": ["fft", "spmv", "stencil", "monte_carlo", "heat_jacobi"],
        "excluded_case": "2d_heat_conduction direct sparse factorization (no portable PyTorch equivalent)",
        "errors": errors,
        "max_abs_error": max(errors.values()),
        "pass": bool(max(errors.values()) < 1.0e-10),
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
