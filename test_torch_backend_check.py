import unittest


try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchBackendAuditTest(unittest.TestCase):
    def test_cross_backend_errors_are_small(self):
        from torch_backend_check import run_audit

        result = run_audit()
        self.assertTrue(result["pass"])
        self.assertLess(result["max_abs_error"], 1.0e-10)
        self.assertEqual(result["checked_kernels"], ["fft", "spmv", "stencil", "monte_carlo", "heat_jacobi"])


if __name__ == "__main__":
    unittest.main()
