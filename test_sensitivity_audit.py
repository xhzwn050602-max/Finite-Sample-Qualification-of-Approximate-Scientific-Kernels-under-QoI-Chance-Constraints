import unittest

import sensitivity_audit as audit


def _record(errors, time):
    return {
        "calibration_error": list(errors),
        "validation_error": list(errors),
        "test_error": list(errors),
        "calibration_time": [time] * len(errors),
        "validation_time": [time] * len(errors),
        "test_time": [time] * len(errors),
    }


class SensitivityAuditTests(unittest.TestCase):
    def test_qualified_reference_is_not_an_empty_gate_fallback(self):
        grouped = {
            ("fft", "light", 101, "fft-64-1.0"): _record([0.0] * 256, 2.0),
            ("fft", "light", 101, "fft-32-1.0"): _record([0.2] * 256, 1.0),
        }
        rows = audit.replay(grouped, {"kernel_epsilon": 0.05, "poisson_epsilon": 0.2},
                            [1.0], [0.1], 0.05, 10)
        self.assertEqual(rows[0]["chosen"], "fft-64-1.0")
        self.assertFalse(rows[0]["fallback_to_exact"])

    def test_cp_lower_is_bounded_for_all_successes(self):
        bound = audit.cp_lower(256, 256, 0.05)
        self.assertGreater(bound, 0.98)
        self.assertLess(bound, 1.0)

    def test_replay_reports_exact_fallback_when_gate_fails(self):
        grouped = {
            ("fft", "light", 101, "fft-64-1.0"): _record([0.0] * 4, 2.0),
            ("fft", "light", 101, "fft-32-1.0"): _record([0.2] * 4, 1.0),
        }
        config = {"kernel_epsilon": 0.05, "poisson_epsilon": 0.2}
        rows = audit.replay(grouped, config, [1.0], [0.1], 0.05, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"], "fft-64-1.0")
        self.assertTrue(rows[0]["fallback_to_exact"])
        self.assertEqual(rows[0]["u_test_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
