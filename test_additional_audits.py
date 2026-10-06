import unittest

from production_heat_audit import _configs, _json_safe, run_cell
from stability_audit import _configs as stability_configs


class AdditionalAuditTest(unittest.TestCase):
    def test_larger_heat_candidate_grid_is_fixed(self):
        names = [cfg.name for cfg in _configs()]
        self.assertEqual(len(names), 13)
        self.assertEqual(names[-1], "heat-exact")
        self.assertEqual(len(set(names)), len(names))

    def test_stability_grid_has_three_precisions(self):
        configs = stability_configs([1, 5, 10])
        self.assertEqual(len(configs), 9)
        self.assertEqual({cfg.precision for cfg in configs}, {16, 32, 64})

    def test_small_larger_heat_cell_is_paired_and_finite(self):
        row, traces = run_cell(
            seed=101,
            distribution="light",
            n=16,
            n_cal=1,
            n_val=1,
            n_test=2,
            epsilon=0.05,
            delta=0.10,
            alpha=0.05,
            n_calls=10,
            oblivious_mean_threshold=0.05,
            include_traces=True,
        )
        self.assertEqual(len(traces), 13 * 4)
        self.assertEqual(row["grid"], 16)
        self.assertTrue(row["selected_finite_rate"] >= 0.0)
        self.assertIn(row["chosen"], {cfg.name for cfg in _configs()})

    def test_nonfinite_trace_values_are_json_null_safe(self):
        self.assertIsNone(_json_safe(float("nan")))
        self.assertEqual(_json_safe({"nested": [float("inf"), -float("inf")]}),
                         {"nested": [None, None]})


if __name__ == "__main__":
    unittest.main()
