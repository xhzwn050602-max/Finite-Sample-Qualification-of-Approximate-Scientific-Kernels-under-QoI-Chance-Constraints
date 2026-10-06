import unittest

import numpy as np

from benchmark import Config, InputRNG, add_cost_accounting, environment_metadata, lower_bound, monte_carlo_case, run_kernel, select_oblivious, selection_key, spmv_case, trial_seed_for
from analyze_results import summarize
from pde_case import Config as PoissonConfig, run as run_poisson, solve_case_details
from heat_case import Config as HeatConfig, run as run_heat, solve_case_details as solve_heat_case_details
from run_sweep import config_provenance, load_config
from scaling_experiment import _fit, _fft_case
from confirmatory_mc import load_locked_config


class ControllerSmokeTest(unittest.TestCase):
    def test_lower_bound_is_monotone(self):
        self.assertLessEqual(lower_bound(2, 8, 0.05), lower_bound(8, 8, 0.05))

    def test_selection_key_prefers_confidence_on_cost_tie(self):
        low = {"selection_total_cost": 1.0, "validation_lower_bound": 0.91,
               "config": {"name": "low"}}
        high = {"selection_total_cost": 1.0, "validation_lower_bound": 0.92,
                "config": {"name": "high"}}
        self.assertEqual(min([low, high], key=selection_key), high)

    def test_selection_uses_calibration_fields(self):
        result = run_kernel("fft", seed=3, n_cal=2, n_val=2, n_test=4, delta=0.5, alpha=0.05, epsilon=0.05)
        self.assertIn("chosen", result)
        self.assertTrue(all("calibration_lower_bound" in row for row in result["records"]))
        self.assertTrue(all("validation_lower_bound" in row for row in result["records"]))
        self.assertTrue(all("test_lower_bound" in row for row in result["records"]))
        self.assertTrue(all("total_cost" in row for row in result["records"]))
        self.assertTrue(all("selection_total_cost" in row for row in result["records"]))
        self.assertEqual(result["selection_cost_basis"], "validation_timing")
        self.assertIn("oblivious_chosen", result)
        self.assertIn("oblivious_threshold", result)

    def test_calibration_overhead_counts_all_profiled_trials(self):
        n_cal = 3
        result = run_kernel("fft", seed=3, n_cal=n_cal, n_val=2, n_test=2,
                            delta=0.5, alpha=0.05, epsilon=0.05)
        expected = n_cal * sum(row["calibration_cost"] for row in result["records"])
        self.assertAlmostEqual(result["calibration_overhead"], expected, places=15)

    def test_qualification_cost_counts_both_profiled_splits(self):
        records = [
            {"calibration_cost": 2.0, "validation_cost": 3.0, "test_cost": 5.0},
            {"calibration_cost": 7.0, "validation_cost": 11.0, "test_cost": 13.0},
        ]
        costs = add_cost_accounting(records, n_cal=2, n_val=3, n_calls=10)
        self.assertEqual(costs["calibration_overhead"], 18.0)
        self.assertEqual(costs["validation_overhead"], 42.0)
        self.assertEqual(costs["qualification_overhead"], 60.0)
        self.assertEqual(records[0]["post_qualification_cost"], 50.0)
        self.assertEqual(records[0]["matched_total_cost"], 110.0)
        self.assertEqual(records[0]["policy_specific_u_total_cost"], 110.0)
        self.assertEqual(records[0]["policy_specific_o_total_cost"], 68.0)

    def test_fallback_flag_means_no_nonexact_configuration_passed(self):
        result = run_kernel("fft", seed=3, n_cal=4, n_val=4, n_test=4,
                            delta=0.5, alpha=0.05, epsilon=1e-12)
        self.assertEqual(result["chosen"]["name"], "fft-64-1.0")
        self.assertTrue(result["fallback_to_exact"])

    def test_spmv_uses_explicit_energy_boundary(self):
        result = run_kernel("spmv", seed=3, n_cal=1, n_val=1, n_test=2, delta=0.5, alpha=0.05, epsilon=0.5)
        self.assertEqual(result["energy_measurement"], "unavailable")
        self.assertTrue(all(row["energy_j"] is None for row in result["records"]))

    def test_spmv_drop_knob_changes_paired_error(self):
        no_drop = spmv_case(InputRNG(41, "light"), Config("spmv-64-0.0", 64, 0.0))[0]
        drop = spmv_case(InputRNG(41, "light"), Config("spmv-64-1.5", 64, 1.5))[0]
        self.assertGreater(drop, no_drop)

    def test_oblivious_baseline_can_select_reference(self):
        exact = {
            "config": {"name": "reference"}, "calibration_mean_error": 0.0,
            "oblivious_selection_total_cost": 1.0,
        }
        approximate = {
            "config": {"name": "approximate"}, "calibration_mean_error": 0.01,
            "oblivious_selection_total_cost": 2.0,
        }
        self.assertIs(select_oblivious([approximate, exact], 0.05, exact), exact)

    def test_declared_input_regimes_are_reproducible(self):
        for distribution in ("light", "heavy", "correlated"):
            first = InputRNG(17, distribution).normal(size=16)
            second = InputRNG(17, distribution).normal(size=16)
            self.assertTrue((first == second).all())
        heavy = InputRNG(17, "heavy").normal(size=128)
        light = InputRNG(17, "light").normal(size=128)
        self.assertFalse((heavy == light).all())

    def test_invalid_statistical_parameters_fail_loudly(self):
        with self.assertRaises(ValueError):
            run_kernel("fft", seed=1, n_cal=0, n_val=2, n_test=2, delta=0.1, alpha=0.05, epsilon=0.1)
        with self.assertRaises(ValueError):
            lower_bound(1, 1, 0.0)

    def test_monte_carlo_measure_is_invariant_to_external_regime(self):
        cfg = Config("mc", 32, 10_000)
        errors = [monte_carlo_case(InputRNG(23, distribution), cfg)[0] for distribution in ("light", "heavy", "correlated")]
        self.assertEqual(errors[0], errors[1])
        self.assertEqual(errors[1], errors[2])

    def test_optional_traces_are_split_labeled_and_reproducible(self):
        result = run_kernel("fft", seed=5, n_cal=1, n_val=1, n_test=1, delta=0.5, alpha=0.05, epsilon=0.5, include_traces=True)
        self.assertEqual(len(result["traces"]), len(result["records"]) * 3)
        self.assertEqual({row["split"] for row in result["traces"]}, {"calibration", "validation", "test"})
        self.assertTrue(all("seed" in row and "wall_time_s" in row for row in result["traces"]))

    def test_trace_seed_ranges_are_disjoint_across_splits(self):
        result = run_kernel("fft", seed=5, n_cal=3, n_val=2, n_test=4,
                            delta=0.5, alpha=0.05, epsilon=0.5,
                            include_traces=True)
        seeds = {
            split: {row["seed"] for row in result["traces"] if row["split"] == split}
            for split in ("calibration", "validation", "test")
        }
        self.assertTrue(seeds["calibration"].isdisjoint(seeds["validation"]))
        self.assertTrue(seeds["validation"].isdisjoint(seeds["test"]))
        self.assertTrue(seeds["calibration"].isdisjoint(seeds["test"]))
        self.assertEqual(seeds["calibration"], {5_000_000, 5_000_001, 5_000_002})
        self.assertEqual(seeds["validation"], {5_100_000, 5_100_001})
        self.assertEqual(seeds["test"], {5_200_000, 5_200_001, 5_200_002, 5_200_003})

    def test_confirmatory_config_is_locked_and_uses_fresh_seed_groups(self):
        from pathlib import Path
        config, digest = load_locked_config(Path("confirmatory-mc-config.json"))
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(config["candidates"]), 4)
        self.assertEqual(config["reference"], "mc-64-1000000")
        broad_seeds = {101, 211, 307, 401, 503, 601, 701, 809, 907, 1009}
        self.assertTrue(broad_seeds.isdisjoint(config["seeds"]))

    def test_trial_seed_schedule_is_disjoint_across_groups(self):
        first = {trial_seed_for(101, split, i) for split, count in (("calibration", 256), ("validation", 256), ("test", 1024)) for i in range(count)}
        second = {trial_seed_for(211, split, i) for split, count in (("calibration", 256), ("validation", 256), ("test", 1024)) for i in range(count)}
        self.assertTrue(first.isdisjoint(second))

    def test_environment_metadata_is_json_friendly(self):
        metadata = environment_metadata()
        self.assertIn("python_version", metadata)
        self.assertIn("thread_env", metadata)

    def test_publication_config_is_valid_and_hashable(self):
        config = load_config("publication-config.json")
        self.assertEqual(config["kernel_calibration"], 256)
        self.assertEqual(config["poisson_validation"], 256)
        self.assertEqual(config["distributions"], ["light", "heavy", "correlated"])
        provenance = config_provenance("publication-config.json")
        self.assertEqual(len(provenance["config_sha256"]), 64)
        self.assertEqual(provenance["config_file"], "publication-config.json")

    def test_summary_does_not_treat_seed_group_rows_as_independent_pool(self):
        summary = summarize("publication-light.csv")
        self.assertTrue(all(row["u_cp_lower"] <= row["u_cp_lower_pooled_descriptive"] for row in summary))
        self.assertTrue(all("minimum per seed-group CP bound" in row["coverage_aggregation"] for row in summary))
        self.assertTrue(all(not row["u_pooled_cp_valid"] for row in summary))

    def test_aggregate_reports_paired_cost_statistics(self):
        summary = summarize("publication-light.csv")
        self.assertEqual(len(summary), 5)
        self.assertIn("config_sha256", summary[0])
        self.assertIn("selection_alpha", summary[0])
        self.assertIn("u_selection_counts", summary[0])
        self.assertTrue(all("u_minus_oblivious_ci95_low" in row for row in summary))
        self.assertTrue(all("u_fallback_rate" in row for row in summary))
        poisson = next(row for row in summary if row["case"] == "2d_poisson")
        self.assertIsNotNone(poisson["selected_residual_mean"])
        self.assertEqual(poisson["selected_finite_rate_min"], 1.0)

    def test_aggregate_rejects_mixed_configuration_provenance(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            path.write_text(
                "case,distribution,config_file,config_sha256,n_test,test_successes,"
                "oblivious_test_successes,total_cost,oblivious_total_cost,"
                "fallback_to_exact,test_lower_bound,oblivious_test_lower_bound\n"
                "fft,light,a.json,aaa,1,1,1,1.0,1.0,False,1.0,1.0\n"
                "fft,light,b.json,bbb,1,1,1,1.0,1.0,False,1.0,1.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                summarize(str(path))

    def test_aggregate_reconstructs_tails_from_jsonl(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "rows.csv"
            csv_path.write_text(
                "case,distribution,seed,chosen,oblivious_chosen,config_file,config_sha256,n_test,test_successes,"
                "oblivious_test_successes,total_cost,oblivious_total_cost,fallback_to_exact,"
                "test_lower_bound,oblivious_test_lower_bound\n"
                "fft,light,7,fft-32-1.0,fft-32-1.0,cfg.json,abc,3,3,3,1.0,1.0,False,0.5,0.5\n",
                encoding="utf-8",
            )
            trace_path = Path(directory) / "traces.jsonl"
            records = [
                {"case": "fft", "distribution": "light", "seed_group": 7,
                 "split": "test", "config": {"name": "fft-32-1.0"}, "error": value}
                for value in (0.01, 0.02, 0.03)
            ]
            trace_path.write_text(
                json.dumps({"record_type": "metadata"}) + "\n"
                + "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            result = summarize(str(csv_path), raw_traces=str(trace_path))[0]
            self.assertAlmostEqual(result["selected_test_p95_error_max"], 0.029, places=3)
            self.assertAlmostEqual(result["selected_test_p99_error_max"], 0.0298, places=4)

    def test_poisson_diagnostics_are_finite_and_iteration_sensitive(self):
        exact = solve_case_details(InputRNG(29, "light"), PoissonConfig("exact", 64, 0, "direct"))
        coarse = solve_case_details(InputRNG(29, "light"), PoissonConfig("coarse", 64, 25))
        refined = solve_case_details(InputRNG(29, "light"), PoissonConfig("refined", 64, 400))
        self.assertLess(exact["field_error"], 1e-3)
        self.assertGreater(exact["field_error"], 0.0)
        self.assertLess(exact["residual_rel"], 1e-10)
        self.assertTrue(coarse["finite"])
        self.assertTrue(refined["finite"])
        self.assertGreaterEqual(coarse["residual_rel"], refined["residual_rel"])
        self.assertGreaterEqual(coarse["center_error"], refined["center_error"])

    def test_poisson_selection_exposes_validation_cost_basis(self):
        result = run_poisson(seed=3, n_cal=1, n_val=1, n_test=2,
                             epsilon=0.5, delta=0.5, alpha=0.05,
                             n_calls=2, distribution="light")
        self.assertEqual(result["selection_cost_basis"], "validation_timing")
        self.assertTrue(all("selection_total_cost" in row for row in result["records"]))

    def test_heat_reference_and_jacobi_are_finite_and_paired(self):
        cache = {}
        exact = solve_heat_case_details(InputRNG(31, "correlated", cache), HeatConfig("heat-exact", 64, 0, "direct"), cache)
        coarse = solve_heat_case_details(InputRNG(31, "correlated", cache), HeatConfig("heat-32-25", 32, 25), cache)
        self.assertTrue(exact["finite"])
        self.assertTrue(coarse["finite"])
        self.assertLess(exact["qoi_error"], 1e-12)
        self.assertLess(exact["residual_rel"], 1e-10)
        self.assertGreaterEqual(coarse["field_error"], 0.0)

    def test_heat_selection_exposes_validation_cost_basis(self):
        result = run_heat(seed=3, n_cal=1, n_val=1, n_test=2,
                          epsilon=0.05, delta=0.5, alpha=0.05,
                          n_calls=2, distribution="light")
        self.assertEqual(result["case"], "2d_heat_conduction")
        self.assertEqual(result["selection_cost_basis"], "validation_timing")
        self.assertTrue(all("validation_lower_bound" in row for row in result["records"]))

    def test_heat_calibration_overhead_counts_all_profiled_trials(self):
        n_cal = 2
        result = run_heat(seed=3, n_cal=n_cal, n_val=1, n_test=1,
                          epsilon=0.05, delta=0.5, alpha=0.05,
                          n_calls=2, distribution="light")
        expected = n_cal * sum(row["calibration_cost"] for row in result["records"])
        self.assertAlmostEqual(result["calibration_overhead"], expected, places=12)

    def test_scaling_audit_rows_are_finite_and_fit(self):
        rows = _fft_case(256, seed=17, repeats=3) + _fft_case(512, seed=17, repeats=3)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["median_time_s"] > 0.0 for row in rows))
        fits = _fit(rows)
        self.assertEqual(len(fits), 2)
        self.assertTrue(all(np.isfinite(item["slope_vs_work"]) for item in fits))


if __name__ == "__main__":
    unittest.main()
