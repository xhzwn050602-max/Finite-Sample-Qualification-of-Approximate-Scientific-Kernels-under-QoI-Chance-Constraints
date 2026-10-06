"""Regression tests for the post hoc distribution-shift audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from distribution_shift_audit import audit_pair, cp_lower


class DistributionShiftAuditTest(unittest.TestCase):
    def test_all_success_cp_bound_matches_exact_binomial_formula(self):
        self.assertAlmostEqual(cp_lower(50, 50, 0.05 / 9), (0.05 / 9) ** (1 / 50))
        self.assertEqual(cp_lower(0, 50, 0.05 / 9), 0.0)

    def test_audit_pair_uses_source_selection_and_target_test_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            fields = ["case", "distribution", "seed", "chosen", "oblivious_chosen", "selection_alpha", "n_configurations", "config_sha256"]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "case": "fft", "distribution": "light", "seed": 7, "chosen": "fft-32-1.0",
                    "oblivious_chosen": "fft-32-1.0", "selection_alpha": 0.05 / 9,
                    "n_configurations": 9, "config_sha256": "digest",
                })
            traces = root / "target.jsonl"
            records = [{"record_type": "metadata", "metadata": {"config_sha256": "digest"}}]
            for split, error in (("calibration", 1.0), ("test", 0.01), ("test", 0.02)):
                records.append({
                    "case": "fft", "distribution": "heavy", "seed_group": 7, "split": split,
                    "config": {"name": "fft-32-1.0"}, "error": error,
                })
            traces.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            rows = audit_pair("light", "heavy", source, traces)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["selection_data_used"] == "source calibration/validation only" for row in rows))
            self.assertTrue(all(row["n_target_test_observations"] == 2 for row in rows))
            self.assertTrue(all(row["target_coverage"] == 1.0 for row in rows))
            self.assertTrue(all(row["guarantee_status"] == "not_applicable_under_distribution_shift" for row in rows))

    def test_target_seed_overlap_check_detects_reused_trial_seed(self):
        from distribution_shift_audit import _target_seed_overlap

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.jsonl"
            records = [
                {"record_type": "metadata", "metadata": {}},
                {"case": "fft", "distribution": "heavy", "split": "test",
                 "seed": 123, "seed_group": 7},
                {"case": "fft", "distribution": "heavy", "split": "test",
                 "seed": 123, "seed_group": 11},
            ]
            path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            self.assertTrue(_target_seed_overlap(path))


if __name__ == "__main__":
    unittest.main()
