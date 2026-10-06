import tempfile
import unittest
from pathlib import Path

from call_count_audit import audit


class CallCountAuditTest(unittest.TestCase):
    def test_replay_has_expected_shape_and_monotone_call_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = audit(
                [Path("publication-light-traces.jsonl")],
                root / "audit.json",
                root / "audit.csv",
                call_counts=(1, 10),
            )
            self.assertEqual(result["rows"], 5 * 10 * 2)
            self.assertEqual(result["summary_rows"], 5 * 2)
            self.assertEqual(result["config_sha256"], "5dcd8768e92253cbbad7169be1999293bbd0a16a5817cc2a6e387ff978d997e9")
            self.assertTrue((root / "audit.json").exists())
            self.assertTrue((root / "audit.csv").exists())


if __name__ == "__main__":
    unittest.main()
