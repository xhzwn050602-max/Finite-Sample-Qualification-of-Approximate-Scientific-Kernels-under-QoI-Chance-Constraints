import unittest

from theory_audit import audit


class TheoryAuditTest(unittest.TestCase):
    def test_cp_formula_and_implementation_agree(self):
        result = audit()
        self.assertTrue(result["all_checks_pass"])
        self.assertEqual([row["minimal_all_success_n"] for row in result["rows"]], [50, 53])


if __name__ == "__main__":
    unittest.main()
