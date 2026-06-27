import unittest


class Exp36ScaffoldTest(unittest.TestCase):
    def test_scaffold_imports(self):
        import exp36_minimax_objective_rescue.code as code

        self.assertEqual(code.EXP36_STATUS, "EXP36_READBACK_COMPLETED")


if __name__ == "__main__":
    unittest.main()

