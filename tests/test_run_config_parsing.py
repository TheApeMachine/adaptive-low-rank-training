import unittest

from production.run_config import _as_opt_int


class TestRunConfigParsing(unittest.TestCase):
    def test_as_opt_int_parses_integral_float_strings(self) -> None:
        self.assertEqual(_as_opt_int("3.0"), 3)
        self.assertEqual(_as_opt_int(3.0), 3)

    def test_as_opt_int_rejects_non_integral_floats(self) -> None:
        self.assertIsNone(_as_opt_int("3.5"))
        self.assertIsNone(_as_opt_int(3.5))

    def test_as_opt_int_parses_ints(self) -> None:
        self.assertEqual(_as_opt_int("7"), 7)
        self.assertEqual(_as_opt_int(7), 7)

    def test_as_opt_int_none_and_invalid(self) -> None:
        self.assertIsNone(_as_opt_int(None))
        self.assertIsNone(_as_opt_int("not-a-number"))


if __name__ == "__main__":
    unittest.main()


