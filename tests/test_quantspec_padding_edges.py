import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import make_quantspec


class TestQuantSpecPaddingEdges(unittest.TestCase):
    def test_q4_requires_even_pad_dim_and_even_qblock(self) -> None:
        # Odd dims should be padded and the pad_dim must end up even for q4 packing.
        spec = make_quantspec("q4_0", dim=15, qblock=32)
        self.assertEqual(spec.kind, "q4_0")
        self.assertEqual(spec.dim, 15)
        self.assertEqual(spec.pad_dim % 2, 0)
        self.assertEqual(spec.qblock % 2, 0)

    def test_nf4_requires_dim_ge_2(self) -> None:
        with self.assertRaises(ValueError):
            _ = make_quantspec("nf4", dim=1, qblock=32)

    def test_q8_allows_any_dim(self) -> None:
        spec = make_quantspec("q8_0", dim=1, qblock=32)
        self.assertEqual(spec.dim, 1)
        self.assertGreaterEqual(spec.pad_dim, 1)


if __name__ == "__main__":
    unittest.main()


