import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import (
    dequantize_nf4,
    dequantize_q4_0,
    dequantize_q8_0,
    make_quantspec,
    quantize_nf4,
    quantize_q4_0,
    quantize_q8_0,
)


class TestQuantRoundtripBounds(unittest.TestCase):
    def test_q8_roundtrip_error_is_bounded(self) -> None:
        torch.manual_seed(0)
        spec = make_quantspec("q8_0", dim=32, qblock=16)
        x = torch.randn((2, 5, 32), dtype=torch.float16) * 0.5
        q, s = quantize_q8_0(x, spec)
        x2 = dequantize_q8_0(q, s, spec).to(torch.float16)
        err = (x2 - x).abs()
        # For symmetric int8 with scale=amax/127, worst-case per-element error is <= scale.
        # Use a small multiple for safety across fp16 casts.
        s_eff = s.to(torch.float32).unsqueeze(-1).repeat(1, 1, 1, spec.qblock).reshape_as(err.to(torch.float32))
        self.assertTrue(bool((err.to(torch.float32) <= (2.5 * s_eff + 1e-6)).all()))

    def test_q4_roundtrip_error_is_bounded(self) -> None:
        torch.manual_seed(0)
        spec = make_quantspec("q4_0", dim=32, qblock=16)
        x = torch.randn((2, 5, 32), dtype=torch.float16) * 0.5
        q, s = quantize_q4_0(x, spec)
        x2 = dequantize_q4_0(q, s, spec).to(torch.float16)
        err = (x2 - x).abs()
        # For q4 with scale=amax/7, error <= scale/2 (rounding) in ideal math;
        # use a conservative multiple for fp16 casts and packing.
        s_eff = s.to(torch.float32).unsqueeze(-1).repeat(1, 1, 1, spec.qblock).reshape_as(err.to(torch.float32))
        self.assertTrue(bool((err.to(torch.float32) <= (4.0 * s_eff + 1e-6)).all()))

    def test_nf4_roundtrip_is_finite(self) -> None:
        torch.manual_seed(0)
        spec = make_quantspec("nf4", dim=32, qblock=16)
        x = torch.randn((2, 5, 32), dtype=torch.float16)
        q, s = quantize_nf4(x, spec)
        x2 = dequantize_nf4(q, s, spec)
        self.assertTrue(bool(torch.isfinite(x2).all()))


if __name__ == "__main__":
    unittest.main()


