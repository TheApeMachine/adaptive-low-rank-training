import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention_impl.decoupled_attention_impl.helpers import neg_inf
from production.kvcache_backend import make_quantspec, quantize_nf4, quantize_q4_0, quantize_q8_0


class TestNumericSentinels(unittest.TestCase):
    def test_neg_inf_underflows_in_exp(self) -> None:
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.tensor(neg_inf(dt), dtype=dt)
            y = torch.exp(x)
            self.assertEqual(float(y.to(torch.float32).item()), 0.0)

    def test_neg_inf_is_safe_for_softmax_masking(self) -> None:
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            scores = torch.randn((2, 7), dtype=dt)
            scores[:, 3] = torch.tensor(neg_inf(dt), dtype=dt)
            p = torch.softmax(scores, dim=-1)
            self.assertTrue(bool(torch.isfinite(p).all()))
            # masked position should be ~0 probability
            self.assertTrue(bool((p[:, 3] == 0).all()) or bool((p[:, 3] < 1e-7).all()))

    def test_quant_scales_never_underflow_to_zero(self) -> None:
        # Use all-zeros; clamp must ensure nonzero fp16 scales.
        x = torch.zeros((2, 5, 32), dtype=torch.float16)
        tiny = float(torch.finfo(torch.float16).tiny)

        spec8 = make_quantspec("q8_0", dim=32, qblock=16)
        _q8, s8 = quantize_q8_0(x, spec8)
        self.assertTrue(bool((s8 > 0).all()))
        self.assertTrue(bool((s8.to(torch.float32) >= tiny).all()))

        spec4 = make_quantspec("q4_0", dim=32, qblock=16)
        _q4, s4 = quantize_q4_0(x, spec4)
        self.assertTrue(bool((s4 > 0).all()))
        self.assertTrue(bool((s4.to(torch.float32) >= tiny).all()))

        specn = make_quantspec("nf4", dim=32, qblock=16)
        _qn, sn = quantize_nf4(x, specn)
        self.assertTrue(bool((sn > 0).all()))
        self.assertTrue(bool((sn.to(torch.float32) >= tiny).all()))


if __name__ == "__main__":
    unittest.main()


