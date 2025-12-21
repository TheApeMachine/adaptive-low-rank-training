import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.rope import RotaryEmbedding


class TestRoPEInvariants(unittest.TestCase):
    def test_rotate_preserves_pairwise_l2_norm(self) -> None:
        # RoPE is a blockwise rotation; it should preserve L2 norm of each rotated pair.
        torch.manual_seed(0)
        B, H, T, D = 2, 3, 5, 12
        rot_dim = 8
        rope = RotaryEmbedding(rot_dim=rot_dim)
        x = torch.randn((B, H, T, D), dtype=torch.float32)

        y = rope.rotate(x, pos_offset=0)
        # RotaryEmbedding stores pairs as two contiguous halves:
        #   x_rot = [x1 (rot/2), x2 (rot/2)] and rotates (x1_i, x2_i) for each i.
        x_rot = x[..., :rot_dim]
        y_rot = y[..., :rot_dim]
        x1, x2 = x_rot[..., : rot_dim // 2], x_rot[..., rot_dim // 2 : rot_dim]
        y1, y2 = y_rot[..., : rot_dim // 2], y_rot[..., rot_dim // 2 : rot_dim]
        x_norm = (x1 * x1) + (x2 * x2)
        y_norm = (y1 * y1) + (y2 * y2)
        self.assertTrue(bool(torch.allclose(x_norm, y_norm, atol=1e-5, rtol=1e-5)))

    def test_cache_grows_amortized_pow2(self) -> None:
        rope = RotaryEmbedding(rot_dim=8)
        dev = torch.device("cpu")
        dt = torch.float32
        # First request: len=3 -> cache should allocate pow2=4.
        _ = rope._cos_sin(3, dev, dt)
        key = (str(dev), str(dt))
        cos, _sin = rope._cache[key]
        self.assertEqual(int(cos.size(0)), 4)

        # Next request: len=5 -> cache should grow to pow2=8.
        _ = rope._cos_sin(5, dev, dt)
        cos2, _sin2 = rope._cache[key]
        self.assertEqual(int(cos2.size(0)), 8)


if __name__ == "__main__":
    unittest.main()


