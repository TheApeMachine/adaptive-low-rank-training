import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import DecoupledLayerKVCache, KVCacheTensorConfig, LayerKVCache, SeqCacheTensor


class TestKVCacheSync(unittest.TestCase):
    def test_layer_cache_append_and_truncate_keeps_kv_in_sync(self) -> None:
        dev = torch.device("cpu")
        k_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        v_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        c = LayerKVCache(batch_size=1, max_seq_len=8, k_dim=4, v_dim=6, k_cfg=k_cfg, v_cfg=v_cfg, device=dev)

        k1 = torch.randn((1, 3, 4), dtype=torch.float16)
        v1 = torch.randn((1, 3, 6), dtype=torch.float16)
        old = c.append(k1, v1)
        self.assertEqual(old, 0)
        self.assertEqual(c.pos, 3)

        c.truncate(1)
        self.assertEqual(c.pos, 1)

        # Truncating beyond current pos should error.
        with self.assertRaises(ValueError):
            c.truncate(99)

    def test_decoupled_cache_append_and_truncate_keeps_all_in_sync(self) -> None:
        dev = torch.device("cpu")
        ks = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=4)
        kg = KVCacheTensorConfig(kind="q8_0", qblock=32, residual_len=4)
        vv = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=4)
        c = DecoupledLayerKVCache(
            batch_size=1,
            max_seq_len=16,
            k_sem_dim=6,
            k_geo_dim=8,
            v_dim=10,
            k_sem_cfg=ks,
            k_geo_cfg=kg,
            v_cfg=vv,
            device=dev,
        )

        k_sem = torch.randn((1, 5, 6), dtype=torch.float16)
        k_geo = torch.randn((1, 5, 8), dtype=torch.float16)
        v = torch.randn((1, 5, 10), dtype=torch.float16)
        old = c.append(k_sem, k_geo, v)
        self.assertEqual(old, 0)
        self.assertEqual(c.pos, 5)

        c.truncate(2)
        self.assertEqual(c.pos, 2)

        with self.assertRaises(ValueError):
            c.truncate(-1)


class TestSeqCacheResidualWindow(unittest.TestCase):
    def test_quantized_cache_residual_window_returns_exact_tail(self) -> None:
        # For quantized caches with a residual fp16 tail, get_slice() should return the exact
        # newest tokens from the residual buffer (no quantization error on the tail).
        dev = torch.device("cpu")
        cfg = KVCacheTensorConfig(kind="q4_0", qblock=4, residual_len=4)
        t = SeqCacheTensor(batch_size=1, max_seq_len=16, dim=6, cfg=cfg, device=dev)

        # Append more than residual_len to force wrap.
        x = torch.randn((1, 7, 6), dtype=torch.float16)
        _ = t.append(x)
        self.assertEqual(t.pos, 7)

        tail = t.get_slice(3, 7, dtype=torch.float16)
        self.assertEqual(tuple(tail.shape), (1, 4, 6))
        # This slice is fully inside the residual window, so it should match exactly.
        self.assertTrue(bool(torch.equal(tail, x[:, -4:])))

    def test_truncate_rebuilds_residual_and_keeps_slices_valid(self) -> None:
        dev = torch.device("cpu")
        cfg = KVCacheTensorConfig(kind="q8_0", qblock=8, residual_len=4)
        t = SeqCacheTensor(batch_size=1, max_seq_len=16, dim=8, cfg=cfg, device=dev)

        x = torch.randn((1, 10, 8), dtype=torch.float16)
        _ = t.append(x)
        self.assertEqual(t.pos, 10)

        # After truncate, residual is rebuilt from authoritative storage; ensure it doesn't crash and slices are consistent.
        t.truncate(6)
        self.assertEqual(t.pos, 6)
        sl = t.get_slice(2, 6, dtype=torch.float16)
        self.assertEqual(tuple(sl.shape), (1, 4, 8))


if __name__ == "__main__":
    unittest.main()


