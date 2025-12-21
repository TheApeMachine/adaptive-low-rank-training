import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import DecoupledLayerKVCache, KVCacheTensorConfig, SeqCacheTensor
from production.runtime_tuning import KVCachePolicy, estimate_decoupled_kvcache_bytes, estimate_seq_cache_bytes


def _nbytes(t: object) -> int:
    if t is None:
        return 0
    if not isinstance(t, torch.Tensor):
        return 0
    return int(t.numel() * t.element_size())


def _seq_alloc_bytes(cache: SeqCacheTensor) -> int:
    return int(_nbytes(cache.buf) + _nbytes(cache.q) + _nbytes(cache.s) + _nbytes(getattr(cache, "_residual", None)))


class TestEstimateSeqCacheBytes(unittest.TestCase):
    def test_estimate_matches_allocated_tensor_bytes(self) -> None:
        dev = torch.device("cpu")
        B, L, D = 2, 11, 15

        for kind in ("fp16", "fp32", "q8_0", "q4_0", "nf4"):
            cfg = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=128)
            cache = SeqCacheTensor(batch_size=B, max_seq_len=L, dim=D, cfg=cfg, device=dev)
            est = estimate_seq_cache_bytes(batch_size=B, max_seq_len=L, dim=D, cfg=cfg)
            self.assertEqual(est, _seq_alloc_bytes(cache), msg=f"kind={kind}")


class TestEstimateDecoupledKVCacheBytes(unittest.TestCase):
    def test_estimate_matches_allocated_bytes_per_layer(self) -> None:
        dev = torch.device("cpu")
        B, L = 1, 8
        sem_dim, geo_dim, v_dim = 7, 10, 12
        policy = KVCachePolicy(
            k_sem_kind="q4_0",
            k_geo_kind="q8_0",
            v_kind="q4_0",
            k_sem_qblock=32,
            k_geo_qblock=32,
            v_qblock=32,
            residual_len=16,
        )
        k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
        cache = DecoupledLayerKVCache(
            batch_size=B,
            max_seq_len=L,
            k_sem_dim=sem_dim,
            k_geo_dim=geo_dim,
            v_dim=v_dim,
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            v_cfg=v_cfg,
            device=dev,
        )

        alloc = _seq_alloc_bytes(cache.k_sem) + _seq_alloc_bytes(cache.k_geo) + _seq_alloc_bytes(cache.v)
        est = estimate_decoupled_kvcache_bytes(
            n_layer=3,
            batch_size=B,
            max_seq_len=L,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            policy=policy,
        )
        self.assertEqual(est, 3 * int(alloc))


if __name__ == "__main__":
    unittest.main()

