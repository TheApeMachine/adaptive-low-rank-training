import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import KVCacheTensorConfig, LayerKVCache
from production.model import GPT, ModelConfig


class TestIncrementalCacheEquivalence(unittest.TestCase):
    def _cfg(self, *, mode: str) -> ModelConfig:
        tie_qk = True
        # production/attention.py disallows tie_qk for gqa unless kv_head == n_head.
        if str(mode) == "gqa":
            tie_qk = False
        return ModelConfig(
            vocab_size=128,
            block_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode=str(mode),
            attn_dim=32,
            kv_head=2 if mode == "gqa" else None,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=bool(tie_qk),
        )

    def _make_std_caches(self, cfg: ModelConfig, *, B: int, max_seq: int) -> list[LayerKVCache]:
        k_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        v_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        if cfg.attn_mode == "standard":
            dim = int(cfg.d_model)
        elif cfg.attn_mode == "gqa":
            head_dim = int(cfg.attn_dim) // int(cfg.n_head)
            dim = int((cfg.kv_head or cfg.n_head) * head_dim)
        else:
            dim = int(cfg.attn_dim)
        return [LayerKVCache(batch_size=B, max_seq_len=max_seq, k_dim=dim, v_dim=dim, k_cfg=k_cfg, v_cfg=v_cfg, device=torch.device("cpu")) for _ in range(int(cfg.n_layer))]

    def _assert_incremental_matches_full(self, *, mode: str) -> None:
        torch.manual_seed(0)
        cfg = self._cfg(mode=mode)
        m = GPT(cfg).eval()
        B, T = 2, 10
        x = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

        # Full forward (no caches)
        logits_full, _ = m(x)

        # Incremental: prefill token-by-token into caches.
        caches = self._make_std_caches(cfg, B=B, max_seq=T)
        logits_steps = []
        pos = 0
        for t in range(T):
            tok = x[:, t : t + 1]
            lg, caches = m(tok, caches=caches, pos_offset=pos)
            logits_steps.append(lg)
            pos += 1
        logits_inc = torch.cat(logits_steps, dim=1)

        self.assertEqual(tuple(logits_inc.shape), tuple(logits_full.shape))
        # Tolerance: cache path uses fp16 KV; still should be very close for this tiny model.
        mx = float((logits_inc.float() - logits_full.float()).abs().max().item())
        self.assertLess(mx, 5e-2)

    def test_standard_incremental_matches_full(self) -> None:
        self._assert_incremental_matches_full(mode="standard")

    def test_bottleneck_incremental_matches_full(self) -> None:
        self._assert_incremental_matches_full(mode="bottleneck")

    def test_gqa_incremental_matches_full(self) -> None:
        self._assert_incremental_matches_full(mode="gqa")


if __name__ == "__main__":
    unittest.main()


