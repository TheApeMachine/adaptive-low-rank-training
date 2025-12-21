import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import DecoupledLayerKVCache, KVCacheTensorConfig
from production.model import GPT, ModelConfig


class TestDecoupledIncrementalEquivalence(unittest.TestCase):
    def _cfg(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=128,
            block_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
        )

    def _make_caches(self, cfg: ModelConfig, *, B: int, max_seq: int, kind: str, residual_len: int) -> list[DecoupledLayerKVCache]:
        ks = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=residual_len)
        kg = KVCacheTensorConfig(kind=("q8_0" if kind == "q4_0" else kind), qblock=32, residual_len=residual_len)
        vv = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=residual_len)
        caches: list[DecoupledLayerKVCache] = []
        for _ in range(int(cfg.n_layer)):
            caches.append(
                DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=int(cfg.sem_dim),
                    k_geo_dim=int(cfg.geo_dim),
                    v_dim=int(cfg.attn_dim),
                    k_sem_cfg=ks,
                    k_geo_cfg=kg,
                    v_cfg=vv,
                    device=torch.device("cpu"),
                )
            )
        return caches

    def test_decoupled_incremental_matches_full(self) -> None:
        torch.manual_seed(0)
        cfg = self._cfg()
        m = GPT(cfg).eval()
        B, T = 2, 12
        x = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

        # Full forward (no caches).
        logits_full, _ = m(x)

        # Incremental decode with fp16 caches.
        caches = self._make_caches(cfg, B=B, max_seq=T, kind="fp16", residual_len=0)
        logits_steps = []
        pos = 0
        for t in range(T):
            tok = x[:, t : t + 1]
            lg, caches = m(tok, caches=caches, pos_offset=pos)
            logits_steps.append(lg)
            pos += 1
        logits_inc = torch.cat(logits_steps, dim=1)

        self.assertEqual(tuple(logits_inc.shape), tuple(logits_full.shape))
        mx = float((logits_inc.float() - logits_full.float()).abs().max().item())
        # decoupled cache path uses fp16 KV; allow small tolerance.
        self.assertLess(mx, 5e-2)

    def test_decoupled_cache_truncate_rollback_is_behaviorally_stable(self) -> None:
        # This approximates speculative decoding rollback:
        #  - build prefix caches
        #  - append a "draft" segment and record logits
        #  - truncate caches back to prefix
        #  - append the same draft again; logits should match
        torch.manual_seed(0)
        cfg = self._cfg()
        m = GPT(cfg).eval()

        B = 1
        prefix_len = 6
        draft_len = 4
        x = torch.randint(0, cfg.vocab_size, (B, prefix_len + draft_len), dtype=torch.long)

        # Use quantized caches + residual tail to stress the rollback code paths.
        caches = self._make_caches(cfg, B=B, max_seq=prefix_len + draft_len + 2, kind="q4_0", residual_len=4)

        # Prefill prefix incrementally
        pos = 0
        for t in range(prefix_len):
            tok = x[:, t : t + 1]
            _lg, caches = m(tok, caches=caches, pos_offset=pos)
            pos += 1

        # Draft segment pass #1
        logits1 = []
        pos0 = pos
        for t in range(prefix_len, prefix_len + draft_len):
            tok = x[:, t : t + 1]
            lg, caches = m(tok, caches=caches, pos_offset=pos)
            logits1.append(lg)
            pos += 1
        logits1 = torch.cat(logits1, dim=1)

        # Rollback caches
        for c in caches:
            c.truncate(pos0)
        pos = pos0

        # Draft segment pass #2
        logits2 = []
        for t in range(prefix_len, prefix_len + draft_len):
            tok = x[:, t : t + 1]
            lg, caches = m(tok, caches=caches, pos_offset=pos)
            logits2.append(lg)
            pos += 1
        logits2 = torch.cat(logits2, dim=1)

        self.assertEqual(tuple(logits1.shape), tuple(logits2.shape))
        mx = float((logits1.float() - logits2.float()).abs().max().item())
        # Quantization introduces some noise; but rollback should restore identical *behavior* given same prefix.
        self.assertLess(mx, 1e-1)


if __name__ == "__main__":
    unittest.main()


