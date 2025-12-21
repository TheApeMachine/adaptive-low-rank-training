import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.model import GPT, ModelConfig
from production.runtime_tuning import KVSelfOptConfig


class TestSelfoptCachePolicyLogging(unittest.TestCase):
    def test_cache_policy_emits_structured_event(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            vocab_size=101,
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=16,
            sem_dim=8,
            geo_dim=8,
            dropout=0.0,
            tie_qk=False,
            null_attn=False,
        )
        m = GPT(cfg)

        prompt = torch.randint(0, int(cfg.vocab_size), (1, 8), dtype=torch.long)

        events = []

        def log_callback(ev):
            events.append(dict(ev))

        self_opt = KVSelfOptConfig(
            mode="startup",
            scope="cache",
            cache_path=None,
            residuals=(0,),
            qblocks=(32,),
            k_sem_kinds=("fp16",),
            k_geo_kinds=("fp16",),
            v_kinds=("fp16",),
            mem_overhead_frac=0.0,
            policy_warmup=0,
            policy_iters=1,
            policy_hysteresis=0.0,
            prefer_lower_mem_within=0.0,
            policy_quality=False,
            policy_quality_long=False,
        )

        out = m.generate(
            prompt,
            max_new_tokens=1,
            temperature=1.0,
            top_k=None,
            kv_cache="fp16",
            kv_qblock=32,
            kv_residual=0,
            kv_decode_block=8,
            kv_fused="none",
            self_opt=self_opt,
            log_callback=log_callback,
        )
        self.assertEqual(tuple(out.shape), (1, 9))

        hits = [e for e in events if e.get("type") == "analysis" and e.get("subtype") == "selfopt_cache_policy"]
        self.assertTrue(hits, "Expected an analysis/selfopt_cache_policy event to be emitted via log_callback")
        ev = hits[-1]
        self.assertIn("chosen_policy", ev)
        self.assertIsInstance(ev["chosen_policy"], str)
        self.assertIn("base_policy", ev)
        self.assertIsInstance(ev["base_policy"], str)


if __name__ == "__main__":
    unittest.main()

