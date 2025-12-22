"""
metrics provides quality math and sampling for generation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

class Metrics:
    """Judge and sampler for model outputs."""
    @staticmethod
    def sample(logits: torch.Tensor, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample with temperature and return (token, probs)."""
        p = F.softmax(logits / max(1e-8, temp), dim=-1)
        return torch.multinomial(p, 1), p

    @staticmethod
    def verify(
        main_next: torch.Tensor,
        main_block: torch.Tensor,
        proposed: torch.Tensor,
        q_probs: list[torch.Tensor]
    ) -> tuple[int, torch.Tensor]:
        """Parallel verification for speculative decoding (rejection sampling)."""
        k = proposed.size(1)
        for i in range(k):
            # 1. Get main model distribution for the current step
            p = F.softmax((main_next if i == 0 else main_block[:, i-1, :]), dim=-1)
            q = q_probs[i]

            # 2. Acceptance probability: p(x)/q(x)
            token = proposed[:, i : i+1]
            p_tok = p.gather(-1, token)
            q_tok = q.gather(-1, token)

            # 3. Rejection check
            if torch.rand_like(p_tok) > (p_tok / q_tok).clamp(max=1.0):
                # Sample from normalized difference: norm(max(0, p - q))
                diff = (p - q).clamp(min=0)
                next_tok = torch.multinomial(diff / diff.sum(-1, keepdim=True), 1)
                return i, next_tok

        # 4. All accepted: Sample the next token from the final main distribution
        p_final = F.softmax(main_block[:, -1, :], dim=-1)
        return k, torch.multinomial(p_final, 1)

    @staticmethod
    def compare(
        lb: torch.Tensor,
        lt: torch.Tensor,
        tgt: torch.Tensor,
        *,
        compute_kl: bool = False,
    ) -> dict[str, float]:
        """Compare base vs test logits for quality gating.

        Returns:
        - max_abs_logit: max |Δlogit| over vocab (last position)
        - delta_nll: CE(test) - CE(base) in nats/token (last position)
        - ppl_ratio: exp(delta_nll) (equivalently PPL(test)/PPL(base))
        - kl_base_cand: KL(p_base || p_test) in nats/token (optional; expensive)
        """
        lb2 = lb[:, -1, :].float()
        lt2 = lt[:, -1, :].float()

        ce_base = F.cross_entropy(lb2, tgt).detach()
        ce_test = F.cross_entropy(lt2, tgt).detach()
        dnll_t = ce_test - ce_base
        dnll = float(dnll_t.item())

        out: dict[str, float] = {
            "max_abs_logit": float((lt2 - lb2).abs().max().item()),
            "delta_nll": dnll,
            "ppl_ratio": float(torch.exp(dnll_t).item()),
        }

        if compute_kl:
            # KL(p_base || p_test) = E_{p_base}[log p_base - log p_test]
            logp_base = F.log_softmax(lb2, dim=-1)
            logp_test = F.log_softmax(lt2, dim=-1)
            p_base = torch.exp(logp_base)
            kl = (p_base * (logp_base - logp_test)).sum(dim=-1).mean()
            out["kl_base_cand"] = float(kl.item())

        return out
