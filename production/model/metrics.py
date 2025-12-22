"""
metrics provides quality math and sampling for generation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

class Metrics:
    """Judge and sampler for model outputs."""
    @staticmethod
    def sampling_probs(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return sampling distribution p(x) after applying (temperature, top_k).

        The returned probabilities are float32 and safe for torch.multinomial.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be (B,V) but got shape={tuple(logits.shape)}")
        if not torch.isfinite(torch.tensor(float(temperature))):
            raise ValueError(f"temperature must be finite (got {temperature!r})")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0 for sampling_probs()")

        x = logits.float() / float(temperature)
        if top_k is not None:
            k = int(top_k)
            if k > 0 and k < int(x.size(-1)):
                vals = torch.topk(x, k, dim=-1).values
                cutoff = vals[:, -1].unsqueeze(-1)
                x = torch.where(x >= cutoff, x, torch.full_like(x, -float("inf")))

        p = torch.softmax(x, dim=-1)
        p_sum = p.sum(dim=-1, keepdim=True)
        ok = torch.isfinite(p_sum) & (p_sum > float(eps))
        if not bool(ok.all()):
            x2 = logits.float() / float(temperature)
            p2 = torch.softmax(x2, dim=-1)
            p = torch.where(ok, p, p2)
            p_sum = p.sum(dim=-1, keepdim=True)

        return p / p_sum.clamp_min(float(eps))

    @staticmethod
    def sample(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample with (temperature, top_k) and return (token, probs)."""
        p = Metrics.sampling_probs(logits, temperature=float(temperature), top_k=top_k)
        return torch.multinomial(p, 1), p

    @staticmethod
    def verify(
        main_next: torch.Tensor,
        main_block: torch.Tensor,
        proposed: torch.Tensor,
        q_probs: list[torch.Tensor],
        *,
        temperature: float,
        top_k: int | None,
        eps: float = 1e-8,
    ) -> tuple[int, torch.Tensor]:
        """Parallel verification for speculative decoding (rejection sampling)."""
        k = proposed.size(1)
        for i in range(k):
            # 1. Get main model distribution for the current step
            p = Metrics.sampling_probs(
                (main_next if i == 0 else main_block[:, i - 1, :]),
                temperature=float(temperature),
                top_k=top_k,
                eps=float(eps),
            )
            q = q_probs[i].float()

            # 2. Acceptance probability: p(x)/q(x)
            token = proposed[:, i : i+1]
            p_tok = p.gather(-1, token)
            q_tok = q.gather(-1, token).clamp_min(float(eps))

            # 3. Rejection check
            if torch.rand_like(p_tok) > (p_tok / q_tok).clamp(max=1.0):
                # Sample from normalized difference: norm(max(0, p - q))
                diff = (p - q).clamp(min=0)
                diff_sum = diff.sum(dim=-1, keepdim=True)
                ok = torch.isfinite(diff_sum) & (diff_sum > float(eps))
                if not bool(ok.all()):
                    next_tok = torch.multinomial(p, 1)
                else:
                    next_tok = torch.multinomial(diff / diff_sum, 1)
                return i, next_tok

        # 4. All accepted: Sample the next token from the final main distribution
        p_final = Metrics.sampling_probs(
            main_block[:, -1, :],
            temperature=float(temperature),
            top_k=top_k,
            eps=float(eps),
        )
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
        if tuple(lb.shape) != tuple(lt.shape):
            raise ValueError(
                f"lb and lt must have the same shape (got lb.shape={tuple(lb.shape)} vs lt.shape={tuple(lt.shape)})"
            )
        if lb.ndim < 2:
            raise ValueError(f"lb must have at least 2 dimensions (got lb.ndim={int(lb.ndim)})")
        if lt.ndim < 2:
            raise ValueError(f"lt must have at least 2 dimensions (got lt.ndim={int(lt.ndim)})")

        if lb.ndim == 2:
            lb2 = lb.float()
            lt2 = lt.float()
        else:
            lb2 = lb[:, -1, :].float()
            lt2 = lt[:, -1, :].float()

        if tgt.ndim == 2:
            tgt_last = tgt[:, -1]
        elif tgt.ndim == 1:
            tgt_last = tgt
        else:
            raise ValueError(f"tgt must be (B,) or (B,T) but got shape={tuple(tgt.shape)}")
        if tgt_last.ndim != 1:
            raise ValueError(f"tgt_last must be 1D but got shape={tuple(tgt_last.shape)}")
        if int(tgt_last.size(0)) != int(lb2.size(0)):
            raise ValueError(
                f"tgt_last batch must match logits batch (got {int(tgt_last.size(0))} vs {int(lb2.size(0))})"
            )
        if tgt_last.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"tgt must contain integer class indices (got dtype={tgt_last.dtype})")

        ce_base = F.cross_entropy(lb2, tgt_last).detach()
        ce_test = F.cross_entropy(lt2, tgt_last).detach()
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
            p_base = F.softmax(lb2, dim=-1)
            kl = (p_base * (logp_base - logp_test)).sum(dim=-1).mean()
            out["kl_base_cand"] = float(kl.item())

        return out
