# Full SDD workflow

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Workflow Steps

### [x] Step: Requirements
<!-- chat-id: 5cb709d6-7c43-4968-a8bd-ab993a56a7fa -->

Create a Product Requirements Document (PRD) based on the feature description.

1. Review existing codebase to understand current architecture and patterns
2. Analyze the feature definition and identify unclear aspects
3. Ask the user for clarifications on aspects that significantly impact scope or user experience
4. Make reasonable decisions for minor details based on context and conventions
5. If user can't clarify, make a decision, state the assumption, and continue

Save the PRD to `{@artifacts_path}/requirements.md`.

### [x] Step: Technical Specification

Create a technical specification based on the PRD in `{@artifacts_path}/requirements.md`.

1. Review existing codebase architecture and identify reusable components
2. Define the implementation approach

Save to `{@artifacts_path}/spec.md` with:
- Technical context (language, dependencies)
- Implementation approach referencing existing code patterns
- Source code structure changes
- Data model / API / interface changes
- Delivery phases (incremental, testable milestones)
- Verification approach using project lint/test commands

### [x] Step: Planning
<!-- chat-id: 49c351d5-4a6c-49ce-ab33-f2bece136689 -->

Create a detailed implementation plan based on `{@artifacts_path}/spec.md`.

1. Break down the work into concrete tasks
2. Each task should reference relevant contracts and include verification steps
3. Replace the Implementation step below with the planned tasks

Rule of thumb for step size: each step should represent a coherent unit of work (e.g. implement a component, add an API endpoint, write tests for a module). Avoid steps that are too granular (single function) or too broad (entire feature).

If the feature is trivial and doesn't warrant full specification, update this workflow to remove unnecessary steps and explain the reasoning to the user.

Save to `{@artifacts_path}/plan.md`.

---

## Implementation Plan (PR-sized slices)

### [ ] Step: Fix decoupled null `out_proj` duplication

- **Files:** `production/attention_impl/decoupled_attention_impl/attention_core.py`
- **Bug:** Manual decoupled `null_attn` training/full-attn path applies `out_proj` twice.
- **Acceptance:** `out_proj` is applied exactly once; no behavior depends on accidental double-projection.
- **Verification:**
  - `make test`
  - Add/extend a unit test that would have caught double-application (e.g., compare against a reference path or assert that the forward equals a single-projection reference when dropout=0).

### [ ] Step: Make long-seq approximation knobs explicit (single runtime source)

- **Files:** `production/model/config.py`, `production/runner_train_impl/trainer.py`, `production/attention_impl/decoupled_attention_impl/attention_core.py`
- **Contract:** Attention must not rely on `getattr(cfg, ...)` for undeclared/hidden config.
- **Approach:**
  - Add explicit long-seq fields to `ModelConfig`:
    - `train_long_seq_enabled`, `train_long_seq_threshold`, `train_long_seq_mem_block`, `train_long_seq_local_window`, `train_long_seq_q_chunk`, and new `train_long_seq_mem_summarizer`.
  - Wire these fields during training model config construction (`Trainer._build_model_cfg`).
  - Update `DecoupledBottleneckAttention` to read the explicit fields (remove `getattr` fallback).
- **Acceptance:**
  - The long-seq branch remains gated to training + `attn_mask is None` and preserves current defaults.
  - `resolved_config.json` includes the long-seq fields.
- **Verification:**
  - `make test`
  - Add/update a config round-trip test ensuring these fields survive `ModelConfig.from_dict()` and influence the attention branch selection.

### [ ] Step: Learned memory summarizer (mean/linear/conv) for long-seq path

- **Files:** `production/attention_impl/decoupled_attention_impl/attention_core.py` (optionally factor into a small helper module under `production/attention_impl/decoupled_attention_impl/`)
- **Contract:** Preserve the additive-logit invariant by keeping a single SDPA over a single token axis: semantic-only memory tokens + full-res local tokens.
- **Implement:**
  - Add summarizer choices: `mean` (existing), `linear`, `conv`.
  - Replace the current `.mean(dim=3)` memory builder with the selected summarizer.
  - Initialize learned summarizers to match mean at init (exactly or to tight tolerance):
    - Prefer “mean + residual” parameterization with residual scale initialized to 0.
  - Default: `conv` on CUDA; fallback to `linear` or `mean` where conv is unsupported.
- **Acceptance:**
  - Output shapes match existing long-seq implementation.
  - Default initialization produces negligible drift vs mean summarizer.
- **Verification:**
  - `make test`
  - Add a unit test that runs the long-seq branch and asserts:
    - shape invariants
    - mean-matching at init (`linear/conv` equals mean for identical inputs, within tolerance)

### [ ] Step: Causal input-conditioned sem/geo gating (default on)

- **Files:** `production/attention_impl/decoupled_attention_impl/attention_core.py`, `production/model/config.py`
- **Contracts:**
  - Strictly causal (no sequence pooling across future tokens).
  - Cache-compatible (must not change how cached K/V are computed or stored).
- **Implement:**
  - Compute a per-token per-head gate `g(x_t) ∈ [0,1]^H` from the current-token hidden state.
  - Apply as query scaling (preserves compatibility with SDPA, streaming decode, and fused decode):
    - `q_sem *= 2*g`, `q_geo *= 2*(1-g)`.
  - Neutral init (`g=0.5`) and checkpoint-compatibility:
    - Either keep `decoupled_gate_logit` as a bias term or provide an explicit migration path.
- **Acceptance:**
  - No train/infer mismatch.
  - Gating is per-token (not sequence-mean).
- **Verification:**
  - `make test`
  - Add regression tests for shape/dtype stability and for “gate depends only on token-local features” (no reduction over the sequence dimension).

### [ ] Step: Null token integration into Triton fused decode (1-pass)

- **Files:** `production/attention_impl/decoupled_attention_impl/kernels_q4q8q4.py`, `production/attention_impl/decoupled_attention_impl/attention_core.py`
- **Kernel contract:**
  - Extend `kv_decode_update_decoupled_q4q8q4` to accept `k_sem_null_ptr`, `k_geo_null_ptr`, `v_null_ptr` and `HAS_NULL: tl.constexpr`.
  - Include strides for null tensors (or enforce/verify contiguous layout).
  - Compile out null logic when `HAS_NULL=False`.
- **Correct math:** `s_null = dot(q_sem, k_sem_null) * SEM_SCALE + dot(q_geo, k_geo_null) * GEO_SCALE`.
- **Correct seeding:** Initialize online-softmax state from null exactly once per decode step:
  - Must work even when `L_prefix == 0`.
  - Must not duplicate null when the update kernel is launched multiple times across prefix blocks.
- **Python plumbing:**
  - Remove fused decode guard that forbids `null_attn`.
  - Pass null tensors/flags into kernel launches.
  - Update fused eligibility checks so `null_attn=True` is allowed when the null-capable kernels are present.
- **Acceptance:** Fused decode with `null_attn=True` matches streaming decode within tolerance for the same inputs/cache.
- **Verification:**
  - Add a CUDA-only parity test comparing fused vs streaming decode with `null_attn=True` for 1-pass mode.

### [ ] Step: Null token integration into Triton fused decode (2-pass)

- **Files:** `production/attention_impl/decoupled_attention_impl/kernels_q4q8q4.py`, `production/attention_impl/decoupled_attention_impl/attention_core.py`
- **Correctness requirement:** Null must be included exactly once globally (not once per partition).
- **Preferred approach (partition-invariant):** Incorporate null into `kv_decode_reduce_partitions` as a “virtual partition”:
  - `m = max(max_p m_part[p], s_null)`
  - `d = sum_p d_part[p] * exp(m_part[p]-m) + exp(s_null - m)`
  - `o = sum_p o_part[p] * exp(m_part[p]-m) + v_null * exp(s_null - m)`
  - Keep all null work behind `HAS_NULL: tl.constexpr`.
- **Acceptance:** Partitioned fused decode with `null_attn=True` matches streaming decode within tolerance.
- **Verification:**
  - Extend parity tests to cover 2-pass mode.
  - Include cases where `P > 0` and `P == 0`.

### [ ] Step: KV policy escape hatch in minimal CLI (`--kv-policy`)

- **Files:** `production/cli.py`, `production/run_config.py`, `production/runner_sample.py`
- **Behavior:**
  - Add a single advanced flag `--kv-policy`.
  - If provided, parse with `KVCachePolicy.parse()` and treat as an atomic override over any derived/default KV flags.
- **UX:** Improve `_MinimalParser.error()` to detect legacy KV flags (e.g. `--kv-cache-k-sem`) and print: `Use --kv-policy for a unified configuration.`
- **Acceptance:** Minimal CLI remains intent-first while still allowing expert override.
- **Verification:**
  - `make test`
  - Extend `tests/test_cli_parse_args.py` and/or KV policy tests to cover:
    - successful parsing
    - error hint for deprecated flags

### [ ] Step: Update training run summaries (KV policy + null attention state)

- **Files:** `production/runner_train_impl/summary.py`
- **Add rows:**
  - `KV Policy`: short-string representation if known.
  - `Null Attention`: `Inactive` / `Active (Unfused)` / `Active (Fused)`.
- **Acceptance:** Summary reflects fused capability after Triton null integration.
- **Verification:**
  - `make test`
  - Add/update a narrow unit test around the summary emitter if one exists; otherwise add a focused test for the new rows.

### [ ] Step: Add long-seq drift validator (metrics-style, feasible)

- **Files:** Add `production/validate_long_seq_drift.py`
- **Approach:**
  - Reuse the existing logit-fidelity metrics approach in `production/optimizer/tuner/metrics.py`.
  - Avoid OOM by using an exact reference at smaller context and/or chunking; keep dropout disabled for comparability.
- **Acceptance:** Script runs end-to-end on a checkpoint and reports metric deltas in a stable, documented format.
- **Verification:**
  - Ensure importability under `make test`.
  - Include a representative invocation in `--help`.

### [ ] Step: Final verification

- **CPU-only:** `make test`
- **CUDA (if available):** Run the new fused-vs-streaming parity tests for both 1-pass and 2-pass kernels with `null_attn=True`.
