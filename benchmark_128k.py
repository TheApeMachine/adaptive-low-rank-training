#!/usr/bin/env python3
"""
benchmark_128k.py
Direct 128k context inference benchmark.

This script measures:
1. Actual memory usage at 128k context
2. Inference throughput (tokens/second)
3. Perplexity at 128k context
4. Comparison between Decoupled and Baseline architectures

Requires: MacBook Pro M4 Max with 128GB unified memory

Usage:
    python3 benchmark_128k.py --ckpt runs/context_1024/best.pt --context 131072
    python3 benchmark_128k.py --compare runs/context_1024/best.pt runs/baseline_context_1024/best.pt
"""
import argparse
import importlib
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from production.config import pick_device
from production.model import DecoupledLayerKVCache, GPT, LayerKVCache, ModelConfig
from production.selfopt_cache import as_object_list, as_object_pair2, as_str_object_dict

# Optional plotting dependency (avoid importing at module load so type checking doesn't require it).
if TYPE_CHECKING:
    HAS_MATPLOTLIB: bool = False
else:
    HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib.pyplot") is not None


def _torch_load_obj(path: str, *, device: torch.device) -> object:
    # torch.load is typed as Any in stubs; isolate it.
    return torch.load(str(path), map_location=device)  # pyright: ignore[reportAny]


def _as_state_dict(o: object) -> dict[str, torch.Tensor] | None:
    m = as_str_object_dict(o)
    if m is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for k, v in m.items():
        if not isinstance(v, torch.Tensor):
            return None
        out[str(k)] = v
    return out


def _load_checkpoint_and_model(ckpt_path: str, device: torch.device) -> tuple[ModelConfig, GPT]:
    """
    Load checkpoint from disk and initialize the model.

    Args:
        ckpt_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (ModelConfig, GPT) with the loaded and initialized model

    Raises:
        ValueError: If checkpoint is malformed or missing required keys
    """
    ckpt_obj = _torch_load_obj(str(ckpt_path), device=device)
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise ValueError("Checkpoint must be a dict-like object")
    cfg_map = as_str_object_dict(ckpt.get("config"))
    if cfg_map is None:
        raise ValueError("Checkpoint missing config")
    cfg = ModelConfig.from_dict(cfg_map, device=device)
    model = GPT(cfg).to(device)
    sd = _as_state_dict(ckpt.get("model"))
    if sd is None:
        raise ValueError("Checkpoint missing model state_dict")
    _ = model.load_state_dict(sd)
    _ = model.eval()
    return (cfg, model)


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB (MPS/CUDA)."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, estimate from system
        import subprocess
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=False
        )
        # Parse vm_stat output (rough approximation)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Pages active' in line:
                pages = int(line.split(':')[1].strip().rstrip('.'))
                return (pages * 16384) / (1024**3)  # 16KB pages to GB
        return 0.0
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def load_tokens(path: str, max_tokens: int = 200_000) -> torch.Tensor:
    """Load token file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tokens = [int(t) for t in content.split()[:max_tokens]]
        return torch.tensor(tokens, dtype=torch.long)
    except ValueError:
        print(f"Could not parse {path} as token file")
        sys.exit(1)


@dataclass
class BenchmarkResult:
    context_length: int
    model_name: str
    attn_mode: str
    memory_gb: float
    inference_time_ms: float
    tokens_per_sec: float
    loss: float
    perplexity: float


def benchmark_inference(
    model: GPT,
    tokens: torch.Tensor,
    context_length: int,
    device: torch.device,
    *,
    warmup_runs: int = 2,
    bench_runs: int = 5,
) -> BenchmarkResult | None:
    """
    Benchmark inference at a specific context length.
    """
    _ = model.eval()

    # Prepare input
    if len(tokens) < context_length + 1:
        print(f"Warning: Not enough tokens ({len(tokens)}) for context {context_length}")
        return None

    # Override block size to allow extrapolation
    if model.cfg.block_size < context_length:
        print(f"  [Extending block_size: {model.cfg.block_size} -> {context_length}]")
        model.cfg.block_size = context_length
        # Note: We do NOT resize self.causal_mask because creating a 128k x 128k bool tensor
        # exceeds INT_MAX elements (2^31) which crashes MPS/CUDA on some setups.
        # Since we use chunked inference (chunk_size=1024), we never need a mask larger
        # than the chunk size, so the original mask (e.g. 1024x1024) is sufficient
        # as long as we don't run a full dense forward pass.

    input_ids = tokens[:context_length].unsqueeze(0).to(device)
    target_ids = tokens[1:context_length + 1].unsqueeze(0).to(device)

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
    # We skip full-context warmup to avoid OOM.
    # Instead we just run a small dummy pass to wake up the GPU.
    dummy_input = input_ids[:, :128]
    for _ in range(int(warmup_runs)):
        with torch.no_grad():
            _logits_dummy, _caches_dummy = model.forward(dummy_input, caches=None, pos_offset=0)

    # Synchronize
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

    print("done")

    # Benchmark
    print(f"  Benchmarking ({bench_runs} runs)...", end=" ", flush=True)
    times: list[float] = []
    losses: list[float] = []

    # We must use chunked inference to avoid OOM on the full 128k matrix
    # processing the sequence in blocks of 'chunk_size' using the KV cache.
    chunk_size = 1024

    for _ in range(int(bench_runs)):
        total_loss = 0.0
        caches: list[DecoupledLayerKVCache | LayerKVCache] | None = None

        # We process the prompt in chunks.
        # For the first chunk, we compute standard attention.
        # For subsequent chunks, we use the cache.

        # NOTE: This effectively benchmarks "Prefill" speed if we pass large chunks,
        # or "Decode" speed if we pass 1 token at a time.
        # To measure 128k context support, we need to process the whole prompt
        # and see if it fits.

        # If we just want to test "can it handle 128k context inference",
        # we should process (Context-1) tokens to fill cache, then measure generation of last token.

        try:
            # 1. Fill Cache (Prefill) in chunks to avoid O(N^2) OOM
            # We don't time the prefill of the whole 128k for the "inference latency" metric,
            # we want to measure the time to generate *at* 128k.

            with torch.no_grad():
                caches = None

                # We only need to process up to the last token to measure the *next* token prediction
                # But to measure perplexity, we need loss on all tokens.
                # Let's separate the two:
                # 1. Perplexity: Sum of losses across chunks.
                # 2. Latency: Time to forward the last chunk.

                # Simplification: We will run the *last chunk* of size 128 to measure
                # performance at deep context.

                # A. Fast-forward cache (simulated) or just run forward on full sequence?
                # We can't run full sequence. We must chunk.

                # Iterate through chunks
                for i in range(0, context_length, chunk_size):
                    # End index for this chunk
                    end = min(i + chunk_size, context_length)

                    chunk_input = input_ids[:, i:end]

                    # Forward pass with cache
                    # Note: v21 model forward signature: forward(idx, caches=None, pos_offset=0)
                    chunk_logits, caches_out = model.forward(chunk_input, caches=caches, pos_offset=i)
                    caches = caches_out

                    # Compute loss for this chunk
                    # Targets match input shifted by 1.
                    # target_ids is tokens[1:context+1]
                    # chunk_target corresponds to target_ids[:, i:end]
                    chunk_target = target_ids[:, i:end]

                    loss = torch.nn.functional.cross_entropy(
                        chunk_logits.view(-1, chunk_logits.size(-1)),
                        chunk_target.view(-1),
                        reduction="sum",  # sum so we can average correctly later
                    )
                    total_loss += loss.item()

                    # Delete logits to save memory
                    del chunk_logits

                # Measure single-token generation time at full context
                # This is the "Inference Time" at 128k.
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                # Generate 1 token
                # Feed the last token again? No, feed a dummy token or next token
                dummy_next = target_ids[:, -1:]
                _logits_next, _caches_next = model.forward(dummy_next, caches=caches, pos_offset=context_length)

                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                times.append(t1 - t0)
                losses.append(total_loss / context_length)

        except RuntimeError as e:
            if "out of memory" in str(e) or "buffer size" in str(e):
                print(f"OOM at context {context_length}")
                return None
            raise e

    print("done")

    # Compute metrics
    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    # tokens_per_sec is now 1 / avg_time (since we timed 1 token generation)
    tokens_per_sec = 1.0 / avg_time
    memory_gb = get_memory_usage_gb()
    perplexity = math.exp(avg_loss)

    return BenchmarkResult(
        context_length=context_length,
        model_name=model.cfg.attn_mode,
        attn_mode=model.cfg.attn_mode,
        memory_gb=memory_gb,
        inference_time_ms=avg_time * 1000,
        tokens_per_sec=tokens_per_sec,
        loss=avg_loss,
        perplexity=perplexity
    )


def theoretical_kv_cache_size(cfg: ModelConfig, context_length: int,
                               quantization_bits: int = 16) -> float:
    """Calculate theoretical KV cache size in GB."""
    if cfg.attn_mode == 'decoupled':
        kv_dim = cfg.sem_dim + cfg.geo_dim
    elif cfg.attn_mode == 'bottleneck':
        kv_dim = cfg.attn_dim
    else:  # standard
        kv_dim = cfg.d_model

    # K + V, for all layers
    cache_elements = 2 * cfg.n_layer * context_length * kv_dim
    cache_bits = cache_elements * quantization_bits
    cache_gb = cache_bits / 8 / (1024**3)

    return cache_gb


def _as_str(o: object, default: str) -> str:
    try:
        s = str(o)
        return s
    except (TypeError, ValueError):
        return str(default)


def _as_int_list(o: object, default: list[int]) -> list[int]:
    lst = as_object_list(o)
    if lst is None:
        return list(default)
    out: list[int] = []
    for item in lst:
        try:
            # Handle numeric types directly
            if isinstance(item, (int, float)):
                out.append(int(item))
            else:
                # Try converting string to int directly
                try:
                    out.append(int(str(item)))
                except ValueError:
                    # If direct int conversion fails, try via float for float-like strings
                    out.append(int(float(str(item))))
        except (TypeError, ValueError):
            continue
    return out if out else list(default)


def _maybe_plot(results: list[BenchmarkResult], cfg: ModelConfig, out_path: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    subplot = getattr(plt, "subplot", None)
    figure = getattr(plt, "figure", None)
    plot = getattr(plt, "plot", None)
    xlabel = getattr(plt, "xlabel", None)
    ylabel = getattr(plt, "ylabel", None)
    title = getattr(plt, "title", None)
    xscale = getattr(plt, "xscale", None)
    yscale = getattr(plt, "yscale", None)
    grid = getattr(plt, "grid", None)
    axvline = getattr(plt, "axvline", None)
    axhline = getattr(plt, "axhline", None)
    legend = getattr(plt, "legend", None)
    tight_layout = getattr(plt, "tight_layout", None)
    savefig = getattr(plt, "savefig", None)
    close = getattr(plt, "close", None)
    suptitle = getattr(plt, "suptitle", None)

    if not (callable(subplot) and callable(figure) and callable(plot) and callable(savefig) and callable(close)):
        return

    contexts = [int(r.context_length) for r in results]
    times_ms = [float(r.inference_time_ms) for r in results]
    tok_s = [float(r.tokens_per_sec) for r in results]
    ppls = [float(r.perplexity) for r in results]

    _ = figure(figsize=(14, 10))

    _ = subplot(2, 2, 1)
    _ = plot(contexts, times_ms, "bo-", linewidth=2)
    if callable(xlabel):
        _ = xlabel("Context Length")
    if callable(ylabel):
        _ = ylabel("Inference Time (ms)")
    if callable(title):
        _ = title("Inference Time vs Context")
    if callable(xscale):
        _ = xscale("log", base=2)
    if callable(grid):
        _ = grid(True, alpha=0.3)

    _ = subplot(2, 2, 2)
    _ = plot(contexts, tok_s, "go-", linewidth=2)
    if callable(xlabel):
        _ = xlabel("Context Length")
    if callable(ylabel):
        _ = ylabel("Tokens/Second")
    if callable(title):
        _ = title("Throughput vs Context")
    if callable(xscale):
        _ = xscale("log", base=2)
    if callable(grid):
        _ = grid(True, alpha=0.3)

    _ = subplot(2, 2, 3)
    _ = plot(contexts, ppls, "ro-", linewidth=2)
    if callable(axvline):
        _ = axvline(x=int(cfg.block_size), color="green", linestyle="--", label=f"Train ctx ({int(cfg.block_size)})")
    if callable(xlabel):
        _ = xlabel("Context Length")
    if callable(ylabel):
        _ = ylabel("Perplexity")
    if callable(title):
        _ = title("Perplexity vs Context (RoPE Extrapolation)")
    if callable(xscale):
        _ = xscale("log", base=2)
    if callable(legend):
        _ = legend()
    if callable(grid):
        _ = grid(True, alpha=0.3)

    _ = subplot(2, 2, 4)
    kv_fp16 = [float(theoretical_kv_cache_size(cfg, int(c), 16)) for c in contexts]
    kv_q4 = [float(theoretical_kv_cache_size(cfg, int(c), 4)) for c in contexts]
    _ = plot(contexts, kv_fp16, "b-", linewidth=2, label="FP16")
    _ = plot(contexts, kv_q4, "g-", linewidth=2, label="Q4")
    if callable(axhline):
        _ = axhline(y=128, color="red", linestyle="--", label="128GB RAM")
    if callable(xlabel):
        _ = xlabel("Context Length")
    if callable(ylabel):
        _ = ylabel("KV Cache Size (GB)")
    if callable(title):
        _ = title("KV Cache Memory Requirements")
    if callable(xscale):
        _ = xscale("log", base=2)
    if callable(yscale):
        _ = yscale("log")
    if callable(legend):
        _ = legend()
    if callable(grid):
        _ = grid(True, alpha=0.3)

    if callable(suptitle):
        _ = suptitle(f"128k Benchmark ({cfg.attn_mode})", fontsize=14)
    if callable(tight_layout):
        _ = tight_layout()
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    _ = savefig(out_path, dpi=150, bbox_inches="tight")
    _ = close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="128k Context Benchmark")
    _ = parser.add_argument("--ckpt", type=str, default=None, help="Model checkpoint")
    _ = parser.add_argument("--compare", type=str, nargs=2, default=None, help="Compare two checkpoints")
    _ = parser.add_argument("--data", type=str, default="fineweb_100m.tokens", help="Token file")
    _ = parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384, 65536, 131072],
        help="Context lengths to benchmark",
    )
    _ = parser.add_argument("--output", type=str, default="assets/benchmark_128k.png", help="Output plot path")
    args = parser.parse_args(argv)
    a = as_str_object_dict(getattr(args, "__dict__", {})) or {}

    device = pick_device()
    contexts = _as_int_list(a.get("contexts"), [1024, 4096, 16384, 65536, 131072])
    print(f"Device: {device}")
    print(f"Contexts to benchmark: {contexts}")

    data_path = _as_str(a.get("data"), "fineweb_100m.tokens")
    print(f"\nLoading tokens from: {data_path}")
    max_tokens = int(max(contexts)) + 1000
    tokens = load_tokens(data_path, max_tokens)
    print(f"Loaded {len(tokens):,} tokens")

    cmp_pair = as_object_pair2(a.get("compare", None))
    compare_paths: tuple[str, str] | None = None
    if cmp_pair is not None:
        p0, p1 = cmp_pair
        if isinstance(p0, str) and isinstance(p1, str):
            compare_paths = (p0, p1)

    if compare_paths is not None:
        # Comparison mode
        ckpts = [compare_paths[0], compare_paths[1]]
        all_results: dict[str, list[BenchmarkResult]] = {}

        for ckpt_path in ckpts:
            print(f"\n{'='*60}")
            print(f"Loading: {ckpt_path}")
            print('='*60)

            cfg, model = _load_checkpoint_and_model(str(ckpt_path), device)

            model_name = f"{cfg.attn_mode}"
            if cfg.attn_mode == 'decoupled':
                model_name += f" ({cfg.sem_dim}/{cfg.geo_dim})"
            elif cfg.attn_mode == 'bottleneck':
                model_name += f" ({cfg.attn_dim})"
            else:
                model_name += f" ({cfg.d_model})"

            print(f"Model: {model_name}")
            print(f"Trained context: {cfg.block_size}")

            results_for_model: list[BenchmarkResult] = []
            for ctx in contexts:
                if ctx > len(tokens) - 1:
                    print(f"Skipping {ctx} (not enough tokens)")
                    continue

                print(f"\nContext {ctx:,}:")
                try:
                    result = benchmark_inference(model, tokens, ctx, device)
                    if result is not None:
                        results_for_model.append(result)
                        kv_cache = theoretical_kv_cache_size(cfg, ctx)
                        print(f"  → Time: {result.inference_time_ms:.1f}ms")
                        print(f"  → Throughput: {result.tokens_per_sec:,.0f} tok/s")
                        print(f"  → Loss: {result.loss:.4f} (PPL: {result.perplexity:.1f})")
                        print(f"  → Theoretical KV cache: {kv_cache:.3f} GB")
                except (RuntimeError, ValueError, OSError) as e:
                    print(f"  → FAILED: {e}")

            all_results[model_name] = results_for_model

            # Clear model from memory
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Print comparison summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        for ctx in contexts:
            print(f"\nContext: {ctx:,}")
            print("-" * 60)
            for model_name, res_list in all_results.items():
                result = next((r for r in res_list if r.context_length == ctx), None)
                if result:
                    msg = (
                        f"  {model_name:<30} "
                        f"Loss: {result.loss:.4f}  "
                        f"Time: {result.inference_time_ms:.1f}ms  "
                        f"Tok/s: {result.tokens_per_sec:,.0f}"
                    )
                    print(msg)

    else:
        # Single model benchmark
        ckpt_path_obj = a.get("ckpt", None)
        if not isinstance(ckpt_path_obj, str) or not ckpt_path_obj:
            print("Error: Provide --ckpt or --compare")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Loading: {ckpt_path_obj}")
        print('='*60)

        cfg, model = _load_checkpoint_and_model(str(ckpt_path_obj), device)

        print(f"Attention mode: {cfg.attn_mode}")
        print(f"Trained context: {cfg.block_size}")
        if cfg.attn_mode == 'decoupled':
            print(f"Dimensions: sem={cfg.sem_dim}, geo={cfg.geo_dim}")

        single_results: list[BenchmarkResult] = []
        for ctx in contexts:
            if ctx > len(tokens) - 1:
                print(f"Skipping {ctx} (not enough tokens)")
                continue

            print(f"\nContext {ctx:,}:")
            try:
                result = benchmark_inference(model, tokens, ctx, device)
                if result is not None:
                    single_results.append(result)
                    kv_cache = theoretical_kv_cache_size(cfg, ctx)
                    kv_cache_q4 = theoretical_kv_cache_size(cfg, ctx, 4)

                    print(f"  → Inference time: {result.inference_time_ms:.1f}ms")
                    print(f"  → Throughput: {result.tokens_per_sec:,.0f} tok/s")
                    print(f"  → Loss: {result.loss:.4f}")
                    print(f"  → Perplexity: {result.perplexity:.1f}")
                    print(f"  → KV cache (FP16): {kv_cache:.3f} GB")
                    print(f"  → KV cache (Q4): {kv_cache_q4:.3f} GB")
            except (RuntimeError, ValueError, OSError) as e:
                print(f"  → FAILED: {e}")

        # Summary table
        if single_results:
            print("\n" + "=" * 80)
            print("BENCHMARK SUMMARY")
            print("=" * 80)
            print(f"{'Context':<12} {'Time (ms)':<12} {'Tok/s':<12} {'Loss':<10} {'PPL':<10}")
            print("-" * 80)

            for r in single_results:
                row = (
                    f"{r.context_length:<12,} {r.inference_time_ms:<12.1f} "
                    f"{r.tokens_per_sec:<12,.0f} {r.loss:<10.4f} {r.perplexity:<10.1f}"
                )
                print(row)

            # Extrapolation quality
            train_ctx = cfg.block_size
            train_result = next((r for r in single_results if r.context_length == train_ctx), None)
            max_result = single_results[-1]

            if train_result and max_result.context_length > train_ctx:
                ppl_degradation = ((max_result.perplexity - train_result.perplexity)
                                   / train_result.perplexity) * 100
                extrap_ratio = max_result.context_length / train_ctx

                print("\n" + "-" * 80)
                print(f"Extrapolation: {train_ctx} → {max_result.context_length} ({extrap_ratio:.0f}x)")
                print(f"PPL degradation: {ppl_degradation:+.1f}%")

                if ppl_degradation < 50:
                    print("✓ EXCELLENT: RoPE extrapolation successful!")
                elif ppl_degradation < 100:
                    print("⚠ MODERATE: Some degradation at long context")
                else:
                    print("✗ POOR: Significant degradation, may need position interpolation")

            out_path = _as_str(a.get("output"), "assets/benchmark_128k.png")
            _maybe_plot(single_results, cfg, out_path)
            if HAS_MATPLOTLIB:
                print(f"\n✓ Saved: {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

