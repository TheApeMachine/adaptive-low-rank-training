.PHONY: train v20 bottleneck_attention decoupled_bottleneck prepare_fineweb suggestions support bigboy replicate_paper visualize setup prepare_wikitext install_deps replicate_ablations replicate_fineweb

train:
	# python3 v1_gradient_grouping.py --mode baseline 
	# python3 v1_gradient_grouping.py --mode grouped --sim-threshold 0.9
	# python3 v1_gradient_grouping.py --mode coarse_to_fine

	# python3 v2_optimized_gradient_grouping.py --mode baseline 
	# python3 v2_optimized_gradient_grouping.py --sim-threshold 0.9 --mode grouped 
	# python3 v2_optimized_gradient_grouping.py --mode coarse_to_fine
	
	# python3 v3_adaptive_lowrank.py
	# python3 v3_adaptive_lowrank.py --epochs 30
	# python3 v3_adaptive_lowrank.py --init-rank1 128 --init-rank2 64

	# python3 v4_adaptive_lowrank_rand.py
	# python3 v4_adaptive_lowrank_rand.py --epochs 30
	# python3 v4_adaptive_lowrank_rand.py --init-rank1 128 --init-rank2 64
	
	# python3 v5_multi_layer_adaptive.py
	# python3 v5_multi_layer_adaptive.py --epochs 30
	# python3 v5_multi_layer_adaptive.py --init-rank1 128 --init-rank2 64
	# python3 v5_1_multi_layer_adaptive_smooth.py --epochs 30
	# python3 v5_1_multi_layer_adaptive_smooth.py --init-rank1 128 --init-rank2 64
	
	# python3 v6_lowrank_backprop.py
	# python3 v6_lowrank_backprop.py --epochs 30
	# python3 v6_lowrank_backprop.py --init-rank1 128 --init-rank2 64
	
	# python3 v7_transformer_dense_baseline.py --epochs 5
	# python3 v7_transformer_dense_baseline.py --epochs 25
	# python3 v7_transformer_dense_baseline.py --epochs 50
	# python3 v7_1_transformer_lowrank.py --epochs 15
	# python3 v7_1_transformer_lowrank.py --epochs 20 --init-rank 32
	# python3 v7_1_transformer_lowrank.py --epochs 30 --init-rank 64
	# python3 v7_2_transformer_lowrank_ema.py --epochs 15
	# python3 v7_2_transformer_lowrank_ema.py --epochs 20 --init-rank 32
	# python3 v7_2_transformer_lowrank_ema.py --epochs 30 --init-rank 64
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 15
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 20 --init-rank 32
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 30 --init-rank 64
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 15
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 20 --init-rank 32
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 30 --init-rank 64
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 15
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 20 --init-rank 32
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 30 --init-rank 64
	
	# python3 v8_transformer_lowrank_spectral.py --epochs 15
	# python3 v8_transformer_lowrank_spectral.py --epochs 20 --init-rank 32
	# python3 v8_transformer_lowrank_spectral.py --epochs 30 --init-rank 64
	
	# python3 v9_transformer_lowrank_spectral_bidirectional.py --epochs 30 --init-rank 64 --data-file wiki.train.tokens --log-file v9_log.jsonl

	# python3 v10_transformer_lowrank_scaled.py \
	# 	--data-file wiki.train.tokens \
	# 	--log-file v10_log.jsonl \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--d-model 512 \
	# 	--n-layers 6 \
	# 	--n-heads 8 \
	# 	--d-ff 2048 \
	# 	--block-size 256 \
	# 	--batch-size 32 \
	# 	--steps-per-epoch 200

	# python3 v11_transformer_lowrank_momentum.py \
	# 	--data-file wiki.train.tokens \
	# 	--log-file v11_log.jsonl \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--d-model 512 \
	# 	--n-layers 6 \
	# 	--n-heads 8 \
	# 	--d-ff 2048 \
	# 	--block-size 256 \
	# 	--batch-size 32 \
	# 	--steps-per-epoch 200

	# python3 v13_transformer_lowrank_lazy_svd_adaptive.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--log-file v13_log.jsonl

	# python3 v14_transformer_adaptive_heads_lowrank.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--log-file v14_log.jsonl

	# python3 v15_transformer_lowrank_adaptive_grad.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--log-file v15_log.jsonl

	# python3.12 v16_transformer_lowrank_pressure_cooker.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--log-file v16_log.jsonl

	# python3.12 v16_transformer_lowrank_pressure_cooker.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--min-rank 4 \
	# 	--compute-target 0.45 \
	# 	--warmup-epochs 1 \
	# 	--pressure-step 0.20 \
	# 	--energy-target-lo 0.85 \
	# 	--lambda-scale 100 \
	# 	--prune-every 200 \
	# 	--svd-interval 200 \
	# 	--log-file v16_log.jsonl

	# python3.12 v17_transformer_lowrank_pressure_cooker.py \
	# 	--mode train \
	# 	--data wiki.train.tokens \
	# 	--out-dir runs/v17

	# python3.12 v17_transformer_lowrank_pressure_cooker.py \
	# 	--mode generate \
	# 	--ckpt runs/v17/best.pt \
	# 	--prompt "Once upon a time" \
	# 	--max-new-tokens 400

	# python3.12 v18_transformer_lowrank_alrt.py \
	# 	--mode train --data wiki.train.tokens --out-dir runs/v18

	# python3.12 v18_transformer_lowrank_alrt.py \
	# 	--mode generate --ckpt runs/v18/best.pt \
	# 	--prompt "Once upon a time" --max-new-tokens 400 \
	# 	--temperature 0.8 --top-k 50


	# python3.12 v19_transformer_attn_bottleneck.py \
	# 	--data ./wiki.train.tokens \
	# 	--out-dir runs/v19_baseline \
	# 	--attn-dim 512

	# python3.12 v19_transformer_attn_bottleneck.py \
	# 	--data ./wiki.train.tokens \
	# 	--out-dir runs/v19_attn128 \
	# 	--attn-dim 128

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null \
		--attn-dim 128 --null-attn

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null_tie \
		--attn-dim 128 --null-attn --tie-qk

v20:
	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_baseline \
		--attn-dim 512 \
		--embed-dim 512

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128 \
		--attn-dim 128 \
		--embed-dim 512

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_embed256 \
		--attn-dim 512 \
		--embed-dim 256

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128_embed256 \
		--attn-dim 128 \
		--embed-dim 256

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128_embed128 \
		--attn-dim 128 \
		--embed-dim 128

bottleneck_attention:
	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
	 	--out-dir runs/v19_baseline \
	 	--attn-dim 512

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128 \
		--attn-dim 128

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null \
		--attn-dim 128 --null-attn

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null_tie \
		--attn-dim 128 --null-attn --tie-qk

decoupled_bottleneck:
	python3.12 v21_transformer_decoupled_bottleneck.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_bottleneck_rope \
		--attn-mode bottleneck \
		--attn-dim 128 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64 \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck.py \
		--mode sample \
		--ckpt runs/v21_decoupled_sem32_geo64/best.pt \
		--prompt-tokens "1 2 3 4 5" \
		--max-new-tokens 200 \
		--kv-cache q4_0

prepare_fineweb:
	python3.12 prepare_fineweb.py --out fineweb_100m.tokens

support:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_gqa_kv2_parammatch \
		--seed 1337 \
		--device mps \
		--attn-mode gqa \
		--kv-head 2 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2059 \
		--block 256 \
		--embed-dim 512 \
		--attn-dim 128 \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_small_d128_standard \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 128 \
		--layers 6 \
		--n-head 4 \
		--d-ff 512 \
		--block 256 \
		--embed-dim 128 \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64 \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64_block1024 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 128 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 1200 \
		--eval-every 200 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 8

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64_block2048 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 2048 \
		--embed-dim 128 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 800 \
		--eval-every 200 \
		--eval-iters 10 \
		--lr 3e-4 \
		--batch-size 4

bigboy:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_baseline \
		--attn-mode standard \
		--d-model 512 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_decoupled \
		--attn-mode decoupled \
		--d-model 512 \
		--n-head 8 \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

suggestions:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_combined_baseline_96 \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--null-attn

# =============================================================================
# SETUP & DATA PREPARATION
# =============================================================================

setup: prepare_wikitext
	@echo "=============================================="
	@echo "Setup complete! WikiText-2 ready."
	@echo "Run 'make prepare_fineweb' for FineWeb-Edu (optional, ~2GB download)"
	@echo "=============================================="

prepare_wikitext:
	@echo ">>> Preparing WikiText-2 dataset..."
	@if [ ! -f wiki.train.tokens ]; then \
		echo "Downloading and tokenizing WikiText-2..."; \
		python3.12 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt', 'ptb.train.txt')" 2>/dev/null || true; \
		python3.12 v21_transformer_decoupled_bottleneck_gqa.py --data wiki.train.tokens --steps 0 2>/dev/null || \
		echo "Note: WikiText-2 will be auto-downloaded on first training run."; \
	else \
		echo "wiki.train.tokens already exists."; \
	fi

install_deps:
	@echo ">>> Installing Python dependencies..."
	pip install torch numpy matplotlib seaborn tqdm
	@echo ">>> Optional: For FineWeb experiments"
	pip install datasets tiktoken || echo "Optional deps failed (ok if not using FineWeb)"

# =============================================================================
# PAPER REPLICATION
# =============================================================================
# Run `make replicate_paper` to reproduce all experiments from the paper.
# Estimated time: 8-12 hours on M1/M2 Mac or CUDA GPU.
# =============================================================================

replicate_paper: setup replicate_wikitext replicate_ablations replicate_fineweb visualize
	@echo "=============================================="
	@echo "All paper experiments complete!"
	@echo "Check runs/ for checkpoints and logs."
	@echo "Check assets/ for generated figures."
	@echo "=============================================="

replicate_wikitext:
	@echo ">>> [1/4] WikiText-2 Core Experiments"
	# Standard Baseline (d=512)
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_baseline_512 \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--steps 6000 \
		--eval-every 200 \
		--lr 3e-4 \
		--batch-size 64

	# Combined Bottleneck 96 (BEST PERPLEXITY)
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_combined_baseline_96 \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--lr 3e-4 \
		--batch-size 64

	# Decoupled Bottleneck (32 sem + 64 geo)
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64 \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--lr 3e-4 \
		--batch-size 64

replicate_ablations:
	@echo ">>> [2/4] Ablation Studies"
	# GQA Comparison (8Q/2KV heads)
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_gqa_kv2_parammatch \
		--attn-mode gqa \
		--kv-head 2 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--attn-dim 128 \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--lr 3e-4 \
		--batch-size 64

	# Small Model Control (d_model=128)
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_small_d128_standard \
		--attn-mode standard \
		--d-model 128 \
		--layers 6 \
		--n-head 4 \
		--d-ff 512 \
		--block 256 \
		--embed-dim 128 \
		--steps 6000 \
		--eval-every 200 \
		--lr 3e-4 \
		--batch-size 64 \
		--null-attn

	# Long Context 1024
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_block1024 \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 512 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 1200 \
		--eval-every 200 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 8

	# Long Context 2048
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_block2048 \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 2048 \
		--embed-dim 512 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 800 \
		--eval-every 200 \
		--eval-iters 10 \
		--lr 3e-4 \
		--batch-size 4

replicate_fineweb:
	@echo ">>> [3/4] FineWeb-Edu Large Scale Validation"
	@if [ ! -f fineweb_100m.tokens ]; then \
		echo "FineWeb dataset not found. Preparing..."; \
		python3.12 prepare_fineweb.py --out fineweb_100m.tokens; \
	fi
	# FineWeb Baseline
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_baseline \
		--attn-mode standard \
		--d-model 512 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

	# FineWeb Decoupled
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_decoupled \
		--attn-mode decoupled \
		--d-model 512 \
		--n-head 8 \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

visualize:
	@echo ">>> [4/4] Generating Figures"
	@mkdir -p assets
	python3.12 plot_results.py || echo "plot_results.py failed (may need log files)"
	python3.12 plot_memory.py || echo "plot_memory.py failed"
	@echo "Figures saved to assets/"