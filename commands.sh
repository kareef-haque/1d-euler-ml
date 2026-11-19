#!/usr/bin/env bash
set -euo pipefail

echo "Starting NFV attention ablation study..."

echo "[1/4] Running variants: attn_c4, attnp_c4"
python ablate_nfv.py \
    --variants attn_c4,attnp_c4 \
    --seeds 0 \
    --out ablation_attn \
    --data_dir generated_nfv

echo "[2/4] Running variants: attn_c4_e8, attn_c4_e32"
python ablate_nfv.py \
    --variants attn_c4_e8,attn_c4_e32 \
    --seeds 0 \
    --out ablation_attn \
    --data_dir generated_nfv

echo "[3/4] Plotting standard Sod ablation results..."
python plot_sod_ablation.py "ablation_attn/*/model_final.pt" \
    --out sod_attn.png

echo "[4/4] Plotting primitive variables (zoomed)..."
python plot_sod_ablation.py "ablation_attn/*/model_final.pt" \
    --out sod_attn_prim.png \
    --vars prim \
    --xlim 0.15 0.3

echo "Ablation and plotting complete."