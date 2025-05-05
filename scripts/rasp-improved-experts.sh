#!/usr/bin/env bash
# Reproduces Appendix‑B results of “Learning Transformer Programs”
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

# ───────────── global knobs (identical for every task) ─────────────
SEED=0
DATASET_SIZE=20000
BATCH=512
EPOCHS=250

LR=2e-2           # the paper’s LR
CLIP=1.0          # grad‑norm clip
GUMBEL_SAMPLES=4
TAU_INIT=3.0
TAU_END=0.1       # cools far enough for discrete convergence
TAU_SCHED=geomspace
# ───────────────────────────────────────────────────────────────────

TASKS=(double_hist hist reverse sort most_freq dyck1 dyck2)

for DATASET in "${TASKS[@]}"; do
  # ───── model sizes from Table 1 (k = MAX = VOCAB for synthetic sets) ─────
  case "$DATASET" in
      hist)         VOCAB=8  MAX=8  L=1 H=4 M=2 ;;
      double_hist)  VOCAB=8  MAX=8  L=3 H=4 M=2 ;;   # 2 cat + 2 num heads
      reverse)      VOCAB=8  MAX=8  L=3 H=8 M=2 ;;
      sort)         VOCAB=8  MAX=8  L=3 H=8 M=4 ;;
      most_freq)    VOCAB=8  MAX=8  L=4 H=8 M=4 ;;
      dyck1)        VOCAB=16 MAX=16 L=3 H=8 M=2 ;;
      dyck2)        VOCAB=16 MAX=16 L=3 H=4 M=4 ;;
  esac
  # ──────────────────────────────────────────────────────────────────────────

  # ───── defaults (cat = num split) ─────
  N_HEADS_CAT=$((H/2)); N_HEADS_NUM=$((H/2))
  N_CAT_MLPS=$((M/2));  N_NUM_MLPS=$((M/2))
  ATT_TYPE="both"
  COUNT_ONLY="--count_only"
  MLP_TYPE="--mlp_type cat"
  # ────────────────────────────────

  case "$DATASET" in
    # histogram only needs numerical heads; keep counts‑only loss
    hist)
      ATT_TYPE="num"
      N_HEADS_CAT=0; N_HEADS_NUM=$H
      ;;

    # ──────────────────────────────────────────────────────────────
    # >>> exact paper recipe that reaches ≈ 98 % on double‑hist <<< 
    double_hist)
      ATT_TYPE="both"
      COUNT_ONLY=""                 # we need the *values*, not just counts
      N_HEADS_CAT=2;  N_HEADS_NUM=2 # 2 + 2 as in Appendix B
      N_CAT_MLPS=1;   N_NUM_MLPS=1  # mixed stack
      MLP_TYPE="--mlp_type mix"     # first cat, second num
      ;;
    # ──────────────────────────────────────────────────────────────
  esac

  echo -e "\n=== Running $DATASET ==="
  python src/run.py \
      --dataset "$DATASET" \
      --vocab_size "$VOCAB" \
      --dataset_size "$DATASET_SIZE" \
      --min_length 1 --max_length "$MAX" \
      --batch_size "$BATCH" --n_epochs "$EPOCHS" \
      --lr "$LR" --max_grad_norm "$CLIP" \
      --sample_fn gumbel_soft --gumbel_samples "$GUMBEL_SAMPLES" \
      --tau_init "$TAU_INIT" --tau_end "$TAU_END" --tau_schedule "$TAU_SCHED" \
      --n_vars_cat 1 --n_vars_num 1 --d_var "$MAX" \
      --n_layers "$L" \
      --n_heads_cat "$N_HEADS_CAT" --n_heads_num "$N_HEADS_NUM" \
      --n_cat_mlps "$N_CAT_MLPS" --n_num_mlps "$N_NUM_MLPS" \
      --attention_type "$ATT_TYPE" \
      --rel_pos_bias fixed --one_hot_embed --dropout 0.0 \
      --mlp_vars_in 2 --d_mlp 64 \
      $COUNT_ONLY $MLP_TYPE \
      --selector_width 0 \
      --seed "$SEED" --unique 1 \
      --use_sparse_expert --n_experts 4 \
      --save --save_code \
      --output_dir "output-improved/rasp-improved-experts/${DATASET}/k${VOCAB}_len${MAX}_L${L}_H${H}_M${M}/s${SEED}"
done