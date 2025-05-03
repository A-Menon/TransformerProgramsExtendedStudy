#!/bin/bash
# Run all RASP tasks with the paper’s exact hyper‑params
SEED=0

# Table‑level params -------------------------------------
declare -A VOCAB_SIZES=( ["reverse"]=8 ["hist"]=8 ["double_hist"]=8 ["sort"]=8 ["most_freq"]=8 ["dyck1"]=16 ["dyck2"]=16 )
declare -A MAX_LENGTHS=( ["reverse"]=8 ["hist"]=8 ["double_hist"]=8 ["sort"]=8 ["most_freq"]=8 ["dyck1"]=16 ["dyck2"]=16 )
declare -A N_LAYERS=(    ["reverse"]=3 ["hist"]=1 ["double_hist"]=3 ["sort"]=3 ["most_freq"]=4 ["dyck1"]=3 ["dyck2"]=3 )
declare -A TOTAL_HEADS=( ["reverse"]=8 ["hist"]=4 ["double_hist"]=4 ["sort"]=8 ["most_freq"]=8 ["dyck1"]=8 ["dyck2"]=4 )
declare -A TOTAL_MLPS=(  ["reverse"]=2 ["hist"]=2 ["double_hist"]=2 ["sort"]=4 ["most_freq"]=4 ["dyck1"]=2 ["dyck2"]=4 )
# --------------------------------------------------------

TASKS=("reverse" "hist" "double_hist" "sort" "most_freq" "dyck1" "dyck2")

for DATASET in "${TASKS[@]}"; do
  echo "=== Running $DATASET ==="

  VOCAB_SIZE=${VOCAB_SIZES[$DATASET]}
  MAX_LENGTH=${MAX_LENGTHS[$DATASET]}
  N_LAYERS_VAL=${N_LAYERS[$DATASET]}
  H=${TOTAL_HEADS[$DATASET]}
  M=${TOTAL_MLPS[$DATASET]}

  N_HEADS_CAT_VAL=$((H / 2))
  N_HEADS_NUM_VAL=$((H / 2))
  N_CAT_MLPS_VAL=$((M / 2))
  N_NUM_MLPS_VAL=$((M / 2))

  python src/run.py \
      --dataset "$DATASET" \
      --vocab_size "$VOCAB_SIZE" \
      --dataset_size 20000 \
      --min_length 1 \
      --max_length "$MAX_LENGTH" \
      --n_epochs 250 \
      --batch_size 512 \
      --lr 5e-2 \
      --gumbel_samples 1 \
      --sample_fn gumbel_soft \
      --tau_init 3.0 \
      --tau_end 0.01 \
      --tau_schedule geomspace \
      --n_vars_cat 1 \
      --d_var "$MAX_LENGTH" \
      --n_vars_num 1 \
      --n_layers "$N_LAYERS_VAL" \
      --n_heads_cat "$N_HEADS_CAT_VAL" \
      --n_heads_num "$N_HEADS_NUM_VAL" \
      --n_cat_mlps "$N_CAT_MLPS_VAL" \
      --n_num_mlps "$N_NUM_MLPS_VAL" \
      --attention_type cat \
      --rel_pos_bias fixed \
      --one_hot_embed \
      --dropout 0.0 \
      --mlp_vars_in 2 \
      --d_mlp 64 \
      --count_only \
      --selector_width 0 \
      --seed "$SEED" \
      --unique 1 \
      --save \
      --save_code \
      --output_dir "output/rasp/${DATASET}/k${VOCAB_SIZE}_len${MAX_LENGTH}_L${N_LAYERS_VAL}_H${H}_M${M}/s${SEED}"
done
