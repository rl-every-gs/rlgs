#!/usr/bin/env bash
set -euo pipefail

# Shared paths for all four scenes. Override them from the shell if needed.
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./data/tandt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/rl_tmgs}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Frozen RL-TMGS settings used by all four camera-ready scene runs.
COMMON_ARGS=(
  --eval
  --quiet
  --disable_viewer
  --iterations 30000
  --seeds 0
  --test_iterations 30000
  --save_iterations 30000
  --checkpoint_iterations 30000
  --optimizer_type default
  --densification_interval 100
  --separate_sh
  --mode final_count
  --policy 1
  --reward_cam_idx 20
  --same_reward_set 1008
  --rl_densi 1
  --densi_rl_innter_step 4
  --rst_opdrds 0
  --init_std -1.5
  --grdclip 2.4
  --reward_scale_v2 0.041666666666666664
  --hidden_state 16
  --reward_scale_final 0.05
  --densi_init_std -2.0
  --rl_densi_cnt 500
  --reward_option 2
  --lr_freq 1
  --shuffle_interval 1000
  --offset 2
  --future_step 20
  --boost_step_loop 8
  --lambda_penalty 70.0
  --timing 0
)

# Tanks and Temples: Train
"${PYTHON}" -u train.py \
  -s "${DATA_ROOT}/train" \
  -m "${OUTPUT_ROOT}/train" \
  --budget 1085480 \
  "${COMMON_ARGS[@]}"

# Tanks and Temples: Truck
"${PYTHON}" -u train.py \
  -s "${DATA_ROOT}/truck" \
  -m "${OUTPUT_ROOT}/truck" \
  --budget 2584171 \
  "${COMMON_ARGS[@]}"

# Deep Blending: DrJohnson
"${PYTHON}" -u train.py \
  -s "${DATA_ROOT}/drjohnson" \
  -m "${OUTPUT_ROOT}/drjohnson" \
  --budget 3273600 \
  "${COMMON_ARGS[@]}"

# Deep Blending: Playroom
"${PYTHON}" -u train.py \
  -s "${DATA_ROOT}/playroom" \
  -m "${OUTPUT_ROOT}/playroom" \
  --budget 2326100 \
  "${COMMON_ARGS[@]}"
