#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage: bash train.sh DATA_PATH OUTPUT_PATH [TRAIN_ARGS...]

Example:
  bash train.sh /data/tandt/train ./output/train
EOF
  exit 2
fi

METHOD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_PATH=$1
OUTPUT_PATH=$2
shift 2

if [[ ! -d "${DATA_PATH}" ]]; then
  printf 'Dataset directory does not exist: %s\n' "${DATA_PATH}" >&2
  exit 1
fi

# Public benchmark configuration is versioned with the code. Select it from
# the dataset directory name instead of requiring a private registry or a
# caller-supplied config path. SCENE can disambiguate a differently named path.
SCENE="${SCENE:-$(basename -- "${DATA_PATH}")}"
CONFIG_PATH="${METHOD_DIR}/configs/${SCENE}.json"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'No released RL-MCMC config for scene %s: %s\n' \
    "${SCENE}" "${CONFIG_PATH}" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"

# Match the official 3DGS-MCMC data path: read the full-resolution `images`
# directory and apply the released scene's downsampling factor at runtime.
IMAGE_DIR="images"
RESOLUTION=$("${PYTHON}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["resolution"])' \
  "${CONFIG_PATH}")

if [[ ! -d "${DATA_PATH}/${IMAGE_DIR}" ]]; then
  printf 'Official RL-MCMC input directory does not exist: %s\n' \
    "${DATA_PATH}/${IMAGE_DIR}" >&2
  exit 1
fi

case "${RESOLUTION}" in
  1|2|4|8) ;;
  *)
    printf 'Unsupported RL-MCMC resolution in %s: %s\n' \
      "${CONFIG_PATH}" "${RESOLUTION}" >&2
    exit 1
    ;;
esac

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

printf 'RL-MCMC input: directory=%s resolution=%s config=%s\n' \
  "${IMAGE_DIR}" "${RESOLUTION}" "${CONFIG_PATH}"

# Frozen RL-MCMC recipe previously hidden in the legacy batch launcher.
COMMON_ARGS=(
  --init_type sfm
  --eval
  --quiet
  --test_iterations 30000
  --save_iterations 30000
  --seed 0
  --policy 1
  --reward_cam_idx 20
  --same_reward_set 1008
  --rl_densi 1
  --densi_rl_innter_step 4
  --rst_opdrds 0
  --grdclip 2.4
  --reward_scale_v2 0.041666666666666664
  --hidden_state 16
  --reward_scale_final 0.05
  --rl_densi_cnt 500
  --reward_option 2
  --lr_freq 1
  --shuffle_interval 1000
  --offset 2
  --future_step 20
  --boost_step_loop 8
  --lambda_penalty 70.0
  --boost_step 29000
  --hidden_interval 1000
  --pl_lr 0.0001
  --pl_lr_final 0.0005
  --opacity_reg 0.01
  --beta_itr 1.0
  --beta_final 0.001
  --init_std -1.5
  --densi_init_std -1.5
)

exec "${PYTHON}" -B -u "${METHOD_DIR}/train_mcmv3_10_smaller.py" \
  -s "${DATA_PATH}" \
  -m "${OUTPUT_PATH}" \
  -i "${IMAGE_DIR}" \
  -r "${RESOLUTION}" \
  --config "${CONFIG_PATH}" \
  "${COMMON_ARGS[@]}" \
  "$@"
