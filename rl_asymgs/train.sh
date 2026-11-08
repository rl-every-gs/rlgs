#!/usr/bin/env bash
set -Eeuo pipefail

METHOD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT="${DATA_ROOT:-${METHOD_DIR}/dataset/robustnerf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${METHOD_DIR}/output/rl_asymgs}"
NERFBASELINES="${NERFBASELINES:-nerfbaselines}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash train.sh [--dry-run]

Runs the frozen RL-AsymGS RobustNeRF recipe sequentially on:
  yoda, crab2, statue, android

Environment variables:
  DATA_ROOT       Prepared RobustNeRF root containing <scene>-undistorted
  OUTPUT_ROOT     Fresh output root for the four scene runs
  NERFBASELINES   NerfBaselines executable (default: nerfbaselines)
  CUDA_VISIBLE_DEVICES  GPU selection (default: 0)

Example:
  DATA_ROOT=/data/robustnerf \
  OUTPUT_ROOT=./output/rl_asymgs \
  bash train.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${NERFBASELINES}" >/dev/null 2>&1; then
  printf 'NerfBaselines executable not found: %s\n' "${NERFBASELINES}" >&2
  printf 'Activate the validated rlgs-asymgs environment first.\n' >&2
  exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
  printf 'sha256sum is required to verify frozen release inputs.\n' >&2
  exit 1
fi

WEIGHTS="${METHOD_DIR}/weights/dinov2_vits14_reg4_pretrain.pth"
EXPECTED_WEIGHTS_SHA256="f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
if [[ ! -f "${WEIGHTS}" ]]; then
  printf 'Missing frozen DINOv2 weights: %s\n' "${WEIGHTS}" >&2
  exit 1
fi
ACTUAL_WEIGHTS_SHA256=$(sha256sum "${WEIGHTS}" | awk '{print $1}')
if [[ "${ACTUAL_WEIGHTS_SHA256}" != "${EXPECTED_WEIGHTS_SHA256}" ]]; then
  printf 'DINOv2 weight SHA-256 mismatch.\nExpected: %s\nActual:   %s\n' \
    "${EXPECTED_WEIGHTS_SHA256}" "${ACTUAL_WEIGHTS_SHA256}" >&2
  exit 1
fi

SCENES=(yoda crab2 statue android)
declare -A MASK_SHA256=(
  [yoda]="f888c1b315819bb9a81fab6a9264681afd45624cfa8454ed70e1910ff06db298"
  [crab2]="1163aca63654a351e321a4724e1a8a013d93c149ac9527b54cab1193c46d9ab3"
  [statue]="9a2a8c307f5e18fd2cc4286612c75b66f6c447523f2e32012d4f552d82a3704a"
  [android]="5b448093748c52dd75e242fffea9f8f4096e827145dc5a65ea2cd623347a27c8"
)

# Preflight every scene before starting the first long training run.
for scene in "${SCENES[@]}"; do
  dataset="${DATA_ROOT}/${scene}-undistorted"
  output="${OUTPUT_ROOT}/${scene}-rlgs"
  required=(
    "${dataset}/images"
    "${dataset}/sparse"
    "${dataset}/train_list.txt"
    "${dataset}/test_list.txt"
    "${dataset}/multi_cue_masks_${scene}-undistorted.npz"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      printf 'Missing required formal-run input for %s: %s\n' "${scene}" "${path}" >&2
      exit 1
    fi
  done
  mask="${dataset}/multi_cue_masks_${scene}-undistorted.npz"
  actual_mask_sha=$(sha256sum "${mask}" | awk '{print $1}')
  if [[ "${actual_mask_sha}" != "${MASK_SHA256[$scene]}" ]]; then
    printf 'Mask SHA-256 mismatch for %s.\nExpected: %s\nActual:   %s\n' \
      "${scene}" "${MASK_SHA256[$scene]}" "${actual_mask_sha}" >&2
    exit 1
  fi
  if [[ -e "${output}" ]]; then
    printf 'Refusing to mix with an existing output: %s\n' "${output}" >&2
    exit 1
  fi
done

export PYTHONPATH="${METHOD_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NERFBASELINES_REGISTER="${METHOD_DIR}/asym_emags_spec.py"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON_ARGS=(
  train
  --method asym_emags
  --set iterations=30000
  --set lambda_mul=0.1
  --set enable_rlgs=True
  --set rlgs_init_log_std=-1.5
  --set rlgs_densi_init_log_std=-2.0
  --backend python
  --eval-all-iters 1000::1000
)

printf 'RL-AsymGS formal recipe: scenes=%s data=%s output=%s gpu=%s\n' \
  "${SCENES[*]}" "${DATA_ROOT}" "${OUTPUT_ROOT}" "${CUDA_VISIBLE_DEVICES}"

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi

for scene in "${SCENES[@]}"; do
  dataset="${DATA_ROOT}/${scene}-undistorted"
  output="${OUTPUT_ROOT}/${scene}-rlgs"
  command=(
    "${NERFBASELINES}"
    "${COMMON_ARGS[@]}"
    --data "${dataset}"
    --output "${output}"
  )

  printf '[%s] ' "${scene}"
  printf '%q ' "${command[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  "${command[@]}"

  result="${output}/results-30000.json"
  if [[ ! -s "${result}" ]]; then
    printf 'Training completed without a non-empty final result: %s\n' "${result}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry run complete; no training or output creation was performed.\n'
else
  printf 'All four frozen RL-AsymGS scene runs completed under: %s\n' "${OUTPUT_ROOT}"
fi
