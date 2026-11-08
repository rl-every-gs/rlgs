#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four outputs created by train.sh with the same deterministic
# runtime used by the audited camera-ready runner.
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/rl_tmgs}"
ITERATION="${ITERATION:-30000}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for scene in train truck drjohnson playroom; do
  model_path="${OUTPUT_ROOT}/${scene}"
  point_cloud="${model_path}/point_cloud/iteration_${ITERATION}/point_cloud.ply"

  if [[ ! -f "${model_path}/cfg_args" || ! -f "${point_cloud}" ]]; then
    printf 'Missing iteration %s checkpoint for %s: %s\n' \
      "${ITERATION}" "${scene}" "${model_path}" >&2
    exit 1
  fi

  "${PYTHON}" -B -u render.py \
    -m "${model_path}" \
    --iteration "${ITERATION}" \
    --skip_train \
    --quiet

  "${PYTHON}" -B -u metrics.py -m "${model_path}"

  if [[ ! -s "${model_path}/results.json" ]]; then
    printf 'metrics.py did not create a non-empty results.json for %s\n' \
      "${scene}" >&2
    exit 1
  fi
done
