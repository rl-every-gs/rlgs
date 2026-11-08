#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<'EOF'
Usage: bash evaluate.sh MODEL_PATH [ITERATION]

Example:
  bash evaluate.sh ./output/train 30000
EOF
  exit 2
fi

METHOD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=$1
ITERATION=${2:-30000}
PYTHON="${PYTHON:-python}"

if [[ ! -f "${MODEL_PATH}/cfg_args" ]]; then
  printf 'Missing checkpoint configuration: %s/cfg_args\n' "${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  printf 'Missing iteration %s point cloud under: %s\n' \
    "${ITERATION}" "${MODEL_PATH}" >&2
  exit 1
fi

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Native 3DGS-MCMC evaluation: render held-out views, then compute metrics.
RESULTS_PATH="${MODEL_PATH}/results.json"
STARTED_AT=$(date +%s)

"${PYTHON}" -B -u "${METHOD_DIR}/render.py" \
  -m "${MODEL_PATH}" \
  --iteration "${ITERATION}" \
  --skip_train \
  --quiet

"${PYTHON}" -B -u "${METHOD_DIR}/metrics.py" -m "${MODEL_PATH}"

if [[ ! -s "${RESULTS_PATH}" ]] || \
   [[ $(stat -c %Y "${RESULTS_PATH}") -lt ${STARTED_AT} ]]; then
  printf 'metrics.py did not create a fresh, non-empty results.json: %s\n' \
    "${RESULTS_PATH}" >&2
  exit 1
fi

"${PYTHON}" -B - "${RESULTS_PATH}" "${ITERATION}" <<'PY'
import json
import math
import sys

results_path, iteration = sys.argv[1:]
with open(results_path, "r", encoding="utf-8") as handle:
    results = json.load(handle)

metrics = results.get(f"ours_{iteration}", {})
for name in ("PSNR", "SSIM", "LPIPS"):
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"Missing or non-finite {name} in {results_path}")

print(json.dumps(metrics, indent=2))
PY
