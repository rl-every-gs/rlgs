#!/bin/bash
# EMA-GS baseline and EMA-GS + RLGS on RobustNeRF Android.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NERFBASELINES_REGISTER="$ROOT/asym_emags_spec.py"

DATA="dataset/robustnerf/android-undistorted"
WEIGHTS="$ROOT/weights/dinov2_vits14_reg4_pretrain.pth"
if [[ ! -f "$WEIGHTS" ]]; then
  mkdir -p "$ROOT/weights"
  curl -L --retry 5 -o "$WEIGHTS" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
fi

if [[ "${1:-both}" == "base" || "${1:-both}" == "both" ]]; then
  nerfbaselines train --method asym_emags \
    --data "$DATA" \
    --output output_emags/android-undistorted \
    --set lambda_mul=0.1 \
    --backend python \
    --eval-all-iters 1000::1000
fi

if [[ "${1:-both}" == "rlgs" || "${1:-both}" == "both" ]]; then
  nerfbaselines train --method asym_emags \
    --data "$DATA" \
    --output output_emags/android-undistorted-rlgs-align \
    --set lambda_mul=0.1 \
    --set enable_rlgs=True \
    --backend python \
    --eval-all-iters 1000::1000
fi
