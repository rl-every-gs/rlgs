#!/bin/bash
# Run AsymGS EMA-GS baseline and/or RL-AsymGS.
# Usage:
#   bash scripts/run_exp.sh --dataset_path dataset/robustnerf/android-undistorted --option rlgs
#   bash scripts/run_exp.sh --dataset_path /abs/path/to/android-undistorted --option baseline
#   bash scripts/run_exp.sh --dataset_path dataset/robustnerf/android-undistorted --option both
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET_PATH=""
OPTION="rlgs"
OUTPUT=""
LAMBDA_MUL="0.1"
EVAL_ITERS="1000::1000"

usage() {
  cat <<EOF
Usage: $0 --dataset_path PATH --option {baseline|rlgs|both} [--output DIR]

  --dataset_path   Path to an undistorted RobustNeRF scene
                   (e.g. dataset/robustnerf/android-undistorted)
  --option         baseline | rlgs | both   (default: rlgs)
  --output         Output root (default: output_emags/<scene>[-rlgs])
  --lambda_mul     EMA-GS lambda_mul (default: 0.1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path) DATASET_PATH="$2"; shift 2 ;;
    --option)       OPTION="$2"; shift 2 ;;
    --output)       OUTPUT="$2"; shift 2 ;;
    --lambda_mul)   LAMBDA_MUL="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "[error] --dataset_path is required" >&2
  usage
  exit 1
fi

# Resolve relative paths against repo root.
if [[ "$DATASET_PATH" != /* ]]; then
  DATASET_PATH="$ROOT/$DATASET_PATH"
fi
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "[error] dataset path not found: $DATASET_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATASET_PATH/train_list.txt" ]]; then
  echo "[error] missing train_list.txt under $DATASET_PATH" >&2
  echo "       Run: bash scripts/download_dataset.sh" >&2
  exit 1
fi

SCENE="$(basename "$DATASET_PATH")"
MASK="$DATASET_PATH/multi_cue_masks_${SCENE}.npz"
if [[ ! -f "$MASK" ]]; then
  echo "[warn] multi-cue mask not found: $MASK"
  echo "       Training may fail until masks are generated."
fi

WEIGHTS="$ROOT/weights/dinov2_vits14_reg4_pretrain.pth"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "[download] DINOv2 weights"
  mkdir -p "$ROOT/weights"
  curl -L --retry 5 -o "$WEIGHTS" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NERFBASELINES_REGISTER="$ROOT/asym_emags_spec.py"

run_one() {
  local enable_rlgs="$1"
  local out="$2"
  local args=(
    nerfbaselines train --method asym_emags
    --data "$DATASET_PATH"
    --output "$out"
    --set "lambda_mul=${LAMBDA_MUL}"
    --backend python
    --eval-all-iters "$EVAL_ITERS"
  )
  if [[ "$enable_rlgs" == "1" ]]; then
    args+=(--set enable_rlgs=True)
  fi
  echo "[train] enable_rlgs=$enable_rlgs -> $out"
  "${args[@]}"
}

case "$OPTION" in
  baseline)
    OUT="${OUTPUT:-$ROOT/output_emags/${SCENE}}"
    run_one 0 "$OUT"
    ;;
  rlgs)
    OUT="${OUTPUT:-$ROOT/output_emags/${SCENE}-rlgs}"
    run_one 1 "$OUT"
    ;;
  both)
    if [[ -n "$OUTPUT" ]]; then
      echo "[error] --output with --option both is ambiguous; omit --output" >&2
      exit 1
    fi
    run_one 0 "$ROOT/output_emags/${SCENE}"
    run_one 1 "$ROOT/output_emags/${SCENE}-rlgs"
    ;;
  *)
    echo "[error] --option must be baseline|rlgs|both (got: $OPTION)" >&2
    exit 1
    ;;
esac

echo "[done] option=$OPTION dataset=$DATASET_PATH"
