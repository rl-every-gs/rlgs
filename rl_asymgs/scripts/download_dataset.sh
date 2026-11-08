#!/bin/bash
# Download RobustNeRF, undistort (COLMAP), and fetch DINOv2 weights for RL-AsymGS.
# Usage:
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh --scene android
#   bash scripts/download_dataset.sh --scene all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCENE="android"
ROBUSTNERF_URL="https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz"
DINOV2_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
DINOV2_WEIGHTS="$ROOT/weights/dinov2_vits14_reg4_pretrain.pth"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
RLGS_ASSETS_REPO="${RLGS_ASSETS_REPO:-stack93/rlgs-assets}"
MASK_BASE_URL="${HF_ENDPOINT%/}/datasets/${RLGS_ASSETS_REPO}/resolve/main/rl_asymgs/robustnerf_masks"
declare -A MASK_SHA256=(
  [yoda]="f888c1b315819bb9a81fab6a9264681afd45624cfa8454ed70e1910ff06db298"
  [crab2]="1163aca63654a351e321a4724e1a8a013d93c149ac9527b54cab1193c46d9ab3"
  [statue]="9a2a8c307f5e18fd2cc4286612c75b66f6c447523f2e32012d4f552d82a3704a"
  [android]="5b448093748c52dd75e242fffea9f8f4096e827145dc5a65ea2cd623347a27c8"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--scene android|crab2|statue|yoda|all]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$ROOT/dataset" "$ROOT/weights"

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "[error] sha256sum is required to verify downloaded release assets." >&2
  exit 1
fi

# ---- DINOv2 ----
if [[ ! -f "$DINOV2_WEIGHTS" ]]; then
  echo "[download] DINOv2 -> $DINOV2_WEIGHTS"
  curl -L --retry 5 -o "$DINOV2_WEIGHTS" "$DINOV2_URL"
else
  echo "[skip] DINOv2 already exists"
fi

# ---- RobustNeRF raw ----
RAW_ROOT="$ROOT/dataset/robustnerf"
NEED_TAR=0
for s in android crab2 statue yoda; do
  if [[ ! -d "$RAW_ROOT/$s/images" || ! -d "$RAW_ROOT/$s/sparse/0" ]]; then
    NEED_TAR=1
    break
  fi
done

if [[ "$NEED_TAR" == "1" ]]; then
  TMP="$ROOT/dataset/robustnerf.tar.gz"
  if [[ ! -f "$TMP" ]]; then
    echo "[download] RobustNeRF -> $TMP"
    curl -L --retry 5 -o "$TMP" "$ROBUSTNERF_URL"
  else
    echo "[skip] archive already exists: $TMP"
  fi
  echo "[extract] $TMP"
  mkdir -p "$RAW_ROOT"
  tar -xzf "$TMP" -C "$ROOT/dataset"
  # Normalize layout if tarball extracts a top-level folder.
  if [[ ! -d "$RAW_ROOT/android" ]]; then
    FOUND="$(find "$ROOT/dataset" -maxdepth 3 -type d -name android | head -1 || true)"
    if [[ -n "${FOUND:-}" ]]; then
      SRC="$(dirname "$FOUND")"
      echo "[normalize] move contents from $SRC -> $RAW_ROOT"
      mkdir -p "$RAW_ROOT"
      shopt -s dotglob nullglob
      mv "$SRC"/* "$RAW_ROOT"/ 2>/dev/null || true
      shopt -u dotglob nullglob
    fi
  fi
else
  echo "[skip] RobustNeRF raw scenes already present under $RAW_ROOT"
fi

undistort_one() {
  local scene="$1"
  local max_size="$2"
  local src="$RAW_ROOT/$scene"
  local dst="$RAW_ROOT/${scene}-undistorted"
  if [[ ! -d "$src/images" || ! -d "$src/sparse/0" ]]; then
    echo "[error] missing raw scene: $src (images + sparse/0 required)" >&2
    exit 1
  fi
  if [[ -d "$dst/images" && -f "$dst/train_list.txt" ]]; then
    echo "[skip] already undistorted: $dst"
  else
    if ! command -v colmap >/dev/null 2>&1; then
      echo "[error] colmap not found; install COLMAP to undistort images." >&2
      exit 1
    fi
    echo "[undistort] $scene (max_image_size=$max_size)"
    colmap image_undistorter \
      --image_path "$src/images" \
      --input_path "$src/sparse/0" \
      --output_path "$dst" \
      --output_type COLMAP \
      --max_image_size "$max_size"
    cp "$src/train_list.txt" "$dst/train_list.txt"
    cp "$src/test_list.txt" "$dst/test_list.txt"
  fi

  # Ship precomputed AsymGS multi-cue masks for Android (same files used in our RL-AsymGS runs).
  local shipped="$ROOT/assets/masks/robustnerf/${scene}-undistorted"
  local mask="$dst/multi_cue_masks_${scene}-undistorted.npz"
  local dilate="$dst/dilated_correspond_points_${scene}-undistorted.npz"
  if [[ -d "$shipped" ]]; then
    mkdir -p "$dst"
    if [[ -f "$shipped/multi_cue_masks_${scene}-undistorted.npz" && ! -f "$mask" ]]; then
      cp -a "$shipped/multi_cue_masks_${scene}-undistorted.npz" "$mask"
      echo "[install] shipped multi-cue mask -> $mask"
    fi
    if [[ -f "$shipped/dilated_correspond_points_${scene}-undistorted.npz" && ! -f "$dilate" ]]; then
      cp -a "$shipped/dilated_correspond_points_${scene}-undistorted.npz" "$dilate"
      echo "[install] shipped dilated correspondences -> $dilate"
    fi
  fi

  local expected_mask_sha="${MASK_SHA256[$scene]}"
  if [[ ! -f "$mask" ]]; then
    local mask_file="multi_cue_masks_${scene}-undistorted.npz"
    local partial="${mask}.part"
    echo "[download] RL-AsymGS mask -> $mask"
    curl -L --fail --retry 5 -o "$partial" "${MASK_BASE_URL}/${mask_file}"
    local downloaded_sha
    downloaded_sha=$(sha256sum "$partial" | awk '{print $1}')
    if [[ "$downloaded_sha" != "$expected_mask_sha" ]]; then
      rm -f "$partial"
      echo "[error] downloaded mask SHA-256 mismatch for $scene" >&2
      echo "        expected: $expected_mask_sha" >&2
      echo "        actual:   $downloaded_sha" >&2
      exit 1
    fi
    mv "$partial" "$mask"
  fi

  local actual_mask_sha
  actual_mask_sha=$(sha256sum "$mask" | awk '{print $1}')
  if [[ "$actual_mask_sha" != "$expected_mask_sha" ]]; then
    echo "[error] mask SHA-256 mismatch for $scene: $mask" >&2
    echo "        expected: $expected_mask_sha" >&2
    echo "        actual:   $actual_mask_sha" >&2
    exit 1
  fi
  echo "[ok] verified mask: $mask"
}

case "$SCENE" in
  android) undistort_one android 504 ;;
  crab2)   undistort_one crab2 432 ;;
  statue)  undistort_one statue 504 ;;
  yoda)    undistort_one yoda 432 ;;
  all)
    undistort_one android 504
    undistort_one crab2 432
    undistort_one statue 504
    undistort_one yoda 432
    ;;
  *)
    echo "[error] unknown --scene $SCENE (use android|crab2|statue|yoda|all)" >&2
    exit 1
    ;;
esac

echo
echo "Done."
echo "Next:"
echo "  bash scripts/run_exp.sh --dataset_path dataset/robustnerf/android-undistorted --option rlgs"
