# RL-TMGS

RL-TMGS applies the RLGS controller to the Taming-3DGS backbone. `train.py` is
the supported trainer, while `train.sh` contains the released four-scene recipe.

## Quick start

Run from this directory on Linux with an NVIDIA GPU:

```bash
bash setup.sh
conda activate rlgs-tmgs
bash test.sh

PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=./data/tandt OUTPUT_ROOT=./output/rl_tmgs \
bash train.sh

PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=./output/rl_tmgs \
bash evaluate.sh
```

`DATA_ROOT` must contain these scene directories:

```text
train/
truck/
drjohnson/
playroom/
```

Training runs the four scenes sequentially for 30,000 iterations with their
released Gaussian budgets. Evaluation renders held-out views and writes PSNR,
SSIM, and LPIPS results. Set `ITERATION=N` to evaluate another saved iteration.

To run one scene, copy its command block from [train.sh](train.sh). For the
Train/Truck reproduction workflow, `reproduce_tnt.sh` can download missing data,
train, render, and evaluate; pass `--dry-run` to inspect its commands first.

See [SETUP_AND_TEST.md](SETUP_AND_TEST.md) for environment selection, CUDA
checks, and single-checkpoint evaluation.

This implementation is derived from Taming-3DGS and the original 3D Gaussian
Splatting codebase. See the included licenses and third-party notices.
