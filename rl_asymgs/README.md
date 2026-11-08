# RL-AsymGS

RL-AsymGS applies the RLGS controller to the AsymGS-EMA backbone. The released
recipe trains the RobustNeRF Yoda, Crab2, Statue, and Android scenes
sequentially for 30,000 iterations.

## Quick start

Run from this directory on Linux with an NVIDIA GPU:

```bash
bash setup.sh
conda activate rlgs-asymgs
bash test.sh

# Download and prepare RobustNeRF, DINOv2, and the released masks.
bash scripts/download_dataset.sh --scene all

# Check all paths and commands without starting training.
DATA_ROOT="$PWD/dataset/robustnerf" \
OUTPUT_ROOT="$PWD/output/rl_asymgs" \
bash train.sh --dry-run

# Train all four scenes.
DATA_ROOT="$PWD/dataset/robustnerf" \
OUTPUT_ROOT="$PWD/output/rl_asymgs" \
bash train.sh
```

For regions using the Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/download_dataset.sh --scene all
```

The downloader verifies the released DINOv2 weight and all four precomputed
masks by SHA-256. The masks are hosted in the public
[`stack93/rlgs-assets`](https://huggingface.co/datasets/stack93/rlgs-assets)
dataset.

Each prepared scene must contain `images/`, `sparse/`, `train_list.txt`,
`test_list.txt`, and its `multi_cue_masks_<scene>-undistorted.npz` file.

## Evaluation

Evaluate an existing NerfBaselines checkpoint into a new directory:

```bash
bash test.sh \
  --model /path/to/checkpoint \
  --data /path/to/prepared/scene \
  --output /path/to/new/evaluation
```

See [SETUP_AND_TEST.md](SETUP_AND_TEST.md) for environment selection, CUDA
checks, and checkpoint requirements.

## Released configuration

- Python 3.11, PyTorch 2.6.0+cu126, and NerfBaselines 1.2.12
- seed `0`, `lambda_mul=0.1`, and evaluation every 1,000 iterations
- RLGS initial log standard deviations: `-1.5` for learning rate and `-2.0`
  for densification

This implementation is based on
[AsymGS](https://github.com/steveli88/AsymGS). See the included licenses,
upstream README, and third-party notices.
