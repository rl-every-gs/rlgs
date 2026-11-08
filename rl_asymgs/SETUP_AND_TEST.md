# rl_asymgs: environment and one-command testing

Run these commands in this method directory. The wrappers locate repository files relative to themselves; no hostname or dataset path is hardcoded.

## New environment

```bash
bash setup.sh --dry-run
bash setup.sh
```

This creates the dedicated Conda environment `rlgs-asymgs` with Python 3.11, PyTorch 2.6.0 + CUDA 12.6 and the matching local CUDA extensions. Conda, an NVIDIA driver and a C++ compiler (`g++`) must already be installed. CUDA Toolkit 12.6 is installed inside the new environment. Use `--env NAME` or `--prefix /path/to/new/environment` to choose a different location; use `--jobs 2` to limit compiler parallelism.

Existing environments are not modified or reinstalled. To inspect one:

```bash
bash setup.sh --check-only --python /absolute/path/to/environment/bin/python
```

## Quick CUDA test

```bash
bash test.sh
# Or use an existing compatible environment:
bash test.sh --python /absolute/path/to/environment/bin/python
```

The default test imports this method's model/policy code and runs a tiny KNN plus rasterizer forward/backward check. It uses eight synthetic points only as a kernel test, writes no scene result, and does not train a model or download a dataset. It does not measure reconstruction quality or validate RL convergence. `--imports-only` skips GPU execution and is explicitly reported as an import-only check.

## Reproduce the four formal RobustNeRF runs

```bash
DATA_ROOT=/path/to/robustnerf \
OUTPUT_ROOT="$PWD/output/rl_asymgs" \
bash train.sh
```

The launcher runs Yoda, Crab2, Statue and Android sequentially with the frozen
30,000-iteration RL-AsymGS configuration. It requires each prepared
`<scene>-undistorted` directory, the matching precomputed multi-cue mask and the
published DINOv2 weight hash. Existing scene output directories are rejected so
a formal run cannot silently resume or mix with an older attempt. Run
`bash train.sh --dry-run` to preflight inputs and print the commands only.

The public preparation script downloads the exact four precomputed masks from
Hugging Face and verifies their SHA-256 hashes:

```bash
bash scripts/download_dataset.sh --scene all
# Optional mirror:
HF_ENDPOINT=https://hf-mirror.com bash scripts/download_dataset.sh --scene all
```

## Evaluate a trained checkpoint

```bash
bash test.sh --model /path/to/checkpoint --data /path/to/prepared/scene \
  --output /path/to/new/evaluation
```

Add `--python`, `--env` or `--prefix` when using a non-default environment. `--dry-run` validates inputs and prints commands without creating output or executing CUDA. Evaluation runs on held-out views and checks that a new results file contains finite PSNR, SSIM and LPIPS values. A zero exit status from an upstream metric script alone is not treated as success. Existing output directories are rejected; the original checkpoint is preserved.

Use a NerfBaselines checkpoint directory containing `nb-info.json`. The script runs `nerfbaselines render` with the Python backend, followed by `nerfbaselines evaluate`. It registers the evaluation-only `evaluation_spec.py`. The adapter restores the saved Gaussian tensors and 3D filter directly from the frozen checkpoint and delegates rendering to the original method. It fixes the old standalone loading path without changing training source or the checkpoint. This loader supports the frozen RobustNeRF checkpoints; appearance-conditioned PhotoTourism checkpoints are rejected explicitly. DINOv2 weights and Semantic-SAM masks are not needed for this saved-checkpoint inference. Training and mask preprocessing have separate requirements documented in the original README. Historical scripts in `scripts/` are not used by this entry point.
