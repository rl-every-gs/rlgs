# rl_tmgs: environment and one-command testing

Run these commands in this method directory. The wrappers locate repository files relative to themselves; no hostname or dataset path is hardcoded.

## New environment

```bash
bash setup.sh --dry-run
bash setup.sh
```

This creates the dedicated Conda environment `rlgs-tmgs` with Python 3.10, PyTorch 2.6.0 + CUDA 12.6 and the matching local CUDA extensions. Conda, an NVIDIA driver and a C++ compiler (`g++`) must already be installed. CUDA Toolkit 12.6 is installed inside the new environment. Use `--env NAME` or `--prefix /path/to/new/environment` to choose a different location; use `--jobs 2` to limit compiler parallelism.

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

## Training entry point

`train.py` is the only training entry point in this directory. It is byte-for-byte identical to the frozen TNT camera-ready RLGS trainer. The old Taming-3DGS baseline entry, experimental variants, unused helpers, search configurations, and legacy launchers have been archived outside this checkout. Training dependencies, `render.py`, `metrics.py`, `setup.sh` and `test.sh` remain in place.

Inspect the supported options in the activated environment:

```bash
python train.py --help
```

For the frozen TNT recipe, use the recorded hyperparameters through the reproduction runner. Set `PYTHON` explicitly to the environment you installed:

```bash
PYTHON=/absolute/path/to/environment/bin/python \
  bash reproduce_tnt.sh --scene train \
  --data-root /path/to/tandt --output-root /path/to/new/run \
  --expected-host "$(hostname -s)" --dry-run
```

Remove `--dry-run` to train, render and evaluate. Select `--scene truck` or `--scene all` as needed. The recipe and checksum manifest now refer to `train.py`; the trainer checksum and all training arguments are unchanged. The original v1 release archive retains its original filenames.

## Evaluate a trained checkpoint

To render and evaluate all four outputs produced by `train.sh`, use the public
evaluation wrapper. It sets the same deterministic CuBLAS environment as the
audited reproduction runner:

```bash
PYTHON=/absolute/path/to/environment/bin/python \
  OUTPUT_ROOT=/path/to/output/rl_tmgs \
  bash evaluate.sh
```

Set `ITERATION=N` to evaluate a saved iteration other than 30000.

To evaluate one checkpoint into a separate new directory, use the runtime test
wrapper below.

```bash
bash test.sh --model /path/to/checkpoint --data /path/to/prepared/scene \
  --output /path/to/new/evaluation
```

Add `--python`, `--env` or `--prefix` when using a non-default environment. `--dry-run` validates inputs and prints commands without creating output or executing CUDA. Evaluation runs on held-out views and checks that a new results file contains finite PSNR, SSIM and LPIPS values. A zero exit status from an upstream metric script alone is not treated as success. Existing output directories are rejected; the original checkpoint is preserved.

The checkpoint directory must contain `cfg_args` and `point_cloud/iteration_N/point_cloud.ply`. The latest saved iteration is selected by default; use `--iteration N` to select one. Saved image/resolution settings are reused; `--images images_2 --resolution 1` can override them. Rendering and metrics are written to the new evaluation directory, with only the original point cloud linked read-only by convention.
