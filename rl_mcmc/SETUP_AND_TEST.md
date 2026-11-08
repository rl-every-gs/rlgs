# rl_mcmc: environment and one-command testing

Run these commands in this method directory. The wrappers locate repository files relative to themselves; no hostname or dataset path is hardcoded.

## New environment

```bash
bash setup.sh --dry-run
bash setup.sh
```

This creates the dedicated Conda environment `rlgs-mcmc` with Python 3.10, PyTorch 2.6.0 + CUDA 12.6 and the matching local CUDA extensions. Conda, an NVIDIA driver and a C++ compiler (`g++`) must already be installed. CUDA Toolkit 12.6 is installed inside the new environment. Use `--env NAME` or `--prefix /path/to/new/environment` to choose a different location; use `--jobs 2` to limit compiler parallelism.

Existing environments are not modified or reinstalled. To inspect one:

```bash
bash setup.sh --check-only --python /absolute/path/to/environment/bin/python
```

The wrappers use `python3` as their bootstrap interpreter by default. On hosts
where it is not on `PATH`, set `PYTHON` or `PYTHON_BOOTSTRAP` to an absolute
Python path; the same value can also be passed through `--python` for checks.

## Train one scene

Pass the dataset and output paths explicitly. The released scene configuration
is selected from `configs/` using the dataset directory name.

```bash
bash train.sh /path/to/tandt/train /path/to/new/output
```

If the dataset directory has a different name, set `SCENE=train` (or another
released scene name) before the command. Additional trainer arguments can be
appended.

## Evaluate one scene

RL-MCMC uses the native MCMC evaluation sequence: render the held-out views,
then compute PSNR, SSIM and LPIPS.

```bash
bash evaluate.sh /path/to/model 30000
```

The model directory must contain `cfg_args` and
`point_cloud/iteration_30000/point_cloud.ply`. The original dataset path is
read from `cfg_args`, and `results.json` is written below the model directory.

## Quick CUDA test

```bash
bash test.sh
# Or use an existing compatible environment:
bash test.sh --python /absolute/path/to/environment/bin/python
```

The default test imports this method's model/policy code and runs a tiny KNN plus rasterizer forward/backward check. It uses eight synthetic points only as a kernel test, writes no scene result, and does not train a model or download a dataset. It does not measure reconstruction quality or validate RL convergence. `--imports-only` skips GPU execution and is explicitly reported as an import-only check.

## Optional isolated checkpoint validation

```bash
bash test.sh --model /path/to/checkpoint --data /path/to/prepared/scene \
  --output /path/to/new/evaluation
```

This shared RLGS validation wrapper is useful when the original checkpoint must
remain untouched. It is separate from the recommended native `evaluate.sh`
path above. Add `--python`, `--env` or `--prefix` when using a non-default
environment. `--dry-run` validates inputs and prints commands without creating
output or executing CUDA. Existing output directories are rejected; the
original checkpoint is preserved.

The checkpoint directory must contain `cfg_args` and `point_cloud/iteration_N/point_cloud.ply`. The latest saved iteration is selected by default; use `--iteration N` to select one. Saved image/resolution settings are reused; `--images images_2 --resolution 1` can override them. Rendering and metrics are written to the new evaluation directory, with only the original point cloud linked read-only by convention.
