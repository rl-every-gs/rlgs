# RL-MCMC

RL-MCMC applies the RLGS controller to 3D Gaussian Splatting as Markov Chain
Monte Carlo. The directory provides one supported training entry point and the
native rendering and evaluation workflow.

## Quick start

Run from this directory on Linux with an NVIDIA GPU:

```bash
bash setup.sh
conda activate rlgs-mcmc
bash test.sh

bash train.sh /path/to/scene ./output/scene
bash evaluate.sh ./output/scene 30000
```

The scene directory name selects `configs/<scene>.json`. If the directory has a
different name, set `SCENE`, for example:

```bash
SCENE=bicycle bash train.sh /path/to/custom-name ./output/bicycle
```

See [SETUP_AND_TEST.md](SETUP_AND_TEST.md) for environment selection, CUDA
checks, and isolated checkpoint evaluation.

This implementation is derived from
[3D Gaussian Splatting as Markov Chain Monte Carlo](https://github.com/ubc-vision/3dgs-mcmc)
and the original 3D Gaussian Splatting codebase. The inherited license permits
non-commercial research and evaluation; see the included license and
third-party notices.
