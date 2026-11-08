#!/usr/bin/env bash
# Frozen RLGS-TMGS TNT reproduction: download -> train -> render -> metrics.
# The validated 4dgs runtime is explicit; never use whichever Python is on PATH.
set -Eeuo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-/home/stack/miniconda3/envs/4dgs/bin/python}
if [[ ! -x "$python_bin" ]]; then
    printf 'Missing validated Python: %s\n' "$python_bin" >&2
    printf '%s\n' \
        'Create environment_rl-tmgs.yml and build its three local CUDA extensions first.' \
        'On another installation, set PYTHON=/absolute/path/to/4dgs/bin/python.' >&2
    exit 1
fi
export MKL_THREADING_LAYER=GNU
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" "$project_root/reproducibility/tnt_camera_ready.py" "$@"
