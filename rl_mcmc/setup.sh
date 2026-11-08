#!/usr/bin/env bash
set -Eeuo pipefail
method_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bootstrap_python="${PYTHON_BOOTSTRAP:-${PYTHON:-python3}}"
exec "$bootstrap_python" "$method_dir/../tools/runtime.py" rl_mcmc setup "$@"
