#!/usr/bin/env bash
set -Eeuo pipefail
method_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$method_dir/../tools/runtime.py" rl_asymgs test "$@"
