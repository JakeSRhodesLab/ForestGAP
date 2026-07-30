#!/usr/bin/env bash
set -euo pipefail

# Runs the full experiment launcher with user/global Python path injection
# disabled to avoid ABI conflicts.
ENV_NAME="${GENERALFOREST_ENV:-generalforest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
unset PYTHONPATH
PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" python experiments/run_all.py "$@"
