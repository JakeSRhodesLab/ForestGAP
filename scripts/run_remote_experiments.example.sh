#!/usr/bin/env bash
# Run the independent application runners on separate SSH hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOSTS_SPEC="${REMOTE_HOSTS:-${SCRIPT_DIR}/remote_hosts.example.txt}"
MANIFEST="$(mktemp "${TMPDIR:-/tmp}/forestgap-remote-jobs.XXXXXX.tsv")"
trap 'rm -f "${MANIFEST}"' EXIT

usage() {
    cat <<'EOF'
Usage: run_remote_experiments.example.sh [--] [experiment arguments]

Create six independent experiment jobs and distribute them with
dispatch_remote_jobs.sh. All arguments after -- are passed to every runner.
Without --, arguments are also forwarded for convenience.

Environment configuration:
  REMOTE_HOSTS              Host file or comma-separated host list.
    REMOTE_UV_BIN             uv executable available on every remote host.
  REMOTE_RUN_ID             Dispatcher run ID.
  REMOTE_MAX_HOSTS          Maximum number of configured hosts to use.
  REMOTE_WORKERS_PER_HOST   Local process-pool workers per remote job.
  REMOTE_RETRIES            Failed-job retry count.
  REMOTE_SKIP_UNAVAILABLE=1 Continue if a selected host fails preflight.
    REMOTE_ACCEPT_NEW_HOST_KEYS=1 Add previously unknown trusted SSH host keys.
  REMOTE_DRY_RUN=1          Preflight and print assignments without dispatching.
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == -- ]]; then
    shift
fi

experiment_args=""
for argument in "$@"; do
    printf -v quoted_argument '%q' "${argument}"
    experiment_args+=" ${quoted_argument}"
done

add_runner() {
    local job_id="$1"
    local runner="$2"
    printf '%s\tpython experiments/%s%s --workers "$HOST_WORKERS"\n' \
        "${job_id}" "${runner}" "${experiment_args}" >> "${MANIFEST}"
}

# These runners write distinct result directories, so they can execute together.
add_runner prediction-identity run_prediction_identity.py
add_runner imputation run_imputation.py
add_runner outlier-detection run_outlier_detection.py
add_runner prototypes run_prototypes.py
add_runner local-feature-importance run_local_feature_importance.py
add_runner mds-visualization run_mds_visualization.py

dispatch_args=(
    --jobs "${MANIFEST}"
    --hosts "${HOSTS_SPEC}"
    --repo-root "${REPO_ROOT}"
)
if [[ -n "${REMOTE_UV_BIN:-}" ]]; then
    dispatch_args+=(--uv-bin "${REMOTE_UV_BIN}")
fi
if [[ -n "${REMOTE_RUN_ID:-}" ]]; then
    dispatch_args+=(--run-id "${REMOTE_RUN_ID}")
fi
if [[ -n "${REMOTE_MAX_HOSTS:-}" ]]; then
    dispatch_args+=(--max-hosts "${REMOTE_MAX_HOSTS}")
fi
if [[ -n "${REMOTE_WORKERS_PER_HOST:-}" ]]; then
    dispatch_args+=(--workers-per-host "${REMOTE_WORKERS_PER_HOST}")
fi
if [[ -n "${REMOTE_RETRIES:-}" ]]; then
    dispatch_args+=(--retries "${REMOTE_RETRIES}")
fi
if [[ "${REMOTE_SKIP_UNAVAILABLE:-0}" == 1 ]]; then
    dispatch_args+=(--skip-unavailable)
fi
if [[ "${REMOTE_ACCEPT_NEW_HOST_KEYS:-0}" == 1 ]]; then
    dispatch_args+=(--accept-new-host-keys)
fi
if [[ "${REMOTE_DRY_RUN:-0}" == 1 ]]; then
    dispatch_args+=(--dry-run)
fi

exec "${SCRIPT_DIR}/dispatch_remote_jobs.sh" "${dispatch_args[@]}"