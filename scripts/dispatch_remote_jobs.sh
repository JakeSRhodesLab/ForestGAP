#!/usr/bin/env bash
# Dispatch independent, trusted shell commands across explicitly named SSH hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JOBS_PATH=""
declare -a DIRECT_COMMANDS=()
HOSTS_SPEC=""
REPO_ROOT="${DEFAULT_REPO_ROOT}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_DIR=""
MAX_HOSTS=0
WORKERS_PER_HOST=0
RETRIES=0
DRY_RUN=false
PREFLIGHT_ONLY=false
SKIP_UNAVAILABLE=false
ACCEPT_NEW_HOST_KEYS=false

usage() {
    cat <<'EOF'
Usage: dispatch_remote_jobs.sh (--jobs JOBS.tsv | --command COMMAND ...) --hosts HOSTS [options]

Schedule trusted, independent commands over passwordless SSH. Each non-comment
manifest row is `job_id<TAB>command`; commands execute in --repo-root through
its local uv project environment. Job IDs must match [A-Za-z0-9][A-Za-z0-9._-]*.

Required:
    --jobs PATH                 Tab-separated job manifest; cannot use --command.
    --command COMMAND           Trusted command to dispatch; repeat for each job.
  --hosts PATH|HOSTS          Host file or comma-separated SSH host names.

Options:
  --repo-root PATH            Shared checkout path on every host.
    --uv-bin PATH               uv executable on every host (default: controller's uv).
  --run-id ID                 Artifact directory name (default: UTC timestamp).
  --runs-dir PATH             Controller and remote artifact base directory.
  --max-hosts N               Use at most the first N configured hosts.
  --workers-per-host N        Override local process-pool workers per job.
  --retries N                 Retry a failed job at most N times (default: 0).
  --dry-run                   Preflight and print assignments without running jobs.
  --preflight-only            Validate selected hosts, then exit.
  --skip-unavailable          Exclude failed preflight hosts instead of failing.
    --accept-new-host-keys      Add previously unknown host keys; reject changed keys.
  --help                      Show this help.

Host file format: one `ssh-target [slots] [workers-per-slot]` entry per line. A slot
allows one simultaneous heavy job. Without an explicit worker value, each slot
gets floor(remote CPUs / slots), at least one worker.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

note() {
    printf '[dispatch] %s\n' "$*"
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

command_digest() {
    printf '%s' "$1" | sha256sum | awk '{print $1}'
}

while (($#)); do
    case "$1" in
        --jobs)
            (($# >= 2)) || die '--jobs requires a path'
            JOBS_PATH="$2"
            shift 2
            ;;
        --command)
            (($# >= 2)) || die '--command requires a command string'
            DIRECT_COMMANDS+=("$2")
            shift 2
            ;;
        --hosts)
            (($# >= 2)) || die '--hosts requires a host file or comma-separated hosts'
            HOSTS_SPEC="$2"
            shift 2
            ;;
        --repo-root)
            (($# >= 2)) || die '--repo-root requires a path'
            REPO_ROOT="$2"
            shift 2
            ;;
        --uv-bin)
            (($# >= 2)) || die '--uv-bin requires a path'
            UV_BIN="$2"
            shift 2
            ;;
        --run-id)
            (($# >= 2)) || die '--run-id requires a value'
            RUN_ID="$2"
            shift 2
            ;;
        --runs-dir)
            (($# >= 2)) || die '--runs-dir requires a path'
            RUNS_DIR="$2"
            shift 2
            ;;
        --max-hosts)
            (($# >= 2)) || die '--max-hosts requires a number'
            MAX_HOSTS="$2"
            shift 2
            ;;
        --workers-per-host)
            (($# >= 2)) || die '--workers-per-host requires a number'
            WORKERS_PER_HOST="$2"
            shift 2
            ;;
        --retries)
            (($# >= 2)) || die '--retries requires a number'
            RETRIES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=true
            shift
            ;;
        --skip-unavailable)
            SKIP_UNAVAILABLE=true
            shift
            ;;
        --accept-new-host-keys)
            ACCEPT_NEW_HOST_KEYS=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ -n "${JOBS_PATH}" && ${#DIRECT_COMMANDS[@]} -gt 0 ]]; then
    die '--jobs and --command are mutually exclusive'
fi
if [[ -z "${JOBS_PATH}" && ${#DIRECT_COMMANDS[@]} -eq 0 ]]; then
    die 'provide --jobs or at least one --command'
fi
[[ -n "${HOSTS_SPEC}" ]] || die '--hosts is required'
if [[ -n "${JOBS_PATH}" ]]; then
    [[ -f "${JOBS_PATH}" ]] || die "job manifest does not exist: ${JOBS_PATH}"
fi
[[ -d "${REPO_ROOT}" ]] || die "repository root does not exist: ${REPO_ROOT}"
[[ -n "${UV_BIN}" ]] || die 'uv is unavailable; install uv or pass --uv-bin'
is_nonnegative_integer "${MAX_HOSTS}" || die '--max-hosts must be non-negative'
is_nonnegative_integer "${WORKERS_PER_HOST}" || die '--workers-per-host must be non-negative'
is_nonnegative_integer "${RETRIES}" || die '--retries must be non-negative'
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'run ID contains unsafe characters'

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/results/remote_runs}"
mkdir -p "${RUNS_DIR}"
RUNS_DIR="$(cd "${RUNS_DIR}" && pwd)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOCAL_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null)" || die 'repository root is not a Git checkout'
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
if [[ "${ACCEPT_NEW_HOST_KEYS}" == true ]]; then
    SSH_OPTIONS+=(-o StrictHostKeyChecking=accept-new)
fi
LOCAL_HOST_SHORT="$(hostname -s)"
LOCAL_HOST_NAME="$(hostname)"

if [[ -e "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "run directory already exists; select a new --run-id: ${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/jobs"

STATUS_PATH="${RUN_DIR}/status.tsv"
HOST_STATUS_PATH="${RUN_DIR}/hosts.tsv"
QUEUE_PATH="${RUN_DIR}/queue.tsv"
QUEUE_NEXT_PATH="${RUN_DIR}/queue.next"
QUEUE_LOCK_PATH="${RUN_DIR}/queue.lock"
STATUS_LOCK_PATH="${RUN_DIR}/status.lock"
FAILURE_PATH="${RUN_DIR}/failures.tsv"
STOP_PATH="${RUN_DIR}/stop-requested"
printf 'event\tjob_id\thost\tslot\tattempt\tstarted_at\tfinished_at\truntime_seconds\treturn_code\tcommand_sha256\thost_workers\n' > "${STATUS_PATH}"
printf 'host\tslots\tworkers_per_slot\tremote_cpus\tremote_host\trevision\tstatus\treason\n' > "${HOST_STATUS_PATH}"
: > "${FAILURE_PATH}"

declare -a JOB_IDS=()
declare -a JOB_COMMANDS=()
declare -A SEEN_JOB_IDS=()
add_job() {
    local job_id="$1"
    local command="$2"
    [[ "${job_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe job ID: ${job_id}"
    [[ -n "$(trim "${command}")" ]] || die "empty command for job: ${job_id}"
    [[ -z "${SEEN_JOB_IDS[${job_id}]+x}" ]] || die "duplicate job ID: ${job_id}"
    SEEN_JOB_IDS["${job_id}"]=1
    JOB_IDS+=("${job_id}")
    JOB_COMMANDS+=("${command}")
}

if [[ -n "${JOBS_PATH}" ]]; then
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
        line="${raw_line%$'\r'}"
        stripped="$(trim "${line}")"
        [[ -z "${stripped}" || "${stripped}" == \#* ]] && continue
        [[ "${line}" == *$'\t'* ]] || die "manifest row lacks a tab delimiter: ${line}"
        add_job "${line%%$'\t'*}" "${line#*$'\t'}"
    done < "${JOBS_PATH}"
else
    for index in "${!DIRECT_COMMANDS[@]}"; do
        add_job "command-$((index + 1))" "${DIRECT_COMMANDS[index]}"
    done
fi

((${#JOB_IDS[@]})) || die 'job manifest contains no jobs'

declare -a HOST_NAMES=()
declare -a HOST_SLOTS=()
declare -a HOST_WORKER_CAPS=()
declare -A SEEN_HOSTS=()

add_host() {
    local host="$1"
    local slots="$2"
    local workers="$3"
    [[ "${host}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe SSH target: ${host}"
    is_positive_integer "${slots}" || die "invalid slot count for ${host}: ${slots}"
    is_nonnegative_integer "${workers}" || die "invalid worker count for ${host}: ${workers}"
    [[ -z "${SEEN_HOSTS[${host}]+x}" ]] || die "duplicate host: ${host}"
    SEEN_HOSTS["${host}"]=1
    HOST_NAMES+=("${host}")
    HOST_SLOTS+=("${slots}")
    HOST_WORKER_CAPS+=("${workers}")
}

if [[ -f "${HOSTS_SPEC}" ]]; then
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
        line="${raw_line%$'\r'}"
        stripped="$(trim "${line}")"
        [[ -z "${stripped}" || "${stripped}" == \#* ]] && continue
        read -r host slots workers extra <<< "${stripped}"
        [[ -z "${extra:-}" ]] || die "invalid host row: ${line}"
        add_host "${host}" "${slots:-1}" "${workers:-0}"
    done < "${HOSTS_SPEC}"
else
    IFS=',' read -r -a requested_hosts <<< "${HOSTS_SPEC}"
    for host in "${requested_hosts[@]}"; do
        add_host "$(trim "${host}")" 1 0
    done
fi

((${#HOST_NAMES[@]})) || die 'no hosts were selected'
if ((MAX_HOSTS > 0 && ${#HOST_NAMES[@]} > MAX_HOSTS)); then
    HOST_NAMES=("${HOST_NAMES[@]:0:MAX_HOSTS}")
    HOST_SLOTS=("${HOST_SLOTS[@]:0:MAX_HOSTS}")
    HOST_WORKER_CAPS=("${HOST_WORKER_CAPS[@]:0:MAX_HOSTS}")
fi

append_host_status() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${HOST_STATUS_PATH}"
}

is_local_target() {
    local target="$1"
    if [[ "${target}" == *@* && "${target%%@*}" != "${USER}" ]]; then
        return 1
    fi
    target="${target#*@}"
    [[ "${target}" == "${LOCAL_HOST_SHORT}" || "${target}" == "${LOCAL_HOST_NAME}" ]]
}

run_host_script() {
    local host="$1"
    shift
    if is_local_target "${host}"; then
        env "$@" bash -s
        return
    fi
    local remote_assignments
    printf -v remote_assignments '%q ' "$@"
    ssh "${SSH_OPTIONS[@]}" "${host}" "${remote_assignments}bash -s"
}

preflight_host() {
    local host="$1"
    local slots="$2"
    local configured_workers="$3"
    local output marker revision cpus remote_host extra workers
    if ! output="$(run_host_script "${host}" \
        "REPO_ROOT=${REPO_ROOT}" "RUN_DIR=${RUN_DIR}" "UV_BIN=${UV_BIN}" "EXPECTED_REVISION=${LOCAL_REVISION}" 2>&1 <<'REMOTE'
set -euo pipefail
command -v bash >/dev/null
command -v "${UV_BIN}" >/dev/null
[[ -d "${REPO_ROOT}" ]]
revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${revision}" == "${EXPECTED_REVISION}" ]]
cpus="$(nproc)"
[[ "${cpus}" =~ ^[1-9][0-9]*$ ]]
mkdir -p "${RUN_DIR}/jobs"
probe_file="${RUN_DIR}/.preflight-${HOSTNAME}-$$"
touch "${probe_file}"
rm -f "${probe_file}"
cd "${REPO_ROOT}"
unset PYTHONPATH
PYTHONNOUSERSITE=1 "${UV_BIN}" run --no-sync --project "${REPO_ROOT}" python -c 'import forestgap'
printf 'ok\t%s\t%s\t%s\n' "${revision}" "${cpus}" "$(hostname -s)"
REMOTE
    )"; then
        append_host_status "${host}" "${slots}" "${configured_workers}" '' '' '' unavailable "${output//$'\n'/ }"
        return 1
    fi
    IFS=$'\t' read -r marker revision cpus remote_host extra <<< "${output}"
    if [[ "${marker}" != ok || -n "${extra:-}" ]] || ! is_positive_integer "${cpus}"; then
        append_host_status "${host}" "${slots}" "${configured_workers}" '' '' '' unavailable "unexpected preflight output: ${output//$'\n'/ }"
        return 1
    fi
    workers="${WORKERS_PER_HOST}"
    if ((workers == 0)); then
        workers="${configured_workers}"
    fi
    if ((workers == 0)); then
        workers=$((cpus / slots))
        ((workers > 0)) || workers=1
    fi
    append_host_status "${host}" "${slots}" "${workers}" "${cpus}" "${remote_host}" "${revision}" ok ''
    printf '%s\t%s\t%s\n' "${host}" "${slots}" "${workers}"
}

declare -a ACTIVE_HOSTS=()
declare -a ACTIVE_SLOTS=()
declare -a ACTIVE_WORKERS=()
for index in "${!HOST_NAMES[@]}"; do
    host="${HOST_NAMES[index]}"
    slots="${HOST_SLOTS[index]}"
    configured_workers="${HOST_WORKER_CAPS[index]}"
    note "preflight ${host}"
    if preflight_line="$(preflight_host "${host}" "${slots}" "${configured_workers}")"; then
        IFS=$'\t' read -r host slots workers <<< "${preflight_line}"
        ACTIVE_HOSTS+=("${host}")
        ACTIVE_SLOTS+=("${slots}")
        ACTIVE_WORKERS+=("${workers}")
    elif [[ "${SKIP_UNAVAILABLE}" == true ]]; then
        note "excluding unavailable host ${host}"
    else
        die "preflight failed for ${host}; pass --skip-unavailable to continue without it"
    fi
done

((${#ACTIVE_HOSTS[@]})) || die 'no hosts passed preflight'
if [[ "${PREFLIGHT_ONLY}" == true ]]; then
    note "preflight completed; host details: ${HOST_STATUS_PATH}"
    exit 0
fi

printf '%s\n' "${LOCAL_REVISION}" > "${RUN_DIR}/revision.txt"
printf '%s\n' "${JOBS_PATH}" > "${RUN_DIR}/jobs-source.txt"
printf '%s\n' "${HOSTS_SPEC}" > "${RUN_DIR}/hosts-source.txt"
: > "${QUEUE_PATH}"
for index in "${!JOB_IDS[@]}"; do
    printf '%s\t%s\n' "${JOB_IDS[index]}" "${JOB_COMMANDS[index]}" >> "${QUEUE_PATH}"
done
printf '1\n' > "${QUEUE_NEXT_PATH}"
: > "${QUEUE_LOCK_PATH}"
: > "${STATUS_LOCK_PATH}"
rm -f "${STOP_PATH}"

write_status() {
    local lock_fd
    exec {lock_fd}>"${STATUS_LOCK_PATH}"
    flock -x "${lock_fd}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${STATUS_PATH}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

record_failure() {
    local lock_fd
    exec {lock_fd}>"${STATUS_LOCK_PATH}"
    flock -x "${lock_fd}"
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${FAILURE_PATH}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

claim_next_job() {
    local lock_fd next_line total line
    exec {lock_fd}>"${QUEUE_LOCK_PATH}"
    flock -x "${lock_fd}"
    if [[ -e "${STOP_PATH}" ]]; then
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    next_line="$(<"${QUEUE_NEXT_PATH}")"
    total="$(wc -l < "${QUEUE_PATH}")"
    if ((next_line > total)); then
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    line="$(sed -n "${next_line}p" "${QUEUE_PATH}")"
    printf '%s\n' "$((next_line + 1))" > "${QUEUE_NEXT_PATH}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
    CLAIMED_JOB_ID="${line%%$'\t'*}"
    CLAIMED_COMMAND="${line#*$'\t'}"
}

run_remote_job() {
    local host="$1"
    local slot="$2"
    local workers="$3"
    local job_id="$4"
    local command="$5"
    local digest attempt maximum_attempts rc=1
    digest="$(command_digest "${command}")"
    attempt=1
    maximum_attempts=$((RETRIES + 1))
    while ((attempt <= maximum_attempts)); do
        [[ ! -e "${STOP_PATH}" ]] || return 130
        local started_at finished_at start_seconds finish_seconds runtime_seconds log_path
        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        start_seconds="$(date +%s)"
        log_path="${RUN_DIR}/logs/${job_id}.attempt-${attempt}.log"
        write_status started "${job_id}" "${host}" "${slot}" "${attempt}" "${started_at}" '' '' '' "${digest}" "${workers}"
        note "starting ${job_id} on ${host} slot=${slot} workers=${workers} attempt=${attempt}"
        set +e
        run_host_script "${host}" \
            "JOB_ID=${job_id}" "RUN_ID=${RUN_ID}" "RUN_DIR=${RUN_DIR}" "JOB_DIR=${RUN_DIR}/jobs/${job_id}" "REPO_ROOT=${REPO_ROOT}" "UV_BIN=${UV_BIN}" "HOST_WORKERS=${workers}" "REMOTE_COMMAND=${command}" <<'REMOTE' 2>&1 | tee "${log_path}" | sed -u "s/^/[${host}:${job_id}] /"
set -euo pipefail
export JOB_ID RUN_ID RUN_DIR JOB_DIR REPO_ROOT HOST_WORKERS
export HOST_NAME="$(hostname -s)"
mkdir -p "${JOB_DIR}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export FORESTGAP_N_JOBS=1
cd "${REPO_ROOT}"
exec "${UV_BIN}" run --no-sync --project "${REPO_ROOT}" bash -lc "${REMOTE_COMMAND}"
REMOTE
        rc=${PIPESTATUS[0]}
        set -e
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        finish_seconds="$(date +%s)"
        runtime_seconds=$((finish_seconds - start_seconds))
        write_status finished "${job_id}" "${host}" "${slot}" "${attempt}" "${started_at}" "${finished_at}" "${runtime_seconds}" "${rc}" "${digest}" "${workers}"
        if ((rc == 0)); then
            note "finished ${job_id} on ${host} in ${runtime_seconds}s"
            return 0
        fi
        if ((attempt < maximum_attempts)); then
            note "retrying ${job_id} after exit ${rc}"
            write_status retrying "${job_id}" "${host}" "${slot}" "${attempt}" "${started_at}" "${finished_at}" "${runtime_seconds}" "${rc}" "${digest}" "${workers}"
        fi
        ((attempt++))
    done
    record_failure "${job_id}" "${host}" "${rc}"
    note "failed ${job_id} on ${host}; log=${log_path}"
    return "${rc}"
}

worker_lane() {
    local host="$1"
    local slot="$2"
    local workers="$3"
    while claim_next_job; do
        run_remote_job "${host}" "${slot}" "${workers}" "${CLAIMED_JOB_ID}" "${CLAIMED_COMMAND}" || true
    done
}

cleanup() {
    if [[ ! -e "${STOP_PATH}" ]]; then
        note 'interrupt received; no new jobs will be scheduled'
        : > "${STOP_PATH}"
    fi
}
trap cleanup INT TERM

if [[ "${DRY_RUN}" == true ]]; then
    note "dry run: ${#JOB_IDS[@]} jobs across ${#ACTIVE_HOSTS[@]} reachable hosts"
    next_host=0
    for index in "${!JOB_IDS[@]}"; do
        host="${ACTIVE_HOSTS[next_host]}"
        workers="${ACTIVE_WORKERS[next_host]}"
        printf 'DRY-RUN\t%s\t%s\tworkers=%s\t%s\n' "${JOB_IDS[index]}" "${host}" "${workers}" "${JOB_COMMANDS[index]}"
        next_host=$(((next_host + 1) % ${#ACTIVE_HOSTS[@]}))
    done
    exit 0
fi

declare -a LANE_PIDS=()
for index in "${!ACTIVE_HOSTS[@]}"; do
    host="${ACTIVE_HOSTS[index]}"
    slots="${ACTIVE_SLOTS[index]}"
    workers="${ACTIVE_WORKERS[index]}"
    for ((slot = 1; slot <= slots; slot++)); do
        worker_lane "${host}" "${slot}" "${workers}" &
        LANE_PIDS+=("$!")
    done
done

for lane_pid in "${LANE_PIDS[@]}"; do
    wait "${lane_pid}" || true
done

if [[ -s "${FAILURE_PATH}" ]]; then
    note "one or more jobs failed; details: ${FAILURE_PATH}"
    exit 1
fi
if [[ -e "${STOP_PATH}" ]]; then
    note 'dispatch stopped before the queue was exhausted'
    exit 130
fi
note "all jobs completed; status: ${STATUS_PATH}"