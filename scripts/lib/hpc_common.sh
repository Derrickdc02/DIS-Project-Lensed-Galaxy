#!/usr/bin/env bash
# Shared helpers for the Wilkes3 SLURM entry points.

hpc_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

hpc_require_command() {
    command -v "$1" >/dev/null 2>&1 || hpc_die "Required command not found: $1"
}

hpc_require_repo_submission() {
    local repo_root="$1"
    local submitted_from

    [[ -n "${SLURM_JOB_ID:-}" ]] || hpc_die "Run this script with sbatch, not directly."
    [[ -n "${SLURM_SUBMIT_DIR:-}" ]] || hpc_die "SLURM_SUBMIT_DIR is not set."

    submitted_from="$(realpath -m -- "$SLURM_SUBMIT_DIR")"
    repo_root="$(realpath -m -- "$repo_root")"
    if [[ "$submitted_from" != "$repo_root" ]]; then
        hpc_die "Submit from the repository root: cd $repo_root && sbatch scripts/<job>.sh"
    fi

    cd -- "$repo_root" || hpc_die "Unable to enter repository root: $repo_root"
}

hpc_load_modules() {
    [[ -r /etc/profile.d/modules.sh ]] || hpc_die "Environment Modules initialisation not found."

    set +u
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
    module purge
    module load rhel8/default-amp
    set -u
}

hpc_activate_environment() {
    VENV="${VENV:-$HOME/rds/hpc-work/venv/dis_proj}"
    [[ -r "$VENV/bin/activate" ]] || hpc_die "Virtual environment not found: $VENV"

    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    hpc_require_command python
}

hpc_require_dir() {
    [[ -d "$1" ]] || hpc_die "Required directory not found: $1"
}

hpc_require_file() {
    [[ -f "$1" ]] || hpc_die "Required file not found: $1"
}

hpc_prepare_output_dir() {
    mkdir -p -- "$1"
    [[ -w "$1" ]] || hpc_die "Output directory is not writable: $1"
}

hpc_print_environment() {
    local repo_root="$1"
    local git_state="clean"

    git -C "$repo_root" diff --quiet --ignore-submodules -- || git_state="dirty"
    git -C "$repo_root" diff --cached --quiet --ignore-submodules -- || git_state="dirty"

    printf 'Job ID: %s\n' "${SLURM_JOB_ID:-unknown}"
    printf 'Job name: %s\n' "${SLURM_JOB_NAME:-unknown}"
    printf 'Node list: %s\n' "${SLURM_JOB_NODELIST:-unknown}"
    printf 'Slurm tasks: %s\n' "${SLURM_NTASKS:-unknown}"
    printf 'Slurm CPUs per task: %s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
    printf 'Slurm GPUs: %s\n' "${SLURM_GPUS:-${SLURM_JOB_GPUS:-unknown}}"
    printf 'Submit directory: %s\n' "${SLURM_SUBMIT_DIR:-unknown}"
    printf 'Git commit: %s (%s)\n' "$(git -C "$repo_root" rev-parse HEAD)" "$git_state"
    printf 'Python: %s\n' "$(python --version 2>&1)"

    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for distribution in ("numpy", "torch", "score-models", "caustics"):
    try:
        value = version(distribution)
    except PackageNotFoundError:
        value = "not installed"
    print(f"{distribution}: {value}")
PY

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        printf 'nvidia-smi: not available\n'
    fi
}

hpc_print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

hpc_run() {
    hpc_print_command srun --ntasks=1 "$@"
    srun --ntasks=1 "$@"
}

hpc_begin() {
    local repo_root="$1"

    HPC_START_EPOCH="$(date +%s)"
    export HPC_START_EPOCH

    hpc_require_command realpath
    hpc_require_repo_submission "$repo_root"
    hpc_load_modules
    hpc_activate_environment
    hpc_require_command srun
    printf 'Started: %s\n' "$(date --iso-8601=seconds)"
    hpc_print_environment "$repo_root"
}

hpc_on_exit() {
    local status="$1"
    local end_epoch elapsed

    trap - EXIT
    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - HPC_START_EPOCH))
    printf 'Finished: %s\n' "$(date --iso-8601=seconds)"
    printf 'Elapsed seconds: %s\n' "$elapsed"
    printf 'Exit status: %s\n' "$status"
    exit "$status"
}
