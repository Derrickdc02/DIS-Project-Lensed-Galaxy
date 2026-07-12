#!/usr/bin/env bash
#SBATCH --job-name=probes_pick70
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/lib/hpc_common.sh
source "$SCRIPT_DIR/lib/hpc_common.sh"

HPC_START_EPOCH="$(date +%s)"
export HPC_START_EPOCH
trap 'hpc_on_exit $?' EXIT

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/gals_gband_norm}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/probes_final/sample_pick70}"
CKPT="${CKPT:-$REPO_ROOT/outputs/probes_final/latest.pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hpc_begin "$REPO_ROOT"
hpc_require_dir "$DATA_DIR"
hpc_require_file "$CKPT"
hpc_prepare_output_dir "$OUTPUT_DIR"

args=(
    --output_dir "$OUTPUT_DIR"
    --data_dir "$DATA_DIR"
    --ckpt "$CKPT"
    --steps 8000
    --n_post 160
    --chunk 32
    --pick 70
    --noise_sigma 0.02
    --seed 21
)

hpc_run python -u -m sample "${args[@]}"
