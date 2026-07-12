#!/usr/bin/env bash
#SBATCH --job-name=probes_stage2
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:30:00
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/probes_diffusion_subset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hpc_begin "$REPO_ROOT"
hpc_require_dir "$DATA_DIR"
hpc_prepare_output_dir "$OUTPUT_DIR"

args=(
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --image_size 128
    --nf 128
    --ch_mult 1 2 2 2
    --sigma_min 1e-4
    --epochs 1000
    --batch_size 64
    --lr 2e-4
    --seed 21
)

hpc_run python -u -m lowres_sample_train "${args[@]}"
