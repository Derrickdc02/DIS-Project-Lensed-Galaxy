#!/usr/bin/env bash
#SBATCH --job-name=probes_stage3
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/probes_final}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

hpc_begin "$REPO_ROOT"
hpc_require_command torchrun
hpc_require_dir "$DATA_DIR"
hpc_prepare_output_dir "$OUTPUT_DIR"

args=(
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --epochs 2700
    --image_size 256
    --nf 128
    --ch_mult 1 1 2 2 2 2 2
    --batch_size 4
    --ckpt_every_steps 1000
    --n_subset -1
    --log_every_steps 50
    --keep_last_n 3
    --max_hours 35
    --clip 1.0
    --lr 1e-4
    --ema_decay 0.9999
    --sigma_min 1e-4
    --sigma_max 263.4
    --resume auto
    --seed 21
)

hpc_run torchrun --standalone --nproc_per_node=4 -m train_prior "${args[@]}"
