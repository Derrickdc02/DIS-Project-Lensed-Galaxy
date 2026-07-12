#!/bin/bash
#! SLURM job script for Wilkes3 — unconditional prior sampling
#! Draws x ~ p(x) from the trained NCSN++ score prior for the PQMass check
#! (single A100; 1000 draws x 1000 steps).

#SBATCH -J probes_prior
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --no-requeue
#SBATCH -o slurm_logs/sample_prior_%j.out
#SBATCH -e slurm_logs/sample_prior_%j.err
#SBATCH -p ampere

#! sample_prior.py checkpoints each chunk to outputs/.../prior_check/chunks/ as it
#! finishes, so if the job hits the walltime before all chunks are done, just
#! resubmit this same script — completed chunks are skipped and it resumes.

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
VENV="${VENV:-$HOME/rds/hpc-work/venv/dis_proj}"
source "$VENV/bin/activate"

application="python"

options="src/sample_prior.py \
    --output_dir ./outputs/probes_final/prior_check \
    --ckpt ../latest.pt \
    --n_samples 1000 \
    --chunk 50 \
    --steps 4000 \
    --image_size 256 \
    --seed 21"

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
CMD="$application $options"


cd $workdir
mkdir -p slurm_logs
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

nvidia-smi

eval $CMD
