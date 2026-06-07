#!/bin/bash
#! SLURM job script for Wilkes3 — posterior sampling
#! Convolved-likelihood source reconstruction from a lensed observation
#! (single A100; 160 draws x 8000 steps).

#SBATCH -J probes_srcfov
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --no-requeue
#SBATCH -o slurm_logs/sample_%j.out
#SBATCH -e slurm_logs/sample_%j.err
#SBATCH -p ampere

#! sample.py checkpoints each chunk to outputs/.../samples/chunks/ as it
#! finishes, so if the job hits the walltime before all chunks are done, just
#! resubmit this same script — completed chunks are skipped and it resumes.

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
source /home/yd388/rds/hpc-work/venv/dis_proj/bin/activate

application="python"

#! These options reproduce the notebooks/full_sample.ipynb full run.
options="src/sample.py \
    --output_dir ./outputs/probes_final/sample_srcfov \
    --data_dir ./data/gals_gband_norm \
    --ckpt ../latest.pt \
    --steps 8000 \
    --n_post 160 \
    --chunk 32 \
    --pick 15 \
    --noise_sigma 0.02 \
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
