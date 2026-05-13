#!/bin/bash
#!
#! SLURM job script for Wilkes3 — Stage 3 full reproduction
#! NCSN++ score-based prior on PROBES at 256×256 (4× A100 DDP, ~24-35 hours)
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J probes_3test
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MPHIL-DIS-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=4
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=20:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Output and error logs:
#SBATCH -o slurm_logs/stage3_%j.out
#SBATCH -e slurm_logs/stage3_%j.err

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.
#! Stage 3 uses all 4 A100s on one node. If the run doesn't finish within
#! 35 hours (max_hours), it saves a checkpoint and exits cleanly — just
#! resubmit this same script and training will resume from where it left off.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
#! Activate your Python environment — EDIT THIS LINE to match your setup:
source /home/yd388/rds/hpc-work/venv/dis_proj/bin/activate
#! Or if using conda:
#! source /path/to/miniconda3/etc/profile.d/conda.sh
#! conda activate probes

#! Full path to application executable:
#! Stage 3 uses torchrun to launch 4 DDP workers (one per GPU)
application="torchrun"

#! Run options for the application:
options="--standalone --nproc_per_node=4 train_prior.py \
    --data_dir ./data/gals_gband_norm \
    --output_dir ./outputs/probes_diffusion_final \
    --image_size 256 \
    --nf 128 \
    --ch_mult 1 1 2 2 2 2 2 \
    --sigma_min 1e-4 \
    --epochs 2700 \
    --batch_size 4 \
    --lr 1e-4 \
    --ema_decay 0.9999 \
    --warmup 5000 \
    --clip 1.0 \
    --max_hours 35.0 \
    --ckpt_every_steps 1000 \
    --log_every_steps 50 \
    --keep_last_n 3 \
    --seed 21"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=8

#! NCCL settings recommended for DDP on Wilkes3 InfiniBand:
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
mkdir -p slurm_logs
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

nvidia-smi

eval $CMD