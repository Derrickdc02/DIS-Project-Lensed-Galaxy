# Wilkes3 training and HPC scripts

Submit every job from the repository root so the safety check and Slurm log paths are
well defined:

```bash
sbatch scripts/run_stage2.sh
sbatch scripts/run_stage3.sh
sbatch scripts/run_sample.sh
sbatch scripts/run_sample_prior.sh
```

## Resource and runtime summary

| Script | Purpose | Slurm resources | Walltime | Default output |
|---|---|---|---:|---|
| `run_stage2.sh` | 128 px pilot training | 1 GPU, 8 CPUs | 08:30 | `outputs/probes_diffusion_subset` |
| `run_stage3.sh` | 256 px DDP training | 4 GPUs, 32 CPUs | 36:00 | `outputs/probes_final` |
| `run_sample.sh` | posterior draws | 1 GPU, 8 CPUs | 07:00 | `outputs/probes_final/sample_pick70` |
| `run_sample_prior.sh` | unconditional draws | 1 GPU, 8 CPUs | 12:00 | `outputs/probes_final/prior_check` |

Actual completion time depends on queue state, GPU type, dataset size and filesystem
load; the table records allocation limits, not guaranteed elapsed time.

## Path overrides

Only the runtime and path variables are designed for environment overrides:
`VENV`, `DATA_DIR`, `OUTPUT_DIR`, `CKPT`, `OMP_NUM_THREADS`, and `NCCL_DEBUG`.
For example:

```bash
sbatch --export=ALL,VENV=/home/user/venvs/dis,DATA_DIR=/rds/project/probes   scripts/run_stage2.sh

sbatch --export=ALL,VENV=/home/user/venvs/dis,CKPT=/rds/project/run/latest.pt,OUTPUT_DIR=/rds/project/posterior   scripts/run_sample.sh
```

Use absolute paths for checkpoints. Scientific hyperparameters remain explicit in the
version-controlled scripts so a log SHA determines the run configuration.

## Resume and shutdown behaviour

Stage 3 uses `--resume auto`: it loads `latest.pt` in the output directory when
present and otherwise starts a new run. Its Slurm limit is 36 hours while
`--max_hours 35` asks the training process to save and exit first. This one-hour
margin is intentional and reduces the risk of a forced termination during a write.
Posterior and prior samplers write numbered chunks and skip completed chunks when
restarted with the same output directory.

All scripts validate the repository root, virtual-environment activation file, data,
checkpoint and output paths before launching `srun`. Logs include Git SHA/dirty state,
Slurm allocation, Python and package versions, GPU information, the shell-escaped
command, timestamps, duration and exit status. `slurm_logs/.gitkeep` makes the log
directory available in a fresh clone while generated `.out` and `.err` files remain
ignored.
