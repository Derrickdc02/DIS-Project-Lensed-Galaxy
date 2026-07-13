# Posterior and prior sampling

## Posterior

The standard Wilkes3 entry point is `sbatch scripts/run_sample.sh`. For a custom run:

```bash
sample-posterior   --data_dir data/gals_gband_norm   --output_dir outputs/custom_posterior   --ckpt /absolute/path/to/latest.pt   --pick 70 --noise_sigma 0.02   --steps 8000 --n_post 160 --chunk 32 --seed 21
```

The source index addresses the sorted preprocessed `.npy` list. Each run stores the
truth, noisy observation, posterior chunks, a merged tensor and compact diagnostics.
Use a new output directory when changing any scientific parameter.

## Prior

Generate unconditional draws with the Slurm script or CLI:

```bash
sample-prior   --output_dir outputs/probes_final/prior_check   --ckpt /absolute/path/to/latest.pt   --n_samples 1000 --chunk 50 --steps 4000 --image_size 256 --seed 21
```

Both samplers are resumable at chunk boundaries. Resume is safe only when the model,
seed and scientific arguments are unchanged.
