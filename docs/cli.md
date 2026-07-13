# Command-line reference

Installation creates these stable commands:

| Command | Role | Hardware expectation |
|---|---|---|
| `preprocess-probes` | raw FITS to normalised NumPy arrays | CPU |
| `train-prior-lowres` | 128 px pilot training | one GPU |
| `train-prior` | resumable 256 px training | four GPUs via `torchrun` |
| `sample-posterior` | likelihood-conditioned posterior draws | one GPU |
| `sample-prior` | unconditional prior draws | one GPU |
| `validate-chi2` | posterior-predictive chi-squared | CPU/GPU by saved tensors |
| `validate-pqmass` | pixel and PCA PQMass | CPU |
| `validate-mira` | posterior calibration scoring | CPU or GPU |
| `reproduce-figure2` | OOD noise-sweep figure | one GPU |

The executable help text is the source of truth for detailed options:

```bash
preprocess-probes --help
train-prior --help
train-prior-lowres --help
sample-posterior --help
sample-prior --help
validate-chi2 --help
validate-pqmass --help
validate-mira --help
reproduce-figure2 --help
```

Underscore option names are retained for compatibility with existing Slurm runs;
new validation commands use hyphenated option names. A non-zero exit status indicates
invalid arguments, missing inputs or a failed workflow.
