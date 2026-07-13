# Reproducibility artifacts

The machine-readable inventory is in
[`artifacts/manifest.json`](https://github.com/Derrickdc02/DIS-Project-Lensed-Galaxy/blob/main/artifacts/manifest.json).
It records the checkpoint, representative posterior/prior draws, a posterior figure,
a chi-squared summary and a PQMass summary retained in private Google Drive storage.

## Access policy

The files are not publicly downloadable. They are available for academic
reproducibility upon reasonable request. Email **derricktang02@gmail.com** with your
name, affiliation and intended reproducibility use. Approved requests receive
individual read-only access.

Private Drive links and file IDs are intentionally omitted from the public repository.
The documented access process, version-controlled generation commands and CPU-tested
environment form the artifact access policy.

The manifest records file names, sizes, timestamps, generation notes and reproduction
commands. Legacy artifacts predate the standard Slurm provenance log; new jobs record
the Git commit, dirty state, packages, resources, command and timing.

## Archived chi-squared baseline

The archived CSV contains 16 runs with 160 draws each. Posterior-mean reduced
chi-squared spans approximately 1.002 to 1.891; the `sample_pick70` run is the clear
high outlier. The historical artifacts therefore include a calibration failure and
must not be summarised as uniformly near one. Report all runs when reproducing the
validation.
