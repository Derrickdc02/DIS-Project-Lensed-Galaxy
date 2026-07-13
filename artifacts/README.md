# Reproducibility artifact record

`manifest.json` records the checkpoint and representative generated outputs retained
in private Google Drive storage. Private Drive links and file IDs are deliberately not
published in this repository.

## Access policy

Checkpoints and large generated artifacts are available for academic reproducibility
upon reasonable request. Email **derricktang02@gmail.com** with:

- your name;
- your institution or affiliation; and
- a short description of the intended reproducibility use.

Approved requests receive individual, read-only Google Drive access. The files do not
need to be publicly accessible because the code, environment, commands and access
procedure are documented, and every artifact can be regenerated with the
version-controlled workflows.

## Run records

The manifest records file names, sizes, timestamps, generation notes and the current
reproduction command. Legacy artifacts predate the standard Slurm provenance log.
New jobs record the Git commit, dirty state, environment, command, resources and
timing in `slurm_logs/`; retain the matching log when sharing results from a new run.
