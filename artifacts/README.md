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
need to be publicly accessible for this release because the code, environment,
commands and access procedure are documented, and every artifact can be regenerated
with the version-controlled workflows.

## Integrity and provenance

SHA-256 is recorded for the three small artifacts whose raw bytes were available
during the audit. The three large legacy tensors remain without SHA-256 because their
owner chose not to download them solely to calculate a checksum. This is disclosed in
the manifest and is not a publication blocker under the private-on-request policy.
Recipients can still calculate a local checksum after download and report any transfer
problem.

Legacy artifacts also predate the new Slurm provenance logging, so their generating
Git SHA is unknown. New jobs record the Git SHA, dirty state, environment, command,
resources and timing in `slurm_logs/`; future shared artifacts should retain that log.
