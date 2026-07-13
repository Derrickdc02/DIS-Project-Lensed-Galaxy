# Reproducibility artifacts

The machine-readable audit is in
[`artifacts/manifest.json`](https://github.com/Derrickdc02/DIS-Project-Lensed-Galaxy/blob/main/artifacts/manifest.json).
It records the checkpoint, representative posterior/prior draws, a posterior figure,
a chi-squared summary and a PQMass summary found in the project Google Drive.

## Publication status

As audited on 2026-07-13, Drive reports `shared: false` for every listed file. The
links therefore are not presented as public downloads, and the release gate remains
closed. The in-app Drive API cannot grant anonymous access, and the browser session
needed to change ?Anyone with the link? permissions was unavailable. The repository
records this limitation explicitly rather than publishing unverifiable links.

SHA-256 was calculated from raw Drive bytes for the chi-squared CSV, PQMass NPZ and
posterior PNG. The checkpoint and large sample tensors have `sha256: null` because
their local originals are absent and the Drive metadata connector does not expose a
SHA-256 field. Legacy files likewise have `source_git_sha: null`; timestamps are not
sufficient evidence for assigning a commit.

## Archived chi-squared baseline

The archived CSV contains 16 runs with 160 draws each. Posterior-mean reduced
chi-squared spans approximately 1.002 to 1.891; the `sample_pick70` run is the clear
high outlier. This is evidence that the historical artifacts include a calibration
failure and should not be summarised as uniformly ?near one?. Re-run the current
validation commands after publishing artifacts and report all runs, not only the
best-performing examples.

See `artifacts/README.md` for the exact permission, checksum, Git-SHA and signed-out
browser checks required before creating the release-candidate tag.
