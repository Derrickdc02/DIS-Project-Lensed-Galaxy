# Reproducibility artifact record

`manifest.json` records the Google Drive files discovered on 2026-07-13. It is an
audit record, not a claim that the current files are publicly reproducible.

## Current release gate

All listed Drive files reported `shared: false`, and an unauthenticated checkpoint
request returned HTML rather than the file. The release candidate must not be tagged
until the owner completes all of the following:

1. Set the checkpoint, posterior/prior draws, summary files and their parent folder to
   **Anyone with the link: Viewer**.
2. Download the three large tensors and calculate checksums:

   ```bash
   sha256sum latest.pt posterior_draws.pt prior_samples.pt
   ```

3. Replace each `null` `sha256` value in `manifest.json` and re-check every recorded
   byte size.
4. Record the generating Git SHA. For a rerun, use the SHA printed in the new Slurm
   log; legacy artifacts do not contain enough evidence to infer it.
5. Verify each URL from a signed-out browser before changing `release_ready` to true.

The connector supplied raw bytes for the three small artifacts, so their SHA-256
values are complete. File sizes, IDs, timestamps and permission state come directly
from Drive metadata. No checksum or Git SHA has been guessed.
