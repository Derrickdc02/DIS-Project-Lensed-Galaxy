# Data preprocessing

The repository does not redistribute PROBES imaging. Obtain the raw multi-band FITS
files under `data/raws/`, then run from the repository root:

```bash
preprocess-probes --raw-dir data/raws --out-dir data/gals_gband_norm
```

The command selects the expected g-band plane, rejects non-finite or excessively
zero-valued images, takes a centred square crop, clips flux to the project range and
maps the result to `[-1, 1]`. Each accepted galaxy is stored as a NumPy array.
Existing output is preserved by default. Pass `--overwrite` only when existing
normalized arrays should be replaced intentionally.

Before submitting a GPU job, verify that the output directory contains `.npy` files:

```bash
find data/gals_gband_norm -maxdepth 1 -name '*.npy' | head
```

The preprocessing implementation is exposed in {doc}`api` and covered by unit tests
for corruption checks, crop geometry and normalisation round trips.
