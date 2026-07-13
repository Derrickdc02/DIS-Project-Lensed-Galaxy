# DIS Project: Lensed Galaxy

This repository implements a reproducible research pipeline for score-based Bayesian
source reconstruction in strong gravitational lensing. A score model trained on
preprocessed PROBES galaxies supplies the source prior; a differentiable SIE plus
external-shear lens model connects source images to noisy observations.

The documentation separates lightweight CPU checks from the Wilkes3 GPU workflow.
Building these pages never executes a notebook or starts model training.

```{toctree}
:maxdepth: 2
:caption: Scientific workflow

overview
installation
preprocessing
hpc
sampling
validation
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli
api
notebooks
ci_container
artifacts
limitations
references
```
