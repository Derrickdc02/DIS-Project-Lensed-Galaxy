# Scientific overview

## Aim

The project reconstructs a distribution over an unlensed background galaxy rather
than returning one best-fit image. Posterior samples therefore expose image-level
uncertainty and can be propagated into later scientific analysis.

## Model

The workflow has three components:

1. PROBES g-band FITS images are centre-cropped, flux-clipped and mapped to `[-1, 1]`.
2. An NCSN++ score network learns a variance-exploding diffusion prior over source
   images.
3. Reverse-diffusion sampling combines that prior score with a convolved Gaussian
   likelihood evaluated through a differentiable SIE plus external-shear lens.

The default mock experiment fixes the lens parameters and varies the source,
observation noise, or prior. It does not infer lens mass parameters.

## Validation

Posterior calibration is checked with reduced chi-squared diagnostics and MIRA.
Prior fidelity is checked with PQMass in pixel and joint-PCA spaces. The Figure 2
workflow probes an out-of-distribution glyph over a noise sweep. These diagnostics
answer different questions and should be reported together rather than treated as
interchangeable scores.
