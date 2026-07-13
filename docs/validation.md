# Validation workflows

## Reduced chi-squared

```bash
validate-chi2 --output_dir outputs/custom_posterior
```

This posterior-predictive check compares re-lensed draws to the saved observation.
Interpret reduced chi-squared jointly with spatial residuals; a value near one alone
does not establish posterior calibration.

## PQMass prior validation

```bash
validate-pqmass   --prior outputs/probes_final/prior_check   --data-dir data/gals_gband_norm   --output outputs/validation/pqmass.json   --max-samples 1000 --num-refs 100 --re-tessellation 200   --pca-components 50 --seed 21
```

The command loads merged or chunked prior draws, selects a deterministic matched real
sample and evaluates PQMass in pixel space and an exact joint-PCA score space. The JSON
records summary statistics and retained PCA variance.

## MIRA posterior validation

Use repeated posterior runs, grouped by model label:

```bash
validate-mira   --model baseline=outputs/mira/baseline   --model candidate=outputs/mira/candidate   --output outputs/validation/mira.json   --pca-components 50 --num-runs 200 --num-bootstrap 200 --device auto
```

A deterministic CPU smoke test is available without research artifacts:

```bash
validate-mira --smoke-test --device cpu --num-runs 4 --num-bootstrap 4
```

## Figure 2

```bash
reproduce-figure2   --ckpt /absolute/path/to/latest.pt   --noises 0.001,0.1,0.8,1.0,5.0   --steps 8000 --seed 21 --out outputs/figure2.png
```

The default source is an out-of-distribution `7` glyph. This is a qualitative stress
test, not a substitute for calibration or two-sample validation.
