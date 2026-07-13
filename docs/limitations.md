# Limitations

- The default experiments fix the lens model; uncertainty in lens mass parameters is
  not propagated.
- Training and diffusion sampling are computationally expensive and have not been
  reduced to a meaningful CPU workload.
- PROBES selection and preprocessing define the learned prior and may underrepresent
  morphologies outside the training distribution.
- PQMass, MIRA and reduced chi-squared diagnose different failure modes; none alone
  proves scientific validity.
- The Figure 2 glyph is deliberately out of distribution and is a qualitative stress
  test.
- Exact floating-point reproducibility across GPU generations and CUDA libraries is
  not guaranteed. Record the Git SHA, artifact checksum and HPC environment metadata.
- Public checkpoints and representative outputs are versioned separately from Git
  because of their size; verify SHA-256 before use.
