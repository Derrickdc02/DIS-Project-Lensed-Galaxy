# Notebook catalogue

These notebooks are thin frontends to the tested modules in src. Algorithm implementations must not be copied back into notebook cells.

| Notebook | Environment | Purpose | Saved outputs |
| --- | --- | --- | --- |
| quickstart.ipynb | CPU/local | Synthetic lensing smoke test and project orientation | None |
| posterior_reconstruction_hpc.ipynb | Wilkes3 | Configure and inspect posterior reconstruction | None |
| posterior_reconstruction_colab.ipynb | Colab + Drive | Public posterior reconstruction entry point | Compact metrics only |
| figure2_ood_colab.ipynb | Colab + Drive | Figure 2 OOD reproduction | Final figure only |
| pqm_prior_validation_hpc.ipynb | Wilkes3 | PQMass production entry point | None |
| pqm_prior_validation_colab.ipynb | Colab + Drive | Public PQMass validation | Compact JSON only |
| mira_posterior_validation_hpc.ipynb | Wilkes3 | MIRA production entry point | None |
| mira_posterior_validation_colab.ipynb | Colab + Drive | Public MIRA validation | Compact JSON only |

All notebooks target Python 3.11. GPU- or data-dependent cells are guarded by RUN = False so opening or inspecting a notebook never starts an expensive job.
