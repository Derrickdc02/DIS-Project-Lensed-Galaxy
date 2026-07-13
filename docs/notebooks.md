# Notebook catalogue

Notebooks are lightweight front ends over the installed package and CLI. They do not
contain copied implementations. HPC notebooks save no output; Colab notebooks retain
only key statistics or final figures when the user executes them.

| Notebook | Environment | Purpose |
|---|---|---|
| `quickstart.ipynb` | local/CPU | lens construction and one forward-model smoke test |
| `posterior_reconstruction_hpc.ipynb` | Wilkes3 | configure and inspect posterior sampling |
| `posterior_reconstruction_colab.ipynb` | Colab | download artifacts and inspect posterior summaries |
| `figure2_ood_colab.ipynb` | Colab | reproduce the OOD noise sweep |
| `pqm_prior_validation_hpc.ipynb` | Wilkes3 | run PQMass on saved prior draws |
| `pqm_prior_validation_colab.ipynb` | Colab | inspect PQMass output from Drive |
| `mira_posterior_validation_hpc.ipynb` | Wilkes3 | run MIRA over posterior run groups |
| `mira_posterior_validation_colab.ipynb` | Colab | inspect or smoke-test MIRA |

Browse the notebooks in the
[GitHub catalogue](https://github.com/Derrickdc02/DIS-Project-Lensed-Galaxy/tree/main/notebooks).
Every Colab badge clones the default `main` branch. The structure test enforces the
exact eight-file inventory, Python 3.11 kernels, portable paths, empty HPC outputs,
per-file size below 2 MB and total saved output below 5 MB.
