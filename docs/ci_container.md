# Continuous integration and CPU container

CI separates fast CPU correctness checks from Wilkes3 science runs. GitHub Actions
and the root `.gitlab-ci.yml` cover Python 3.11 tests, Ruff, wheel contents, CLI
smoke tests, Bash syntax, ShellCheck, shfmt, notebook structure, NBQA, Sphinx
warnings-as-errors and a CPU container build/run.

Documentation is built on every relevant change and deployed from `main` to:

<https://derrickdc02.github.io/DIS-Project-Lensed-Galaxy/>

## GitLab pipeline

GitLab runs independent `python`, `shell`, `docs` and `container` jobs. The
first two jobs form the test stage; documentation and the CPU image are built only
after that stage succeeds. On the GitLab default branch, `deploy-pages` publishes
the strict Sphinx output when Pages is available.

Before relying on the pipeline, confirm that the target project has an available
runner under `Settings > CI/CD > Runners`. The container job uses
Docker-in-Docker, which requires runner support for Docker services. A pending job
usually means no matching runner is available; a Docker daemon connection error
usually means the runner is not configured for Docker-in-Docker.

No CI job downloads private checkpoints or PROBES data, submits a Slurm job, or
uses a GPU. Do not add private Drive links, checkpoint files or credentials as CI
variables.

GitLab Pages availability and its final URL depend on the institutional GitLab
configuration. After a successful default-branch pipeline, inspect
`Deploy > Pages` in the project. Pages deployment is allowed to fail without
masking failures in the required Python, shell, documentation or container jobs.

The container validates installation and CPU-safe commands only. It does not emulate
Slurm, CUDA, multi-GPU DDP, model training or full diffusion sampling. A successful
container run therefore complements, but cannot replace, a recorded Wilkes3 run.
