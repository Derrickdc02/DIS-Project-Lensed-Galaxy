# Continuous integration and CPU container

CI separates fast CPU correctness checks from Wilkes3 science runs. The automated
suite covers Python 3.11 tests, Ruff, wheel contents, CLI smoke tests, Bash syntax,
ShellCheck, shfmt, notebook structure, NBQA, Sphinx warnings-as-errors and a CPU
container build/run.

Documentation is built on every relevant change and deployed from `main` to:

<https://derrickdc02.github.io/DIS-Project-Lensed-Galaxy/>

The container validates installation and CPU-safe commands only. It does not emulate
Slurm, CUDA, multi-GPU DDP, model training or full diffusion sampling. A successful
container run therefore complements, but cannot replace, a recorded Wilkes3 run.
