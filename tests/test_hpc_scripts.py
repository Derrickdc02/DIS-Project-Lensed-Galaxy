from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ENTRY_POINTS = {
    "run_stage2.sh": ("--ntasks=1", "--cpus-per-task=8", "--gres=gpu:1"),
    "run_stage3.sh": ("--ntasks=1", "--cpus-per-task=32", "--gres=gpu:4"),
    "run_sample.sh": ("--ntasks=1", "--cpus-per-task=8", "--gres=gpu:1"),
    "run_sample_prior.sh": ("--ntasks=1", "--cpus-per-task=8", "--gres=gpu:1"),
}
LEGACY_PATTERNS = (
    r"\beval\b",
    r"`[^`]+`",
    r"machine\.file",
    r"PBS_NODEFILE",
    r"\bmpirun\b",
    r"\bmpiexec\b",
    r"OMPI_MCA",
    r"\bnp=",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_hpc_entry_points_have_strict_launcher_contract():
    assert {path.name for path in SCRIPTS.glob("run_*.sh")} == set(ENTRY_POINTS)
    for name, resources in ENTRY_POINTS.items():
        text = _text(SCRIPTS / name)
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -Eeuo pipefail" in text
        assert 'source "$SCRIPT_DIR/lib/hpc_common.sh"' in text
        assert "args=(" in text and '"${args[@]}"' in text
        assert all(resource in text for resource in resources)
        assert not any(re.search(pattern, text) for pattern in LEGACY_PATTERNS)


def test_hpc_launch_commands_and_stage3_timeout_are_explicit():
    common = _text(SCRIPTS / "lib" / "hpc_common.sh")
    assert 'srun --ntasks=1 "$@"' in common
    assert "eval" not in common and "`" not in common

    stage3 = _text(SCRIPTS / "run_stage3.sh")
    assert "#SBATCH --time=36:00:00" in stage3
    assert "--max_hours 35" in stage3
    assert "hpc_run torchrun --standalone --nproc_per_node=4 -m train_prior" in stage3

    for name in ("run_stage2.sh", "run_sample.sh", "run_sample_prior.sh"):
        assert "hpc_run python -u -m" in _text(SCRIPTS / name)


def test_hpc_paths_are_validated_and_used():
    for name in ENTRY_POINTS:
        text = _text(SCRIPTS / name)
        for variable in ("DATA_DIR", "OUTPUT_DIR", "CKPT"):
            if re.search(rf"^{variable}=", text, flags=re.MULTILINE):
                assert text.count(f'"${variable}"') >= 2, f"{name}: {variable} is not validated and used"
        assert 'hpc_prepare_output_dir "$OUTPUT_DIR"' in text

    common = _text(SCRIPTS / "lib" / "hpc_common.sh")
    for required in (
        "SLURM_SUBMIT_DIR",
        "VENV/bin/activate",
        "Git commit:",
        "Slurm tasks:",
        "Slurm CPUs per task:",
        "Slurm GPUs:",
        "Started:",
        "Finished:",
        "Elapsed seconds:",
        "Exit status:",
        "nvidia-smi",
    ):
        assert required in common


def test_slurm_log_directory_is_present_but_logs_are_ignored():
    assert (ROOT / "slurm_logs" / ".gitkeep").is_file()
    ignore = _text(ROOT / ".gitignore")
    assert "slurm_logs/*" in ignore
    assert "!slurm_logs/.gitkeep" in ignore
