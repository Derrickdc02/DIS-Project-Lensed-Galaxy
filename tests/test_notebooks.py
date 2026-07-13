import json
from pathlib import Path


NOTEBOOK_ROOT = Path(__file__).parents[1] / "notebooks"
EXPECTED_NOTEBOOKS = {
    "figure2_ood_colab.ipynb",
    "mira_posterior_validation_colab.ipynb",
    "mira_posterior_validation_hpc.ipynb",
    "posterior_reconstruction_colab.ipynb",
    "posterior_reconstruction_hpc.ipynb",
    "pqm_prior_validation_colab.ipynb",
    "pqm_prior_validation_hpc.ipynb",
    "quickstart.ipynb",
}
BANNED_TEXT = (
    "/home/yd388",
    "/rds/user/yd388",
    "--branch local",
    "branch local",
    "def posterior_sample",
    "def _load_run",
    "pqm_chi2(",
    "from mira_score import",
)


def _load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebook_inventory_is_curated():
    actual = {
        path.relative_to(NOTEBOOK_ROOT).as_posix()
        for path in NOTEBOOK_ROOT.rglob("*.ipynb")
    }
    assert actual == EXPECTED_NOTEBOOKS
    assert not (NOTEBOOK_ROOT / "archive").exists()


def test_notebooks_are_portable_and_small():
    total_output_bytes = 0
    for name in sorted(EXPECTED_NOTEBOOKS):
        path = NOTEBOOK_ROOT / name
        notebook = _load_notebook(path)
        text = path.read_text(encoding="utf-8")

        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert notebook["metadata"]["language_info"]["version"] == "3.11"
        assert path.stat().st_size < 2 * 1024 * 1024

        for banned in BANNED_TEXT:
            assert banned not in text

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                outputs = cell.get("outputs", [])
                total_output_bytes += len(json.dumps(outputs).encode("utf-8"))
                if name.endswith("_hpc.ipynb") or name == "quickstart.ipynb":
                    assert outputs == []

    assert total_output_bytes < 5 * 1024 * 1024


def test_colab_badges_target_main_branch():
    for name in sorted(EXPECTED_NOTEBOOKS):
        if not name.endswith("_colab.ipynb"):
            continue
        text = (NOTEBOOK_ROOT / name).read_text(encoding="utf-8")
        expected = (
            "github/Derrickdc02/DIS-Project-Lensed-Galaxy/"
            f"blob/main/notebooks/{name}"
        )
        assert expected in text
        assert "colab-badge.svg" in text
