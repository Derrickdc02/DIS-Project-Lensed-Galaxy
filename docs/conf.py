from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "DIS Project: Lensed Galaxy"
author = "Derrick Tang"
copyright = "2026, Derrick Tang"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["caustics", "score_models", "pqm", "mira_score"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = project
html_theme_options = {
    "source_repository": "https://github.com/Derrickdc02/DIS-Project-Lensed-Galaxy/",
    "source_branch": "main",
    "source_directory": "docs/",
}
