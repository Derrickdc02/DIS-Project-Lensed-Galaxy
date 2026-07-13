"""MIRA posterior-calibration validation for saved reconstruction runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def discover_runs(root: str | Path) -> list[Path]:
    """Find run directories containing samples/posterior_draws.pt."""
    root = Path(root)
    files = sorted(root.rglob("samples/posterior_draws.pt"))
    return [path.parent.parent for path in files]


def parse_model_spec(specification: str) -> tuple[str, Path]:
    """Parse a NAME=PATH model specification."""
    if "=" not in specification:
        raise ValueError(f"Model specification must be NAME=PATH, got {specification!r}")
    name, raw_path = specification.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"Model specification must be NAME=PATH, got {specification!r}")
    return name.strip(), Path(raw_path).expanduser().resolve()


def load_posterior_run(run_dir: str | Path) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Load one truth and its posterior draws from a sampling run."""
    run_dir = Path(run_dir)
    artifact = run_dir / "samples" / "posterior_draws.pt"
    if not artifact.is_file():
        raise FileNotFoundError(f"Posterior artifact not found: {artifact}")

    data = torch.load(artifact, map_location="cpu", weights_only=False)
    for key in ("post", "src"):
        if key not in data:
            raise KeyError(f"{artifact} does not contain required key {key!r}")

    posterior = data["post"].float()
    truth = data["src"].float()
    if posterior.ndim == 4 and posterior.shape[1] == 1:
        posterior = posterior[:, 0]
    if truth.ndim == 3 and truth.shape[0] == 1:
        truth = truth[0]
    if posterior.ndim != 3 or truth.ndim != 2:
        raise ValueError(
            f"Expected posterior (S,H,W) and truth (H,W), got {posterior.shape} and {truth.shape}"
        )
    if posterior.shape[1:] != truth.shape:
        raise ValueError(f"Posterior/truth image shapes differ in {artifact}")
    if not torch.isfinite(posterior).all() or not torch.isfinite(truth).all():
        raise ValueError(f"Non-finite posterior data in {artifact}")

    name = str(data.get("src_name", run_dir.name))
    return name, truth.reshape(-1), posterior.reshape(posterior.shape[0], -1)


def assemble_model_tensors(
    model_runs: dict[str, list[Path]],
) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
    """Align multiple models on their common truths.

    Returns model names, truth names, truth tensor (T,D), and posterior tensor
    (M,T,S,D). The smallest available posterior sample count is used for all
    model/truth pairs.
    """
    if not model_runs:
        raise ValueError("At least one model is required")

    per_model: dict[str, dict[str, torch.Tensor]] = {}
    truth_by_name: dict[str, torch.Tensor] = {}
    for model_name, run_dirs in model_runs.items():
        if not run_dirs:
            raise ValueError(f"Model {model_name!r} has no posterior runs")
        per_model[model_name] = {}
        for run_dir in run_dirs:
            truth_name, truth, posterior = load_posterior_run(run_dir)
            if truth_name in per_model[model_name]:
                raise ValueError(f"Duplicate truth {truth_name!r} for model {model_name!r}")
            per_model[model_name][truth_name] = posterior
            previous = truth_by_name.setdefault(truth_name, truth)
            if previous.shape != truth.shape or not torch.allclose(previous, truth):
                raise ValueError(f"Truth {truth_name!r} differs between model runs")

    common_truths = sorted(set.intersection(*(set(runs) for runs in per_model.values())))
    if not common_truths:
        raise ValueError("No truth sources are shared by every model")

    sample_count = min(
        per_model[model][truth].shape[0]
        for model in per_model
        for truth in common_truths
    )
    if sample_count < 2:
        raise ValueError("Each truth requires at least two posterior draws")

    model_names = list(per_model)
    truths = torch.stack([truth_by_name[name] for name in common_truths])
    posterior = torch.stack(
        [
            torch.stack(
                [per_model[model][truth][:sample_count] for truth in common_truths],
                dim=0,
            )
            for model in model_names
        ],
        dim=0,
    )
    return model_names, common_truths, truths, posterior


def pca_project(
    truth: torch.Tensor,
    posterior: torch.Tensor,
    n_components: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project truths and posterior draws into a joint low-rank PCA space."""
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    model_count, truth_count, sample_count, dimension = posterior.shape
    stack = torch.cat([truth, posterior.reshape(-1, dimension)], dim=0).float()
    component_count = min(n_components, stack.shape[0] - 1, dimension)
    if component_count < 1:
        raise ValueError("Not enough observations for PCA")

    torch.manual_seed(seed)
    mean = stack.mean(dim=0, keepdim=True)
    centered = stack - mean
    _, _, vectors = torch.pca_lowrank(centered, q=component_count, center=False)
    projected = centered @ vectors
    retained = float(projected.square().sum() / centered.square().sum())
    projected_truth = projected[:truth_count]
    projected_post = projected[truth_count:].reshape(
        model_count,
        truth_count,
        sample_count,
        component_count,
    )
    return projected_truth, projected_post, retained


def gaussian_smoke_case(
    *,
    truth_count: int = 64,
    sample_count: int = 64,
    dimension: int = 5,
    seed: int = 1,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Create matched, over-confident, and under-confident Gaussian posteriors."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(truth_count, dimension, generator=generator)
    truth = centres + torch.randn(truth_count, dimension, generator=generator)

    def draws(scale: float) -> torch.Tensor:
        noise = torch.randn(
            truth_count,
            sample_count,
            dimension,
            generator=generator,
        )
        return centres[:, None, :] + scale * noise

    posterior = torch.stack(
        [draws(1.0), draws(1.0 / math.sqrt(3.0)), draws(math.sqrt(2.0))],
        dim=0,
    )
    names = [
        "matched",
        "over-confident (covariance / 3)",
        "under-confident (covariance x 2)",
    ]
    return names, truth, posterior


def add_directional_baselines(
    names: list[str],
    posterior: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    """Append narrowed and broadened versions of the first model posterior."""
    if not names:
        raise ValueError("At least one model name is required")
    if posterior.ndim != 4:
        raise ValueError("Posterior must have shape (M,T,S,D)")
    if posterior.shape[0] != len(names):
        raise ValueError("Model-name count does not match posterior model axis")
    if posterior.shape[2] < 1:
        raise ValueError("Each model posterior requires at least one sample")
    base = posterior[0:1]
    mean = base.mean(dim=2, keepdim=True)
    narrowed = mean + (base - mean) / math.sqrt(3.0)
    broadened = mean + math.sqrt(2.0) * (base - mean)
    labels = [
        *names,
        f"{names[0]} [over-confident: covariance / 3]",
        f"{names[0]} [under-confident: covariance x 2]",
    ]
    return labels, torch.cat([posterior, narrowed, broadened], dim=0)


def run_mira_scores(
    names: list[str],
    truth: torch.Tensor,
    posterior: torch.Tensor,
    *,
    num_runs: int,
    num_bootstrap: int,
    norm: bool,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run MIRA and return JSON-serializable score summaries."""
    from mira_score import mira, mira_bootstrap

    if posterior.ndim != 4 or truth.ndim != 2:
        raise ValueError("MIRA expects truth (T,q) and posterior (M,T,S,q)")
    if posterior.shape[0] != len(names):
        raise ValueError("Model-name count does not match posterior model axis")
    if posterior.shape[1] != truth.shape[0] or posterior.shape[3] != truth.shape[1]:
        raise ValueError("Truth and posterior dimensions do not align")
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap cannot be negative")

    torch.manual_seed(seed)
    np.random.seed(seed)
    score, monte_carlo_std = mira(
        truth,
        posterior,
        num_runs=num_runs,
        norm=norm,
        disable_tqdm=True,
        device=device,
    )

    bootstrap_mean = None
    bootstrap_std = None
    if num_bootstrap:
        bootstrap_mean_tensor, bootstrap_std_tensor = mira_bootstrap(
            truth,
            posterior,
            num_bootstrap=num_bootstrap,
            num_runs=1,
            norm=norm,
            disable_tqdm=True,
            device=device,
        )
        bootstrap_mean = [float(value) for value in bootstrap_mean_tensor]
        bootstrap_std = [float(value) for value in bootstrap_std_tensor]

    sample_count = posterior.shape[2]
    theory_mean = (2 * sample_count + 1) / (3 * sample_count)
    theory_std = math.sqrt(1.0 / (18.0 * truth.shape[0]))
    return {
        "models": names,
        "truth_count": int(truth.shape[0]),
        "samples_per_truth": int(sample_count),
        "feature_dimension": int(truth.shape[1]),
        "score": [float(value) for value in score],
        "monte_carlo_std": [float(value) for value in monte_carlo_std],
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,
        "matched_reference_mean": theory_mean,
        "matched_reference_std": theory_std,
    }


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MIRA command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model name and root containing posterior runs; repeat to compare models",
    )
    parser.add_argument("--output", type=Path, default=Path("mira_results.json"))
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> int:
    """Run MIRA on saved posterior runs or a deterministic synthetic smoke case."""
    args = build_arg_parser().parse_args()
    device = _device_from_name(args.device)

    if args.smoke_test:
        names, truth, posterior = gaussian_smoke_case(seed=args.seed)
        retained = None
    else:
        model_runs: dict[str, list[Path]] = {}
        if args.model:
            for specification in args.model:
                name, root = parse_model_spec(specification)
                if name in model_runs:
                    raise SystemExit(f"Duplicate --model name: {name}")
                model_runs[name] = discover_runs(root)
        else:
            model_runs["model"] = discover_runs(args.runs_root)

        names, truth_names, truth, posterior = assemble_model_tensors(model_runs)
        print(f"Shared truths ({len(truth_names)}): {', '.join(truth_names)}")
        retained = None
        if not args.no_pca:
            truth, posterior, retained = pca_project(
                truth,
                posterior,
                args.pca_components,
                seed=args.seed,
            )
        if not args.no_baselines:
            names, posterior = add_directional_baselines(names, posterior)

    result = run_mira_scores(
        names,
        truth,
        posterior,
        num_runs=args.num_runs,
        num_bootstrap=args.num_bootstrap,
        norm=not args.smoke_test,
        device=device,
        seed=args.seed,
    )
    result["device"] = str(device)
    result["seed"] = args.seed
    result["pca_retained_variance"] = retained

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
