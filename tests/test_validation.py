from pathlib import Path

import numpy as np
import pytest
import torch

from sample import discover_sources, load_source
from validate_mira import (
    add_directional_baselines,
    gaussian_smoke_case,
    parse_model_spec,
    run_mira_scores,
)
from validate_pqmass import (
    load_prior_samples,
    pca_scores,
    pqmass_statistics,
)


def test_source_discovery_and_loading(tmp_path):
    np.save(tmp_path / "b.npy", np.ones((1, 4, 4), dtype=np.float32))
    np.save(tmp_path / "a.npy", np.zeros((1, 4, 4), dtype=np.float32))

    sources = discover_sources(tmp_path)
    assert [path.name for path in sources] == ["a.npy", "b.npy"]

    source, name = load_source(tmp_path, 1, torch.device("cpu"))
    assert name == "b"
    assert source.shape == (4, 4)
    assert torch.equal(source, torch.ones(4, 4))

    with pytest.raises(IndexError):
        load_source(tmp_path, 2, torch.device("cpu"))


def test_source_discovery_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_sources(tmp_path)


def test_load_prior_samples_from_chunks(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    torch.save(torch.zeros(2, 1, 4, 4), chunks / "chunk_000.pt")
    torch.save(torch.ones(3, 1, 4, 4), chunks / "chunk_001.pt")

    samples = load_prior_samples(tmp_path)
    assert samples.shape == (5, 1, 4, 4)
    assert torch.equal(samples[:2], torch.zeros(2, 1, 4, 4))
    assert torch.equal(samples[2:], torch.ones(3, 1, 4, 4))


def test_pca_scores_joint_projection():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(16, 8)).astype(np.float32)
    prior = rng.normal(size=(16, 8)).astype(np.float32)

    real_scores, prior_scores, retained = pca_scores(real, prior, 3)
    assert real_scores.shape == (16, 3)
    assert prior_scores.shape == (16, 3)
    assert 0.0 < retained <= 1.0


def test_pqmass_statistics_smoke():
    pytest.importorskip("pqm")
    rng = np.random.default_rng(1)
    real = rng.normal(size=(32, 4)).astype(np.float32)
    prior = rng.normal(size=(32, 4)).astype(np.float32)

    result = pqmass_statistics(
        prior,
        real,
        num_refs=4,
        re_tessellation=2,
        seed=1,
    )
    assert result["num_refs"] == 4
    assert result["degrees_of_freedom"] == 3
    assert np.isfinite(result["chi2_over_dof_mean"])
    assert 0.0 <= result["pvalue_mean"] <= 1.0


def test_mira_smoke_case_and_baselines():
    pytest.importorskip("mira_score")
    names, truth, posterior = gaussian_smoke_case(
        truth_count=16,
        sample_count=16,
        dimension=3,
        seed=2,
    )
    result = run_mira_scores(
        names,
        truth,
        posterior,
        num_runs=2,
        num_bootstrap=0,
        norm=False,
        device=torch.device("cpu"),
        seed=2,
    )
    assert result["models"] == names
    assert len(result["score"]) == 3
    assert result["bootstrap_std"] is None

    expanded_names, expanded = add_directional_baselines(["base"], posterior[:1])
    assert len(expanded_names) == 3
    assert expanded.shape[0] == 3


def test_parse_model_spec(tmp_path):
    name, path = parse_model_spec(f"model_a={tmp_path}")
    assert name == "model_a"
    assert path == Path(tmp_path).resolve()

    with pytest.raises(ValueError):
        parse_model_spec("missing-separator")
