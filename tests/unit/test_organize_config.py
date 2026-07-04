"""Config smoke test for OrganizeConfig."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, "src/")

import config
from config import OrganizeConfig


def test_organize_singleton_exists():
    assert isinstance(config.organize, OrganizeConfig)


def test_defaults_anchored_to_paths():
    assert config.organize.vocab_path == config.paths.vocab_labels_path
    assert config.organize.vocab_emb_path == config.paths.vocab_embeddings_path
    assert config.organize.radlex_csv_path == config.paths.data_dir / "radlex.csv"
    assert config.organize.output_dir == config.paths.results_dir / "concept_organization"


def test_n_clusters_and_distance_mutually_exclusive():
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        OrganizeConfig(n_clusters=5, distance_threshold=0.5)


def test_invalid_linkage_rejected():
    import pytest
    with pytest.raises(ValueError, match="linkage"):
        OrganizeConfig(linkage="ward")  # ward incompatible with cosine


def test_invalid_metric_rejected():
    import pytest
    with pytest.raises(ValueError, match="metric"):
        OrganizeConfig(metric="euclidean")
