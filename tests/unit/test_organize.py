"""Unit tests for concept_discovery.organize."""
from __future__ import annotations

import sys
sys.path.insert(0, "src/")

import torch
from concept_discovery.organize import ConceptSet, ImageConcepts


class TestDataclasses:
    def test_concept_set_constructs(self):
        cs = ConceptSet(
            names=["a", "b"],
            embeddings=torch.randn(2, 512),
            name_to_idx={"a": 0, "b": 1},
            per_image=[ImageConcepts(image_id="x", activations={"a": 1.0})],
        )
        assert cs.names == ["a", "b"]
        assert cs.embeddings.shape == (2, 512)
