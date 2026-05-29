"""
test_concept_naming.py — Tests for concept naming logic.

Verifies that name_concepts assigns correct labels from vocabulary
using cosine similarity with decoder weights.
"""

import pytest
import torch

from sae_module import SAEManager


class TestNameConcepts:
    def test_output_structure(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=3)

        assert len(names) == 4096
        assert 0 in names
        assert "name" in names[0]
        assert "score" in names[0]
        assert "candidates" in names[0]

    def test_candidates_count(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=5)

        assert len(names[0]["candidates"]) == 5

    def test_name_from_vocab(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=1)

        # Every assigned name should come from the vocabulary
        for feat_id, info in names.items():
            assert info["name"] in fake_vocab_labels

    def test_scores_in_range(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=1)

        # Cosine similarity is in [-1, 1]
        for feat_id, info in names.items():
            assert -1.0 <= info["score"] <= 1.0

    def test_candidates_sorted_descending(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=5)

        for feat_id, info in list(names.items())[:10]:
            scores = [c["score"] for c in info["candidates"]]
            assert scores == sorted(scores, reverse=True)
