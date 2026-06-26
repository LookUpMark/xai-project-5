"""
test_concept_naming.py — Tests for concept naming logic.

Verifies that name_concepts assigns correct labels from vocabulary
using cosine similarity with decoder weights.
"""

import config
import pytest
import torch

from autoencoder.sae_module import SAEManager


class TestNameConcepts:
    def test_output_structure(
        self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels
    ):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=3)

        assert len(names) == config.sae.dict_size
        assert 0 in names
        assert "name" in names[0]
        assert "score" in names[0]
        assert "candidates" in names[0]

    def test_candidates_count(
        self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels
    ):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=5)

        assert len(names[0]["candidates"]) == 5

    def test_name_from_vocab(
        self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels
    ):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=1)

        # Every assigned name should come from the vocabulary (or be DEAD_FEATURE)
        for feat_id, info in names.items():
            assert info["name"] in fake_vocab_labels or info["name"] == "DEAD_FEATURE"

    def test_scores_in_range(
        self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels
    ):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=1)

        # Cosine similarity is in [-1, 1]
        for feat_id, info in names.items():
            assert -1.0 <= info["score"] <= 1.0

    def test_candidates_sorted_descending(
        self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels
    ):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=5)

        for feat_id, info in list(names.items())[:10]:
            scores = [c["score"] for c in info["candidates"]]
            assert scores == sorted(scores, reverse=True)

    def test_shape_validation_wrong_dim(self, tmp_model_dir, fake_vocab_labels):
        """Wrong embedding dimension should raise ValueError."""
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        wrong_emb = torch.randn(50, 256)  # 256 != 512

        with pytest.raises(ValueError, match="activation_dim"):
            mgr.name_concepts(wrong_emb, fake_vocab_labels)

    def test_length_validation_mismatch(self, tmp_model_dir, fake_vocab_embeddings):
        """Mismatched labels length should raise ValueError."""
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        wrong_labels = ["only_one_label"]

        with pytest.raises(ValueError, match="vocab_labels length"):
            mgr.name_concepts(fake_vocab_embeddings, wrong_labels)

    def test_dead_features_flagged(
        self, tmp_path, fake_vocab_embeddings, fake_vocab_labels
    ):
        """Dead features (zero decoder vectors) should be flagged with is_dead=True."""
        # Create model with some zero decoder rows (dead features)
        model_dir = tmp_path / "sae_dead"
        model_dir.mkdir()

        decoder_weight = torch.randn(512, config.sae.dict_size)
        # Set features 0, 1, 2 as dead (zero columns → zero rows after transpose)
        decoder_weight[:, 0] = 0.0
        decoder_weight[:, 1] = 0.0
        decoder_weight[:, 2] = 0.0

        state_dict = {
            "encoder.weight": torch.randn(config.sae.dict_size, 512),
            "encoder.bias": torch.zeros(config.sae.dict_size),
            "decoder.weight": decoder_weight,
            "b_dec": torch.zeros(512),
            "k": torch.tensor(32),
            "threshold": torch.tensor(-1.0),
        }
        torch.save(state_dict, model_dir / "ae.pt")

        mgr = SAEManager({"device": "cpu"})
        mgr.load(model_dir)
        names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels, top_n=3)

        # Dead features should be flagged
        assert names[0]["is_dead"] is True
        assert names[0]["name"] == "DEAD_FEATURE"
        assert names[0]["score"] == 0.0
        assert names[0]["candidates"] == []
        assert names[1]["is_dead"] is True
        assert names[2]["is_dead"] is True

        # Non-dead features should NOT be flagged
        assert names[3]["is_dead"] is False
        assert names[3]["name"] in fake_vocab_labels
        assert names[3]["score"] != 0.0
