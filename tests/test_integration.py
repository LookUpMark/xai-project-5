"""
test_integration.py — Integration tests for the full SAE pipeline.

Tests the end-to-end flow: load model -> encode -> get concepts -> name -> explain.
Uses real AutoEncoderTopK with random weights (no training needed).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sae_module import SAEManager


class TestFullPipelineFlow:
    """Test the complete pipeline that Member 3 would execute."""

    def test_load_encode_decode_roundtrip(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        x = fake_embeddings[:5]
        sparse = mgr.encode(x)
        x_hat = mgr.decode(sparse)

        assert x.shape == x_hat.shape
        # Reconstruction won't be perfect with random weights, but shapes must match

    def test_encode_to_concepts_pipeline(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        concepts = mgr.get_top_concepts(fake_embeddings[:3], n=5)

        # Verify structure
        assert len(concepts) == 3
        for sample_concepts in concepts:
            assert len(sample_concepts) == 5
            for feat_id, activation in sample_concepts:
                assert 0 <= feat_id < 4096
                assert activation >= 0

    def test_naming_pipeline(self, tmp_model_dir, fake_vocab_embeddings, fake_vocab_labels):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        concept_names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels)

        # All 4096 features should have names
        assert len(concept_names) == 4096

        # Names should come from vocabulary
        all_names = {info["name"] for info in concept_names.values()}
        assert all_names.issubset(set(fake_vocab_labels))

    def test_full_explanation_flow(
        self, tmp_model_dir, fake_embeddings, fake_vocab_embeddings, fake_vocab_labels
    ):
        """Simulate the complete flow: load -> name -> explain."""
        from importlib import import_module

        gen_module = import_module("02c_generate_explanations")

        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        # Step 1: Name concepts
        concept_names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels)

        # Step 2: Get top concepts for samples
        top_concepts = mgr.get_top_concepts(fake_embeddings[:3], n=5)

        # Step 3: Generate explanations
        # Convert concept_names keys to strings (as JSON would)
        str_names = {str(k): v for k, v in concept_names.items()}

        explanations = []
        for idx, sample_concepts in enumerate(top_concepts):
            explanation = gen_module.generate_explanation(sample_concepts, str_names)
            explanation["sample_idx"] = idx
            explanations.append(explanation)

        assert len(explanations) == 3
        for exp in explanations:
            assert "pseudo_report" in exp
            assert "findings" in exp
            assert len(exp["findings"]) == 5

    def test_metrics_after_load(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        mse = mgr.compute_reconstruction_mse(fake_embeddings[:10])
        sparsity = mgr.compute_sparsity_metrics(fake_embeddings[:10])

        assert mse > 0
        assert sparsity["l0_mean"] > 0
        assert 0 <= sparsity["dead_features_pct"] <= 100

    def test_json_serializable_output(
        self, tmp_model_dir, fake_embeddings, fake_vocab_embeddings, fake_vocab_labels
    ):
        """Verify that all outputs can be serialized to JSON (for results/)."""
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        concept_names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels)
        top_concepts = mgr.get_top_concepts(fake_embeddings[:2], n=3)
        metrics = mgr.compute_sparsity_metrics(fake_embeddings[:10])

        # concept_names should serialize
        json.dumps(concept_names)

        # top_concepts should serialize
        json.dumps(top_concepts)

        # metrics should serialize
        json.dumps(metrics)
