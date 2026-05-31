"""
test_generate_explanations.py — Tests for explanation generation.

Verifies that pseudo-reports are correctly structured from SAE activations.
"""

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def generate_explanation():
    """Import generate_explanation from generate_explanations module."""
    spec = import_module("autoencoder.generate_explanations")
    return spec.generate_explanation


@pytest.fixture
def sample_concept_names():
    """Fake concept_names.json structure."""
    return {
        "10": {"name": "cardiomegaly", "score": 0.85, "candidates": []},
        "20": {"name": "pleural_effusion", "score": 0.79, "candidates": []},
        "30": {"name": "pneumothorax", "score": 0.72, "candidates": []},
        "40": {"name": "consolidation", "score": 0.68, "candidates": []},
        "50": {"name": "atelectasis", "score": 0.65, "candidates": []},
    }


class TestGenerateExplanation:
    def test_output_has_required_keys(self, generate_explanation, sample_concept_names):
        top_concepts = [(10, 0.95), (20, 0.80), (30, 0.65), (40, 0.50), (50, 0.30)]
        result = generate_explanation(top_concepts, sample_concept_names)

        assert "findings" in result
        assert "pseudo_report" in result
        assert "n_active_concepts" in result

    def test_findings_count(self, generate_explanation, sample_concept_names):
        top_concepts = [(10, 0.95), (20, 0.80), (30, 0.65)]
        result = generate_explanation(top_concepts, sample_concept_names)

        assert result["n_active_concepts"] == 3
        assert len(result["findings"]) == 3

    def test_finding_structure(self, generate_explanation, sample_concept_names):
        top_concepts = [(10, 0.95)]
        result = generate_explanation(top_concepts, sample_concept_names)

        finding = result["findings"][0]
        assert finding["concept"] == "cardiomegaly"
        assert finding["feature_id"] == 10
        assert finding["activation"] == 0.95
        assert finding["naming_confidence"] == 0.85

    def test_unknown_feature_handled(self, generate_explanation, sample_concept_names):
        top_concepts = [(9999, 0.5)]  # not in concept_names
        result = generate_explanation(top_concepts, sample_concept_names)

        assert "unknown_feature_9999" in result["findings"][0]["concept"]

    def test_pseudo_report_is_string(self, generate_explanation, sample_concept_names):
        top_concepts = [(10, 0.95), (20, 0.80)]
        result = generate_explanation(top_concepts, sample_concept_names)

        assert isinstance(result["pseudo_report"], str)
        assert "cardiomegaly" in result["pseudo_report"]

    def test_pseudo_report_mentions_dominant(
        self, generate_explanation, sample_concept_names
    ):
        top_concepts = [(20, 0.99), (10, 0.50)]
        result = generate_explanation(top_concepts, sample_concept_names)

        assert "pleural_effusion" in result["pseudo_report"]
        assert "dominant" in result["pseudo_report"]

    def test_empty_findings_guard(self, generate_explanation, sample_concept_names):
        """Empty top_concepts should return a valid response, not crash."""
        result = generate_explanation([], sample_concept_names)

        assert result["n_active_concepts"] == 0
        assert result["findings"] == []
        assert "No active concepts" in result["pseudo_report"]
