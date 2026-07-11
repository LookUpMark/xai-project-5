"""Integration tests for SPLiCE pipeline."""

from __future__ import annotations

import pytest
import json

import sys
sys.path.insert(0, "src/")
from concept_discovery.spliece import run
import config
from utils import load_tensor


class TestSpliCEPipeline:
    """End-to-end tests for SPLiCE pipeline."""

    def test_spliece_end_to_end_subset(self, tmp_path):
        """Test full pipeline on subset of test data (real fixtures; skip if absent)."""
        from dataclasses import replace
        if not config.paths.test_embeddings_path.exists():
            pytest.skip("real embeddings not present (gitignored)")
        # Load vocabulary
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        # Load subset of test embeddings (100 images)
        test_emb = load_tensor(config.paths.test_embeddings_path)[:100]

        with open(config.paths.test_image_ids_path) as f:
            test_ids = json.load(f)[:100]

        # Run SPLiCE into an isolated dir (never clobber results/spliece)
        cfg = replace(config.spliece, output_dir=tmp_path)
        results = run(cfg, test_emb, test_ids, vocab_terms)

        # Verify output structure
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"
        assert all("image_id" in r for r in results), "Missing image_id field"
        assert all("top_k_concepts" in r for r in results), "Missing top_k_concepts field"
        assert all("pseudo_report" in r for r in results), "Missing pseudo_report field"

        # Verify each result has k or fewer concepts (NNLS may zero some atoms)
        for r in results:
            assert len(r["top_k_concepts"]) <= config.spliece.k, \
                f"Expected ≤{config.spliece.k} concepts, got {len(r['top_k_concepts'])}"

            # Verify each concept has the SAE-compatible schema (F-001)
            for c in r["top_k_concepts"]:
                assert "feature_id" in c, "Concept missing 'feature_id' field"
                assert "name" in c, "Concept missing 'name' field"
                assert "activation" in c, "Concept missing 'activation' field"
                assert isinstance(c["activation"], float), "Activation must be float"
                assert c["activation"] > 0, "Activation must be positive"

    def test_output_file_created(self, tmp_path):
        """Verify output file is written correctly (isolated dir)."""
        from dataclasses import replace
        if not config.paths.test_embeddings_path.exists():
            pytest.skip("real embeddings not present (gitignored)")
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        test_emb = load_tensor(config.paths.test_embeddings_path)[:10]
        test_ids = [f"test_{i}" for i in range(10)]

        cfg = replace(config.spliece, output_dir=tmp_path)
        results = run(cfg, test_emb, test_ids, vocab_terms)

        output_path = tmp_path / "sample_explanations.json"
        assert output_path.exists(), f"Output file not found: {output_path}"

        with open(output_path) as f:
            loaded_results = json.load(f)

        assert len(loaded_results) == len(results), "File content mismatch"
        assert loaded_results[0]["image_id"] == results[0]["image_id"], "ID mismatch"

    def test_gap_correction_config_respected(self, tmp_path):
        """Test that use_gap_correction flag is respected (isolated dirs)."""
        from dataclasses import replace
        if not config.paths.test_embeddings_path.exists():
            pytest.skip("real embeddings not present (gitignored)")
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        test_emb = load_tensor(config.paths.test_embeddings_path)[:5]
        test_ids = [f"test_{i}" for i in range(5)]

        cfg_with_gap = replace(config.spliece, use_gap_correction=True,
                               output_dir=tmp_path / "gap")
        results_with_gap = run(cfg_with_gap, test_emb, test_ids, vocab_terms)

        cfg_no_gap = replace(config.spliece, use_gap_correction=False,
                             output_dir=tmp_path / "nogap")
        results_no_gap = run(cfg_no_gap, test_emb, test_ids, vocab_terms)

        # Results should differ
        assert results_with_gap[0]["top_k_concepts"] != results_no_gap[0]["top_k_concepts"], \
            "Gap correction should affect results"
