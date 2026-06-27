"""Integration tests for SPLiCE pipeline."""

from __future__ import annotations

import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, "src/")
from concept_discovery.spliece import run
import config
from utils import load_tensor


class TestSpliCEPipeline:
    """End-to-end tests for SPLiCE pipeline."""

    def test_spliece_end_to_end_subset(self):
        """Test full pipeline on subset of test data."""
        # Load vocabulary
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        # Load subset of test embeddings (100 images)
        test_emb = load_tensor(config.paths.test_embeddings_path)[:100]

        # Create dummy image IDs (or load if available)
        test_ids_path = Path("data/test_image_ids.json")
        if test_ids_path.exists():
            with open(test_ids_path) as f:
                all_ids = json.load(f)
                test_ids = all_ids[:100]
        else:
            test_ids = [f"test_{i}" for i in range(100)]

        # Run SPLiCE
        results = run(config.spliece, test_emb, test_ids, vocab_terms)

        # Verify output structure
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"
        assert all("image_id" in r for r in results), "Missing image_id field"
        assert all("top_k_concepts" in r for r in results), "Missing top_k_concepts field"
        assert all("pseudo_report" in r for r in results), "Missing pseudo_report field"

        # Verify each result has k or fewer concepts (clamp may create zeros)
        for r in results:
            assert len(r["top_k_concepts"]) <= config.spliece.k, \
                f"Expected ≤{config.spliece.k} concepts, got {len(r['top_k_concepts'])}"

            # Verify each concept has required fields
            for c in r["top_k_concepts"]:
                assert "term" in c, "Concept missing 'term' field"
                assert "coefficient" in c, "Concept missing 'coefficient' field"
                assert isinstance(c["coefficient"], float), "Coefficient must be float"
                assert c["coefficient"] > 0, "Coefficient must be positive"

    def test_output_file_created(self):
        """Verify output file is written correctly."""
        # Load vocabulary
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        # Load tiny subset
        test_emb = load_tensor(config.paths.test_embeddings_path)[:10]
        test_ids = [f"test_{i}" for i in range(10)]

        # Run SPLiCE
        results = run(config.spliece, test_emb, test_ids, vocab_terms)

        # Verify output file exists
        output_path = config.spliece.output_dir / "sample_explanations.json"
        assert output_path.exists(), f"Output file not found: {output_path}"

        # Verify file can be loaded and matches results
        with open(output_path) as f:
            loaded_results = json.load(f)

        assert len(loaded_results) == len(results), "File content mismatch"
        assert loaded_results[0]["image_id"] == results[0]["image_id"], "ID mismatch"

    def test_gap_correction_config_respected(self):
        """Test that use_gap_correction flag is respected."""
        # Load vocabulary
        with open(config.paths.vocab_labels_path) as f:
            vocab_terms = json.load(f)

        test_emb = load_tensor(config.paths.test_embeddings_path)[:5]
        test_ids = [f"test_{i}" for i in range(5)]

        # Run with gap correction enabled
        from dataclasses import replace
        cfg_with_gap = replace(config.spliece, use_gap_correction=True,
                               output_dir=config.paths.results_dir / "spliece_test_gap")
        results_with_gap = run(cfg_with_gap, test_emb, test_ids, vocab_terms)

        # Run without gap correction
        cfg_no_gap = replace(config.spliece, use_gap_correction=False,
                              output_dir=config.paths.results_dir / "spliece_test_no_gap")
        results_no_gap = run(cfg_no_gap, test_emb, test_ids, vocab_terms)

        # Results should differ
        assert results_with_gap[0]["top_k_concepts"] != results_no_gap[0]["top_k_concepts"], \
            "Gap correction should affect results"
