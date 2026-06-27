"""Unit tests for SPLiCE sparse decomposition."""

from __future__ import annotations

import pytest
import torch

import sys
sys.path.insert(0, "src/")
from concept_discovery.spliece import decompose_image


class TestSpliCE:
    """Test suite for SPLiCE decomposition."""

    def test_decompose_image_returns_k_nonzero(self):
        """Verify exactly k non-zero coefficients are returned."""
        vocab_emb = torch.randn(1030, 512)
        vocab_emb = vocab_emb / vocab_emb.norm(dim=1, keepdim=True)
        image_emb = torch.randn(512)
        image_emb = image_emb / image_emb.norm()

        coeffs = decompose_image(image_emb, vocab_emb, gap=None, k=32)

        assert coeffs.shape == (1030,), f"Expected shape (1030,), got {coeffs.shape}"
        # OMP may produce slightly more or fewer than k due to numerical precision
        nonzero_count = (coeffs > 0).sum().item()
        assert nonzero_count <= 32, \
            f"Expected ≤32 non-zero coeffs, got {nonzero_count}"
        assert (coeffs >= 0).all(), "Found negative coefficients"

    def test_gap_correction_changes_result(self):
        """Verify that modality gap correction actually changes the output."""
        vocab_emb = torch.randn(100, 512)
        vocab_emb = vocab_emb / vocab_emb.norm(dim=1, keepdim=True)
        image_emb = torch.randn(512)
        image_emb = image_emb / image_emb.norm()
        gap = torch.randn(512)

        coeffs_no_gap = decompose_image(image_emb, vocab_emb, gap=None, k=10)
        coeffs_with_gap = decompose_image(image_emb, vocab_emb, gap=gap, k=10)

        assert not torch.allclose(coeffs_no_gap, coeffs_with_gap), \
            "Gap correction should change decomposition"

    def test_all_coefficients_non_negative(self):
        """Verify clamp(min=0) enforces non-negativity."""
        vocab_emb = torch.randn(200, 512)
        vocab_emb = vocab_emb / vocab_emb.norm(dim=1, keepdim=True)
        image_emb = torch.randn(512)
        image_emb = image_emb / image_emb.norm()

        coeffs = decompose_image(image_emb, vocab_emb, gap=None, k=20)

        assert (coeffs >= 0).all(), "All coefficients must be non-negative"

    def test_vocab_shape_mismatch_raises(self):
        """Test that vocabulary embedding shape mismatches are handled."""
        vocab_emb = torch.randn(50, 512)  # Wrong vocab size
        image_emb = torch.randn(512)
        image_emb = image_emb / image_emb.norm()

        # Should not raise, but should handle gracefully
        coeffs = decompose_image(image_emb, vocab_emb, gap=None, k=10)
        assert coeffs.shape == (50,), "Should handle smaller vocab"

    def test_k_larger_than_vocab_raises(self):
        """Test that k > vocab_size raises ValueError (sklearn constraint)."""
        vocab_emb = torch.randn(20, 512)
        vocab_emb = vocab_emb / vocab_emb.norm(dim=1, keepdim=True)
        image_emb = torch.randn(512)
        image_emb = image_emb / image_emb.norm()

        # k=25 > vocab_size=20; sklearn OMP raises ValueError
        with pytest.raises(ValueError, match="cannot be more than the number of features"):
            decompose_image(image_emb, vocab_emb, gap=None, k=25)

    def test_run_emits_sae_schema(self, tmp_path):
        """F-001/F-009: run() must emit SAE-compatible keys
        (feature_id/name/activation) — the exact schema the LLM judge reads.
        Synthetic fixtures so it runs without the gitignored embeddings."""
        from dataclasses import replace
        from concept_discovery.spliece import run
        import config

        V, D = 60, 512
        vocab_emb = torch.randn(V, D)
        vocab_emb = vocab_emb / vocab_emb.norm(dim=1, keepdim=True)
        vocab_terms = [{"term": f"term_{i}"} for i in range(V)]  # real contract: list[dict]

        # images built as positive combos of vocab atoms -> NNLS yields >0 concepts
        image_embs = torch.stack([
            (vocab_emb[3] + vocab_emb[7]) / 2,
            (vocab_emb[10] + vocab_emb[20]) / 2,
            vocab_emb[33],
        ])
        image_ids = ["img_a", "img_b", "img_c"]

        emb_file = tmp_path / "vocab_emb.pt"
        torch.save(vocab_emb, emb_file)
        cfg = replace(
            config.spliece, k=10, use_gap_correction=False,
            vocab_emb_path=emb_file, output_dir=tmp_path,
        )

        results = run(cfg, image_embs, image_ids, vocab_terms)

        assert len(results) == 3
        for r in results:
            assert set(r.keys()) == {"image_id", "top_k_concepts", "pseudo_report"}
            assert len(r["top_k_concepts"]) <= 10
            for c in r["top_k_concepts"]:
                assert set(c.keys()) == {"feature_id", "name", "activation"}
                assert isinstance(c["feature_id"], int)
                assert isinstance(c["name"], str)
                assert isinstance(c["activation"], float)
                assert c["activation"] > 0

    def test_run_length_guards(self, tmp_path):
        """F-007: run() raises on test_embeddings/image_ids and vocab/vocab_emb mismatches."""
        from dataclasses import replace
        from concept_discovery.spliece import run
        import config

        vocab_terms = [{"term": f"t{i}"} for i in range(30)]
        emb_file = tmp_path / "v.pt"
        torch.save(torch.randn(30, 512), emb_file)
        cfg = replace(config.spliece, vocab_emb_path=emb_file, output_dir=tmp_path)

        # 5 embeddings vs 3 ids -> first guard fires before vocab is loaded
        with pytest.raises(ValueError, match="test_embeddings"):
            run(cfg, torch.randn(5, 512), ["a", "b", "c"], vocab_terms)

        # vocab_terms (30) != vocab_emb rows (10) -> second guard
        small_file = tmp_path / "small.pt"
        torch.save(torch.randn(10, 512), small_file)
        cfg2 = replace(config.spliece, vocab_emb_path=small_file, output_dir=tmp_path)
        with pytest.raises(ValueError, match="vocab_terms"):
            run(cfg2, torch.randn(3, 512), ["a", "b", "c"], vocab_terms)
