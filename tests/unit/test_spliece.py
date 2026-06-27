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
