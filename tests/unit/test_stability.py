"""
test_stability.py — Tests for multi-seed stability analysis.

Verifies Jaccard computation and stability metrics using real
AutoEncoderTopK with random weights.
"""

import pytest
import torch

from autoencoder.sae_module import SAEManager


class TestComputeStability:
    def test_identical_models_perfect_jaccard(self, tmp_model_dir, fake_embeddings):
        """Two copies of the same model should have Jaccard = 1.0."""
        dirs = [tmp_model_dir, tmp_model_dir]
        result = SAEManager.compute_stability(
            dirs, fake_embeddings[:20], config={"device": "cpu"}
        )

        assert result["mean_jaccard"] == pytest.approx(1.0, abs=1e-6)
        assert result["jaccard_matrix"].shape == (2, 2)

    def test_jaccard_matrix_symmetric(self, tmp_model_dir, fake_embeddings):
        dirs = [tmp_model_dir, tmp_model_dir, tmp_model_dir]
        result = SAEManager.compute_stability(
            dirs, fake_embeddings[:20], config={"device": "cpu"}
        )

        matrix = result["jaccard_matrix"]
        assert torch.allclose(matrix, matrix.T)

    def test_jaccard_diagonal_is_one(self, tmp_model_dir, fake_embeddings):
        dirs = [tmp_model_dir, tmp_model_dir]
        result = SAEManager.compute_stability(
            dirs, fake_embeddings[:20], config={"device": "cpu"}
        )

        matrix = result["jaccard_matrix"]
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)

    def test_output_keys(self, tmp_model_dir, fake_embeddings):
        dirs = [tmp_model_dir, tmp_model_dir]
        result = SAEManager.compute_stability(
            dirs, fake_embeddings[:10], config={"device": "cpu"}
        )

        assert "jaccard_matrix" in result
        assert "mean_jaccard" in result
        assert "std_jaccard" in result

    def test_different_seeds_produce_results(self, tmp_path, fake_embeddings):
        """Create two models with different random weights to test < 1.0 Jaccard."""
        # Model A
        dir_a = tmp_path / "sae_seedA"
        dir_a.mkdir()
        torch.manual_seed(0)
        state_a = {
            "encoder.weight": torch.randn(1024, 512),
            "encoder.bias": torch.zeros(1024),
            "decoder.weight": torch.randn(512, 1024),
            "b_dec": torch.zeros(512),
            "k": torch.tensor(32),
            "threshold": torch.tensor(-1.0),
        }
        torch.save(state_a, dir_a / "ae.pt")

        # Model B (different random weights)
        dir_b = tmp_path / "sae_seedB"
        dir_b.mkdir()
        torch.manual_seed(999)
        state_b = {
            "encoder.weight": torch.randn(1024, 512),
            "encoder.bias": torch.zeros(1024),
            "decoder.weight": torch.randn(512, 1024),
            "b_dec": torch.zeros(512),
            "k": torch.tensor(32),
            "threshold": torch.tensor(-1.0),
        }
        torch.save(state_b, dir_b / "ae.pt")

        result = SAEManager.compute_stability(
            [dir_a, dir_b], fake_embeddings[:20], config={"device": "cpu"}
        )

        # Different models should have Jaccard < 1.0
        assert result["mean_jaccard"] < 1.0
        assert result["mean_jaccard"] >= 0.0

    def test_trainer0_subdir_loading(self, tmp_model_dir_trainer0, fake_embeddings):
        """Model saved under trainer_0/ should load correctly."""
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir_trainer0)
        assert mgr.is_loaded

        sparse = mgr.encode(fake_embeddings[:5])
        assert sparse.shape == (5, 1024)

    def test_n_parameter_limits_comparison(self, tmp_model_dir, fake_embeddings):
        """n parameter should limit which features are compared."""
        dirs = [tmp_model_dir, tmp_model_dir]

        # n=32 (all features) should give Jaccard 1.0 for identical models
        result_full = SAEManager.compute_stability(
            dirs, fake_embeddings[:10], config={"device": "cpu"}, n=32
        )
        assert result_full["mean_jaccard"] == pytest.approx(1.0, abs=1e-6)

        # n=5 (only top-5 features) should also give 1.0 for identical models
        result_n5 = SAEManager.compute_stability(
            dirs, fake_embeddings[:10], config={"device": "cpu"}, n=5
        )
        assert result_n5["mean_jaccard"] == pytest.approx(1.0, abs=1e-6)
