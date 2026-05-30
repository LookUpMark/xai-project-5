"""
test_sae_module.py — Unit tests for SAEManager.

Tests the facade interface with mocked AutoEncoderTopK.
No GPU or real model weights required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from autoencoder.sae_module import SAEManager, DEFAULT_CONFIG


class TestSAEManagerInit:
    def test_default_config(self):
        mgr = SAEManager()
        assert mgr.config["activation_dim"] == 512
        assert mgr.config["dict_size"] == 4096
        assert mgr.config["k"] == 32

    def test_custom_config(self):
        mgr = SAEManager({"k": 64, "device": "cpu"})
        assert mgr.config["k"] == 64
        assert mgr.config["device"] == "cpu"
        assert mgr.config["activation_dim"] == 512  # default preserved

    def test_not_loaded_initially(self):
        mgr = SAEManager()
        assert not mgr.is_loaded


class TestSAEManagerLoad:
    def test_load_success(self, tmp_model_dir):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        assert mgr.is_loaded

    def test_load_missing_file(self, tmp_path):
        mgr = SAEManager({"device": "cpu"})
        with pytest.raises(FileNotFoundError):
            mgr.load(tmp_path / "nonexistent")

    def test_load_string_path(self, tmp_model_dir):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(str(tmp_model_dir))
        assert mgr.is_loaded


class TestSAEManagerEncode:
    def test_encode_not_loaded_raises(self, fake_embeddings):
        mgr = SAEManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.encode(fake_embeddings)

    def test_encode_output_shape(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        sparse = mgr.encode(fake_embeddings[:10])
        assert sparse.shape == (10, 4096)

    def test_encode_sparsity(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        sparse = mgr.encode(fake_embeddings[:10])
        # Each row should have at most k non-zero values
        l0 = (sparse != 0).sum(dim=1)
        assert (l0 <= 32).all()

    def test_encode_topk_shapes(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        sparse, values, indices = mgr.encode_topk(fake_embeddings[:10])
        assert sparse.shape == (10, 4096)
        assert values.shape == (10, 32)
        assert indices.shape == (10, 32)


class TestSAEManagerDecode:
    def test_decode_output_shape(self, tmp_model_dir, fake_sparse):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        reconstructed = mgr.decode(fake_sparse)
        assert reconstructed.shape == (10, 512)

    def test_reconstruct_output_shape(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        x_hat = mgr.reconstruct(fake_embeddings[:10])
        assert x_hat.shape == (10, 512)


class TestSAEManagerConcepts:
    def test_get_decoder_weights_shape(self, tmp_model_dir):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        W = mgr.get_decoder_weights()
        assert W.shape == (4096, 512)

    def test_get_top_concepts_length(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        concepts = mgr.get_top_concepts(fake_embeddings[:5], n=3)
        assert len(concepts) == 5
        assert len(concepts[0]) == 3

    def test_get_top_concepts_tuple_format(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        concepts = mgr.get_top_concepts(fake_embeddings[:1], n=2)
        feat_id, activation = concepts[0][0]
        assert isinstance(feat_id, int)
        assert isinstance(activation, float)
        assert 0 <= feat_id < 4096


class TestSAEManagerMetrics:
    def test_reconstruction_mse_is_float(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        mse = mgr.compute_reconstruction_mse(fake_embeddings[:10])
        assert isinstance(mse, float)
        assert mse >= 0

    def test_sparsity_metrics_keys(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        metrics = mgr.compute_sparsity_metrics(fake_embeddings[:10])
        assert "l0_mean" in metrics
        assert "l0_std" in metrics
        assert "hoyer_mean" in metrics
        assert "dead_features_pct" in metrics

    def test_sparsity_l0_reasonable(self, tmp_model_dir, fake_embeddings):
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        metrics = mgr.compute_sparsity_metrics(fake_embeddings[:10])
        # L0 should be <= k (32) since TopK enforces this
        assert metrics["l0_mean"] <= 32
