"""
conftest.py — Shared fixtures for SAE tests.

Provides mock SAE model, fake embeddings, and temp directories
so tests run without GPU or real model weights.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def fake_embeddings():
    """Random embeddings tensor (100, 512) simulating BiomedCLIP output."""
    torch.manual_seed(42)
    return torch.randn(100, 512)


@pytest.fixture
def fake_vocab_embeddings():
    """Random vocabulary embeddings (50, 512) simulating text encoder output."""
    torch.manual_seed(123)
    return torch.randn(50, 512)


@pytest.fixture
def fake_vocab_labels():
    """Fake medical vocabulary labels."""
    return [f"concept_{i}" for i in range(50)]


@pytest.fixture
def fake_sparse():
    """Fake sparse activations (10, 4096) with k=32 non-zero per row."""
    torch.manual_seed(42)
    sparse = torch.zeros(10, 4096)
    for i in range(10):
        indices = torch.randperm(4096)[:32]
        sparse[i, indices] = torch.randn(32).abs()
    return sparse


@pytest.fixture
def mock_ae():
    """Mock AutoEncoderTopK that returns predictable outputs."""
    ae = MagicMock()
    ae.eval.return_value = ae

    def mock_encode(x, return_topk=False, use_threshold=False):
        batch_size = x.shape[0]
        sparse = torch.zeros(batch_size, 4096)
        for i in range(batch_size):
            indices = torch.randperm(4096)[:32]
            sparse[i, indices] = torch.randn(32).abs()
        if return_topk:
            topk_vals = []
            topk_idx = []
            for row in sparse:
                topk = row.topk(32)
                topk_vals.append(topk.values)
                topk_idx.append(topk.indices)
            return sparse, torch.stack(topk_vals), torch.stack(topk_idx), sparse
        return sparse

    def mock_decode(x):
        return torch.randn(x.shape[0], 512)

    def mock_forward(x, output_features=False):
        encoded = mock_encode(x)
        reconstructed = torch.randn(x.shape[0], 512)
        if output_features:
            return reconstructed, encoded
        return reconstructed

    ae.encode = mock_encode
    ae.decode = mock_decode
    ae.__call__ = mock_forward

    # Fake decoder weights (512, 4096) — transposed to (4096, 512)
    ae.decoder = MagicMock()
    ae.decoder.weight = MagicMock()
    ae.decoder.weight.data = torch.randn(512, 4096)

    return ae


@pytest.fixture
def tmp_model_dir(tmp_path, mock_ae):
    """Create a temporary model directory with a fake ae.pt."""
    model_dir = tmp_path / "sae_seed42"
    model_dir.mkdir()

    state_dict = {
        "encoder.weight": torch.randn(4096, 512),
        "encoder.bias": torch.zeros(4096),
        "decoder.weight": torch.randn(512, 4096),
        "b_dec": torch.zeros(512),
        "k": torch.tensor(32),
        "threshold": torch.tensor(-1.0),
    }
    torch.save(state_dict, model_dir / "ae.pt")

    config_data = {
        "trainer": {
            "dict_class": "AutoEncoderTopK",
            "k": 32,
            "activation_dim": 512,
            "dict_size": 4096,
        }
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config_data, f)

    return model_dir


@pytest.fixture
def tmp_model_dir_trainer0(tmp_path):
    """Model saved under trainer_0/ subdirectory (library convention)."""
    model_dir = tmp_path / "sae_seed0"
    trainer_dir = model_dir / "trainer_0"
    trainer_dir.mkdir(parents=True)

    state_dict = {
        "encoder.weight": torch.randn(4096, 512),
        "encoder.bias": torch.zeros(4096),
        "decoder.weight": torch.randn(512, 4096),
        "b_dec": torch.zeros(512),
        "k": torch.tensor(32),
        "threshold": torch.tensor(-1.0),
    }
    torch.save(state_dict, trainer_dir / "ae.pt")

    return model_dir


@pytest.fixture
def tmp_embeddings_file(tmp_path, fake_embeddings):
    """Save fake embeddings to a temp .pt file."""
    path = tmp_path / "visual_embeddings.pt"
    torch.save(fake_embeddings, path)
    return path


@pytest.fixture
def tmp_vocab_files(tmp_path, fake_vocab_embeddings, fake_vocab_labels):
    """Save fake vocab embeddings and labels to temp files."""
    emb_path = tmp_path / "text_vocab_embeddings.pt"
    labels_path = tmp_path / "vocabulary.json"
    torch.save(fake_vocab_embeddings, emb_path)
    with open(labels_path, "w") as f:
        json.dump(fake_vocab_labels, f)
    return emb_path, labels_path
