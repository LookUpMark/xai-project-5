"""
test_vlm_autoencoder.py — Integration tests: VLM embedding extraction → SAE pipeline.

Tests the complete flow from mock VLM (BiomedCLIP-like) through embedding
extraction, SAE encoding, concept naming, and explanation generation.
All models are mocked — no GPU or real weights required.
"""

import json
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset

from config import VLMConfig
from autoencoder.sae_module import SAEManager
import extract_embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeImageDataset(Dataset):
    """Minimal image dataset returning dummy PIL images."""

    def __init__(self, n: int = 10):
        from PIL import Image
        import numpy as np

        self.images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], f"image_{idx}.png"


class FakeTextDataset(Dataset):
    """Minimal text dataset returning medical report strings."""

    REPORTS = [
        "Cardiomegaly with bilateral pleural effusion.",
        "No acute cardiopulmonary disease.",
        "Right lower lobe consolidation suggesting pneumonia.",
        "Mild pulmonary edema. No pneumothorax.",
        "Stable appearance of the chest. No focal consolidation.",
    ]

    def __init__(self, n: int = 10):
        self.texts = [self.REPORTS[i % len(self.REPORTS)] for i in range(n)]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], f"report_{idx}.txt"


@pytest.fixture
def vlm_config(tmp_path):
    """VLMConfig pointing to tmp output paths."""
    return VLMConfig(
        model_name="mock-biomed-clip",
        processor_name="mock-biomed-clip",
        device="cpu",
        batch_size=4,
        num_workers=0,
        image_dir=str(tmp_path / "images"),
        reports_dir=str(tmp_path / "reports"),
        output_dir=str(tmp_path / "embeddings"),
    )


@pytest.fixture
def mock_vlm():
    """Mock BiomedCLIP model that returns 512-dim normalized embeddings."""
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model

    def get_image_features(**kwargs):
        batch_size = kwargs["pixel_values"].shape[0]
        emb = torch.randn(batch_size, 512)
        return emb / emb.norm(dim=-1, keepdim=True)

    def get_text_features(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        emb = torch.randn(batch_size, 512)
        return emb / emb.norm(dim=-1, keepdim=True)

    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    return model


@pytest.fixture
def mock_processor():
    """Mock processor with image_processor and tokenizer."""
    processor = MagicMock()

    def process_images(images, return_tensors="pt"):
        result = MagicMock()
        result.to.return_value = {"pixel_values": torch.randn(len(images), 3, 224, 224)}
        return result

    def process_text(text, return_tensors="pt", padding=True, truncation=True):
        result = MagicMock()
        n = len(text) if isinstance(text, list) else 1
        result.to.return_value = {
            "input_ids": torch.randint(0, 1000, (n, 20)),
            "attention_mask": torch.ones(n, 20),
        }
        return result

    processor.image_processor = process_images
    processor.tokenizer = process_text
    return processor


# ---------------------------------------------------------------------------
# Integration Tests: VLM → Embeddings
# ---------------------------------------------------------------------------


class TestVLMEmbeddingExtraction:
    """Test VLM produces correctly-shaped embeddings for SAE input."""

    def test_visual_extraction_produces_normalized_512d(
        self, mock_vlm, mock_processor, vlm_config
    ):
        """Full visual extraction pipeline: dataset → embeddings file."""
        dataset = FakeImageDataset(n=8)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        saved = torch.load(vlm_config.visual_output_path, weights_only=True)
        assert saved.shape == (8, 512)
        # Check L2-normalized (each row norm ≈ 1.0)
        norms = saved.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_text_extraction_produces_normalized_512d(
        self, mock_vlm, mock_processor, vlm_config
    ):
        """Full text extraction pipeline: reports → embeddings file."""
        dataset = FakeTextDataset(n=6)
        extract_embeddings.extract_text_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        saved = torch.load(vlm_config.text_output_path, weights_only=True)
        assert saved.shape == (6, 512)
        norms = saved.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    def test_batched_extraction_consistency(
        self, mock_vlm, mock_processor, vlm_config
    ):
        """Batch size doesn't affect output count."""
        dataset = FakeImageDataset(n=10)
        # batch_size=4 → 3 batches (4+4+2)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        saved = torch.load(vlm_config.visual_output_path, weights_only=True)
        assert saved.shape[0] == 10


# ---------------------------------------------------------------------------
# Integration Tests: Embeddings → SAE → Concepts
# ---------------------------------------------------------------------------


class TestEmbeddingsToSAE:
    """Test that VLM embeddings flow correctly into SAE encoding."""

    def test_embeddings_encode_with_sae(
        self, mock_vlm, mock_processor, vlm_config, tmp_model_dir
    ):
        """Extract embeddings → load into SAE → encode to sparse features."""
        # Step 1: Extract embeddings
        dataset = FakeImageDataset(n=5)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        # Step 2: Load embeddings and feed to SAE
        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        assert embeddings.shape == (5, 512)

        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        sparse = mgr.encode(embeddings)
        assert sparse.shape == (5, 4096)

    def test_sae_reconstruction_from_vlm_embeddings(
        self, mock_vlm, mock_processor, vlm_config, tmp_model_dir
    ):
        """VLM embeddings → SAE encode → decode → same shape."""
        dataset = FakeImageDataset(n=3)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        sparse = mgr.encode(embeddings)
        reconstructed = mgr.decode(sparse)

        assert reconstructed.shape == embeddings.shape

    def test_top_concepts_from_vlm_embeddings(
        self, mock_vlm, mock_processor, vlm_config, tmp_model_dir
    ):
        """VLM embeddings → SAE → top-k concepts extraction."""
        dataset = FakeImageDataset(n=4)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        top_concepts = mgr.get_top_concepts(embeddings, n=5)
        assert len(top_concepts) == 4
        for sample in top_concepts:
            assert len(sample) <= 5
            for feat_id, activation in sample:
                assert 0 <= feat_id < 4096
                assert activation >= 0


# ---------------------------------------------------------------------------
# Integration Tests: Full Pipeline (VLM → SAE → Naming → Explanation)
# ---------------------------------------------------------------------------


class TestFullVLMSAEPipeline:
    """End-to-end: VLM extraction → SAE → concept naming → explanations."""

    def test_end_to_end_vlm_to_explanations(
        self,
        mock_vlm,
        mock_processor,
        vlm_config,
        tmp_model_dir,
        fake_vocab_embeddings,
        fake_vocab_labels,
    ):
        """Complete pipeline from image input to structured explanations."""
        from importlib import import_module

        gen_module = import_module("autoencoder.generate_explanations")

        # Step 1: Extract visual embeddings
        dataset = FakeImageDataset(n=3)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        # Step 2: Load embeddings into SAE
        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        # Step 3: Name concepts using text embeddings
        concept_names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels)
        assert len(concept_names) == 4096

        # Step 4: Get top concepts for each sample
        top_concepts = mgr.get_top_concepts(embeddings, n=5)
        assert len(top_concepts) == 3

        # Step 5: Generate explanations
        str_names = {str(k): v for k, v in concept_names.items()}
        explanations = []
        for idx, sample_concepts in enumerate(top_concepts):
            explanation = gen_module.generate_explanation(sample_concepts, str_names)
            explanation["image_id"] = f"image_{idx}.png"
            explanations.append(explanation)

        assert len(explanations) == 3
        for exp in explanations:
            assert "image_id" in exp
            assert "pseudo_report" in exp
            assert "top_k_concepts" in exp
            assert isinstance(exp["pseudo_report"], str)
            assert len(exp["pseudo_report"]) > 0

    def test_text_embeddings_as_vocabulary(
        self, mock_vlm, mock_processor, vlm_config, tmp_model_dir
    ):
        """Use text extraction output as vocabulary for concept naming."""
        # Extract text embeddings (simulating vocabulary creation)
        text_dataset = FakeTextDataset(n=5)
        extract_embeddings.extract_text_embeddings(
            mock_vlm, mock_processor, text_dataset, vlm_config
        )

        # Use extracted text embeddings as vocab
        vocab_emb = torch.load(vlm_config.text_output_path, weights_only=True)
        vocab_labels = [f"concept_{i}" for i in range(5)]

        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)
        concept_names = mgr.name_concepts(vocab_emb, vocab_labels)

        assert len(concept_names) == 4096
        all_names = {info["name"] for info in concept_names.values()}
        assert all_names.issubset(set(vocab_labels))

    def test_metrics_on_vlm_embeddings(
        self, mock_vlm, mock_processor, vlm_config, tmp_model_dir
    ):
        """Compute SAE quality metrics on VLM-produced embeddings."""
        dataset = FakeImageDataset(n=10)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        mse = mgr.compute_reconstruction_mse(embeddings)
        cosine = mgr.compute_cosine_reconstruction(embeddings)
        sparsity = mgr.compute_sparsity_metrics(embeddings)

        assert mse > 0
        assert -1.0 <= cosine <= 1.0
        assert sparsity["l0_mean"] > 0
        assert 0 <= sparsity["dead_features_pct"] <= 100

    def test_output_json_serializable(
        self,
        mock_vlm,
        mock_processor,
        vlm_config,
        tmp_model_dir,
        fake_vocab_embeddings,
        fake_vocab_labels,
    ):
        """All pipeline outputs can be serialized to JSON for results/."""
        from importlib import import_module

        gen_module = import_module("autoencoder.generate_explanations")

        dataset = FakeImageDataset(n=2)
        extract_embeddings.extract_visual_embeddings(
            mock_vlm, mock_processor, dataset, vlm_config
        )

        embeddings = torch.load(vlm_config.visual_output_path, weights_only=True)
        mgr = SAEManager({"device": "cpu"})
        mgr.load(tmp_model_dir)

        concept_names = mgr.name_concepts(fake_vocab_embeddings, fake_vocab_labels)
        top_concepts = mgr.get_top_concepts(embeddings, n=3)
        metrics = mgr.compute_sparsity_metrics(embeddings)

        str_names = {str(k): v for k, v in concept_names.items()}
        explanations = [
            gen_module.generate_explanation(sc, str_names) for sc in top_concepts
        ]

        # All must serialize without error
        json.dumps(concept_names)
        json.dumps(top_concepts)
        json.dumps(metrics)
        json.dumps(explanations)
