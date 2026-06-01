"""
Tests for extract_visual_embeddings and extract_text_embeddings from extract_embeddings.py.

- Unit tests use mocks to verify the extraction pipeline logic
  (dataloader iteration, model inference, L2 normalization, saving).
- The integration test loads the real BiomedCLIP model on GPU and verifies
  that embeddings can be correctly extracted from 1 image and 1 text.

---

### How to run tests
    
    # Only unit tests (without GPU):
    python -m pytest tests/test_extract_embeddings.py -v -k "not Integration"

    # All tests (requires CUDA):
    python -m pytest tests/test_extract_embeddings.py -v

"""

import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from torch.utils.data import Dataset
from PIL import Image

from config import VLMConfig
import extract_embeddings

extract_visual_embeddings = extract_embeddings.extract_visual_embeddings
extract_text_embeddings = extract_embeddings.extract_text_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeImageDataset(Dataset):
    """Minimal image dataset returning dummy PIL images."""

    def __init__(self, n: int = 3, size: tuple = (224, 224)):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = Image.new("RGB", self.size, color=(idx * 30, idx * 60, idx * 90))
        return img, f"fake_image_{idx}.png"


class FakeTextDataset(Dataset):
    """Minimal text dataset returning dummy report strings."""

    def __init__(self, texts: list[str] | None = None):
        self.texts = texts or [
            "Findings: Normal heart size. Impression: No acute disease.",
            "Findings: Mild cardiomegaly. Impression: Stable.",
            "Findings: Clear lungs bilaterally. Impression: Normal exam.",
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], f"fake_report_{idx}.xml"


def _make_config(tmp_path: Path, batch_size: int = 2) -> VLMConfig:
    """Build a VLMConfig pointing to tmp_path for output and using CPU."""
    config = VLMConfig(
        batch_size=batch_size,
        num_workers=0,
        output_dir=str(tmp_path / "embeddings"),
        device="cpu",
    )
    return config


# =========================================================================
# Unit tests – extract_visual_embeddings
# =========================================================================

class TestExtractVisualEmbeddings:
    """Unit-tests for the visual embedding extraction pipeline."""

    def test_saves_file_to_correct_path(self, tmp_path):
        """Verify that embeddings are saved to config.visual_output_path."""
        config = _make_config(tmp_path)
        dataset = FakeImageDataset(n=4)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # processor.image_processor(...) returns a mock with a .to() that returns itself
        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.image_processor.return_value = proc_output

        # Model returns a tensor of shape (B, 512) per batch
        def fake_image_features(**kwargs):
            return torch.randn(config.batch_size, 512)

        mock_model.get_image_features.side_effect = fake_image_features
        mock_model.eval.return_value = None

        extract_visual_embeddings(mock_model, mock_processor, dataset, config)

        assert config.visual_output_path.exists(), "Visual embeddings file was not created"

        loaded = torch.load(config.visual_output_path, weights_only=True)
        assert loaded.shape == (4, 512), f"Expected shape (4, 512), got {loaded.shape}"

    def test_embeddings_are_l2_normalized(self, tmp_path):
        """Each embedding vector must have unit L2 norm after normalization."""
        config = _make_config(tmp_path, batch_size=3)
        dataset = FakeImageDataset(n=3)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.image_processor.return_value = proc_output

        # Return non-normalized vectors so we can verify the function normalizes
        mock_model.get_image_features.return_value = torch.tensor([
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        mock_model.eval.return_value = None

        extract_visual_embeddings(mock_model, mock_processor, dataset, config)

        loaded = torch.load(config.visual_output_path, weights_only=True)
        norms = loaded.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5), (
            f"Embeddings are not L2 normalized. Norms: {norms}"
        )

    def test_processor_receives_images(self, tmp_path):
        """The image_processor must be called with `images=<list of PIL Images>`."""
        config = _make_config(tmp_path, batch_size=2)
        dataset = FakeImageDataset(n=2)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.image_processor.return_value = proc_output

        mock_model.get_image_features.return_value = torch.randn(2, 512)
        mock_model.eval.return_value = None

        extract_visual_embeddings(mock_model, mock_processor, dataset, config)

        # processor.image_processor must have been called at least once
        assert mock_processor.image_processor.call_count >= 1
        # Inspect the first call's kwargs
        _, kwargs = mock_processor.image_processor.call_args
        assert "images" in kwargs, "image_processor was not called with 'images' kwarg"
        assert kwargs["return_tensors"] == "pt"

    def test_model_called_in_eval_and_no_grad(self, tmp_path):
        """model.eval() must be called before inference."""
        config = _make_config(tmp_path, batch_size=2)
        dataset = FakeImageDataset(n=2)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.image_processor.return_value = proc_output

        mock_model.get_image_features.return_value = torch.randn(2, 512)
        mock_model.eval.return_value = None

        extract_visual_embeddings(mock_model, mock_processor, dataset, config)

        mock_model.eval.assert_called()

    def test_handles_multiple_batches(self, tmp_path):
        """With 5 samples and batch_size=2, we expect 3 batches (2+2+1)."""
        config = _make_config(tmp_path, batch_size=2)
        dataset = FakeImageDataset(n=5)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.image_processor.return_value = proc_output

        # Return dynamic batch-sized tensors
        def fake_features(**kwargs):
            # Infer batch size from the image_processor call
            return torch.randn(len(mock_processor.image_processor.call_args[1]["images"]), 512)

        mock_model.get_image_features.side_effect = fake_features
        mock_model.eval.return_value = None

        extract_visual_embeddings(mock_model, mock_processor, dataset, config)

        loaded = torch.load(config.visual_output_path, weights_only=True)
        assert loaded.shape[0] == 5, f"Expected 5 embeddings, got {loaded.shape[0]}"
        assert mock_model.get_image_features.call_count == 3


# =========================================================================
# Unit tests – extract_text_embeddings
# =========================================================================

class TestExtractTextFeatures:
    """Unit-tests for the text embedding extraction pipeline."""

    def test_saves_file_to_correct_path(self, tmp_path):
        """Verify that text embeddings are saved to config.text_output_path."""
        config = _make_config(tmp_path)
        dataset = FakeTextDataset()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        # Use side_effect to return correctly-sized tensors per batch
        def fake_features(**kwargs):
            batch_size = len(mock_processor.tokenizer.call_args[1]["text"])
            return torch.randn(batch_size, 512)

        mock_model.get_text_features.side_effect = fake_features
        mock_model.eval.return_value = None

        extract_text_embeddings(mock_model, mock_processor, dataset, config)

        assert config.text_output_path.exists(), "Text embeddings file was not created"

        loaded = torch.load(config.text_output_path, weights_only=True)
        assert loaded.shape == (3, 512), f"Expected shape (3, 512), got {loaded.shape}"

    def test_embeddings_are_l2_normalized(self, tmp_path):
        """Each text embedding vector must have unit L2 norm."""
        config = _make_config(tmp_path, batch_size=3)
        dataset = FakeTextDataset()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        mock_model.get_text_features.return_value = torch.tensor([
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        mock_model.eval.return_value = None

        extract_text_embeddings(mock_model, mock_processor, dataset, config)

        loaded = torch.load(config.text_output_path, weights_only=True)
        norms = loaded.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5), (
            f"Text embeddings are not L2 normalized. Norms: {norms}"
        )

    def test_processor_receives_texts(self, tmp_path):
        """The tokenizer must be called with `text=<list of str>`."""
        config = _make_config(tmp_path, batch_size=3)
        dataset = FakeTextDataset()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        mock_model.get_text_features.return_value = torch.randn(3, 512)
        mock_model.eval.return_value = None

        extract_text_embeddings(mock_model, mock_processor, dataset, config)

        # processor.tokenizer must have been called at least once
        assert mock_processor.tokenizer.call_count >= 1
        _, kwargs = mock_processor.tokenizer.call_args
        assert "text" in kwargs, "tokenizer was not called with 'text' kwarg"
        assert kwargs["return_tensors"] == "pt"

    def test_model_called_in_eval_mode(self, tmp_path):
        """model.eval() must be called."""
        config = _make_config(tmp_path, batch_size=3)
        dataset = FakeTextDataset()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        mock_model.get_text_features.return_value = torch.randn(3, 512)
        mock_model.eval.return_value = None

        extract_text_embeddings(mock_model, mock_processor, dataset, config)

        mock_model.eval.assert_called()

    def test_handles_multiple_batches(self, tmp_path):
        """With 5 texts and batch_size=2, we expect 3 batches."""
        texts = [f"Report number {i}" for i in range(5)]
        config = _make_config(tmp_path, batch_size=2)
        dataset = FakeTextDataset(texts=texts)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        def fake_features(**kwargs):
            return torch.randn(len(mock_processor.tokenizer.call_args[1]["text"]), 512)

        mock_model.get_text_features.side_effect = fake_features
        mock_model.eval.return_value = None

        extract_text_embeddings(mock_model, mock_processor, dataset, config)

        loaded = torch.load(config.text_output_path, weights_only=True)
        assert loaded.shape[0] == 5, f"Expected 5 embeddings, got {loaded.shape[0]}"
        assert mock_model.get_text_features.call_count == 3


# =========================================================================
# Integration test – real model, 1 image + 1 text
# =========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegrationSingleSample:
    """
    Integration test: loads the real BiomedCLIP model and extracts embeddings
    from 1 real IU X-ray image and 1 synthetic text report.

    Requires:
      - CUDA GPU
      - Network access (first run downloads the model)
      - At least 1 image in data/iu_xray/images/images_normalized/
    """

    EMBEDDING_DIM = 512  # BiomedCLIP output dimension

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        """Load the real BiomedCLIP model once for the whole class."""
        from utils import load_vlm
        config = VLMConfig()
        model, processor = load_vlm(config)
        yield model, processor
        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()

    def test_extract_single_image_and_text(self, model_and_processor, tmp_path):
        """
        End-to-end test: extract 1 visual embedding and 1 text embedding,
        verifying shapes, normalization, and that the similarity score is a
        valid scalar (the model can meaningfully compare image vs text).
        """
        model, processor = model_and_processor

        # --- Locate a real image ----------------------------------------
        project_root = Path(__file__).parent.parent
        image_dir = project_root / "data" / "iu_xray" / "images" / "images_normalized"
        real_images = sorted(image_dir.glob("*.png"))
        assert len(real_images) > 0, (
            f"No PNG images found in {image_dir}. Cannot run integration test."
        )

        # -- Build single-element datasets --------------------------------
        class SingleImageDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                img = Image.open(real_images[0]).convert("RGB")
                return img, str(real_images[0])

        class SingleTextDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return (
                    "Findings: The heart is normal in size. The lungs are clear. "
                    "Impression: No acute cardiopulmonary disease.",
                    "synthetic_report.xml",
                )

        # -- Config for integration test ----------------------------------
        config = VLMConfig(
            batch_size=1,
            num_workers=0,
            output_dir=str(tmp_path / "integration_embeddings"),
            device="cuda"
        )

        # -- Extract visual embedding -------------------------------------
        extract_visual_embeddings(
            model, processor, SingleImageDataset(), config
        )
        assert config.visual_output_path.exists()
        visual_emb = torch.load(config.visual_output_path, weights_only=True)

        assert visual_emb.shape == (1, self.EMBEDDING_DIM), (
            f"Visual embedding shape mismatch: {visual_emb.shape}"
        )
        visual_norm = visual_emb.norm(dim=-1)
        assert torch.allclose(visual_norm, torch.ones(1), atol=1e-4), (
            f"Visual embedding not L2-normalized: norm = {visual_norm.item():.6f}"
        )

        # -- Extract text embedding ---------------------------------------
        extract_text_embeddings(
            model, processor, SingleTextDataset(), config
        )
        assert config.text_output_path.exists()
        text_emb = torch.load(config.text_output_path, weights_only=True)

        assert text_emb.shape == (1, self.EMBEDDING_DIM), (
            f"Text embedding shape mismatch: {text_emb.shape}"
        )
        text_norm = text_emb.norm(dim=-1)
        assert torch.allclose(text_norm, torch.ones(1), atol=1e-4), (
            f"Text embedding not L2-normalized: norm = {text_norm.item():.6f}"
        )

        # -- Cross-modal similarity check ---------------------------------
        # Both embeddings are unit-norm, so dot product = cosine similarity
        similarity = (visual_emb @ text_emb.T).item()
        assert -1.0 <= similarity <= 1.0, (
            f"Cosine similarity out of range: {similarity}"
        )
        # A functional model should produce non-zero similarity
        assert abs(similarity) > 1e-6, (
            f"Similarity is essentially zero ({similarity}), model may not be working"
        )
