"""
Tests for load_vlm from utils.py.

- Unit tests use mocks to verify that load_vlm correctly calls
  AutoModel and AutoProcessor, sets eval mode, and moves the model to CUDA.
- The integration test loads the real BiomedCLIP model on GPU and verifies
  that model and processor are correctly initialized (device, eval mode,
  sub-components, embedding dimension).

---

### How to run tests

    # Only unit tests (without GPU):
    python -m pytest tests/test_load_vlm.py -v -k "not Integration"

    # All tests (requires CUDA):
    python -m pytest tests/test_load_vlm.py -v

"""

from unittest.mock import MagicMock, patch
import torch
import pytest

from config import VLMConfig
import utils

load_vlm = utils.load_vlm


def test_load_vlm_success():
    """Test load_vlm successfully loads the model and processor, sets eval mode, and moves model to CUDA."""
    config = VLMConfig(model_name="dummy-model", processor_name="dummy-processor")

    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with (
        patch.object(utils, "AutoModel") as mock_auto_model,
        patch.object(utils, "AutoProcessor") as mock_auto_processor,
    ):
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_processor.from_pretrained.return_value = mock_processor

        model, processor = load_vlm(config)

        mock_auto_model.from_pretrained.assert_called_once_with(
            "dummy-model", trust_remote_code=True
        )

        mock_auto_processor.from_pretrained.assert_called_once_with(
            "dummy-processor", trust_remote_code=True
        )

        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with(config.device)

        assert model == mock_model
        assert processor == mock_processor


def test_load_vlm_model_loading_failure():
    """Test that load_vlm propagates any exception raised during model loading."""
    config = VLMConfig(model_name="invalid-model", processor_name="dummy-processor")

    with patch.object(utils, "AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained.side_effect = RuntimeError(
            "Failed to download model weights"
        )

        with pytest.raises(RuntimeError, match="Failed to download model weights"):
            load_vlm(config)


def test_load_vlm_processor_loading_failure():
    """Test that load_vlm propagates any exception raised during processor loading."""
    config = VLMConfig(model_name="dummy-model", processor_name="invalid-processor")

    mock_model = MagicMock()

    with (
        patch.object(utils, "AutoModel") as mock_auto_model,
        patch.object(utils, "AutoProcessor") as mock_auto_processor,
    ):
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_processor.from_pretrained.side_effect = ValueError(
            "Invalid processor name"
        )

        with pytest.raises(ValueError, match="Invalid processor name"):
            load_vlm(config)


# =========================================================================
# Integration test – real model loading
# =========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLoadVLMIntegration:
    """
    Integration test: loads the real BiomedCLIP model and verifies
    that model and processor are correctly initialized.

    Requires:
      - CUDA GPU
      - Network access (first run downloads the model)
    """

    @pytest.fixture(scope="class")
    def loaded_model_and_processor(self):
        """Load the real BiomedCLIP model once for the whole class."""
        config = VLMConfig()
        model, processor = load_vlm(config)
        yield model, processor
        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()

    def test_model_is_on_cuda(self, loaded_model_and_processor):
        """The model must be placed on CUDA after loading."""
        model, _ = loaded_model_and_processor
        device = next(model.parameters()).device
        assert device.type == "cuda", f"Model is on {device}, expected cuda"

    def test_model_is_in_eval_mode(self, loaded_model_and_processor):
        """The model must be in evaluation mode (no dropout, no batchnorm updates)."""
        model, _ = loaded_model_and_processor
        assert not model.training, "Model should be in eval mode, but training=True"

    def test_processor_has_image_processor(self, loaded_model_and_processor):
        """The processor must expose an image_processor sub-component."""
        _, processor = loaded_model_and_processor
        assert hasattr(processor, "image_processor"), (
            "Processor does not have 'image_processor' attribute"
        )
        assert processor.image_processor is not None

    def test_processor_has_tokenizer(self, loaded_model_and_processor):
        """The processor must expose a tokenizer sub-component."""
        _, processor = loaded_model_and_processor
        assert hasattr(processor, "tokenizer"), (
            "Processor does not have 'tokenizer' attribute"
        )
        assert processor.tokenizer is not None

    def test_model_output_dimension(self, loaded_model_and_processor):
        """The model's projection dimension must be 512 (BiomedCLIP standard)."""
        model, _ = loaded_model_and_processor
        # BiomedCLIP exposes projection_dim on the config
        expected_dim = 512
        actual_dim = model.config.projection_dim
        assert actual_dim == expected_dim, (
            f"Expected projection_dim={expected_dim}, got {actual_dim}"
        )

    def test_returns_tuple(self, loaded_model_and_processor):
        """load_vlm must return exactly (model, processor)."""
        model, processor = loaded_model_and_processor
        assert model is not None
        assert processor is not None
