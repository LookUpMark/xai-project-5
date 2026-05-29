import sys
import importlib
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

# Ensuring src is in python path so that imports (like from config import VLMConfig) work correctly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import VLMConfig
import extract_embeddings

load_vlm = extract_embeddings.load_vlm

def test_load_vlm_success():
    """Test load_vlm successfully loads the model and processor, sets eval mode, and moves model to CUDA."""
    config = VLMConfig(
        model_name="dummy-model",
        processor_name="dummy-processor"
    )

    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch.object(extract_embeddings, "AutoModel") as mock_auto_model, \
         patch.object(extract_embeddings, "AutoProcessor") as mock_auto_processor:
        
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_processor.from_pretrained.return_value = mock_processor

        model, processor = load_vlm(config)

        mock_auto_model.from_pretrained.assert_called_once_with(
            "dummy-model",
            trust_remote_code=True
        )

        mock_auto_processor.from_pretrained.assert_called_once_with(
            "dummy-processor",
            trust_remote_code=True
        )

        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cuda")

        assert model == mock_model
        assert processor == mock_processor


def test_load_vlm_model_loading_failure():
    """Test that load_vlm propagates any exception raised during model loading."""
    config = VLMConfig(
        model_name="invalid-model",
        processor_name="dummy-processor"
    )

    with patch.object(extract_embeddings, "AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained.side_effect = RuntimeError("Failed to download model weights")
        
        with pytest.raises(RuntimeError, match="Failed to download model weights"):
            load_vlm(config)


def test_load_vlm_processor_loading_failure():
    """Test that load_vlm propagates any exception raised during processor loading."""
    config = VLMConfig(
        model_name="dummy-model",
        processor_name="invalid-processor"
    )

    mock_model = MagicMock()

    with patch.object(extract_embeddings, "AutoModel") as mock_auto_model, \
         patch.object(extract_embeddings, "AutoProcessor") as mock_auto_processor:
        
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_processor.from_pretrained.side_effect = ValueError("Invalid processor name")

        with pytest.raises(ValueError, match="Invalid processor name"):
            load_vlm(config)
