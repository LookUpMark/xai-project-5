"""Shared utilities for the SAE concept-discovery pipeline."""

from __future__ import annotations

import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from config import VLMConfig


def load_vlm(config: VLMConfig):
    """Load BiomedCLIP model and processor.

    Args:
        config (VLMConfig): dataclass containing parameters.

    Returns:
        tuple: (model, processor) loaded.
    """
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.processor_name,
        trust_remote_code=True,
    )

    model.eval().to(config.device)

    return model, processor


def set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tensor(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Safely load a tensor with weights_only=True.

    Args:
        path: Path to the .pt file.
        device: Target device for map_location.

    Returns:
        Loaded tensor.
    """
    return torch.load(path, map_location=device, weights_only=True)


def ensure_dir(path: Path) -> None:
    """Create parent directories if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure logging with a standard format and return a logger.

    Args:
        name: Logger name (typically __name__ from caller).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def dataclass_to_dict(obj) -> dict:
    """Convert a frozen/regular dataclass to a plain dict.

    Args:
        obj: A dataclass instance.

    Returns:
        Dict with field names as keys.
    """
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
