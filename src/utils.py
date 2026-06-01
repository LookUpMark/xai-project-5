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


def load_state_dict(path: str | Path, device: str = "cpu") -> dict:
    """Safely load a model state dict with weights_only=True.

    Args:
        path: Path to the .pt file containing a state dict.
        device: Target device for map_location.

    Returns:
        Loaded state dict (OrderedDict).
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


def split_embeddings(
    source_path: Path,
    train_path: Path,
    test_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create and save a train/test split from a embeddings tensor.

    If both output files already exist, loads and returns them without
    recomputing the split.

    Args:
        source_path: Path to the source embeddings .pt file (N x D).
        train_path: Destination path for train embeddings.
        test_path: Destination path for test embeddings.
        train_ratio: Fraction of samples for the train set. Defaults to 0.8.
        seed: Random seed for reproducible splitting. Defaults to 42.

    Returns:
        Tuple of (train_embeddings, test_embeddings) tensors.
    """
    if train_path.exists() and test_path.exists():
        train_emb = load_tensor(train_path)
        test_emb = load_tensor(test_path)
        return train_emb, test_emb

    from sklearn.model_selection import train_test_split as _sklearn_split

    embeddings = load_tensor(source_path)
    if embeddings.dim() != 2:
        raise ValueError(
            f"Expected 2D embeddings tensor, got shape {embeddings.shape}"
        )
    indices = np.arange(len(embeddings))
    train_idx, test_idx = _sklearn_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
    )

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_emb, train_path)
    torch.save(test_emb, test_path)

    return train_emb, test_emb
