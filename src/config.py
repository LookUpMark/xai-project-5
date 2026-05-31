"""
config.py - Central configuration for all pipeline scripts.

Uses dataclasses to group related settings. Each dataclass represents
a logical component of the pipeline. Frozen dataclasses provide
immutability guarantees; validation happens in __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class PathsConfig:
    """Project directory layout and derived file paths."""

    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    visual_embeddings_path: Path = field(init=False)
    train_embeddings_path: Path = field(init=False)
    test_embeddings_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.embeddings_dir = self.project_root / "embeddings"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.results_dir / "figures"
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.train_embeddings_path = self.embeddings_dir / "train_embeddings.pt"
        self.test_embeddings_path = self.embeddings_dir / "test_embeddings.pt"
        self.vocab_embeddings_path = self.embeddings_dir / "text_vocab_embeddings.pt"
        self.vocab_labels_path = self.data_dir / "vocabulary.json"


@dataclass(frozen=True)
class BackboneConfig:
    """BiomedCLIP backbone model settings."""

    model_id: str = "chuhac/BiomedCLIP-vit-bert-hf"
    embedding_dim: int = 512


@dataclass
class VLMConfig:
    """VLM embedding extraction settings (Member 1 pipeline).

    Used by utils.load_vlm() and 01_extract_embeddings.py.
    """

    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir: str = "data/iu_xray/images/images_normalized"
    reports_dir: str = "data/iu_xray/reports"
    output_dir: str = "embeddings"
    visual_output_filename: str = "visual_embeddings.pt"
    text_output_filename: str = "text_embeddings.pt"

    @property
    def visual_output_path(self) -> Path:
        return Path(self.output_dir) / self.visual_output_filename

    @property
    def text_output_path(self) -> Path:
        return Path(self.output_dir) / self.text_output_filename


@dataclass(frozen=True)
class SAEConfig:
    """Sparse Autoencoder (Top-K) hyperparameters.

    Ablation presets for small datasets (N ~ 7400):
        Conservative:  k=16, dict_size=2048, lr=None, steps=30_000
        Default:       k=32, dict_size=4096, lr=None, steps=50_000
        Aggressive:    k=64, dict_size=4096, lr=None, steps=80_000

    lr=None triggers the library's auto-scaling: 2e-4 / sqrt(dict_size / 16384).
    For dict_size=4096 this gives ~4e-4. For small datasets, consider overriding
    to a lower value (e.g. 5e-5) to avoid overfitting.
    """

    activation_dim: int = 512
    dict_size: int = 4096
    k: int = 32
    lr: Optional[float] = None  # None = auto-scale from library
    steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    log_steps: int = 1_000
    decay_start_frac: float = 0.8  # fraction of steps to start LR decay

    def __post_init__(self):
        if self.dict_size <= self.activation_dim:
            raise ValueError(
                f"dict_size ({self.dict_size}) must exceed "
                f"activation_dim ({self.activation_dim})"
            )
        if self.k >= self.dict_size:
            raise ValueError(
                f"k ({self.k}) must be less than dict_size ({self.dict_size})"
            )
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.warmup_steps >= self.steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < steps ({self.steps})"
            )
        if not (0.0 < self.decay_start_frac <= 1.0):
            raise ValueError(
                f"decay_start_frac must be in (0, 1], got {self.decay_start_frac}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Multi-seed training and stability analysis settings."""

    seeds: tuple[int, ...] = (0, 42, 123, 456, 789)
    primary_seed: int = 42  # reference model for naming/explanations
    sanity_check_samples: int = 256
    train_split_ratio: float = 0.8  # 80/20 train/test split
    split_seed: int = 42  # deterministic split
    stability_max_samples: Optional[int] = None
    correlation_threshold: float = 0.7

    def __post_init__(self):
        if self.primary_seed not in self.seeds:
            raise ValueError(
                f"primary_seed ({self.primary_seed}) must be in seeds {self.seeds}"
            )
        if not (0.0 < self.train_split_ratio < 1.0):
            raise ValueError(
                f"train_split_ratio must be in (0, 1), got {self.train_split_ratio}"
            )


@dataclass(frozen=True)
class ExplanationConfig:
    """Concept naming and explanation generation settings."""

    concept_top_n: int = 3
    explanation_top_n: int = 5
    explanation_max_samples: Optional[int] = None


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases experiment tracking."""

    enabled: bool = False
    project: str = "sae-concept-discovery"
    entity: Optional[str] = None


@dataclass(frozen=True)
class HardwareConfig:
    """Device and compute settings."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Instantiate configs ──────────────────────────────────────────────

paths = PathsConfig()
backbone = BackboneConfig()
vlm = VLMConfig()
sae = SAEConfig()
training = TrainingConfig()
explanation = ExplanationConfig()
wandb_cfg = WandbConfig()
hardware = HardwareConfig()

# Backward compatibility alias
DEVICE = hardware.device
