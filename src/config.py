"""
config.py - Central configuration for all pipeline scripts.

Uses dataclasses to group related settings. Each dataclass represents
a logical component of the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathsConfig:
    """Project directory layout and derived file paths."""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    visual_embeddings_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.embeddings_dir = self.project_root / "embeddings"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.vocab_embeddings_path = self.embeddings_dir / "text_vocab_embeddings.pt"
        self.vocab_labels_path = self.data_dir / "vocabulary.json"


@dataclass(frozen=True)
class BackboneConfig:
    """BiomedCLIP backbone model settings."""
    model_id: str = "chuhac/BiomedCLIP-vit-bert-hf"
    embedding_dim: int = 512


@dataclass(frozen=True)
class SAEConfig:
    """Sparse Autoencoder (Top-K) hyperparameters."""
    activation_dim: int = 512
    dict_size: int = 4096
    k: int = 32
    lr: float = 5e-5
    steps: int = 50_000
    warmup_steps: int = 1000
    batch_size: int = 256


@dataclass(frozen=True)
class TrainingConfig:
    """Multi-seed training and stability analysis settings."""
    seeds: tuple[int, ...] = (0, 42, 123, 456, 789)
    stability_max_samples: Optional[int] = None


@dataclass(frozen=True)
class ExplanationConfig:
    """Concept naming and explanation generation settings."""
    concept_top_n: int = 3
    explanation_top_n: int = 5
    explanation_max_samples: Optional[int] = None


@dataclass(frozen=True)
class HardwareConfig:
    """Device and compute settings."""
    device: str = "cuda"


# Instantiate configs
paths = PathsConfig()
backbone = BackboneConfig()
sae = SAEConfig()
training = TrainingConfig()
explanation = ExplanationConfig()
hardware = HardwareConfig()

# Hardware
DEVICE = hardware.device
