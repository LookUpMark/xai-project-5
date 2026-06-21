"""
config.py - Central configuration for all pipeline scripts.

Uses dataclasses to group related settings. Each dataclass represents
a logical component of the pipeline. Frozen dataclasses provide
immutability guarantees; validation happens in __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    visual_image_ids_path: Path = field(init=False)
    train_image_ids_path: Path = field(init=False)
    test_image_ids_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        subfolder = "augmented" if augmentation.enabled else "standard"
        self.embeddings_dir = self.project_root / "embeddings" / subfolder
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.results_dir / "figures"
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.train_embeddings_path = self.embeddings_dir / "train_embeddings.pt"
        self.test_embeddings_path = self.embeddings_dir / "test_embeddings.pt"
        # Sidecar image-id lists (basename per row) kept in lockstep with the
        # embedding tensors so downstream stages can recover the image identity
        # (the tensors are bare (N, 512) with no row metadata).
        self.visual_image_ids_path = self.embeddings_dir / "visual_image_ids.json"
        self.train_image_ids_path = self.embeddings_dir / "train_image_ids.json"
        self.test_image_ids_path = self.embeddings_dir / "test_image_ids.json"
        self.vocab_embeddings_path = self.embeddings_dir / "text_vocab_embeddings.pt"
        self.vocab_labels_path = self.data_dir / "vocabulary.json"


@dataclass(frozen=True)
class BackboneConfig:
    """BiomedCLIP backbone model settings."""

    model_id: str = "chuhac/BiomedCLIP-vit-bert-hf"
    embedding_dim: int = 512


@dataclass
class VLMConfig:
    """Vision-Language Model configuration (model identity and runtime)."""
    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


@dataclass
class EmbeddingConfig:
    """I/O configuration for the embedding extraction pipeline."""
    # Input directories
    image_dir: str = "data/iu_xray/images/images_normalized"
    reports_dir: str = "data/iu_xray/reports"

    # Output
    output_base: str = "embeddings"

    @property
    def output_dir(self) -> str:
        subfolder = "augmented" if augmentation.enabled else "standard"
        return f"{self.output_base}/{subfolder}"
    visual_output_filename: str = "visual_embeddings.pt"
    text_output_filename: str = "text_embeddings.pt"

    @property
    def visual_output_path(self) -> Path:
        return Path(self.output_dir) / self.visual_output_filename


    @property
    def text_output_path(self) -> Path:
        return Path(self.output_dir) / self.text_output_filename


@dataclass
class VocabularyConfig:
    """Configuration for the vocabulary building pipeline."""
    # I/O paths
    input_csv_path: str = "data/radlex.csv"
    # Aligned to PathsConfig.vocab_labels_path / vocab_embeddings_path (the
    # canonical consumer names) so the builder writes where concept_naming reads.
    output_path: str = "data/vocabulary.json"
    @property
    def embeddings_output_path(self) -> str:
        subfolder = "augmented" if augmentation.enabled else "standard"
        return f"embeddings/{subfolder}/text_vocab_embeddings.pt"

    # Filtering parameters
    top_k: int = 500

    # NIH ChestX-ray14 seed terms (always included in the final vocabulary)
    nih_seed_terms: List[str] = field(default_factory=lambda: [
        "atelectasis",
        "cardiomegaly",
        "effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "edema",
        "emphysema",
        "fibrosis",
        "pleural thickening",
        "hernia",
    ])

    # Hard-clustered anchor groups for multi-centroid vocabulary filtering.
    # Each key is a clinical sub-domain; its values are the anchor queries
    # whose mean embedding will form that domain's centroid.
    # 39 anchors across 13 clinically distinct groups.
    anchor_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "pulmonary_parenchymal": [
            "pulmonary opacity",
            "airspace disease",
            "interstitial lung disease",
            "lung mass",
            "pulmonary edema",
            "atelectasis",
            "emphysema",
            "pulmonary fibrosis",
        ],
        "pulmonary_infection": [
            "pneumonia",
            "lung consolidation",
            "calcified granuloma",
        ],
        "cardiac": [
            "cardiac abnormality",
            "cardiomegaly",
            "pericardial effusion",
        ],
        "pleural": [
            "pleural abnormality",
            "pleural effusion",
            "pneumothorax",
        ],
        "mediastinal": [
            "mediastinal abnormality",
            "hilar abnormality",
            "mediastinal lymphadenopathy",
        ],
        "vascular": [
            "pulmonary vascular congestion",
            "aortic abnormality",
            "vascular calcification",
        ],
        "airway": [
            "tracheal deviation",
            "bronchial abnormality",
        ],
        "skeletal": [
            "rib fracture",
            "spinal degenerative change",
            "scoliosis",
        ],
        "diaphragm": [
            "diaphragmatic abnormality",
            "elevated hemidiaphragm",
        ],
        "soft_tissue": [
            "chest wall abnormality",
            "soft tissue abnormality",
        ],
        "medical_devices": [
            "support device",
            "endotracheal tube",
            "central venous catheter",
        ],
        "post_surgical": [
            "post-surgical change",
            "median sternotomy",
        ],
        "normal_findings": [
            "normal chest radiograph",
            "no acute cardiopulmonary abnormality",
        ],
    })

    @property
    def anchor_queries(self) -> List[str]:
        """Flat list of all anchor queries (for backward compatibility)."""
        return [q for group in self.anchor_groups.values() for q in group]

    @property
    def input_csv(self) -> Path:
        return Path(self.input_csv_path)

    @property
    def output_file(self) -> Path:
        return Path(self.output_path)

    @property
    def embeddings_file(self) -> Path:
        return Path(self.embeddings_output_path)


@dataclass(frozen=True)
class AugmentationConfig:
    """Settings for data augmentation on Chest X-Rays."""
    enabled: bool = True
    num_augmentations: int = 2
    rotation_degrees: int = 5
    crop_scale: tuple[float, float] = (0.95, 1.0)


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
    dict_size: int = 2048
    k: int = 32
    lr: Optional[float] = None  # None = auto-scale from library
    steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    log_steps: int = 1_000
    decay_start_frac: float = 0.8  # fraction of steps to start LR decay
    lr_base: float = 2e-4  # base LR for auto-scaling formula
    lr_ref_dict_size: int = 16384  # reference dict_size for auto-scaling

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
        decay_start = int(self.steps * self.decay_start_frac)
        if decay_start <= self.warmup_steps:
            raise ValueError(
                f"decay_start ({decay_start} = steps*decay_start_frac) must exceed "
                f"warmup_steps ({self.warmup_steps})"
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

    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ── Instantiate configs ──────────────────────────────────────────────

augmentation = AugmentationConfig()
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
