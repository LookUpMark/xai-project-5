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
    baseline_results_dir: Path = field(init=False)
    visual_embeddings_path: Path = field(init=False)
    train_embeddings_path: Path = field(init=False)
    test_embeddings_path: Path = field(init=False)
    visual_image_ids_path: Path = field(init=False)
    train_image_ids_path: Path = field(init=False)
    test_image_ids_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    # ── Path A: SAE on 768-d pre-projection hidden state (additive, baseline-agnostic) ──
    # Lives under embeddings/standard_hidden/ + models/sae_hidden/ + results/sae_hidden/
    # so the 512-d baseline dirs are never touched. train() appends sae_seed{N}, so
    # hidden_models_dir is the *base* dir (one model per seed sits under it).
    hidden_embeddings_dir: Path = field(init=False)
    hidden_visual_embeddings_path: Path = field(init=False)
    hidden_train_embeddings_path: Path = field(init=False)
    hidden_test_embeddings_path: Path = field(init=False)
    hidden_visual_image_ids_path: Path = field(init=False)
    hidden_train_image_ids_path: Path = field(init=False)
    hidden_test_image_ids_path: Path = field(init=False)
    hidden_models_dir: Path = field(init=False)
    hidden_results_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        # All artifact paths (incl. vocabulary.json) are dataset-routed (re-derived
        # by _set_dataset_paths so config.select_dataset can re-route in place).
        self._set_dataset_paths()

    def _set_dataset_paths(self) -> None:
        """(Re)derive the dataset- + augmentation-routed artifact paths.

        Reads the ``active_dataset`` and ``augmentation`` singletons live, so
        :func:`config.select_dataset` can re-route in place (PathsConfig is
        mutable; the variant swap also sets these fields directly). Each dataset's
        artifacts are isolated under their own segment so PadChest never overwrites
        IU X-Ray::

            embeddings/<dataset>/<standard|augmented>/  tensors + sidecars + vocab emb
            embeddings/<dataset>/standard_hidden/        Path A (768-d)
            models/<dataset>/                            SAE weights + modality_gap
            models/<dataset>/sae_hidden/                 Path A models
            results/<dataset>/{baseline,sae_hidden,figures}/
        """
        aug = "augmented" if augmentation.enabled else "standard"
        ds = active_dataset.name
        emb_root = self.project_root / "embeddings" / ds
        models_root = self.project_root / "models" / ds
        results_root = self.project_root / "results" / ds

        # Embeddings (tensors + sidecars + vocab embeddings).
        self.embeddings_dir = emb_root / aug
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.train_embeddings_path = self.embeddings_dir / "train_embeddings.pt"
        self.test_embeddings_path = self.embeddings_dir / "test_embeddings.pt"
        # Sidecar image-id lists (basename per row) kept in lockstep with the
        # embedding tensors so downstream stages recover image identity (the
        # tensors are bare (N, 512) with no row metadata).
        self.visual_image_ids_path = self.embeddings_dir / "visual_image_ids.json"
        self.train_image_ids_path = self.embeddings_dir / "train_image_ids.json"
        self.test_image_ids_path = self.embeddings_dir / "test_image_ids.json"
        self.vocab_embeddings_path = self.embeddings_dir / "text_vocab_embeddings.pt"
        # vocabulary.json is per-dataset (RadLex-chest for IU/PadChest, UMLS for
        # ROCOv2) so a UMLS build never overwrites a RadLex one.
        self.vocab_labels_path = emb_root / "vocabulary.json"

        # Path A (768-d pre-projection hidden state). Raw CLS tokens, no per-sample
        # L2 norm (SAE-on-residual-stream literature trains on raw activations).
        self.hidden_embeddings_dir = emb_root / "standard_hidden"
        self.hidden_visual_embeddings_path = (
            self.hidden_embeddings_dir / "visual_embeddings_768.pt"
        )
        self.hidden_train_embeddings_path = (
            self.hidden_embeddings_dir / "train_embeddings_768.pt"
        )
        self.hidden_test_embeddings_path = (
            self.hidden_embeddings_dir / "test_embeddings_768.pt"
        )
        self.hidden_visual_image_ids_path = (
            self.hidden_embeddings_dir / "visual_image_ids.json"
        )
        self.hidden_train_image_ids_path = (
            self.hidden_embeddings_dir / "train_image_ids.json"
        )
        self.hidden_test_image_ids_path = (
            self.hidden_embeddings_dir / "test_image_ids.json"
        )

        # Models + results (per-dataset isolation).
        self.models_dir = models_root
        self.hidden_models_dir = models_root / "sae_hidden"
        self.results_dir = results_root
        self.figures_dir = results_root / "figures"
        # Baseline (512-d) outputs live in their own subdir so results/<dataset>/
        # root stays clean.
        self.baseline_results_dir = results_root / "baseline"
        self.hidden_results_dir = results_root / "sae_hidden"


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
    batch_size: int = 64
    num_workers: int = 4
    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    # AMP fp16 autocast on CUDA (~5-8x forward speedup on T4; fp32 weights are
    # kept, autocast casts compute on the fly). Outputs are cast back to fp32
    # before L2-norm + save, so downstream stages (SAE/SPLiCE) see no dtype
    # change. Ignored on MPS/CPU (autocast enabled on cuda only).
    use_half: bool = True


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
        return f"{self.output_base}/{active_dataset.name}/{subfolder}"
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
    # None => per-dataset vocabulary.json (paths.vocab_labels_path); set this to
    # write to a custom location (e.g. tests redirect to a tmp path).
    output_path: str | None = None
    @property
    def embeddings_output_path(self) -> str:
        subfolder = "augmented" if augmentation.enabled else "standard"
        return f"embeddings/{active_dataset.name}/{subfolder}/text_vocab_embeddings.pt"

    # Filtering parameters
    top_k: int = 1024

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
        # Explicit override (e.g. tests) wins; otherwise the per-dataset
        # vocabulary.json (RadLex-chest vs UMLS-multimodal), so the builder writes
        # where concept_naming reads.
        return Path(self.output_path) if self.output_path is not None else paths.vocab_labels_path

    @property
    def embeddings_file(self) -> Path:
        return Path(self.embeddings_output_path)


@dataclass(frozen=True)
class AugmentationConfig:
    """Settings for data augmentation on Chest X-Rays.

    Default disabled: augmentation is an optional extension (member-1 ablation
    over embeddings/augmented/). The standard pipeline runs on
    embeddings/standard/ with augmentation off. Enable explicitly when
    generating/running the augmented ablation.
    """
    enabled: bool = False  # docstring: "Default disabled" — standard pipeline runs on embeddings/standard/
    num_augmentations: int = 2
    rotation_degrees: int = 5
    crop_scale: tuple[float, float] = (0.95, 1.0)


@dataclass(frozen=True)
class DatasetSelection:
    """Active dataset selector (Phase 0 of the multi-dataset refactor).

    ``name`` keys into ``xai_datasets.spec.DATASETS``. Stages resolve the active
    spec via ``get_dataset(active_dataset.name)``. The per-dataset output-path
    dimension (``embeddings/<dataset>/...``) lands in Phase 2 with PadChest;
    Phase 0 keeps the existing ``embeddings/<standard|augmented>/`` layout so IU
    X-Ray artifacts stay byte-identical.
    """

    name: str = "iu_xray"


@dataclass(frozen=True)
class SAEConfig:
    """Sparse Autoencoder (Top-K) hyperparameters.

    Ablation presets for small datasets (N ~ 7400):
        Conservative:  k=16, dict_size=2048, lr=None, steps=30_000
        Default:       k=32, dict_size=2048, lr=5e-5, steps=8_000  # matched to Path A (F-003)
        Aggressive:    k=64, dict_size=4096, lr=None, steps=80_000

    lr=None triggers the library's auto-scaling: 2e-4 / sqrt(dict_size / 16384).
    For dict_size=2048 this gives ~5.7e-4. For small datasets, consider overriding
    to a lower value (e.g. 5e-5) to avoid overfitting.
    """

    activation_dim: int = 512
    dict_size: int = 2048  # matches Path A default (config.sae_hidden) for baseline↔Path A comparison
    k: int = 32
    lr: Optional[float] = 5e-5  # None = library auto-scale (~4e-4)
    steps: int = 8_000  # F-003: matched to Path A (config.sae_hidden) for fair baseline↔Path A comparison
    warmup_steps: int = 1_000
    batch_size: int = 256
    log_steps: int = 1_000
    dead_threshold: float = 1e-8  # F-013: mirror SAEHiddenConfig (single source of truth)
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
class SAEHiddenConfig:
    """Sparse Autoencoder (Top-K) hyperparameters for Path A — SAE on the 768-d
    pre-projection CLS hidden state.

    Audit-corrected (ML-AUDIT-2026-06-25): M-006 drops steps 50k→8k and pins lr=5e-5
    (avoid ~2,140-epoch overfit + auto-scaled 4e-4); M-002 lowers dict_size from the
    baseline's 4096 (1.5 samples/feat) to 2048 (2.9 samples/feat, 2.7x overcomplete of 768).
    Input is RAW (no per-sample L2 norm) per SAE-on-residual-stream literature.
    """

    activation_dim: int = 768
    dict_size: int = 2048
    k: int = 32
    lr: Optional[float] = 5e-5
    steps: int = 8_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    log_steps: int = 1_000
    decay_start_frac: float = 0.8
    dead_threshold: float = 1e-8
    # Permutation-invariant stability (compute_stability_matched, ML-AUDIT-2026-06-26 F-001):
    n_perm: int = 200  # random-pairing null samples (Lan et al. 2024 use 1000)
    match_thresholds: tuple[float, ...] = (0.7, 0.9)  # cosine cutoffs for "fraction matched"

    def __post_init__(self):
        if self.activation_dim != 768:
            raise ValueError(
                f"Path A activation_dim must be 768, got {self.activation_dim}"
            )
        if self.dict_size <= self.activation_dim:
            raise ValueError(
                f"dict_size ({self.dict_size}) must exceed activation_dim (768)"
            )
        if self.k >= self.dict_size:
            raise ValueError(
                f"k ({self.k}) must be less than dict_size ({self.dict_size})"
            )
        if self.warmup_steps >= self.steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < steps ({self.steps})"
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
class JudgeConfig:
    """LLM Judge evaluation settings.

    Controls the MedGemma-based concept judge: model identity, generation
    parameters, retry budget, checkpoint frequency, and reproducibility seed.
    """

    model_name: str = "unsloth/medgemma-4b-it"
    max_new_tokens: int = 64
    max_retries: int = 2
    batch_save_every: int = 25
    seed: int = 42


@dataclass(frozen=True)
class HardwareConfig:
    """Device and compute settings."""

    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


@dataclass(frozen=True)
class SpliCEConfig:
    """Configuration for SPLiCE sparse decomposition (Path B).

    SPLiCE performs deterministic sparse coding directly on the RadLex
    vocabulary, avoiding the non-identifiability issues that plague
    autoencoder-based approaches.

    Args:
        k: Number of concepts per image (top-k active coefficients).
        use_gap_correction: Whether to subtract modality gap before decomposition.
        vocab_path: Path to vocabulary JSON file.
        vocab_emb_path: Path to vocabulary text embeddings (.pt file).
        gap_path: Path to modality gap vector (.pt file).
        output_dir: Directory for output files.
    """

    k: int = 32  # number of concepts per image (top-k active coefficients)
    use_gap_correction: bool = True
    # F-008: anchored to config.paths.project_root (the `paths` singleton is
    # instantiated just below, so the late-bound default_factory resolves it at
    # SpliCEConfig() construction time). Resolves correctly regardless of CWD.
    vocab_path: Path = field(default_factory=lambda: paths.vocab_labels_path)
    vocab_emb_path: Path = field(default_factory=lambda: paths.vocab_embeddings_path)
    gap_path: Path = field(default_factory=lambda: paths.models_dir / "modality_gap.pt")
    output_dir: Path = field(default_factory=lambda: paths.results_dir / "spliece")


@dataclass(frozen=True)
class OrganizeConfig:
    """Configuration for the concept-organization extension (brief §3).

    Clusters discovered concepts (SPLiCE or SAE) by cosine similarity of their
    RadLex text embeddings, annotates each cluster with a best-effort RadLex
    anatomical ancestor, and re-expresses per-image explanations as concept
    families with a redundancy metric. Method-agnostic and dataset-portable.

    Args:
        n_clusters: Target number of clusters. If None and distance_threshold is
            None, defaults to max(2, round(sqrt(M))) at runtime.
        distance_threshold: Agglomerative linkage-distance cut. Mutually exclusive
            with n_clusters (enforced in __post_init__).
        linkage: Agglomerative linkage criterion ('average' | 'complete' | 'single').
        metric: Distance metric; 'cosine' is the only supported value for this design.
        radlex_csv_path: Path to the RadLex ontology CSV (for ancestor annotation).
        vocab_path: Path to vocabulary.json (list of {"term": str, ...} dicts).
        vocab_emb_path: Path to text_vocab_embeddings.pt (V, 512).
        output_dir: Directory for output artifacts.
    """

    n_clusters: Optional[int] = None
    distance_threshold: Optional[float] = None
    linkage: str = "average"
    metric: str = "cosine"
    radlex_csv_path: Path = field(default_factory=lambda: paths.data_dir / "radlex.csv")
    vocab_path: Path = field(default_factory=lambda: paths.vocab_labels_path)
    vocab_emb_path: Path = field(default_factory=lambda: paths.vocab_embeddings_path)
    output_dir: Path = field(default_factory=lambda: paths.results_dir / "concept_organization")

    def __post_init__(self):
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise ValueError(
                "n_clusters and distance_threshold are mutually exclusive in OrganizeConfig"
            )
        if self.linkage not in {"average", "complete", "single"}:
            raise ValueError(
                f"OrganizeConfig.linkage must be average|complete|single, got {self.linkage}"
            )
        if self.metric != "cosine":
            raise ValueError(
                f"OrganizeConfig.metric must be 'cosine', got {self.metric}"
            )


# ── Instantiate configs ──────────────────────────────────────────────

augmentation = AugmentationConfig()
active_dataset = DatasetSelection()
paths = PathsConfig()
backbone = BackboneConfig()
vlm = VLMConfig()
sae = SAEConfig()
sae_hidden = SAEHiddenConfig()
training = TrainingConfig()
explanation = ExplanationConfig()
judge = JudgeConfig()
hardware = HardwareConfig()
spliece = SpliCEConfig()
organize = OrganizeConfig()

# Backward compatibility alias
DEVICE = hardware.device


def select_dataset(name: str) -> None:
    """Switch the active dataset and re-route the embedding paths to match.

    Reassigns the ``active_dataset`` singleton and re-derives ``paths``'
    embedding/hidden paths in place (PathsConfig is mutable). The lazy
    ``@property`` paths (``EmbeddingConfig.output_dir``,
    ``VocabularyConfig.embeddings_output_path``) pick up the new value on next
    access. Call this BEFORE reading any path. Unknown names surface as a
    ``KeyError`` when ``get_dataset`` is next called (e.g. by ``prepare_split``).

    Args:
        name: dataset key in ``xai_datasets.spec.DATASETS`` (e.g. ``"padchest"``).
    """
    global active_dataset
    active_dataset = DatasetSelection(name=name)
    paths._set_dataset_paths()
