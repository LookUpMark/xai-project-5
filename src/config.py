# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class VLMConfig:
    """Vision-Language Model configuration (model identity and runtime)."""
    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class EmbeddingConfig:
    """I/O configuration for the embedding extraction pipeline."""
    # Input directories
    image_dir: str = "data/iu_xray/images/images_normalized"
    reports_dir: str = "data/iu_xray/reports"

    # Output
    output_dir: str = "embeddings"
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
    radlex_csv_path: str = "data/radlex.csv"
    output_path: str = "data/medical_vocabulary.json"
    embeddings_output_path: str = "embeddings/vocab_embeddings.pt"

    # Filtering parameters
    top_k: int = 300

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

    # CXR-specific anchor queries used to compute the relevance centroid
    anchor_queries: List[str] = field(default_factory=lambda: [
        "chest radiograph finding",
        "lung pathology",
        "cardiac abnormality",
        "pleural abnormality",
        "pulmonary opacity",
        "mediastinal finding",
        "thoracic abnormality",
        "airspace disease",
        "interstitial lung disease",
        "chest x-ray diagnosis",
        "cardiopulmonary finding",
        "vascular abnormality of the chest",
        "bone abnormality on chest radiograph",
        "lung mass or nodule",
        "pleural effusion finding",
        "pneumothorax finding",
        "pulmonary edema",
        "chest wall abnormality",
        "diaphragmatic abnormality",
        "hilar abnormality",
    ])

    @property
    def radlex_csv(self) -> Path:
        return Path(self.radlex_csv_path)

    @property
    def output_file(self) -> Path:
        return Path(self.output_path)

    @property
    def embeddings_file(self) -> Path:
        return Path(self.embeddings_output_path)