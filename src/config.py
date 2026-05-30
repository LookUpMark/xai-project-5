# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VLMConfig:
    """Vision-Language Model configuration."""
    # Model parameters
    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"

    # Data parameters
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