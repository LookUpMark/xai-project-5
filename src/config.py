# config.py
from dataclasses import dataclass

@dataclass
class VLMConfig:
    """Vision-Language Model configuration."""
    # Model parameters
    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    batch_size: int = 64

    # Data parameters
    image_dir: str = "data/iu_xray/images/images_normalized"
    image_ext: str = "*.png"
    output_dir: str = "embeddings"
    visual_output_filename: str = "visual_embeddings.pt"
    textual_output_filename: str = "textual_embeddings.pt"