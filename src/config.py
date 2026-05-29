# config.py
from dataclasses import dataclass

@dataclass
class VLMConfig:
    """Vision-Language Model configuration."""
    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"