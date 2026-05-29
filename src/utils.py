"""
utils.py — Shared configuration and paths for the pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SEEDS = [0, 42, 123, 456, 789]

BIOMEDCLIP_MODEL_ID = "chuhac/BiomedCLIP-vit-bert-hf"
EMBEDDING_DIM = 512

SAE_CONFIG = {
    "activation_dim": EMBEDDING_DIM,
    "dict_size": 4096,
    "k": 32,
    "lr": 5e-5,
    "steps": 50_000,
    "warmup_steps": 1000,
    "batch_size": 256,
}


def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
