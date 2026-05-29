"""
config.py — Central configuration for all pipeline scripts.

Edit variables here to control the entire pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

VISUAL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "visual_embeddings.pt"
VOCAB_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "text_vocab_embeddings.pt"
VOCAB_LABELS_PATH = DATA_DIR / "vocabulary.json"

# BiomedCLIP
BIOMEDCLIP_MODEL_ID = "chuhac/BiomedCLIP-vit-bert-hf"
EMBEDDING_DIM = 512

# SAE hyperparameters
SAE_ACTIVATION_DIM = 512
SAE_DICT_SIZE = 4096
SAE_K = 32
SAE_LR = 5e-5
SAE_STEPS = 50_000
SAE_WARMUP_STEPS = 1000
SAE_BATCH_SIZE = 256

# Training seeds for stability analysis
SEEDS = [0, 42, 123, 456, 789]

# Concept naming
CONCEPT_TOP_N = 3

# Explanation generation
EXPLANATION_TOP_N = 5
EXPLANATION_MAX_SAMPLES = None  # None = all samples

# Stability analysis
STABILITY_MAX_SAMPLES = None  # None = all samples

# Hardware
DEVICE = "cuda"
