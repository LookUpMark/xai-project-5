"""
train_sae.py - Train Sparse Autoencoders (Top-K)

Train SAEs on BiomedCLIP embeddings with multiple seeds for stability analysis.

Prerequisites:
    - embeddings/visual_embeddings.pt (output of 01_extract_embeddings.py)

Run:
    python src/autoencoder/train_sae.py
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_single(seed: int):
    """Train a single SAE with the given seed."""
    logger.info(f"Training SAE with seed={seed}")

    mgr = SAEManager({"device": config.hardware.device})
    model_dir = mgr.train(
        embeddings_path=config.paths.visual_embeddings_path,
        seed=seed,
        save_dir=config.paths.models_dir,
        steps=config.sae.steps,
        batch_size=config.sae.batch_size,
    )

    # Sanity check on a small subset
    embeddings = torch.load(config.paths.visual_embeddings_path, map_location="cpu", weights_only=True)
    sample = embeddings[:256]

    mse = mgr.compute_reconstruction_mse(sample)
    sparsity = mgr.compute_sparsity_metrics(sample)

    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  L0 mean: {sparsity['l0_mean']:.1f} (expected ~{config.sae.k})")
    logger.info(f"  Dead features: {sparsity['dead_features_pct']:.1f}%")
    logger.info(f"  Saved to: {model_dir}")

    return model_dir


def main():
    if not config.paths.visual_embeddings_path.exists():
        logger.error(f"Embeddings not found: {config.paths.visual_embeddings_path}")
        logger.error("Run first: python src/01_extract_embeddings.py")
        sys.exit(1)

    logger.info(f"Training {len(config.training.seeds)} SAEs: seeds={config.training.seeds}")

    model_dirs = []
    for seed in config.training.seeds:
        model_dir = train_single(seed)
        model_dirs.append(model_dir)

    logger.info(f"All {len(model_dirs)} SAEs trained successfully.")


if __name__ == "__main__":
    main()
