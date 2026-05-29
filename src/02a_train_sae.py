"""
02a_train_sae.py — Train Sparse Autoencoders (Top-K)

Train SAEs on BiomedCLIP embeddings with multiple seeds for stability analysis.

Prerequisites:
    - embeddings/visual_embeddings.pt (output of 01_extract_embeddings.py)

Usage:
    python src/02a_train_sae.py                          # all 5 seeds
    python src/02a_train_sae.py --seed 42                # single seed
    python src/02a_train_sae.py --steps 10000 --seed 0   # quick test
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sae_module import SAEManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEEDS = [0, 42, 123, 456, 789]
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "visual_embeddings.pt"
MODELS_DIR = PROJECT_ROOT / "models"


def train_single(seed: int, steps: int | None = None, batch_size: int | None = None):
    """Train a single SAE with the given seed."""
    logger.info(f"{'='*60}")
    logger.info(f"Training SAE with seed={seed}")
    logger.info(f"{'='*60}")

    mgr = SAEManager()

    kwargs = {"embeddings_path": EMBEDDINGS_PATH, "seed": seed, "save_dir": MODELS_DIR}
    if steps is not None:
        kwargs["steps"] = steps
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    model_dir = mgr.train(**kwargs)

    # Post-training sanity check
    import torch

    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    sample = embeddings[:256]

    mse = mgr.compute_reconstruction_mse(sample)
    sparsity = mgr.compute_sparsity_metrics(sample)

    logger.info(f"Post-training check (seed={seed}):")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  L0 mean: {sparsity['l0_mean']:.1f} (expected ~{mgr.config['k']})")
    logger.info(f"  Dead features: {sparsity['dead_features_pct']:.1f}%")
    logger.info(f"  Saved to: {model_dir}")

    return model_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAE on BiomedCLIP embeddings")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (default: all 5)")
    parser.add_argument("--steps", type=int, default=None, help="Override steps (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: 256)")
    args = parser.parse_args()

    if not EMBEDDINGS_PATH.exists():
        logger.error(f"Embeddings not found: {EMBEDDINGS_PATH}")
        logger.error("Run first: python src/01_extract_embeddings.py")
        sys.exit(1)

    seeds = [args.seed] if args.seed is not None else SEEDS

    logger.info(f"Training {len(seeds)} SAE(s): seeds={seeds}")
    logger.info(f"Embeddings: {EMBEDDINGS_PATH}")
    logger.info(f"Output: {MODELS_DIR}/")

    model_dirs = []
    for seed in seeds:
        model_dir = train_single(seed, steps=args.steps, batch_size=args.batch_size)
        model_dirs.append(model_dir)

    if len(model_dirs) == 5:
        logger.info(f"\n{'='*60}")
        logger.info("All 5 SAEs trained. Models saved:")
        for d in model_dirs:
            logger.info(f"  {d}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
