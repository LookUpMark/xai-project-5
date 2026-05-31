"""
train_sae.py — Train Sparse Autoencoders (Top-K)

Train SAEs on BiomedCLIP embeddings with multiple seeds for stability analysis.
Creates train/test split if not already on disk, trains on train split only,
evaluates sanity checks on held-out test set.

Prerequisites:
    - embeddings/visual_embeddings.pt (output of 01_extract_embeddings.py)

Run:
    python src/autoencoder/train_sae.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
from autoencoder.tracking import (
    init_tracking,
    log_metrics,
    log_artifact,
    finish_tracking,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_split() -> None:
    """Create train/test split from visual_embeddings.pt if not on disk."""
    train_path = config.paths.train_embeddings_path
    test_path = config.paths.test_embeddings_path

    if train_path.exists() and test_path.exists():
        logger.info("Train/test splits already exist — skipping.")
        return

    from sklearn.model_selection import train_test_split

    source = config.paths.visual_embeddings_path
    if not source.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {source}. "
            f"Run first: python src/01_extract_embeddings.py"
        )

    embeddings = torch.load(source, map_location="cpu", weights_only=True)
    logger.info(
        f"Creating {config.training.train_split_ratio:.0%} / "
        f"{1 - config.training.train_split_ratio:.0%} split "
        f"from {embeddings.shape[0]} embeddings"
    )

    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=config.training.train_split_ratio,
        random_state=config.training.split_seed,
    )

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_emb, train_path)
    torch.save(test_emb, test_path)

    logger.info(f"Train: {train_emb.shape[0]} samples → {train_path}")
    logger.info(f"Test:  {test_emb.shape[0]} samples → {test_path}")


def train_single(seed: int) -> Path:
    """Train a single SAE with the given seed."""
    logger.info(f"Training SAE with seed={seed}")

    mgr = SAEManager(
        {
            "device": config.hardware.device,
            "activation_dim": config.sae.activation_dim,
            "dict_size": config.sae.dict_size,
            "k": config.sae.k,
            "lr": config.sae.lr,
            "warmup_steps": config.sae.warmup_steps,
            "log_steps": config.sae.log_steps,
            "decay_start_frac": config.sae.decay_start_frac,
        }
    )

    model_dir = mgr.train(
        embeddings_path=config.paths.train_embeddings_path,
        seed=seed,
        save_dir=config.paths.models_dir,
        steps=config.sae.steps,
        batch_size=config.sae.batch_size,
    )

    # Sanity check on HELD-OUT test set (not training data)
    test_emb = torch.load(
        config.paths.test_embeddings_path, map_location="cpu", weights_only=True
    )
    n_check = min(config.training.sanity_check_samples, len(test_emb))
    # Random subset, not positional slice
    rng = np.random.default_rng(seed)
    check_idx = rng.choice(len(test_emb), size=n_check, replace=False)
    sample = test_emb[check_idx]

    mse = mgr.compute_reconstruction_mse(sample)
    cosine = mgr.compute_cosine_reconstruction(sample)
    sparsity = mgr.compute_sparsity_metrics(sample)

    logger.info(f"  Test MSE: {mse:.6f}")
    logger.info(f"  Test Cosine: {cosine:.4f}")
    logger.info(f"  L0 mean: {sparsity['l0_mean']:.1f} (expected ~{config.sae.k})")
    logger.info(f"  Dead features: {sparsity['dead_features_pct']:.1f}%")
    logger.info(f"  Dict utilization: {sparsity['dict_utilization_pct']:.1f}%")
    logger.info(f"  Saved to: {model_dir}")

    # Log to wandb if enabled
    if config.wandb_cfg.enabled:
        log_metrics(
            {
                f"train/seed{seed}/test_mse": mse,
                f"train/seed{seed}/test_cosine": cosine,
                f"train/seed{seed}/dead_pct": sparsity["dead_features_pct"],
                f"train/seed{seed}/dict_util": sparsity["dict_utilization_pct"],
            }
        )
        log_artifact(
            model_dir / "training_manifest.json", f"sae_seed{seed}_manifest", "manifest"
        )

    return model_dir


def main():
    # Step 1: Create train/test split
    prepare_split()

    if not config.paths.train_embeddings_path.exists():
        logger.error(
            f"Train embeddings not found: {config.paths.train_embeddings_path}"
        )
        sys.exit(1)

    # Log environment info
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda or 'N/A'}")
    logger.info(f"Device: {config.hardware.device}")
    logger.info(
        f"Training {len(config.training.seeds)} SAEs: seeds={config.training.seeds}"
    )
    logger.info(
        f"SAE config: k={config.sae.k}, dict_size={config.sae.dict_size}, "
        f"lr={'auto' if config.sae.lr is None else config.sae.lr}, "
        f"steps={config.sae.steps}"
    )

    # Init tracking
    if config.wandb_cfg.enabled:
        init_tracking(
            "train_sae",
            {
                "project": config.wandb_cfg.project,
                "entity": config.wandb_cfg.entity,
                "seeds": list(config.training.seeds),
                "k": config.sae.k,
                "dict_size": config.sae.dict_size,
                "steps": config.sae.steps,
                "lr": config.sae.lr,
            },
        )

    # Step 2: Train all seeds
    model_dirs = []
    for seed in config.training.seeds:
        model_dir = train_single(seed)
        model_dirs.append(model_dir)

    logger.info(f"All {len(model_dirs)} SAEs trained successfully.")
    finish_tracking()


if __name__ == "__main__":
    main()
