"""
run_sae_training.py — Train Top-K Sparse Autoencoders on BiomedCLIP embeddings.

Pipeline:
    1. Create the group-aware train/test split from visual_embeddings.pt
       (leak-free by radiograph study; recomputed on every run).
    2. Compute and save the modality gap (visual centroid - text centroid).
    3. Train one Top-K SAE per seed on the train split.
    4. Sanity-check each model on the held-out test split.

The split / modality-gap / per-seed training logic lives in
``src/autoencoder/train_sae.py``; this script is only the CLI entry point and
lets you override SAE/training hyperparameters without editing ``config.py``.
Defaults are sourced from the config singletons 
(currently: dict_size = 2 * activation_dim = 1024, lr = 5e-5, k = 32).

Prerequisites:
    - embeddings/<standard|augmented>/visual_embeddings.pt + visual_image_ids.json
      (output of embedding_extraction/extract_embeddings.py)
    - embeddings/<...>/text_vocab_embeddings.pt
      (output of the vocabulary builder; needed for the modality gap)

Usage:
    python scripts/run_sae_training.py
    python scripts/run_sae_training.py --dict-size 2048 --lr 5e-5 --steps 50000
    python scripts/run_sae_training.py --seeds 0,42,123 --device cuda
    python scripts/run_sae_training.py --lr auto           # library auto-scale (~4e-4)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
import torch
import utils
from autoencoder.train_sae import (
    compute_and_save_modality_gap,
    prepare_split,
    train_single,
)
from autoencoder.tracking import finish_tracking, init_tracking


def _parse_seeds(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list, e.g. '0,42,123' -> (0, 42, 123)."""
    try:
        seeds = tuple(int(s.strip()) for s in raw.split(",") if s.strip() != "")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --seeds value '{raw}': {e}")
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _parse_lr(raw: str):
    """Parse lr: 'auto'/'none' -> None (library auto-scale), else float."""
    if raw.strip().lower() in ("auto", "none", ""):
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --lr value '{raw}': {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Top-K SAEs on BiomedCLIP embeddings (multi-seed)."
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=config.training.seeds,
        help=f"Comma-separated seeds (default: {','.join(map(str, config.training.seeds))}).",
    )
    parser.add_argument(
        "--dict-size",
        type=int,
        default=config.sae.dict_size,
        help=f"SAE dictionary size, must exceed activation_dim (default: {config.sae.dict_size}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.sae.k,
        help=f"Top-K active features per sample, must be < dict_size (default: {config.sae.k}).",
    )
    parser.add_argument(
        "--lr",
        type=_parse_lr,
        default=config.sae.lr,
        help="Learning rate, or 'auto' for the library's auto-scale "
        f"(default: {config.sae.lr}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=config.sae.steps,
        help=f"Training steps per seed (default: {config.sae.steps}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.sae.batch_size,
        help=f"Batch size (default: {config.sae.batch_size}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.hardware.device,
        help=f"Device (default: {config.hardware.device}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply overrides by rebuilding the frozen, __post_init__-validated
    # singletons (SAEConfig/TrainingConfig are frozen, so we reconstruct them
    # rather than mutating fields — this keeps the validation active, e.g.
    # dict_size > activation_dim and k < dict_size). train_sae.py reads these
    # singletons at call time, so the reassignment propagates.
    config.sae = config.SAEConfig(
        dict_size=args.dict_size,
        k=args.k,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    # primary_seed must be a member of seeds (TrainingConfig validates this).
    primary_seed = (
        config.training.primary_seed
        if config.training.primary_seed in args.seeds
        else args.seeds[0]
    )
    config.training = config.TrainingConfig(
        seeds=args.seeds,
        primary_seed=primary_seed,
    )
    object.__setattr__(config.hardware, "device", args.device)

    # Deterministic split seed
    utils.set_global_seed(config.training.split_seed)

    lr_label = "auto" if config.sae.lr is None else f"{config.sae.lr:.1e}"
    print("=" * 60)
    print("  SAE Training")
    print("=" * 60)
    print(f"  Seeds       : {','.join(map(str, config.training.seeds))}")
    print(f"  Device      : {config.hardware.device}")
    print(f"  dict_size   : {config.sae.dict_size}")
    print(f"  k (TopK)    : {config.sae.k}")
    print(f"  lr          : {lr_label}")
    print(f"  steps       : {config.sae.steps}")
    print(f"  batch_size  : {config.sae.batch_size}")
    print(f"  Embeddings  : {config.paths.embeddings_dir}")
    print("=" * 60)

    # Step 1: group-aware train/test split (always recomputed, leak-free)
    prepare_split()

    if not config.paths.train_embeddings_path.exists():
        print(
            f"ERROR: train embeddings not found: {config.paths.train_embeddings_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 2: modality gap (visual centroid - text centroid)
    compute_and_save_modality_gap()

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda or 'N/A'}")

    # Init tracking (no-op when WandbConfig.enabled is False)
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

    # Step 3: train one SAE per seed
    model_dirs = [train_single(seed) for seed in config.training.seeds]

    print(f"All {len(model_dirs)} SAEs trained successfully.")
    finish_tracking()


if __name__ == "__main__":
    main()
