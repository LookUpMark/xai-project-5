"""
02d_stability_analysis.py — Multi-seed stability analysis and clustering

Evaluate robustness of SAE concepts by comparing activations across 5 SAEs
trained with different seeds. Computes Jaccard similarity and clustering metrics.

Prerequisites:
    - models/sae_seed{0,42,123,456,789}/ae.pt (all 5 seeds)
    - embeddings/visual_embeddings.pt

Usage:
    python src/02d_stability_analysis.py
    python src/02d_stability_analysis.py --max-samples 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sae_module import SAEManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "visual_embeddings.pt"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SEEDS = [0, 42, 123, 456, 789]


def compute_feature_frequency(mgr: SAEManager, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute activation frequency of each feature across the dataset."""
    with torch.no_grad():
        sparse = mgr.encode(embeddings)
    return (sparse != 0).float().mean(dim=0)  # (4096,)


def compute_concept_clustering(model_dirs: list[Path], embeddings: torch.Tensor) -> dict:
    """
    Compute correlation between concept activation patterns.
    Useful for identifying redundant or correlated concepts.
    """
    mgr = SAEManager()
    mgr.load(model_dirs[0])  # Use primary seed

    with torch.no_grad():
        sparse = mgr.encode(embeddings)  # (B, 4096)

    # Active features (not all 4096 will be active)
    active_mask = (sparse != 0).float().sum(dim=0) > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    n_active = active_indices.shape[0]

    logger.info(f"  Active features: {n_active}/{sparse.shape[1]}")

    # Correlation between active features (co-occurrence)
    sparse_active = sparse[:, active_indices]  # (B, n_active)
    binary = (sparse_active != 0).float()

    # Cosine similarity between co-activation patterns
    norms = binary.norm(dim=0, keepdim=True) + 1e-8
    binary_norm = binary / norms
    co_occurrence = (binary_norm.T @ binary_norm).cpu()  # (n_active, n_active)

    # Identify clusters (features that always activate together)
    threshold = 0.7
    high_corr_pairs = (co_occurrence > threshold).sum().item() - n_active  # exclude diagonal
    high_corr_pairs //= 2  # symmetric

    return {
        "n_active_features": n_active,
        "n_dead_features": sparse.shape[1] - n_active,
        "high_correlation_pairs": high_corr_pairs,
        "correlation_threshold": threshold,
        "mean_co_occurrence": co_occurrence.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="SAE stability analysis (multi-seed)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for speed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else RESULTS_DIR / "stability_analysis.json"

    # Verify all models exist
    model_dirs = [MODELS_DIR / f"sae_seed{s}" for s in SEEDS]
    missing = [d for d in model_dirs if not (d / "ae.pt").exists()]
    if missing:
        logger.error(f"Missing models: {[str(m) for m in missing]}")
        logger.error("Run first: python src/02a_train_sae.py (all seeds)")
        sys.exit(1)

    if not EMBEDDINGS_PATH.exists():
        logger.error(f"Embeddings not found: {EMBEDDINGS_PATH}")
        sys.exit(1)

    # Load embeddings
    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    if args.max_samples:
        embeddings = embeddings[: args.max_samples]
    logger.info(f"Embeddings: {embeddings.shape}")

    # 1. Jaccard stability (main metric)
    logger.info("Computing Jaccard stability across seeds...")
    stability = SAEManager.compute_stability(model_dirs, embeddings)

    logger.info(f"  Mean Jaccard: {stability['mean_jaccard']:.4f}")
    logger.info(f"  Std Jaccard:  {stability['std_jaccard']:.4f}")
    logger.info(f"  Matrix:\n{stability['jaccard_matrix']}")

    # 2. Per-seed metrics
    logger.info("\nPer-seed metrics:")
    per_seed_metrics = {}
    for seed, model_dir in zip(SEEDS, model_dirs):
        mgr = SAEManager()
        mgr.load(model_dir)

        mse = mgr.compute_reconstruction_mse(embeddings)
        sparsity = mgr.compute_sparsity_metrics(embeddings)
        freq = compute_feature_frequency(mgr, embeddings)

        per_seed_metrics[seed] = {
            "mse": mse,
            **sparsity,
            "feature_frequency_mean": freq.mean().item(),
            "feature_frequency_std": freq.std().item(),
        }
        logger.info(
            f"  Seed {seed:3d}: MSE={mse:.6f}, L0={sparsity['l0_mean']:.1f}, "
            f"Dead={sparsity['dead_features_pct']:.1f}%"
        )

    # 3. Concept clustering analysis
    logger.info("\nConcept clustering analysis...")
    clustering = compute_concept_clustering(model_dirs, embeddings)
    logger.info(f"  Active features: {clustering['n_active_features']}")
    logger.info(f"  Dead features: {clustering['n_dead_features']}")
    logger.info(f"  High-correlation pairs (>{clustering['correlation_threshold']}): "
                f"{clustering['high_correlation_pairs']}")

    # 4. Save results
    results = {
        "stability": {
            "mean_jaccard": stability["mean_jaccard"],
            "std_jaccard": stability["std_jaccard"],
            "jaccard_matrix": stability["jaccard_matrix"].tolist(),
        },
        "per_seed_metrics": per_seed_metrics,
        "clustering": clustering,
        "config": {
            "seeds": SEEDS,
            "n_samples": embeddings.shape[0],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
