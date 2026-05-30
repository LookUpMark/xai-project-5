"""
stability_analysis.py - Multi-seed stability analysis and clustering

Evaluate robustness of SAE concepts by comparing activations across 5 SAEs
trained with different seeds. Computes Jaccard similarity and clustering metrics.

Prerequisites:
    - models/sae_seed{0,42,123,456,789}/ae.pt (all 5 seeds)
    - embeddings/visual_embeddings.pt

Run:
    python src/autoencoder/stability_analysis.py
"""

import json
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

OUTPUT_PATH = config.paths.results_dir / "stability_analysis.json"


def compute_feature_frequency(mgr: SAEManager, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute activation frequency of each feature across the dataset."""
    with torch.no_grad():
        sparse = mgr.encode(embeddings)
    return (sparse != 0).float().mean(dim=0)


def compute_concept_clustering(model_dirs: list[Path], embeddings: torch.Tensor) -> dict:
    """Compute correlation between concept activation patterns."""
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dirs[0])

    with torch.no_grad():
        sparse = mgr.encode(embeddings)

    # Filter out dead features
    active_mask = (sparse != 0).float().sum(dim=0) > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    n_active = active_indices.shape[0]

    logger.info(f"  Active features: {n_active}/{sparse.shape[1]}")

    # Normalized co-occurrence matrix
    sparse_active = sparse[:, active_indices]
    binary = (sparse_active != 0).float()

    norms = binary.norm(dim=0, keepdim=True) + 1e-8
    binary_norm = binary / norms
    co_occurrence = (binary_norm.T @ binary_norm).cpu()

    # Count highly correlated pairs (potential redundancy)
    threshold = 0.7
    high_corr_pairs = (co_occurrence > threshold).sum().item() - n_active
    high_corr_pairs //= 2

    return {
        "n_active_features": n_active,
        "n_dead_features": sparse.shape[1] - n_active,
        "high_correlation_pairs": high_corr_pairs,
        "correlation_threshold": threshold,
        "mean_co_occurrence": co_occurrence.mean().item(),
    }


def main():
    model_dirs = [config.paths.models_dir / f"sae_seed{s}" for s in config.training.seeds]
    missing = [d for d in model_dirs if not (d / "ae.pt").exists()]
    if missing:
        logger.error(f"Missing models: {[str(m) for m in missing]}")
        logger.error("Run first: python src/02a_train_sae.py")
        sys.exit(1)

    if not config.paths.visual_embeddings_path.exists():
        logger.error(f"Embeddings not found: {config.paths.visual_embeddings_path}")
        sys.exit(1)

    embeddings = torch.load(config.paths.visual_embeddings_path, map_location="cpu", weights_only=True)
    if config.training.stability_max_samples:
        embeddings = embeddings[: config.training.stability_max_samples]
    logger.info(f"Embeddings: {embeddings.shape}")

    # 1. Cross-seed Jaccard stability
    logger.info("Computing Jaccard stability across seeds...")
    stability = SAEManager.compute_stability(
        model_dirs, embeddings, config={"device": config.hardware.device}
    )
    logger.info(f"  Mean Jaccard: {stability['mean_jaccard']:.4f}")
    logger.info(f"  Std Jaccard:  {stability['std_jaccard']:.4f}")

    # 2. Per-seed metrics
    logger.info("\nPer-seed metrics:")
    per_seed_metrics = {}
    for seed, model_dir in zip(config.training.seeds, model_dirs):
        mgr = SAEManager({"device": config.hardware.device})
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

    # 3. Clustering
    logger.info("\nConcept clustering analysis...")
    clustering = compute_concept_clustering(model_dirs, embeddings)
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
        "config": {"seeds": list(config.training.seeds), "n_samples": embeddings.shape[0]},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
