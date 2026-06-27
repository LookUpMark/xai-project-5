"""
stability_analysis.py — Multi-seed stability analysis and clustering

Evaluate robustness of SAE concepts by comparing activations across
multiple SAEs trained with different seeds. Uses HELD-OUT test embeddings.
Computes Jaccard similarity, per-seed metrics, and concept clustering.

Prerequisites:
    - models/sae_seed{0,42,123,456,789}/ae.pt (all 5 seeds)
    - embeddings/test_embeddings.pt

Run:
    python src/autoencoder/stability_analysis.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import utils
from autoencoder.sae_module import SAEManager
from autoencoder.visualization import plot_jaccard_heatmap, plot_per_seed_metrics

logger = utils.setup_logging(__name__)

OUTPUT_PATH = config.paths.results_dir / "stability_analysis.json"


def compute_feature_frequency(
    mgr: SAEManager, embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute activation frequency of each feature across the dataset.

    Args:
        mgr: Loaded SAEManager instance.
        embeddings: Input tensor of shape (N, 512).

    Returns:
        Tensor of shape (dict_size,) with per-feature activation frequencies.
    """
    with torch.no_grad():
        sparse = mgr.encode(embeddings)
    return (sparse != 0).float().mean(dim=0)


def compute_concept_clustering(
    model_dirs: list[Path], embeddings: torch.Tensor, device: str
) -> dict:
    """Compute co-activation similarity between concept activation patterns.

    Analyzes one seed model (configurable — uses primary_seed by default).

    Args:
        model_dirs: Paths to trained SAE model directories.
        embeddings: Input tensor of shape (N, 512).
        device: Torch device string (e.g. "cpu", "cuda").

    Returns:
        Dict with keys: n_active_features, n_dead_features,
        high_correlation_pairs, correlation_threshold, mean_co_occurrence.
    """
    mgr = SAEManager({"device": device})
    mgr.load(model_dirs[0])

    with torch.no_grad():
        sparse = mgr.encode(embeddings)

    # Filter out dead features
    active_mask = (sparse != 0).float().sum(dim=0) > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    n_active = active_indices.shape[0]

    logger.info(f"  Active features: {n_active}/{sparse.shape[1]}")

    # Cosine similarity of binary activation patterns across samples
    sparse_active = sparse[:, active_indices]
    binary = (sparse_active != 0).float()

    # Epsilon prevents div-by-zero for features with single-sample activation.
    # For such features, the resulting unit-normalized column has magnitude ~1,
    # producing meaningful (not inflated) cosine similarity values.
    norms = binary.norm(dim=0, keepdim=True) + 1e-8
    binary_norm = binary / norms
    co_occurrence = (binary_norm.T @ binary_norm).cpu()

    # Count highly correlated pairs (potential redundancy)
    threshold = config.training.correlation_threshold
    high_corr_pairs = (co_occurrence > threshold).sum().item() - n_active
    high_corr_pairs //= 2

    return {
        "n_active_features": n_active,
        "n_dead_features": sparse.shape[1] - n_active,
        "high_correlation_pairs": high_corr_pairs,
        "correlation_threshold": threshold,
        "mean_co_occurrence": co_occurrence.mean().item(),
    }


def run() -> Path:
    """Run stability analysis stage. Returns path to output file."""
    model_dirs = [
        config.paths.models_dir / f"sae_seed{s}" for s in config.training.seeds
    ]
    missing = [
        d
        for d in model_dirs
        if not (d / "ae.pt").exists() and not (d / "trainer_0" / "ae.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing models: {[str(m) for m in missing]}. "
            f"Run first: python scripts/run_sae_training.py"
        )

    # Use TEST embeddings for stability analysis
    embeddings_path = config.paths.test_embeddings_path
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Test embeddings not found: {embeddings_path}. "
            f"Run first: python scripts/run_sae_training.py"
        )

    embeddings = utils.load_tensor(embeddings_path)
    if config.training.stability_max_samples:
        embeddings = embeddings[: config.training.stability_max_samples]
    logger.info(f"Test embeddings: {embeddings.shape}")

    # 1. Cross-seed Jaccard stability
    logger.info("Computing Jaccard stability across seeds...")
    stability = SAEManager.compute_stability(
        model_dirs, embeddings, config={"device": config.hardware.device}
    )
    logger.info(f"  Mean Jaccard: {stability['mean_jaccard']:.4f}")
    logger.info(f"  Std Jaccard:  {stability['std_jaccard']:.4f}")

    # 1b. Permutation-invariant matched stability (decoder best-match cosine).
    # The slot-wise Jaccard above is ~0 by construction for SAEs with no canonical
    # feature ordering and cannot establish non-identifiability (ML-AUDIT-2026-06-26
    # F-001). This matched metric is the sound cross-seed signal.
    logger.info("Computing permutation-invariant matched stability...")
    matched = SAEManager.compute_stability_matched(
        model_dirs, config={"device": config.hardware.device}
    )
    matched_path = config.paths.results_dir / "stability_matched.json"
    with open(matched_path, "w") as f:
        json.dump(matched, f, indent=2)
    _mc, _nc = matched["mean_best_match_cosine"], matched["null_mean"]
    logger.info(
        f"  Best-match cosine: {_mc:.4f} vs null {_nc:.4f} "
        f"({_mc / _nc:.2f}x), frac>=0.9: {matched['mean_frac_matched_0.9']:.4f}"
    )

    # 2. Per-seed metrics
    logger.info("\nPer-seed metrics:")
    per_seed_metrics = {}
    for seed, model_dir in zip(config.training.seeds, model_dirs):
        mgr = SAEManager({"device": config.hardware.device})
        mgr.load(model_dir)

        mse = mgr.compute_reconstruction_mse(embeddings)
        cosine = mgr.compute_cosine_reconstruction(embeddings)
        sparsity = mgr.compute_sparsity_metrics(embeddings)
        freq = compute_feature_frequency(mgr, embeddings)

        per_seed_metrics[seed] = {
            "mse": mse,
            "cosine": cosine,
            **sparsity,
            "feature_frequency_mean": freq.mean().item(),
            "feature_frequency_std": freq.std().item(),
        }
        logger.info(
            f"  Seed {seed:3d}: MSE={mse:.6f}, Cosine={cosine:.4f}, "
            f"L0={sparsity['l0_mean']:.1f}, Dead={sparsity['dead_features_pct']:.1f}%, "
            f"Util={sparsity['dict_utilization_pct']:.1f}%"
        )

    # 3. Clustering
    logger.info("\nConcept clustering analysis...")
    clustering = compute_concept_clustering(
        model_dirs, embeddings, config.hardware.device
    )
    logger.info(
        f"  High-similarity pairs (>{clustering['correlation_threshold']}): "
        f"{clustering['high_correlation_pairs']}"
    )

    # 4. Visualization
    jaccard_np = stability["jaccard_matrix"].numpy()
    plot_jaccard_heatmap(
        jaccard_np,
        list(config.training.seeds),
        config.paths.figures_dir / "jaccard_heatmap.png",
    )
    plot_per_seed_metrics(
        per_seed_metrics,
        config.paths.figures_dir / "per_seed_metrics.png",
    )

    # 5. Save results
    results = {
        "stability": {
            "mean_jaccard": stability["mean_jaccard"],
            "std_jaccard": stability["std_jaccard"],
            "jaccard_matrix": stability["jaccard_matrix"].tolist(),
        },
        "per_seed_metrics": per_seed_metrics,
        "clustering": clustering,
        "config": {
            "seeds": list(config.training.seeds),
            "n_samples": embeddings.shape[0],
            "dataset": "test",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {OUTPUT_PATH}")

    return OUTPUT_PATH


def main() -> None:
    """CLI entry point for stability analysis."""
    run()


if __name__ == "__main__":
    main()
