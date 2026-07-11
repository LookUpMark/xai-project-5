"""
visualization.py — Standard SAE visualizations using seaborn/matplotlib.

Generates and saves figures for training diagnostics, concept analysis,
and stability evaluation. Figures are saved to results/figures/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

logger = logging.getLogger(__name__)


def plot_jaccard_heatmap(
    jaccard_matrix: np.ndarray,
    seeds: list[int],
    save_path: Path,
) -> Path:
    """Plot pairwise Jaccard similarity heatmap across seeds.

    Args:
        jaccard_matrix: Array of shape (n_seeds, n_seeds) with Jaccard values.
        seeds: List of seed integers used as axis labels.
        save_path: Destination path for the saved figure.

    Returns:
        The save_path after writing the figure to disk.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = [str(s) for s in seeds]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Cross-Seed Jaccard Similarity")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    utils.ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Jaccard heatmap to {save_path}")
    return save_path


def plot_concept_score_distribution(
    scores: list[float],
    save_path: Path,
) -> Path:
    """Plot histogram of concept naming cosine similarity scores.

    Args:
        scores: List of cosine similarity scores (one per feature).
        save_path: Destination path for the saved figure.

    Returns:
        The save_path after writing the figure to disk.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True, ax=ax)
    ax.set_title("Concept Naming Score Distribution")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    mean_score = float(np.mean(scores))
    ax.axvline(mean_score, color="red", linestyle="--", label=f"Mean={mean_score:.3f}")
    ax.legend()
    utils.ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved concept score distribution to {save_path}")
    return save_path


def plot_per_seed_metrics(
    metrics: dict[int, dict],
    save_path: Path,
) -> Path:
    """Plot grouped bar chart comparing metrics across seeds.

    Args:
        metrics: Dict mapping seed int to metric dict
            (keys: mse, dead_features_pct, etc.).
        save_path: Destination path for the saved figure.

    Returns:
        The save_path after writing the figure to disk.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for seed, m in metrics.items():
        rows.append(
            {
                "seed": str(seed),
                "MSE": m.get("mse", 0),
                "Dead %": m.get("dead_features_pct", 0),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=df,
        x="seed",
        y="MSE",
        ax=axes[0],
        hue="seed",
        palette="Blues_d",
        legend=False,
    )
    axes[0].set_title("Reconstruction MSE per Seed")
    sns.barplot(
        data=df,
        x="seed",
        y="Dead %",
        ax=axes[1],
        hue="seed",
        palette="Reds_d",
        legend=False,
    )
    axes[1].set_title("Dead Features % per Seed")
    fig.suptitle("Per-Seed Metrics Comparison")
    fig.tight_layout()
    utils.ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved per-seed metrics to {save_path}")
    return save_path




