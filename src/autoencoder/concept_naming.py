"""
concept_naming.py — Assign names to SAE concepts

Library stage: assign medical names to the SAE features of a single
reference model using cosine similarity between gap-corrected decoder
weights and vocabulary embeddings. 

Writes ``concept_names.json``.

Invoked by ``scripts/run_baseline.py`` (and the ablation driver).

Prerequisites:
    - models/sae_seed{seed}/ae.pt
    - models/modality_gap.pt
    - embeddings/<...>/text_vocab_embeddings.pt
    - data/vocabulary.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import utils
from autoencoder.sae_module import SAEManager
from autoencoder.visualization import plot_concept_score_distribution

logger = utils.setup_logging(__name__)

OUTPUT_PATH = config.paths.results_dir / "concept_names.json"


def build_concept_names(seed: int) -> Path:
    """Name every feature of the ``sae_seed{seed}`` model.

    Args:
        seed: The trained-model seed whose decoder features are named.

    Returns:
        Path to the written ``concept_names.json``.
    """
    model_dir = config.paths.models_dir / f"sae_seed{seed}"

    for path, desc in [
        (model_dir, "SAE model"),
        (config.paths.vocab_embeddings_path, "Vocab embeddings"),
        (config.paths.vocab_labels_path, "Vocabulary labels"),
        (config.paths.models_dir / "modality_gap.pt", "Modality gap"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    with open(config.paths.vocab_labels_path) as f:
        raw_vocab = json.load(f)
    # Vocabulary JSON is a list of {"term","similarity_score","source"} dicts
    # (builder output) — normalize to term strings for label lookups.
    # Also tolerates a legacy list of plain strings.
    vocab_labels = [
        entry["term"] if isinstance(entry, dict) else entry
        for entry in raw_vocab
    ]
    logger.info(f"Vocabulary: {len(vocab_labels)} terms")

    vocab_embeddings = utils.load_tensor(config.paths.vocab_embeddings_path)
    logger.info(f"Vocab embeddings shape: {vocab_embeddings.shape}")

    if vocab_embeddings.shape[0] != len(vocab_labels):
        raise ValueError(
            f"Vocab embeddings ({vocab_embeddings.shape[0]}) and labels "
            f"({len(vocab_labels)}) count mismatch — rebuild embeddings."
        )

    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)
    
    # Load modality gap
    gap_path = config.paths.models_dir / "modality_gap.pt"
    modality_gap = utils.load_tensor(gap_path)
    logger.info(f"Loaded modality gap from {gap_path}")

    logger.info(
        f"Computing concept names (top_n={config.explanation.concept_top_n})..."
    )
    concept_names = mgr.name_concepts(
        vocab_embeddings,
        vocab_labels,
        top_n=config.explanation.concept_top_n,
        modality_gap=modality_gap,
    )

    # Persist results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(concept_names, f, indent=2, ensure_ascii=False)

    # Summary statistics
    scores = [v["score"] for v in concept_names.values()]
    mean_score = sum(scores) / len(scores)
    logger.info("Concept naming complete:")
    logger.info(f"  Total features: {len(concept_names)}")
    logger.info(f"  Mean score: {mean_score:.4f}")
    logger.info(f"  Min/Max: {min(scores):.4f} / {max(scores):.4f}")
    logger.info(f"  Saved to: {OUTPUT_PATH}")

    # Top-10 by score
    sorted_concepts = sorted(
        concept_names.items(), key=lambda x: x[1]["score"], reverse=True
    )
    logger.info("\nTop-10 concepts:")
    for feat_id, info in sorted_concepts[:10]:
        logger.info(f"  Feature {feat_id:>4}: {info['name']:30s} ({info['score']:.4f})")

    # Visualization
    fig_path = config.paths.figures_dir / "concept_score_distribution.png"
    plot_concept_score_distribution(scores, fig_path)

    return OUTPUT_PATH


# Invoked by scripts/run_baseline.py (and the ablation driver) — no __main__.
