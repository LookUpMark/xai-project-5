"""
02b_concept_naming.py - Assign names to SAE concepts

Assign medical names to the 4096 SAE features using cosine similarity
between decoder weights and vocabulary embeddings.

Prerequisites:
    - models/sae_seed{SEED}/ae.pt
    - embeddings/text_vocab_embeddings.pt
    - data/vocabulary.json

Run:
    python src/02b_concept_naming.py
"""

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from sae_module import SAEManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = config.training.seeds[1]  # Use seed 42 as the primary/reference model
OUTPUT_PATH = config.paths.results_dir / "concept_names.json"


def main():
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"

    for path, desc in [
        (model_dir, "SAE model"),
        (config.paths.vocab_embeddings_path, "Vocab embeddings"),
        (config.paths.vocab_labels_path, "Vocabulary labels"),
    ]:
        if not path.exists():
            logger.error(f"{desc} not found: {path}")
            sys.exit(1)

    with open(config.paths.vocab_labels_path) as f:
        vocab_labels = json.load(f)
    logger.info(f"Vocabulary: {len(vocab_labels)} terms")

    vocab_embeddings = torch.load(config.paths.vocab_embeddings_path, map_location="cpu", weights_only=True)
    logger.info(f"Vocab embeddings shape: {vocab_embeddings.shape}")

    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    logger.info(f"Computing concept names (top_n={config.explanation.concept_top_n})...")
    concept_names = mgr.name_concepts(vocab_embeddings, vocab_labels, top_n=config.explanation.concept_top_n)

    # Persist results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(concept_names, f, indent=2, ensure_ascii=False)

    # Summary statistics
    scores = [v["score"] for v in concept_names.values()]
    logger.info(f"Concept naming complete:")
    logger.info(f"  Total features: {len(concept_names)}")
    logger.info(f"  Mean score: {sum(scores)/len(scores):.4f}")
    logger.info(f"  Min/Max: {min(scores):.4f} / {max(scores):.4f}")
    logger.info(f"  Saved to: {OUTPUT_PATH}")

    # Top-10 by score
    sorted_concepts = sorted(concept_names.items(), key=lambda x: x[1]["score"], reverse=True)
    logger.info(f"\nTop-10 concepts:")
    for feat_id, info in sorted_concepts[:10]:
        logger.info(f"  Feature {feat_id:>4s}: {info['name']:30s} ({info['score']:.4f})")


if __name__ == "__main__":
    main()
