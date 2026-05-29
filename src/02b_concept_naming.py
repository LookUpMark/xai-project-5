"""
02b_concept_naming.py — Assign names to SAE concepts

Assign medical names to the 4096 SAE features using cosine similarity
between decoder weights and vocabulary embeddings.

Prerequisites:
    - models/sae_seed42/ae.pt (or any seed)
    - embeddings/text_vocab_embeddings.pt
    - data/vocabulary.json (list of medical terms)

Usage:
    python src/02b_concept_naming.py                             # seed 42 default
    python src/02b_concept_naming.py --seed 0 --top-n 5         # seed 0, 5 candidates
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sae_module import SAEManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "text_vocab_embeddings.pt"
VOCAB_LABELS_PATH = PROJECT_ROOT / "data" / "vocabulary.json"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description="Name SAE concepts via cosine similarity")
    parser.add_argument("--seed", type=int, default=42, help="Which SAE seed to use")
    parser.add_argument("--top-n", type=int, default=3, help="Candidates per feature")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: results/concept_names.json)")
    args = parser.parse_args()

    model_dir = MODELS_DIR / f"sae_seed{args.seed}"
    output_path = Path(args.output) if args.output else RESULTS_DIR / "concept_names.json"

    # Check prerequisites
    if not model_dir.exists():
        logger.error(f"Model not found: {model_dir}")
        sys.exit(1)
    if not VOCAB_EMBEDDINGS_PATH.exists():
        logger.error(f"Vocab embeddings not found: {VOCAB_EMBEDDINGS_PATH}")
        sys.exit(1)
    if not VOCAB_LABELS_PATH.exists():
        logger.error(f"Vocabulary not found: {VOCAB_LABELS_PATH}")
        sys.exit(1)

    # Load vocabulary
    with open(VOCAB_LABELS_PATH) as f:
        vocab_labels = json.load(f)
    logger.info(f"Vocabulary: {len(vocab_labels)} terms")

    # Load vocab embeddings
    vocab_embeddings = torch.load(VOCAB_EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    logger.info(f"Vocab embeddings shape: {vocab_embeddings.shape}")

    # Load SAE and assign names
    mgr = SAEManager()
    mgr.load(model_dir)

    logger.info(f"Computing concept names (top_n={args.top_n})...")
    concept_names = mgr.name_concepts(vocab_embeddings, vocab_labels, top_n=args.top_n)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(concept_names, f, indent=2, ensure_ascii=False)

    # Statistics
    scores = [v["score"] for v in concept_names.values()]
    logger.info(f"Concept naming complete:")
    logger.info(f"  Total features: {len(concept_names)}")
    logger.info(f"  Mean score: {sum(scores)/len(scores):.4f}")
    logger.info(f"  Min/Max score: {min(scores):.4f} / {max(scores):.4f}")
    logger.info(f"  Saved to: {output_path}")

    # Show top-10 concepts by score
    sorted_concepts = sorted(concept_names.items(), key=lambda x: x[1]["score"], reverse=True)
    logger.info(f"\nTop-10 concepts (by similarity score):")
    for feat_id, info in sorted_concepts[:10]:
        logger.info(f"  Feature {feat_id:4s}: {info['name']:30s} (score={info['score']:.4f})")


if __name__ == "__main__":
    main()
