"""
02c_generate_explanations.py — Generate SAE-based explanations

For each image, extract the top-k activated SAE concepts and generate
a structured explanation (pseudo-report) for the LLM Judge.

Prerequisites:
    - models/sae_seed42/ae.pt
    - embeddings/visual_embeddings.pt
    - results/concept_names.json (output of 02b_concept_naming.py)

Usage:
    python src/02c_generate_explanations.py
    python src/02c_generate_explanations.py --seed 42 --top-n 5 --max-samples 100
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
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "visual_embeddings.pt"
CONCEPT_NAMES_PATH = PROJECT_ROOT / "results" / "concept_names.json"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def generate_explanation(
    top_concepts: list[tuple[int, float]],
    concept_names: dict[str, dict],
) -> dict:
    """
    Generate a structured explanation for a single sample.

    Args:
        top_concepts: List of (feature_id, activation_value).
        concept_names: Dict {feature_id: {"name": str, "score": float}}.

    Returns:
        Dict with structured pseudo-report for the LLM Judge.
    """
    findings = []
    for feat_id, activation in top_concepts:
        feat_key = str(feat_id)
        if feat_key in concept_names:
            name = concept_names[feat_key]["name"]
            similarity = concept_names[feat_key]["score"]
        else:
            name = f"unknown_feature_{feat_id}"
            similarity = 0.0

        findings.append({
            "concept": name,
            "feature_id": feat_id,
            "activation": round(activation, 4),
            "naming_confidence": round(similarity, 4),
        })

    # Textual pseudo-report for the LLM Judge
    concept_list = ", ".join(f["concept"] for f in findings[:5])
    pseudo_report = (
        f"The model identifies the following visual concepts in this radiograph: {concept_list}. "
        f"The dominant concept is '{findings[0]['concept']}' "
        f"(activation={findings[0]['activation']:.3f})."
    )

    return {
        "findings": findings,
        "pseudo_report": pseudo_report,
        "n_active_concepts": len(findings),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate SAE-based explanations")
    parser.add_argument("--seed", type=int, default=42, help="Which SAE seed to use")
    parser.add_argument("--top-n", type=int, default=5, help="Top-n concepts per sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    model_dir = MODELS_DIR / f"sae_seed{args.seed}"
    output_path = Path(args.output) if args.output else RESULTS_DIR / "sample_explanations.json"

    # Check prerequisites
    for path, desc in [
        (model_dir, "SAE model"),
        (EMBEDDINGS_PATH, "Visual embeddings"),
        (CONCEPT_NAMES_PATH, "Concept names"),
    ]:
        if not path.exists():
            logger.error(f"{desc} not found: {path}")
            sys.exit(1)

    # Load data
    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    with open(CONCEPT_NAMES_PATH) as f:
        concept_names = json.load(f)

    if args.max_samples:
        embeddings = embeddings[: args.max_samples]

    logger.info(f"Generating explanations for {embeddings.shape[0]} samples...")

    # Load SAE
    mgr = SAEManager()
    mgr.load(model_dir)

    # Generate explanations for each sample
    all_top_concepts = mgr.get_top_concepts(embeddings, n=args.top_n)

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["sample_idx"] = idx
        explanations.append(explanation)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info(f"Saved to: {output_path}")

    # Show example
    if explanations:
        logger.info(f"\nExample (sample 0):")
        logger.info(f"  {explanations[0]['pseudo_report']}")


if __name__ == "__main__":
    main()
