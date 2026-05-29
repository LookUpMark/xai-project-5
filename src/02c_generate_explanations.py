"""
02c_generate_explanations.py — Generate SAE-based explanations

For each image, extract the top-k activated SAE concepts and generate
a structured explanation (pseudo-report) for the LLM Judge.

Prerequisites:
    - models/sae_seed{SEED}/ae.pt
    - embeddings/visual_embeddings.pt
    - results/concept_names.json (output of 02b)

Run:
    python src/02c_generate_explanations.py
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

SEED = config.SEEDS[1]  # Use seed 42 as primary
CONCEPT_NAMES_PATH = config.RESULTS_DIR / "concept_names.json"
OUTPUT_PATH = config.RESULTS_DIR / "sample_explanations.json"


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
    model_dir = config.MODELS_DIR / f"sae_seed{SEED}"

    for path, desc in [
        (model_dir, "SAE model"),
        (config.VISUAL_EMBEDDINGS_PATH, "Visual embeddings"),
        (CONCEPT_NAMES_PATH, "Concept names"),
    ]:
        if not path.exists():
            logger.error(f"{desc} not found: {path}")
            sys.exit(1)

    # Load data
    embeddings = torch.load(config.VISUAL_EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
    with open(CONCEPT_NAMES_PATH) as f:
        concept_names = json.load(f)

    if config.EXPLANATION_MAX_SAMPLES:
        embeddings = embeddings[: config.EXPLANATION_MAX_SAMPLES]

    logger.info(f"Generating explanations for {embeddings.shape[0]} samples...")

    # Load SAE
    mgr = SAEManager({"device": config.DEVICE})
    mgr.load(model_dir)

    # Generate
    all_top_concepts = mgr.get_top_concepts(embeddings, n=config.EXPLANATION_TOP_N)

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["sample_idx"] = idx
        explanations.append(explanation)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info(f"Saved to: {OUTPUT_PATH}")

    if explanations:
        logger.info(f"\nExample (sample 0):\n  {explanations[0]['pseudo_report']}")


if __name__ == "__main__":
    main()
