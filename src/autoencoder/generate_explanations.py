"""
generate_explanations.py — Generate SAE-based explanations

For each image, extract the top-k activated SAE concepts and generate
a structured explanation (pseudo-report) for the LLM Judge.

Uses HELD-OUT test embeddings for evaluation.

Prerequisites:
    - models/sae_seed{PRIMARY_SEED}/ae.pt
    - embeddings/test_embeddings.pt
    - results/concept_names.json (output of concept_naming.py)

Run:
    python src/autoencoder/generate_explanations.py
"""

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
from autoencoder.tracking import init_tracking, log_artifact, finish_tracking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = config.training.primary_seed
CONCEPT_NAMES_PATH = config.paths.results_dir / "concept_names.json"
OUTPUT_PATH = config.paths.results_dir / "sample_explanations.json"


def generate_explanation(
    top_concepts: list[tuple[int, float]],
    concept_names: dict[str, dict],
) -> dict:
    """
    Generate a structured explanation for a single sample.

    Args:
        top_concepts: List of (feature_id, activation_value).
        concept_names: Dict {feature_id_str: {"name": str, "score": float}}.
            Keys are strings (after JSON serialization).

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

    # Guard against empty findings
    if not findings:
        return {
            "findings": [],
            "pseudo_report": "No active concepts detected.",
            "n_active_concepts": 0,
        }

    # Build natural-language pseudo-report
    concept_list = ", ".join(f["concept"] for f in findings[:5])
    pseudo_report = (
        f"The model identifies the following visual concepts in this "
        f"radiograph: {concept_list}. "
        f"The dominant concept is '{findings[0]['concept']}' "
        f"(activation={findings[0]['activation']:.3f})."
    )

    return {
        "findings": findings,
        "pseudo_report": pseudo_report,
        "n_active_concepts": len(findings),
    }


def run() -> Path:
    """Run explanation generation stage. Returns path to output file."""
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"

    # Use TEST embeddings for evaluation (not training data)
    embeddings_path = config.paths.test_embeddings_path

    for path, desc in [
        (model_dir, "SAE model"),
        (embeddings_path, "Test embeddings"),
        (CONCEPT_NAMES_PATH, "Concept names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    with open(CONCEPT_NAMES_PATH) as f:
        concept_names = json.load(f)

    if config.explanation.explanation_max_samples:
        embeddings = embeddings[: config.explanation.explanation_max_samples]

    logger.info(f"Generating explanations for {embeddings.shape[0]} test samples...")

    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    all_top_concepts = mgr.get_top_concepts(
        embeddings, n=config.explanation.explanation_top_n
    )

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["sample_idx"] = idx
        explanations.append(explanation)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info(f"Saved to: {OUTPUT_PATH}")

    if explanations:
        logger.info(f"\nExample (sample 0):\n  {explanations[0]['pseudo_report']}")

    # Tracking
    if config.wandb_cfg.enabled:
        init_tracking("generate_explanations", {
            "project": config.wandb_cfg.project,
            "seed": SEED,
            "n_samples": len(explanations),
        })
        log_artifact(OUTPUT_PATH, "sample_explanations", "results")
        finish_tracking()

    return OUTPUT_PATH


def main():
    run()


if __name__ == "__main__":
    main()
