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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import utils
from autoencoder.sae_module import SAEManager

logger = utils.setup_logging(__name__)

SEED = config.training.primary_seed


def generate_explanation(
    top_concepts: list[tuple[int, float]],
    concept_names: dict[str, dict],
) -> dict:
    """Generate a structured explanation for a single sample.

    Args:
        top_concepts: List of (feature_id, activation_value) tuples.
        concept_names: Dict mapping feature_id strings to
            {"name": str, "score": float} (JSON-serialized keys).

    Returns:
        Dict with keys: top_k_concepts, pseudo_report. ``image_id`` is added by
        ``run()`` once the test image-id sidecar is available. This shape
        matches what ``evaluate_llm_judge.py`` consumes: each concept carries
        ``feature_id`` / ``name`` / ``activation``.
    """
    top_k_concepts = []
    for feat_id, activation in top_concepts:
        feat_key = str(feat_id)
        if feat_key in concept_names:
            name = concept_names[feat_key]["name"]
        else:
            name = f"unknown_feature_{feat_id}"

        top_k_concepts.append(
            {
                "feature_id": feat_id,
                "name": name,
                "activation": round(activation, 4),
            }
        )

    if not top_k_concepts:
        return {
            "top_k_concepts": [],
            "pseudo_report": "No active concepts detected.",
        }

    # Build natural-language pseudo-report
    concept_list = ", ".join(c["name"] for c in top_k_concepts[:5])
    pseudo_report = (
        f"The model identifies the following visual concepts in this "
        f"radiograph: {concept_list}. "
        f"The dominant concept is '{top_k_concepts[0]['name']}' "
        f"(activation={top_k_concepts[0]['activation']:.3f})."
    )

    return {
        "top_k_concepts": top_k_concepts,
        "pseudo_report": pseudo_report,
    }


def run() -> Path:
    """Run explanation generation stage. Returns path to output file."""
    utils.set_global_seed(SEED)  # F-009: deterministic generation
    # F-002: resolve under the active results_dir (baseline_variant swaps it at
    # runtime), not the import-time value, so outputs land in results/baseline/.
    concept_names_path = config.paths.results_dir / "concept_names.json"
    output_path = config.paths.results_dir / "sample_explanations.json"
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"

    # Use TEST embeddings for evaluation (not training data)
    embeddings_path = config.paths.test_embeddings_path

    for path, desc in [
        (model_dir, "SAE model"),
        (embeddings_path, "Test embeddings"),
        (concept_names_path, "Concept names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    embeddings = utils.load_tensor(embeddings_path)
    with open(concept_names_path) as f:
        concept_names = json.load(f)

    # Load the per-row image ids (basename) for the test split so each
    # explanation carries the image_id the LLM judge joins reports.csv on.
    # Falls back to a positional placeholder when the sidecar is absent.
    # Resolve at run() time (after select_dataset) — the import-time value is
    # frozen to the default dataset and would misroute under --dataset.
    test_image_ids_path = config.paths.test_image_ids_path
    if test_image_ids_path.exists():
        with open(test_image_ids_path) as f:
            test_image_ids = json.load(f)
    else:
        test_image_ids = None
        logger.warning(
            "Test image-id sidecar not found (%s); falling back to "
            "positional sample ids. Re-run extraction + split to populate it.",
            test_image_ids_path,
        )

    if config.explanation.explanation_max_samples:
        embeddings = embeddings[: config.explanation.explanation_max_samples]
        if test_image_ids is not None:
            test_image_ids = test_image_ids[
                : config.explanation.explanation_max_samples
            ]

    logger.info(f"Generating explanations for {embeddings.shape[0]} test samples...")

    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    all_top_concepts = mgr.get_top_concepts(
        embeddings, n=config.explanation.explanation_top_n
    )

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["image_id"] = (
            test_image_ids[idx] if test_image_ids else f"sample_{idx}"
        )
        explanations.append(explanation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info(f"Saved to: {output_path}")

    if explanations:
        logger.info(f"\nExample (sample 0):\n  {explanations[0]['pseudo_report']}")

    return output_path


def main() -> None:
    """CLI entry point for explanation generation."""
    run()


if __name__ == "__main__":
    main()
