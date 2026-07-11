"""generate_explanations_hidden.py — Path A per-sample concept explanations.

For each held-out test image, take the top-k active 768-d SAE features and render
a pseudo-report from the named concepts (frozen-projection bridge output). Writes
sample_explanations.json for the LLM judge, plus a preview REPORT.

Run:
    python src/sae_hidden/generate_explanations_hidden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import utils
from autoencoder.sae_module import SAEManager
from sae_hidden.reports import md_table, write_report
from sae_hidden.train_hidden import hidden_sae_config

log = utils.setup_logging(__name__)

SEED = config.training.primary_seed
# F-C1: paths resolved inside run() (after select_dataset) — module-level binding
# froze them to the default dataset, so on ROCOv2 the explain stage read IU concept
# names and overwrote the IU output. Mirrors the baseline generate_explanations fix.


def build_explanation(top_concepts, concept_names):
    """Render one pseudo-report from (feat_id, activation) tuples."""
    items = []
    for feat_id, activation in top_concepts:
        key = str(feat_id)
        name = concept_names.get(key, {}).get("name", f"unknown_feature_{feat_id}")
        items.append({"feature_id": feat_id, "name": name, "activation": round(activation, 4)})
    if not items:
        return {"top_k_concepts": [], "pseudo_report": "No active concepts detected."}
    listing = ", ".join(c["name"] for c in items[:5])
    pseudo = (
        f"The model identifies the following visual concepts in this radiograph: "
        f"{listing}. The dominant concept is '{items[0]['name']}' "
        f"(activation={items[0]['activation']:.3f})."
    )
    return {"top_k_concepts": items, "pseudo_report": pseudo}


def run() -> Path:
    utils.set_global_seed(SEED)  # F-009: deterministic generation
    # F-C1: resolve per-dataset paths here, not at import.
    concept_names_path = config.paths.hidden_results_dir / "concept_names.json"
    output_path = config.paths.hidden_results_dir / "sample_explanations.json"
    model_dir = config.paths.hidden_models_dir / f"sae_seed{SEED}"
    for path, desc in [
        (model_dir, "Primary-seed SAE"),
        (config.paths.hidden_test_embeddings_path, "768-d test embeddings"),
        (concept_names_path, "Concept names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    embeddings = utils.load_tensor(config.paths.hidden_test_embeddings_path)
    with open(concept_names_path) as f:
        concept_names = json.load(f)

    ids_path = config.paths.hidden_test_image_ids_path
    test_image_ids = json.load(open(ids_path)) if ids_path.exists() else None

    max_samples = config.explanation.explanation_max_samples
    if max_samples:
        embeddings = embeddings[:max_samples]
        if test_image_ids is not None:
            test_image_ids = test_image_ids[:max_samples]

    mgr = SAEManager({**hidden_sae_config(), "device": config.hardware.device})  # F-017: explicit device wins
    mgr.load(model_dir)

    top_per_sample = mgr.get_top_concepts(embeddings, n=config.explanation.explanation_top_n)
    explanations = []
    for idx, top in enumerate(top_per_sample):
        ex = build_explanation(top, concept_names)
        ex["image_id"] = test_image_ids[idx] if test_image_ids else f"sample_{idx}"
        explanations.append(ex)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    # Report
    n = len(explanations)
    preview = explanations[:5]
    summary = (
        f"Generated {n} per-sample explanations from the 768-d SAE (seed {SEED}) on the "
        f"held-out test set, using named (non-dead) concepts. Saved for the LLM judge."
    )
    sections = [
        (
            "Counts",
            md_table(
                ["item", "value"],
                [
                    ["test samples", n],
                    ["top-k per sample", config.explanation.explanation_top_n],
                    ["named concept source", "results/sae_hidden/concept_names.json"],
                ],
            ),
        ),
        (
            "Sample previews (first 5)",
            "\n\n".join(
                f"- `{e['image_id']}`: {e['pseudo_report']}" for e in preview
            ),
        ),
    ]
    report_path = config.paths.hidden_results_dir / "REPORT_explanations.md"
    write_report(report_path, "Path A — Per-sample Explanations (768-d)", sections, summary)
    log.info(f"Explanations: {output_path}")
    log.info(f"Report: {report_path}")
    return output_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
