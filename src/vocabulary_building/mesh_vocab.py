"""mesh_vocab.py — Build a radiology MeSH vocabulary for ROCOv2 (Phase 3).

External, license-free alternative to UMLS: the concept lexicon is MeSH
descriptors filtered to the radiology-visible branches (anatomy / diseases /
procedures), then ranked by cosine similarity to radiology anchor centroids and
truncated to top-k — the **same flow as the RadLex builder**, just on a broader
(radiology) domain and a free ontology. Produces ``vocabulary.json`` +
``text_vocab_embeddings.pt`` with the **same shape** as RadLex, so
``concept_naming`` consumes it unchanged. Independent of ROCOv2's labels (no
circularity with the evaluation).

Driven by ``scripts/run_vocab_building_pipeline.py --dataset rocov2`` (the CLI
dispatches on ``DatasetSpec.vocab_source``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from config import VLMConfig, VocabularyConfig
from vocabulary_building.build_vocabulary import (
    compute_anchor_centroids,
    encode_texts,
    rank_terms_by_relevance,
    save_vocab_embeddings,
    save_vocabulary,
)
from vocabulary_building.mesh_support import load_and_filter_mesh

# Broader than the chest-only anchors: spans the radiology sub-domains a
# multimodal SAE may discover (thoracic, abdominal, neuro, MSK, modalities,
# devices, general findings). Used to rank the MeSH subset to a top-k lexicon.
RADIOLOGY_ANCHOR_GROUPS = {
    "thoracic": ["lung", "heart", "pleura", "mediastinum", "pulmonary opacity"],
    "abdominal": ["liver", "kidney", "bowel", "pancreas", "abdominal mass"],
    "neuro": ["brain", "cerebral", "spinal cord", "skull"],
    "musculoskeletal": ["bone", "joint", "fracture", "skeleton"],
    "modality": [
        "computed tomography",
        "magnetic resonance imaging",
        "radiograph",
        "ultrasound",
    ],
    "devices": ["catheter", "tube", "stent", "prosthesis"],
    "findings": ["mass", "nodule", "opacity", "effusion", "edema", "calcification"],
}


def build_mesh_vocabulary(
    mesh_file,
    model,
    processor,
    vlm_config: VLMConfig,
    vocab_config: VocabularyConfig,
    top_k: int | None = None,
    categories=("A", "C", "E"),
) -> List[dict]:
    """Build the radiology MeSH vocabulary.

    Args:
        mesh_file: Path to the MeSH descriptor file (XML ``desc<year>.gz`` or
            ``desc<year>.xml``).
        model/processor: loaded BiomedCLIP.
        vlm_config: model runtime parameters.
        vocab_config: vocabulary I/O config (output paths are per-dataset).
        top_k: number of terms to keep (default ``vocab_config.top_k``).
        categories: MeSH tree branches to keep (default A/C/E radiology-visible).

    Returns:
        The vocabulary list (same shape as the RadLex builder output).
    """
    terms = load_and_filter_mesh(mesh_file, categories=categories)
    if not terms:
        raise ValueError(
            f"No MeSH terms loaded from {mesh_file} (check the file / categories)."
        )
    print(f"MeSH radiology vocab: {len(terms)} descriptors (branches {categories}).")

    all_embeddings = encode_texts(terms, model, processor, vlm_config)

    # Rank by radiology anchor centroids (reuse the RadLex machinery on broader
    # anchors); top-k truncation keeps the lexicon at a RadLex-comparable size.
    mesh_cfg = replace(vocab_config, anchor_groups=RADIOLOGY_ANCHOR_GROUPS)
    centroids = compute_anchor_centroids(model, processor, vlm_config, mesh_cfg)
    ranked = rank_terms_by_relevance(terms, all_embeddings, centroids)

    k = top_k if top_k is not None else vocab_config.top_k
    top = ranked[:k]
    vocabulary = [
        {"term": term, "similarity_score": round(score, 6), "source": "mesh"}
        for term, score in top
    ]

    save_vocabulary(vocabulary, vocab_config)
    save_vocab_embeddings(vocabulary, terms, all_embeddings, vocab_config)

    print("\nTop-10 MeSH concepts (by radiology-anchor similarity):")
    for i, (term, score) in enumerate(top[:10]):
        print(f"  {i+1:2d}. {term} ({score:.4f})")

    return vocabulary
