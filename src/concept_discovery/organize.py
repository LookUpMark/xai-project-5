"""Concept organization extension (brief §3).

Method-agnostic post-processing that clusters discovered concepts (from SPLiCE
or SAE explanations) by RadLex text-embedding cosine similarity, annotates each
cluster with a best-effort RadLex anatomical ancestor, and re-expresses
per-image explanations as concept families with a redundancy metric.

Three layers:
  - Adapters (from_spliece_explanations / from_sae_explanations) -> ConceptSet
  - Core (cluster_concepts / annotate_radlex / build_structured_explanations /
    compute_metrics / run) operates only on ConceptSet
  - scripts/run_concept_organization.py drives end-to-end

Standalone output: does NOT feed the LLM judge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

sys.path.insert(0, "src/")

import torch  # noqa: E402


@dataclass
class ImageConcepts:
    """Per-image concept activations (normalized)."""
    image_id: str
    activations: dict[str, float]  # name -> activation (>0); subset of ConceptSet.names


@dataclass
class ConceptSet:
    """Method-agnostic concept collection produced by adapters.

    Invariants:
      - embeddings.shape[0] == len(names) == len(name_to_idx)
      - name_to_idx[name] == index of name in names
      - every key in every per_image[].activations is in names
    """
    names: list[str]
    embeddings: torch.Tensor               # (M, 512)
    name_to_idx: dict[str, int]
    per_image: list[ImageConcepts]


@dataclass
class Cluster:
    """Result of clustering concept names."""
    cluster_id: int
    members: list[str]                     # sorted for stable ids
    medoid: str                             # member with max mean cosine to others


@dataclass
class AnnotatedCluster(Cluster):
    """Cluster + best-effort RadLex ancestor."""
    radlex_label: str | None = None
    radlex_rid: str | None = None
    n_resolved: int = 0                    # members that resolved to a RadLex RID
    n_members: int = 0

    @property
    def display_label(self) -> str:
        return self.radlex_label if self.radlex_label else self.medoid


def _build_term_to_idx(vocab_terms: list[dict], vocab_emb: torch.Tensor) -> dict[str, int]:
    """term -> vocab row index, with a count guard (mirrors spliece F-007)."""
    if len(vocab_terms) != vocab_emb.shape[0]:
        raise ValueError(
            f"vocab_terms ({len(vocab_terms)}) != vocab_emb rows ({vocab_emb.shape[0]}); "
            "coefficient index would not map to the correct term — re-embed the vocabulary."
        )
    return {t["term"]: i for i, t in enumerate(vocab_terms) if isinstance(t, dict) and "term" in t}


def _finalize_concept_set(
    per_image: list[ImageConcepts], term_to_idx: dict[str, int], vocab_emb: torch.Tensor
) -> ConceptSet:
    """Collect active names, slice their embeddings, build the ConceptSet.

    Names are sorted for deterministic cluster ids. Names absent from the
    vocabulary have already been skipped by the caller (per-image activations
    only contain resolvable names).
    """
    active = sorted({name for img in per_image for name in img.activations})
    name_to_idx = {n: i for i, n in enumerate(active)}
    if active:
        rows = torch.stack([vocab_emb[term_to_idx[n]] for n in active])
    else:
        rows = torch.empty((0, vocab_emb.shape[1]), dtype=vocab_emb.dtype)
    return ConceptSet(names=active, embeddings=rows, name_to_idx=name_to_idx, per_image=per_image)


def from_spliece_explanations(
    explanations: list[dict],
    vocab_terms: list[dict],
    vocab_emb: torch.Tensor,
) -> ConceptSet:
    """Normalize SPLiCE sample_explanations.json into a ConceptSet.

    SPLiCE feature_id == vocab index; name == vocab term directly. Drops
    non-positive activations and names absent from the vocabulary (graceful,
    no crash). Missing image_id falls back to f"img_{i}".
    """
    term_to_idx = _build_term_to_idx(vocab_terms, vocab_emb)
    per_image: list[ImageConcepts] = []
    for i, img in enumerate(explanations):
        activations: dict[str, float] = {}
        for c in img.get("top_k_concepts", []):
            name = c.get("name")
            act = float(c.get("activation", 0.0))
            if act > 0 and name in term_to_idx:
                # keep the max activation if a name repeats
                activations[name] = max(activations.get(name, 0.0), act)
        image_id = img.get("image_id") or f"img_{i}"
        per_image.append(ImageConcepts(image_id=image_id, activations=activations))
    return _finalize_concept_set(per_image, term_to_idx, vocab_emb)
