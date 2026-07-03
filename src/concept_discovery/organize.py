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
