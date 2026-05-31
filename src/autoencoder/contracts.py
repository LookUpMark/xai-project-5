"""
contracts.py — Typed data contracts for the SAE pipeline.

These dataclasses define the shape of data flowing between pipeline stages.
They wrap tensors and raw types in structured containers for type safety
and validation at stage boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateName:
    """A candidate concept name with its similarity score."""

    label: str
    score: float


@dataclass(frozen=True)
class ConceptName:
    """Named SAE feature with candidates from vocabulary."""

    feature_id: int
    name: str
    score: float
    candidates: list[CandidateName]


@dataclass(frozen=True)
class ConceptMap:
    """Output of concept_naming stage — maps feature IDs to names."""

    concepts: dict[int, ConceptName]
    source_model_seed: int
    total_features: int
    mean_score: float


@dataclass(frozen=True)
class Finding:
    """A single concept activation in an explanation."""

    concept: str
    feature_id: int
    activation: float
    naming_confidence: float


@dataclass(frozen=True)
class Explanation:
    """Output of generate_explanations stage — per-sample explanation."""

    sample_idx: int
    findings: list[Finding]
    pseudo_report: str
    n_active_concepts: int


@dataclass(frozen=True)
class SeedMetrics:
    """Per-seed training metrics."""

    seed: int
    mse: float
    l0_mean: float
    l0_std: float
    dead_features_pct: float
    dict_utilization_pct: float
    activation_entropy: float
    feature_frequency_mean: float
    feature_frequency_std: float


@dataclass(frozen=True)
class ClusteringResult:
    """Concept clustering analysis result."""

    n_active_features: int
    n_dead_features: int
    high_correlation_pairs: int
    correlation_threshold: float
    mean_co_occurrence: float


@dataclass(frozen=True)
class StabilityResult:
    """Output of stability_analysis stage — cross-seed comparison."""

    mean_jaccard: float
    std_jaccard: float
    jaccard_matrix: list[list[float]]
    per_seed_metrics: dict[int, SeedMetrics]
    clustering: ClusteringResult
    config_snapshot: dict
