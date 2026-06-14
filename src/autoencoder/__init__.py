"""autoencoder — Sparse Autoencoder pipeline for medical VLM concept discovery."""

from autoencoder.sae_module import SAEManager
from autoencoder.contracts import (
    CandidateName,
    ConceptName,
    ConceptMap,
    ConceptActivation,
    Explanation,
    SeedMetrics,
    ClusteringResult,
    StabilityResult,
)
from autoencoder.protocols import PipelineStage, TrackedStage

__all__ = [
    "SAEManager",
    "CandidateName",
    "ConceptName",
    "ConceptMap",
    "ConceptActivation",
    "Explanation",
    "SeedMetrics",
    "ClusteringResult",
    "StabilityResult",
    "PipelineStage",
    "TrackedStage",
]
