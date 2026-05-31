"""
protocols.py — Interface definitions for SAE pipeline stages.

Each pipeline stage should implement the relevant protocol,
enabling type-safe composition, testing with mocks, and
independent development of each stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PipelineStage(Protocol):
    """Base protocol for any pipeline stage."""

    @property
    def name(self) -> str:
        """Human-readable stage name."""
        ...

    def run(self) -> Path:
        """Execute the stage, return path to primary output artifact."""
        ...


@runtime_checkable
class TrackedStage(PipelineStage, Protocol):
    """Pipeline stage with experiment tracking support."""

    def run(self, run_id: Optional[str] = None) -> tuple[Path, dict]:
        """Execute stage, return (output_path, metrics_dict)."""
        ...
