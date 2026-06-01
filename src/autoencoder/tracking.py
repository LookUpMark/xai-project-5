"""
tracking.py — Experiment tracking integration (Weights & Biases).

Thin wrapper around wandb that degrades gracefully when wandb
is not installed or not enabled. All functions are no-ops when
tracking is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_tracking_enabled: bool = False


def init_tracking(stage_name: str, config: dict[str, Any]) -> None:
    """Initialize wandb run for a pipeline stage.

    Args:
        stage_name: Human-readable stage identifier (used as run name).
        config: Dict with project, entity, and stage-specific hyperparams.
    """
    global _tracking_enabled
    try:
        import os
        import subprocess

        import torch
        import wandb

        wandb.init(
            project=config.get("project", "sae-concept-discovery"),
            entity=config.get("entity"),
            name=stage_name,
            config=config,
        )

        # Log reproducibility metadata
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            git_commit = "unknown"

        wandb.config.update(
            {
                "git_commit": git_commit,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "N/A",
                "mps_available": torch.backends.mps.is_available(),
                "pythonhashseed": os.getenv("PYTHONHASHSEED", "not_set"),
            },
            allow_val_change=True,
        )

        _tracking_enabled = True
        logger.info(f"wandb tracking enabled: {stage_name}")
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
    except Exception as e:
        logger.warning(f"wandb init failed: {e}. Tracking disabled.")


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to wandb (no-op if tracking disabled)."""
    if not _tracking_enabled:
        return
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception:
        pass


def log_artifact(path: Path, name: str, artifact_type: str) -> None:
    """Log a file artifact to wandb (no-op if tracking disabled)."""
    if not _tracking_enabled:
        return
    try:
        import wandb

        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    except Exception:
        pass


def finish_tracking() -> None:
    """Finish the current wandb run."""
    global _tracking_enabled
    if _tracking_enabled:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
        _tracking_enabled = False
