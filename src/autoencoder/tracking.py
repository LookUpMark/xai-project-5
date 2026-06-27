"""tracking.py — experiment-tracking wrapper (no-op by default).

All functions are inert unless a real backend is wired in. Restored 2026-06-27 after
the module was deleted in 6c53328: ``scripts/run_sae_training.py`` still calls
``init_tracking``/``finish_tracking``, and ``config.wandb_cfg.enabled`` is False by
default, so these no-ops keep the training entrypoint runnable without a wandb account.
"""
from __future__ import annotations


def init_tracking(name: str, config: dict | None = None) -> None:
    """No-op tracking init (wandb disabled by default)."""
    return None


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """No-op metric logging."""
    return None


def log_artifact(path, name: str, type: str = "dataset") -> None:
    """No-op artifact logging."""
    return None


def finish_tracking() -> None:
    """No-op tracking finish."""
    return None
