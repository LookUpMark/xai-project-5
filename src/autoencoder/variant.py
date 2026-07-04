"""variant.py — run baseline (512-d) stages under a config/paths variant.

One context manager swaps the module-global ``config.sae`` (frozen, so replaced
wholesale) and the ``config.paths.{models,results,figures}_dir`` attributes,
restoring both on exit. This lets ``scripts/02_baseline.py`` and the ablation
harness drive the existing ``src/autoencoder`` stage ``run()`` / ``main()``
functions verbatim under a different dict_size / k / steps or an isolated output
directory (per-tag or per-preset ablation).

Embeddings / vocab paths are NOT swapped: the baseline always reads
``embeddings/standard/`` + ``data/vocabulary.json`` regardless of variant.

# ponytail: swaps module-global config singletons; single-threaded, restored in finally.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import config


@contextmanager
def baseline_variant(
    *,
    sae=None,
    models_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> Iterator[None]:
    """Temporarily override baseline config + model/results dirs for a variant run.

    Args:
        sae: an SAEConfig (build with ``dataclasses.replace``) to swap in.
            ``None`` keeps the default ``config.sae``.
        models_dir / results_dir: override the model/results base dirs (used for
            ``--tag`` isolation and ablation presets). ``None`` leaves them, so the
            default run reuses the canonical ``models/`` + ``results/`` (matching
            ``train_sae.py``). ``figures_dir`` tracks ``results_dir``.
    """
    saved_sae = config.sae
    saved = {
        "models": config.paths.models_dir,
        "results": config.paths.results_dir,
        "figures": config.paths.figures_dir,
    }
    try:
        if sae is not None:
            config.sae = sae
        if models_dir is not None:
            config.paths.models_dir = Path(models_dir)
        if results_dir is not None:
            rd = Path(results_dir)
            config.paths.results_dir = rd
            config.paths.figures_dir = rd / "figures"
        yield
    finally:
        config.sae = saved_sae
        config.paths.models_dir = saved["models"]
        config.paths.results_dir = saved["results"]
        config.paths.figures_dir = saved["figures"]
