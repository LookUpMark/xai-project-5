"""variant.py — run Path A stages under a config/paths variant without editing them.

One context manager swaps the module-global ``config.sae_hidden`` (frozen, so it is
replaced wholesale) and the ``config.paths.hidden_*`` I/O attributes, restoring
both on exit. This lets ``scripts/03_hidden.py`` and the ablation harness drive
the existing ``src/sae_hidden`` stage ``run()`` functions verbatim under a different
dict_size / k / steps or a different output directory (standard vs augmented, or a
per-preset ablation dir).

# ponytail: swaps module-global config singletons; single-threaded, restored in finally.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import config

# The hidden I/O attributes on config.paths that define where a Path A run reads
# its embeddings from and writes its models/results. Kept in sync with
# PathsConfig.__post_init__ (src/config.py).
_HIDDEN_PATH_ATTRS = (
    "hidden_embeddings_dir",
    "hidden_visual_embeddings_path",
    "hidden_train_embeddings_path",
    "hidden_test_embeddings_path",
    "hidden_visual_image_ids_path",
    "hidden_train_image_ids_path",
    "hidden_test_image_ids_path",
    "hidden_models_dir",
    "hidden_results_dir",
)


def _hidden_paths_under(embeddings_dir: Path) -> dict[str, Path]:
    """Recompute the 768-d embedding + id paths under a base dir (mirrors PathsConfig)."""
    return {
        "hidden_embeddings_dir": embeddings_dir,
        "hidden_visual_embeddings_path": embeddings_dir / "visual_embeddings_768.pt",
        "hidden_train_embeddings_path": embeddings_dir / "train_embeddings_768.pt",
        "hidden_test_embeddings_path": embeddings_dir / "test_embeddings_768.pt",
        "hidden_visual_image_ids_path": embeddings_dir / "visual_image_ids.json",
        "hidden_train_image_ids_path": embeddings_dir / "train_image_ids.json",
        "hidden_test_image_ids_path": embeddings_dir / "test_image_ids.json",
    }


@contextmanager
def hidden_variant(
    *,
    sae_hidden=None,
    embeddings_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> Iterator[None]:
    """Temporarily override Path A config + hidden I/O paths for a variant run.

    Args:
        sae_hidden: a SAEHiddenConfig (build with ``dataclasses.replace``) to swap in.
            ``None`` keeps the default ``config.sae_hidden``.
        embeddings_dir: when set, recompute all 7 hidden embedding/id paths under it
            (use for augmented extraction). ``None`` leaves them (ablations reuse the
            cached standard_hidden tensors).
        models_dir / results_dir: override the model/results base dirs.
    """
    saved_sae = config.sae_hidden
    saved_paths = {k: getattr(config.paths, k) for k in _HIDDEN_PATH_ATTRS}
    try:
        if sae_hidden is not None:
            config.sae_hidden = sae_hidden
        if embeddings_dir is not None:
            for k, v in _hidden_paths_under(Path(embeddings_dir)).items():
                setattr(config.paths, k, v)
        if models_dir is not None:
            config.paths.hidden_models_dir = Path(models_dir)
        if results_dir is not None:
            config.paths.hidden_results_dir = Path(results_dir)
        yield
    finally:
        config.sae_hidden = saved_sae
        for k, v in saved_paths.items():
            setattr(config.paths, k, v)
