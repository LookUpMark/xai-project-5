"""Shared utilities for the SAE concept-discovery pipeline."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from config import VLMConfig


def load_vlm(config: VLMConfig):
    """Load BiomedCLIP model and processor.

    Args:
        config (VLMConfig): dataclass containing parameters.

    Returns:
        tuple: (model, processor) loaded.
    """
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.processor_name,
        trust_remote_code=True,
    )

    model.eval().to(config.device)

    return model, processor


def set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # F-015: full op-level determinism when the workspace env is configured (CUDA
    # requires CUBLAS_WORKSPACE_CONFIG); warn_only avoids hard errors on MPS/CPU.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def repro_info(input_paths: list[tuple[str, Path]]) -> list[str]:
    """Git SHA + package versions + sha256 of input files — F-010 reproducibility block.

    Inputs are gitignored and notebook-regenerated, so recording their sha256 lets a
    future run verify the committed outputs came from identical inputs.
    """
    import hashlib
    import subprocess

    import numpy
    import sklearn

    lines: list[str] = []
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
        lines.append(f"- git commit: `{sha or 'unknown'}`")
    except Exception:
        lines.append("- git commit: unknown")
    lines.append(
        f"- versions: scikit-learn {sklearn.__version__} | "
        f"torch {torch.__version__} | numpy {numpy.__version__}"
    )
    for label, path in input_paths:
        path = Path(path)
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            lines.append(f"- sha256({label}) [{path.name}]: `{digest}`")
        else:
            lines.append(f"- sha256({label}): <missing>")
    return lines


def load_tensor(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Safely load a tensor with weights_only=True.

    Args:
        path: Path to the .pt file.
        device: Target device for map_location.

    Returns:
        Loaded tensor.
    """
    return torch.load(path, map_location=device, weights_only=True)


def load_state_dict(path: str | Path, device: str = "cpu") -> dict:
    """Safely load a model state dict with weights_only=True.

    Args:
        path: Path to the .pt file containing a state dict.
        device: Target device for map_location.

    Returns:
        Loaded state dict (OrderedDict).
    """
    return torch.load(path, map_location=device, weights_only=True)


def ensure_dir(path: Path) -> None:
    """Create parent directories if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure logging with a standard format and return a logger.

    Args:
        name: Logger name (typically __name__ from caller).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


logger = setup_logging(__name__)


def dataclass_to_dict(obj) -> dict:
    """Convert a frozen/regular dataclass to a plain dict (shallow).

    Args:
        obj: A dataclass instance.

    Returns:
        Dict with field names as keys.
    """
    # ponytail: deliberately SHALLOW — unlike dataclasses.asdict, this does not
    # recurse into nested dataclasses/containers and so preserves tuple fields
    # (e.g. SAEHiddenConfig.match_thresholds = (0.7, 0.9)) as tuples. Callers
    # JSON-serialise the result, where the distinction is moot, but keeping it
    # shallow avoids surprising recursion if a config gains a nested field.
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _split_ids(
    train_idx,
    test_idx,
    source_ids_path: Path | None,
    train_ids_path: Path | None,
    test_ids_path: Path | None,
) -> None:
    """Split a sidecar image-id list with precomputed indices and write JSON files.

    No-op when ``source_ids_path`` is None. ``train_idx``/``test_idx`` MUST be
    the exact permutation used to split the embeddings tensor so the id list
    stays row-aligned with it. All three id paths must be provided together.

    Args:
        train_idx: Train indices (same array used to slice the embeddings).
        test_idx: Test indices.
        source_ids_path: Source sidecar JSON (ordered list of image ids).
        train_ids_path: Destination for the train split of ids.
        test_ids_path: Destination for the test split of ids.
    """
    if source_ids_path is None:
        return
    if train_ids_path is None or test_ids_path is None:
        raise ValueError(
            "source_ids_path requires both train_ids_path and test_ids_path"
        )
    with open(source_ids_path) as f:
        image_ids = json.load(f)
    expected = len(train_idx) + len(test_idx)
    if len(image_ids) != expected:
        raise ValueError(
            f"image-id count ({len(image_ids)}) != train+test rows ({expected}); "
            f"the sidecar is not aligned with the embeddings tensor."
        )
    train_ids = [image_ids[i] for i in train_idx]
    test_ids = [image_ids[i] for i in test_idx]
    train_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_ids_path, "w") as f:
        json.dump(train_ids, f)
    with open(test_ids_path, "w") as f:
        json.dump(test_ids, f)


def split_embeddings(
    source_path: Path,
    train_path: Path,
    test_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    group_key_fn: Callable[[str], str] | None = None,
    source_ids_path: Path | None = None,
    train_ids_path: Path | None = None,
    test_ids_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create and save a train/test split from an embeddings tensor.

    Args:
        source_path: Path to the source embeddings .pt file (N x D).
        train_path: Destination path for train embeddings.
        test_path: Destination path for test embeddings.
        train_ratio: Fraction of *groups* (with a sidecar) or *samples* (without)
            assigned to train. Defaults to 0.8.
        seed: Random seed for reproducible splitting. Defaults to 42.
        group_key_fn: Optional function mapping an image-id basename to a group
            key, so all views / augmented copies of one exam land in one
            partition (no patient/study leakage). When None (or no sidecar), each
            image is its own group => a random, sidecar-aligned split (e.g.
            ROCOv2's independent figures). Callers pass the active dataset's
            group-key via ``DatasetSpec.make_group_key_fn``.
        source_ids_path: Optional source sidecar JSON of image ids. Treated as
            absent when the file does not exist.
        train_ids_path: Optional destination for the train split of ids.
        test_ids_path: Optional destination for the test split of ids.

    Returns:
        Tuple of (train_embeddings, test_embeddings) tensors.
    """
    from sklearn.model_selection import train_test_split as _sklearn_split

    # A missing sidecar file is equivalent to "no grouping" so callers can pass
    # the path unconditionally and still degrade gracefully (e.g. mock runs).
    if source_ids_path is not None and not Path(source_ids_path).exists():
        source_ids_path = None

    embeddings = load_tensor(source_path)
    if embeddings.dim() != 2:
        raise ValueError(
            f"Expected 2D embeddings tensor, got shape {embeddings.shape}"
        )

    if source_ids_path is not None:
        with open(source_ids_path, "r", encoding="utf-8") as f:
            image_ids = json.load(f)

        if len(image_ids) != len(embeddings):
            raise ValueError("ID sidecar length does not match embeddings length")

        # Group by the dataset's group key. ``group_key_fn=None`` (or identity)
        # => each image its own group => a random split, but still sidecar-aligned
        # (augmented copies share a basename, so they stay in one partition).
        group_fn = group_key_fn if group_key_fn is not None else (lambda name: name)
        group_keys = [group_fn(img_id) for img_id in image_ids]
        unique_groups = sorted(set(group_keys))
        train_groups, test_groups = _sklearn_split(
            unique_groups,
            train_size=train_ratio,
            random_state=seed,
        )

        train_set = set(train_groups)
        train_idx = [i for i, key in enumerate(group_keys) if key in train_set]
        test_idx = [i for i, key in enumerate(group_keys) if key not in train_set]

        # Safety net: no group may straddle both partitions.
        train_keys = {group_keys[i] for i in train_idx}
        test_keys = {group_keys[i] for i in test_idx}
        assert not (train_keys & test_keys), "group leakage across train/test"

        n_total = len(embeddings)
        logger.info(
            f"Group-aware split: {len(train_idx)}/{len(test_idx)} samples "
            f"({len(train_idx) / n_total:.1%}/{len(test_idx) / n_total:.1%}) "
            f"across {len(train_groups)}/{len(test_groups)} of {len(unique_groups)} "
            f"groups (group overlap = 0)."
        )
    else:
        indices = np.arange(len(embeddings))
        train_idx, test_idx = _sklearn_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
        )
        logger.info(
            f"Random split (no image-id sidecar): {len(train_idx)}/{len(test_idx)} "
            f"samples."
        )

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_emb, train_path)
    torch.save(test_emb, test_path)

    _split_ids(train_idx, test_idx, source_ids_path, train_ids_path, test_ids_path)

    return train_emb, test_emb


# RadLex preferred-label loading moved to vocabulary_building/radlex_support.py
# (preferred_labels); load_radlex_terms was removed to avoid a duplicate CSV read.
