"""Shared utilities for the SAE concept-discovery pipeline."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
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
    """Convert a frozen/regular dataclass to a plain dict.

    Args:
        obj: A dataclass instance.

    Returns:
        Dict with field names as keys.
    """
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


def study_key_from_basename(name: str) -> str:
    """Derive the radiograph-study group key from an image-id basename.

    IU X-Ray filenames follow ``{patient}_IM-{study}-{view}.dcm.png``, where the
    same study (patient + exam) is captured across multiple views (frontal /
    lateral). Grouping the train/test split on the study key keeps every view of
    one exam in a single partition, preventing patient/study leakage.

    Unrecognised names (no ``_IM-`` marker) are returned unchanged so they become
    singleton groups rather than crashing the split.

    Args:
        name: Image id as stored in the sidecar (a PNG basename).

    Returns:
        ``"{patient}_IM-{study}"`` for well-formed names, else ``name`` verbatim.
    """
    marker = "_IM-"
    idx = name.find(marker)
    if idx < 0:
        return name
    patient = name[:idx]
    rest = name[idx + len(marker):]
    study = rest.split("-", 1)[0]
    return f"{patient}_IM-{study}"


def split_embeddings(
    source_path: Path,
    train_path: Path,
    test_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
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

        # Group by study key
        study_keys = [study_key_from_basename(img_id) for img_id in image_ids]
        unique_groups = sorted(set(study_keys))
        train_groups, test_groups = _sklearn_split(
            unique_groups,
            train_size=train_ratio,
            random_state=seed,
        )

        train_set = set(train_groups)
        train_idx = [i for i, key in enumerate(study_keys) if key in train_set]
        test_idx = [i for i, key in enumerate(study_keys) if key not in train_set]

        # Safety net: no study may straddle both partitions.
        train_keys = {study_keys[i] for i in train_idx}
        test_keys = {study_keys[i] for i in test_idx}
        assert not (train_keys & test_keys), "study leakage across train/test"

        n_total = len(embeddings)
        logger.info(
            f"Group-aware split: {len(train_idx)}/{len(test_idx)} samples "
            f"({len(train_idx) / n_total:.1%}/{len(test_idx) / n_total:.1%}) "
            f"across {len(train_groups)}/{len(test_groups)} of {len(unique_groups)} "
            f"studies (group overlap = 0)."
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



def load_radlex_terms(csv_path: str) -> list[str]:
    """
    Load RadLex CSV and return a deduplicated list of non-obsolete
    preferred labels.

    Args:
        csv_path (str): path to the RadLex CSV file.

    Returns:
        List[str]: cleaned RadLex terms.
    """
    import csv

    terms = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip obsolete
            if row.get("Obsolete", "").strip().upper() == "TRUE":
                continue

            label = row.get("Preferred Label")
            if label and label.strip():
                terms.append(label.strip())

    # Deduplicate while preserving order, drop very short labels
    seen = set()
    unique_terms = []
    for t in terms:
        t_lower = t.lower()
        if t_lower not in seen and len(t) > 1:
            seen.add(t_lower)
            unique_terms.append(t)

    print(f"Loaded {len(unique_terms)} unique non-obsolete RadLex terms "
          f"(from {len(terms)} raw rows).")
    return unique_terms
