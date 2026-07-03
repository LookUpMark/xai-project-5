"""spec.py — DatasetSpec: one description per dataset, consumed by every stage.

The pipeline historically hard-coded IU X-Ray assumptions (image/report paths,
the study-key split grouping, the judge prompt and its two-hop report lookup).
:class:`DatasetSpec` collects everything a dataset brings into a single frozen
description, so each stage reads from the active spec instead. Adding a dataset
then means adding a spec (see ``DATASETS``), not patching half a dozen modules.

The active dataset is ``config.active_dataset.name``; resolve it with
:func:`get_dataset`. Only IU X-Ray is registered in Phase 0 — PadChest (Phase 2)
and ROCOv2 (Phase 3) are added by their respective phases.

Field population by phase:
  - Phase 0 (this file): name, language, domain, image_dataset_cls,
    text_dataset_cls, image_dir, text_source, make_group_key_fn.
  - Phase 2/3: judge_prompt, build_report_lookup (wired when
    ``evaluate_llm_judge.py`` is refactored to read from the spec).
  - Phase 3: vocab_config (RadLex chest vs UMLS multimodal dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from xai_datasets.iu_xray import (
    IUXrayImageDataset,
    IUXrayTextDataset,
    study_key_from_basename,
)

# Maps an image-id basename to a group key so the train/test split keeps every
# view / augmented copy of one exam in one partition. ``None`` => random split.
GroupKeyFn = Optional[Callable[[str], str]]
# Zero-arg factory: materializes the group-key fn (may read a CSV at call time,
# e.g. PadChest's ImageID->PatientID lookup). Returns None for ungrouped data.
GroupKeyFactory = Callable[[], GroupKeyFn]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset, consumed by every pipeline stage."""

    name: str
    language: str
    domain: str

    # Extraction stage (notebooks/vlm/extract_embeddings.ipynb).
    image_dataset_cls: type
    text_dataset_cls: type
    image_dir: Path
    text_source: Path  # reports dir (IU/PadChest) or captions CSV (ROCOv2)

    # Split stage: anti-leakage grouping factory; None => random split.
    make_group_key_fn: GroupKeyFactory

    # LLM-judge stage (Phase 2/3 placeholders — no behavior until the judge is
    # rewired to read from the spec):
    judge_prompt: str = ""
    build_report_lookup: Callable[[], dict] = field(default=lambda: {})


# ---------------------------------------------------------------------------
# IU X-Ray (chest X-ray, English) — the existing reference dataset.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

IU_XRAY_SPEC = DatasetSpec(
    name="iu_xray",
    language="en",
    domain="chest_xray",
    image_dataset_cls=IUXrayImageDataset,
    text_dataset_cls=IUXrayTextDataset,
    image_dir=_REPO_ROOT / "data" / "iu_xray" / "images" / "images_normalized",
    text_source=_REPO_ROOT / "data" / "iu_xray" / "reports",
    make_group_key_fn=lambda: study_key_from_basename,
    # judge_prompt / build_report_lookup: populated in Phase 2/3 when
    # evaluate_llm_judge.py is refactored to consume the spec.
)


# Central registry — the single place new datasets are declared.
DATASETS: dict[str, DatasetSpec] = {
    IU_XRAY_SPEC.name: IU_XRAY_SPEC,
}


def get_dataset(name: str) -> DatasetSpec:
    """Resolve a dataset spec by name.

    Args:
        name: dataset key in ``DATASETS`` (e.g. ``"iu_xray"``).

    Returns:
        The matching :class:`DatasetSpec`.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered datasets: {sorted(DATASETS)}."
        ) from None
