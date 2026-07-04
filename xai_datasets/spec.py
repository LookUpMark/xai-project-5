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
  - Phase 3: vocab_source (RadLex chest vs external MeSH multimodal dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from xai_datasets.iu_xray import (
    IU_XRAY_JUDGE_PROMPT,
    IUXrayImageDataset,
    IUXrayTextDataset,
    build_iu_xray_report_lookup,
    study_key_from_basename,
)
from xai_datasets.padchest import (
    PADCHEST_JUDGE_PROMPT,
    PadChestImageDataset,
    PadChestTextDataset,
    _load_padchest_reports,
    make_padchest_group_key,
)
from xai_datasets.rocov2 import (
    ROCOImageDataset,
    ROCOCaptionDataset,
    ROCOV2_JUDGE_PROMPT,
    _load_rocov2_captions,
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

    # Vocabulary stage (Phase 3): which external lexicon to build the naming
    # vocabulary from. ``vocab_source="radlex"`` (default, chest) or ``"mesh"``
    # (ROCOv2 — free MeSH lexicon, independent of the dataset's labels).
    vocab_source: str = "radlex"
    mesh_file: Optional[Path] = None  # MeSH XML (desc<year>.gz) for vocab_source="mesh"


# ---------------------------------------------------------------------------
# IU X-Ray (chest X-ray, English) — the existing reference dataset.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_IU_XRAY_DIR = _REPO_ROOT / "data" / "iu_xray"

IU_XRAY_SPEC = DatasetSpec(
    name="iu_xray",
    language="en",
    domain="chest_xray",
    image_dataset_cls=IUXrayImageDataset,
    text_dataset_cls=IUXrayTextDataset,
    image_dir=_IU_XRAY_DIR / "images" / "images_normalized",
    # text_source is the iu_xray ROOT: indiana_reports.csv + indiana_projections.csv
    # live here (the reports/ subdir is empty); IUXrayTextDataset reads
    # <text_source>/indiana_reports.csv and the 2-hop lookup reads both CSVs.
    text_source=_IU_XRAY_DIR,
    make_group_key_fn=lambda: study_key_from_basename,
    judge_prompt=IU_XRAY_JUDGE_PROMPT,
    build_report_lookup=lambda: build_iu_xray_report_lookup(_IU_XRAY_DIR),
)


# ---------------------------------------------------------------------------
# PadChest (chest X-ray, Spanish) — Phase 2 scale-test dataset.
# ---------------------------------------------------------------------------
_PADCHEST_DIR = _REPO_ROOT / "data" / "padchest"
_PADCHEST_CSV = _PADCHEST_DIR / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
_PADCHEST_IMAGES = _PADCHEST_DIR / "images"

PADCHEST_SPEC = DatasetSpec(
    name="padchest",
    language="es",
    domain="chest_xray",
    image_dataset_cls=PadChestImageDataset,
    text_dataset_cls=PadChestTextDataset,
    image_dir=_PADCHEST_IMAGES,
    text_source=_PADCHEST_CSV,
    # PatientID is a CSV column (not the filename prefix) -> closure lookup.
    make_group_key_fn=lambda: make_padchest_group_key(_PADCHEST_CSV),
    # Direct ImageID -> Report join (ImageID carries ".png", like the sidecar).
    build_report_lookup=lambda: _load_padchest_reports(_PADCHEST_CSV, _PADCHEST_IMAGES),
    judge_prompt=PADCHEST_JUDGE_PROMPT,
)


# ---------------------------------------------------------------------------
# ROCOv2 (multimodal radiology, English captions) — Phase 3 generalization.
# ---------------------------------------------------------------------------
_ROCOV2_DIR = _REPO_ROOT / "data" / "rocov2"
_ROCOV2_CAPTIONS = _ROCOV2_DIR / "captions.csv"
_ROCOV2_IMAGES = _ROCOV2_DIR / "images"

ROCOV2_SPEC = DatasetSpec(
    name="rocov2",
    language="en",
    domain="multimodal_radiology",
    image_dataset_cls=ROCOImageDataset,
    text_dataset_cls=ROCOCaptionDataset,
    image_dir=_ROCOV2_IMAGES,
    text_source=_ROCOV2_CAPTIONS,
    # Independent figures (no patient/study) -> no grouping -> random split.
    make_group_key_fn=lambda: None,
    # Caption is the judge's textual evidence (figure caption, not a report).
    build_report_lookup=lambda: _load_rocov2_captions(_ROCOV2_CAPTIONS, _ROCOV2_IMAGES),
    judge_prompt=ROCOV2_JUDGE_PROMPT,
    # External MeSH lexicon (free, no UMLS license), independent of ROCOv2 labels.
    # XML descriptor file (gzipped) downloaded by xai_datasets/download_mesh.py.
    vocab_source="mesh",
    mesh_file=_REPO_ROOT / "data" / "mesh" / "desc2026.gz",
)


# Central registry — the single place new datasets are declared.
DATASETS: dict[str, DatasetSpec] = {
    IU_XRAY_SPEC.name: IU_XRAY_SPEC,
    PADCHEST_SPEC.name: PADCHEST_SPEC,
    ROCOV2_SPEC.name: ROCOV2_SPEC,
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
