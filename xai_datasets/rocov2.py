# xai_datasets/rocov2.py
"""ROCOv2 (Radiology Objects in COntext v2) dataset adapter — multimodal radiology.

ROCOv2 (~80k radiological figures + English captions; PMC Open Access, CC BY/CC
BY-NC) is the Phase 3 multimodal dataset. Unlike the chest-only datasets, it
spans all imaging modalities and body regions, so the concept space is broader
and the naming vocabulary is built from an **external MeSH lexicon**
(``src/vocabulary_building/mesh_vocab.py``), independent of the dataset's labels.

Staging layout (after download_rocov2.py)::
    data/rocov2/images/                       *.jpg (flattened across train/valid/test;
                                              filenames like ROCOv2_2023_train_NNNNNN.jpg)
    data/rocov2/captions.csv                  image_name, caption (English)

Schema notes (Rückert et al., Sci. Data 2024):
  - ``captions.csv`` is a 2-column (filename, caption) file with no guaranteed
    header → the reader skips rows whose first cell isn't an image filename.
  - Images are independent figures (no patient/study grouping) → group-key is
    None ⇒ a random, sidecar-aligned train/test split.

Note: ROCOv2 ships per-image UMLS CUIs, but we do NOT use them — building the
vocabulary from them would be circular with any CUI-based evaluation, and the
free UMLS CUI crosswalk is unavailable (license). The caption judge + an external
MeSH lexicon keep naming and evaluation independent. See ``docs/FINDINGS.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _looks_like_image(cell: str) -> bool:
    return cell.lower().endswith(_IMG_EXTS)


# ---------------------------------------------------------------------------
# Captions (the judge ground truth for ROCOv2 — figure captions, not reports)
# ---------------------------------------------------------------------------

def _load_rocov2_captions(captions_csv: Path, image_dir: Path | None = None) -> dict[str, str]:
    """Load ``{image_name: caption}`` from the 2-column captions CSV.

    Args:
        captions_csv: Path to ``captions.csv`` (filename, caption).
        image_dir: If given, keep only rows whose image exists on disk (a partial
            download then yields a self-consistent subset). ``None`` => all rows.

    Returns:
        ``{image_name: caption}`` keyed by the basename (incl. extension) used in
        the embedding sidecar.
    """
    on_disk: set[str] | None = None
    if image_dir is not None:
        on_disk = set()
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            on_disk |= {p.name for p in Path(image_dir).rglob(ext)}

    out: dict[str, str] = {}
    with Path(captions_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            name, caption = row[0].strip(), row[1].strip()
            if not _looks_like_image(name):
                continue  # header / spurious row
            if on_disk is not None and name not in on_disk:
                continue
            out[name] = caption
    return out


class ROCOImageDataset(Dataset):
    """JPEG radiological figures from ROCOv2 (rglobbed, so train/valid/test subdirs OK).

    Returns ``(image, image_path)``; the extractor keeps the basename
    (``Path(p).name``, e.g. ``ROCOv2_2023_train_000001.jpg``) as the row id, which
    matches the ``image_name`` column of ``captions.csv``.
    """

    def __init__(self, image_dir: Path, image_ext: str = "*.jpg"):
        # rglob so a preserved train/valid/test/ layout still works.
        self.image_paths = sorted(Path(image_dir).rglob(image_ext))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, str(img_path)


class ROCOCaptionDataset(Dataset):
    """English figure captions from ROCOv2 (the judge's textual ground truth).

    Yields ``(caption, image_name)`` per row, filtered to images present on disk
    when ``image_dir`` is given.
    """

    def __init__(self, captions_csv: Path, image_dir: Path | None = None) -> None:
        captions = _load_rocov2_captions(captions_csv, image_dir)
        self._items: list[tuple[str, str]] = [
            (cap, name) for name, cap in sorted(captions.items())
        ]
        if not self._items:
            raise FileNotFoundError(
                f"No ROCOv2 captions loaded from {captions_csv} "
                f"(image_dir={image_dir}). Check the captions CSV / image download."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self._items[idx]  # (caption, image_name)


# English caption-based judge prompt. ROCOv2's textual evidence is a figure
# CAPTION (editorial, single sentence), not a clinical report — so the prompt
# reframes the rules: "Unaligned" means not supported by the caption text, not
# a clinical contradiction. Verdict labels stay English (shared parser).
ROCOV2_JUDGE_PROMPT = """You are a clinical AI evaluator specializing in radiology.

You are given a figure caption from a biomedical publication and an AI-generated
pseudo-report based on concepts discovered by a model from that figure. Determine
whether the caption supports the pseudo-report. Focus primarily on the **dominant
concept** mentioned at the end of the pseudo-report.

Important: the caption is a short figure description (not a full radiology report),
so it often mentions only the main finding. Absence from the caption does NOT imply
the finding is clinically absent — only that it is not supported by the caption text.

Rules:
- SUPPORTS (Aligned): the caption explicitly mentions or implies the findings/concepts in the pseudo-report.
- NOT SUPPORTED (Unaligned): the caption does not mention the dominant concept AND the dominant concept is a concrete pathology/abnormality (e.g., mass, effusion, fracture) the caption would plausibly have noted.
- AMBIGUOUS (Uncertain): the dominant concept is a normal structure, modality descriptor, or anatomical label that may be in the figure but is simply not mentioned in the caption.

Now evaluate the following:

Figure caption:
"{report}"

AI-generated pseudo-report:
"{pseudo_report}"

Answer format: <max 25 words explanation> | Verdict: <Aligned/Unaligned/Uncertain>"""
