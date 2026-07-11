"""mesh_support.py — Load + filter MeSH to a radiology-relevant term set.

MeSH (Medical Subject Headings, NLM) is a **free, license-free** controlled
vocabulary — the most-used medical thesaurus after UMLS itself, and a source
*within* UMLS. Unlike UMLS it carries **no CUIs**, so it is used here purely as an
**EXTERNAL naming lexicon** for ROCOv2: independent of the dataset's own labels,
which fixes the circularity the (dropped) UMLS-from-CUIs vocabulary introduced
(see ``docs/FINDINGS.md``).

NLM **discontinued the ASCII ``d<year>.bin`` serialization in January 2026**; the
current production distribution is the MeSH **XML** descriptor file
(``desc<year>.xml`` / gzipped ``desc<year>.gz``), downloaded by
``xai_datasets/download_mesh.py`` to ``data/mesh/desc<year>.gz``. This parser
streams that XML (gzip or plain) with :func:`xml.etree.ElementTree.iterparse`,
keeping memory bounded on the ~300 MB uncompressed file.

Filters the MeSH descriptor hierarchy to the radiology-visible branches by
tree-number prefix:

  - ``A*`` Anatomy          (imaged anatomy)
  - ``C*`` Diseases         (visible pathology)
  - ``E*`` Procedures / Equipment (diagnosis, devices)

Drugs/chemicals (D), organisms (B), psychiatry (F), etc. are excluded as not
radiology-visible. Returns preferred descriptor names (deduped, order-preserving).

Run (driven by ``scripts/run_vocab_building_pipeline.py --dataset rocov2``).
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator


def _local(tag: str) -> str:
    """Strip an XML namespace from a tag (``{ns}Name`` -> ``Name``)."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _open_text(path: Path):
    """Open *path* as a text stream, transparently gunzipping ``.gz`` inputs.

    Gzip is detected by ``.gz`` suffix or the ``1f 8b`` magic bytes, so the real
    production file (``desc<year>.gz``) and plain-text XML test fixtures both work.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if path.suffix == ".gz" or magic == b"\x1f\x8b":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _descriptor_name(record) -> str | None:
    """Preferred name of a ``<DescriptorRecord>``: the text of ``DescriptorName/String``."""
    for el in record.iter():
        if _local(el.tag) != "DescriptorName":
            continue
        for child in el.iter():
            if _local(child.tag) == "String":
                return (child.text or "").strip()
    return None


def _tree_numbers(record) -> list[str]:
    """All ``<TreeNumber>`` values under a ``<DescriptorRecord>`` (e.g. ``A04.411``)."""
    out: list[str] = []
    for el in record.iter():
        if _local(el.tag) == "TreeNumber":
            txt = (el.text or "").strip()
            if txt:
                out.append(txt)
    return out


def _iter_descriptor_records(path: Path) -> Iterator[tuple[str | None, list[str]]]:
    """Stream MeSH XML -> ``(descriptor_name, [tree_numbers])`` per record.

    Uses ``iterparse`` with per-record ``clear()`` so the ~300 MB descriptor file
    streams at constant memory. Each record is yielded when its
    ``<DescriptorRecord>`` end-tag fires (children still attached), then cleared.
    """
    fp = _open_text(path)
    try:
        for _event, elem in ET.iterparse(fp, events=("end",)):
            if _local(elem.tag) != "DescriptorRecord":
                continue
            yield _descriptor_name(elem), _tree_numbers(elem)
            elem.clear()  # bound memory: drop the record once consumed
    finally:
        fp.close()


def load_and_filter_mesh(
    mesh_file,
    categories: Iterable[str] = ("A", "C", "E"),
) -> list[str]:
    """Load MeSH descriptors and keep those in the given tree-branch categories.

    Args:
        mesh_file: Path to the MeSH descriptor file (XML ``desc<year>.xml`` or
            gzipped ``desc<year>.gz``).
        categories: Tree-number prefixes to keep (default A=Anatomy, C=Diseases,
            E=Procedures/Equipment — the radiology-visible branches).

    Returns:
        Deduplicated, order-preserving list of preferred descriptor names.
    """
    prefixes = tuple(categories)
    seen: set[str] = set()
    terms: list[str] = []
    for name, trees in _iter_descriptor_records(Path(mesh_file)):
        if not name or not trees:
            continue
        if any(tn.startswith(prefixes) for tn in trees):
            key = name.lower()
            if key not in seen and len(name) > 1:
                seen.add(key)
                terms.append(name)
    return terms
