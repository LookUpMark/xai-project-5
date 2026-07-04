# xai_datasets/padchest.py
"""PadChest (BIMCV) dataset adapter — chest X-rays + Spanish radiology reports.

PadChest (>160k chest X-rays, hospital San Juan, Alicante) is staged by
``download_padchest.py`` into ``data/padchest/{images,PADCHEST_..._*.csv}``.
License: BIMCV terms forbid redistribution — the data is gitignored.

Schema notes (verified against ``PADCHEST_chest_x_ray_images_labels_160K_*.csv``):
  - ``ImageID`` is the full PNG filename **including** ``.png`` (matches the
    embedding sidecar basename, so the join is direct — no 2-hop bridge like
    IU X-Ray's ``indiana_projections.csv``).
  - ``PatientID`` is a **separate column**, NOT the filename prefix → the
    anti-leakage group-key needs a CSV lookup (a closure), unlike IU X-Ray's
    pure filename parse.
  - ``Report`` is free-text Spanish (note: in the available CSV the reports
    appear systematically word-truncated — flag for the judge, P2.5).
  - ``labelCUIS`` carries UMLS CUIs (space-separated inside a list repr) — a
    bonus asset, unused by the core pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# Columns in PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
_IMAGEID_COL = "ImageID"     # full filename incl ".png"
_PATIENTID_COL = "PatientID"
_REPORT_COL = "Report"


class PadChestImageDataset(Dataset):
    """PNG radiographs from PadChest.

    Returns ``(image, image_path)``; the embedding extractor keeps only the
    basename (``Path(p).name``, e.g. ``..._rarh4r.png``) as the row id, which
    matches the CSV ``ImageID`` column (also includes ``.png``).
    """

    def __init__(self, image_dir: Path, image_ext: str = "*.png"):
        self.image_paths = sorted(Path(image_dir).glob(image_ext))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, str(img_path)


def _load_padchest_reports(
    csv_path: Path, image_dir: Path | None = None
) -> dict[str, str]:
    """Load ``{ImageID: Report}`` from the PadChest labels CSV.

    Args:
        csv_path: Path to ``PADCHEST_chest_x_ray_images_labels_160K_*.csv``.
        image_dir: If given, keep only rows whose PNG exists on disk (a partial
            download then yields a self-consistent subset). ``None`` => all rows.

    Returns:
        ``{ImageID: report_text}`` (ImageID includes the ``.png`` suffix).
    """
    on_disk: set[str] | None = None
    if image_dir is not None:
        on_disk = {p.name for p in Path(image_dir).glob("*.png")}

    out: dict[str, str] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iid = row.get(_IMAGEID_COL)
            if not iid:
                continue
            if on_disk is not None and iid not in on_disk:
                continue
            out[iid] = (row.get(_REPORT_COL) or "").strip()
    return out


class PadChestTextDataset(Dataset):
    """Spanish free-text reports from the PadChest CSV.

    Yields ``(report_text, ImageID)`` per row, filtered to images present on disk
    when ``image_dir`` is given (so a partial download is self-consistent).
    """

    def __init__(self, csv_path: Path, image_dir: Path | None = None) -> None:
        reports = _load_padchest_reports(csv_path, image_dir)
        # (report_text, ImageID) — text FIRST, matching IUXrayTextDataset's
        # (text, uid) contract so extract_text_embeddings unpacks (text, _).
        # Sorted by ImageID for reproducibility.
        self._items: list[tuple[str, str]] = [
            (report, iid) for iid, report in sorted(reports.items())
        ]
        if not self._items:
            raise FileNotFoundError(
                f"No PadChest reports loaded from {csv_path} "
                f"(image_dir={image_dir}). Check the CSV / image download."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self._items[idx]  # (report_text, ImageID)


def make_padchest_group_key(csv_path) -> "Callable[[str], str]":
    """Build the anti-leakage group-key for PadChest: ``ImageID -> PatientID``.

    ``PatientID`` is a separate CSV column (NOT the filename prefix), so the
    group-key needs the CSV lookup (a closure), unlike IU X-Ray's pure filename
    parse. The train/test split then keeps every image of one patient in one
    partition — no patient leakage.

    Args:
        csv_path: Path to the PadChest labels CSV.

    Returns:
        A function ``group_key(image_id) -> patient_id``; unknown ids fall back
        to themselves (singleton group) rather than crashing the split.
    """
    id_to_patient: dict[str, str] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iid = row.get(_IMAGEID_COL)
            pid = row.get(_PATIENTID_COL)
            if iid and pid:
                id_to_patient[iid] = pid

    def group_key(image_id: str) -> str:
        return id_to_patient.get(image_id, image_id)

    return group_key


# Spanish chest-radiology judge prompt. Verdict labels stay English
# (Aligned/Unaligned/Uncertain) so the parser (evaluate_llm_judge._extract_verdict)
# is shared across datasets. NOTE: the available PadChest reports are anonymised
# and systematically word-truncated (see docs/FINDINGS.md A2), so the prompt tells
# the judge to infer the intended medical term. ``{report}``/``{pseudo_report}``
# are filled per pair.
PADCHEST_JUDGE_PROMPT = """Eres un evaluador clínico de IA especializado en radiología.

Dado un informe radiológico y un pseudo-informe generado por IA basado en conceptos descubiertos, determina si el informe original respalda los hallazgos del pseudo-informe. Centra tu evaluación principalmente en el **concepto dominante** mencionado al final del pseudo-informe.

Nota importante: los informes pueden contener palabras abreviadas o truncadas (p. ej. "neumoni" por "neumonía", "izquierd" por "izquierdo", "hallazg" por "hallazgo"). Infiere el término médico pretendido a partir del contexto antes de juzgar.

Reglas:
- SOPORTA (Aligned): el informe original menciona o implica explícitamente los hallazgos/conceptos del pseudo-informe.
- CONTRADICE (Unaligned):
    1. El informe original niega explícitamente estos conceptos.
    2. O el pseudo-informe menciona una patología/anomalía (p. ej., neumonía, masa, fractura) y el informe original NO la menciona. En radiología, las patologías no mencionadas se asumen ausentes.
- AMBIGUO (Uncertain): el pseudo-informe menciona estructuras anatómicas normales o artefactos (p. ej., costillas, columna, dispositivos) que pueden estar en la imagen pero que el radiólogo no menciona por ser normales o irrelevantes.

Ejemplos:

Informe radiológico: "Aumento de la opacidad en lóbulo superior derecho con atelectasia asociada."
Pseudo-informe generado por IA: "El modelo identifica los siguientes conceptos visuales en esta radiografía: linfonodo broncopulmonar, sonda flexible, signo del pico gemelo. El concepto dominante es 'sonda flexible' (activación=0.161)."
Formato de respuesta: El informe habla de opacidades pulmonares, pero el concepto dominante 'sonda flexible' es un artefacto no relacionado. | Verdict: Uncertain

Informe radiológico: "Silueta cardíaca y mediastino de tamaño normal. Pulmones claros. Sin enfermedad aguda."
Pseudo-informe generado por IA: "El modelo identifica los siguientes conceptos visuales en esta radiografía: cardiomegalia, platisma, reborde delgado. El concepto dominante es 'cardiomegalia' (activación=0.180)."
Formato de respuesta: El informe indica corazón de tamaño normal, lo que contradice explícitamente cardiomegalia. | Verdict: Unaligned

Informe radiológico: "Aumento de la opacidad en lóbulo superior derecho con posible masa."
Pseudo-informe generado por IA: "El modelo identifica los siguientes conceptos visuales en esta radiografía: lesión de masa, signo de la banda grasa, signo de la órbita desnuda. El concepto dominante es 'lesión de masa' (activación=0.150)."
Formato de respuesta: El informe menciona explícitamente una posible masa, lo que se alinea con el concepto dominante. | Verdict: Aligned

Ahora evalúa lo siguiente:

Informe radiológico:
"{report}"

Pseudo-informe generado por IA:
"{pseudo_report}"

Formato de respuesta: <explicación de máximo 25 palabras> | Verdict: <Aligned/Unaligned/Uncertain>"""
