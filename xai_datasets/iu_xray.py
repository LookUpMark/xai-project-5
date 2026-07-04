# datasets/iu_xray.py
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class IUXrayImageDataset(Dataset):
    """Dataset handling images from IU Xray."""
    def __init__(self, image_dir: Path, image_ext: str = "*.png"):
        self.image_paths = sorted(list(image_dir.glob(image_ext)))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, str(img_path)


class IUXrayTextDataset(Dataset):
    """Dataset handling reports from IU X-Ray (indiana_reports.csv).

    Legge ``indiana_reports.csv`` da *reports_dir* e concatena
    i campi ``findings`` e ``impression`` in un'unica stringa clinica.
    """

    _CSV_NAME = "indiana_reports.csv"
    _FINDINGS_COL = "findings"
    _IMPRESSION_COL = "impression"
    _UID_COL = "uid"

    def __init__(self, reports_dir: Path, image_dir: Path | None = None) -> None:
        # ``image_dir`` is accepted for signature uniformity with
        # ``PadChestTextDataset`` (which filters reports to on-disk images) but
        # ignored here: IU X-Ray loads every report from indiana_reports.csv.
        csv_path = reports_dir / self._CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected {self._CSV_NAME} at {csv_path}. "
                "Run datasets/download_iu_xray.py first."
            )
        self._records: list[dict[str, str]] = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._records.append(dict(row))

    def __len__(self) -> int:
        return len(self._records)

    def _build_text(self, row: dict[str, str]) -> str:
        findings = (row.get(self._FINDINGS_COL) or "").strip()
        impression = (row.get(self._IMPRESSION_COL) or "").strip()
        parts: list[str] = []
        if findings:
            parts.append(f"Findings: {findings}")
        if impression:
            parts.append(f"Impression: {impression}")
        if not parts:
            return "No clinical report available."
        return " ".join(parts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        row = self._records[idx]
        text = self._build_text(row)
        uid = row.get(self._UID_COL, str(idx))
        return text, uid


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


def build_iu_xray_report_lookup(iu_xray_dir) -> dict:
    """Build ``{image_filename: "Findings ... Impression ..."}`` for IU X-Ray.

    IU X-Ray splits the report across two CSVs: ``indiana_projections.csv``
    maps ``filename → uid`` and ``indiana_reports.csv`` maps ``uid → findings``
    + ``impression``. This 2-hop bridge is IU-specific (PadChest joins directly).
    Both CSVs live at the ``iu_xray_dir`` root (NOT under ``reports/``).

    Args:
        iu_xray_dir: Path to ``data/iu_xray/`` (holding both indiana CSVs).

    Returns:
        ``{filename: combined_report_text}`` keyed by the image-id basename used
        in the explanations sidecar (e.g. ``"3222_IM-1522-2001.dcm.png"``).
    """
    iu_xray_dir = Path(iu_xray_dir)
    reports_csv = iu_xray_dir / "indiana_reports.csv"
    projections_csv = iu_xray_dir / "indiana_projections.csv"

    uid_to_report: dict[str, str] = {}
    with reports_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            parts = [p for p in (findings, impression) if p]
            if parts:
                uid_to_report[str(row.get("uid"))] = " ".join(parts)

    lookup: dict[str, str] = {}
    with projections_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            filename = row.get("filename")
            uid = str(row.get("uid") or "")
            if filename and uid in uid_to_report:
                lookup[filename] = uid_to_report[uid]
    return lookup


# English chest-radiology judge prompt (the LLM-judge protocol). Verdict labels
# stay English so the parser (evaluate_llm_judge._extract_verdict) is shared
# across datasets. ``{report}`` and ``{pseudo_report}`` are filled per pair.
IU_XRAY_JUDGE_PROMPT = """You are a clinical AI evaluator specializing in radiology.

Given a radiology report and an AI-generated pseudo-report based on discovered concepts,
determine whether the original report supports the findings in the pseudo-report.
Focus your evaluation primarily on the **dominant concept** mentioned at the end of the pseudo-report.

Rules:
- SUPPORTS (Aligned): The original report explicitly mentions or implies the findings/concepts in the pseudo-report.
- CONTRADICTS (Unaligned):
    1. The original report explicitly denies these concepts.
    2. OR the pseudo-report mentions a pathology/abnormality (e.g., pneumonia, mass, fracture) and the original report does NOT mention it. In radiology, unmentioned pathologies are assumed absent.
- AMBIGUOUS (Uncertain): The pseudo-report mentions normal anatomical structures or artifacts (e.g., ribs, spine, devices) that might be in the image but are simply not mentioned by the radiologist because they are normal or irrelevant.

Examples:

Radiology report: "There is an increased opacity in the right upper lobe with associated atelectasis."
AI-generated pseudo-report: "The model identifies the following visual concepts in this radiograph: bronchopulmonary lymph node, flexible Spule, twin peak sign. The dominant concept is 'flexible Spule' (activation=0.161)."
Answer format: The report discusses lung opacities, but the dominant concept flexible Spule (coil) is a completely unrelated artifact. | Verdict: Uncertain

Radiology report: "The heart is top normal in size. The lungs are clear. No acute disease."
AI-generated pseudo-report: "The model identifies the following visual concepts in this radiograph: cardiomegaly, strap muscle of neck, thin rim. The dominant concept is 'cardiomegaly' (activation=0.180)."
Answer format: The report states the heart is normal size, which explicitly contradicts cardiomegaly (enlarged heart). | Verdict: Unaligned

Radiology report: "There is an increased opacity in the right upper lobe with possible mass."
AI-generated pseudo-report: "The model identifies the following visual concepts in this radiograph: mass lesion, navicular fat stripe sign, bare orbit sign. The dominant concept is 'mass lesion' (activation=0.150)."
Answer format: The report explicitly mentions a possible mass, which aligns with the dominant concept of a mass lesion. | Verdict: Aligned

Now evaluate the following:

Radiology report:
"{report}"

AI-generated pseudo-report:
"{pseudo_report}"

Answer format: <max 25 words explanation> | Verdict: <Aligned/Unaligned/Uncertain>"""