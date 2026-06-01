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

    def __init__(self, reports_dir: Path) -> None:
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