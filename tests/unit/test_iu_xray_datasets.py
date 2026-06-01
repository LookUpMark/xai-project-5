"""
tests/unit/test_iu_xray_datasets.py — Unit tests for IUXrayImageDataset e IUXrayTextDataset.

Tutti i test girano senza dati reali: usa tmp_path per creare
fixture sintetiche (PNG stub + indiana_reports.csv).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

# datasets/ è nella root del progetto, non in src/ — aggiunge al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "datasets"))

from iu_xray import IUXrayImageDataset, IUXrayTextDataset


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_png(path: Path) -> None:
    """Crea un PNG RGB 4×4 minimo."""
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


def _make_reports_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Scrive indiana_reports.csv con i campi minimi attesi da IUXrayTextDataset."""
    fieldnames = ["uid", "findings", "impression", "tags", "image"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# IUXrayImageDataset
# ---------------------------------------------------------------------------

class TestIUXrayImageDataset:
    def test_len_matches_png_count(self, tmp_path: Path) -> None:
        for i in range(3):
            _make_png(tmp_path / f"img_{i:03d}.png")
        ds = IUXrayImageDataset(tmp_path)
        assert len(ds) == 3

    def test_empty_dir_gives_len_zero(self, tmp_path: Path) -> None:
        ds = IUXrayImageDataset(tmp_path)
        assert len(ds) == 0

    def test_getitem_returns_rgb_image_and_path(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "scan.png")
        ds = IUXrayImageDataset(tmp_path)
        img, path_str = ds[0]
        assert img.mode == "RGB"
        assert path_str.endswith("scan.png")

    def test_ignores_non_png_files(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "real.png")
        (tmp_path / "notes.txt").write_text("ignore me")
        ds = IUXrayImageDataset(tmp_path)
        assert len(ds) == 1

    def test_paths_are_sorted(self, tmp_path: Path) -> None:
        for name in ["c.png", "a.png", "b.png"]:
            _make_png(tmp_path / name)
        ds = IUXrayImageDataset(tmp_path)
        names = [Path(ds[i][1]).name for i in range(len(ds))]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# IUXrayTextDataset
# ---------------------------------------------------------------------------

class TestIUXrayTextDataset:
    def _make_dir(self, tmp_path: Path, rows: list[dict[str, str]]) -> Path:
        _make_reports_csv(tmp_path / "indiana_reports.csv", rows)
        return tmp_path

    def test_len_matches_csv_rows(self, tmp_path: Path) -> None:
        rows = [
            {"uid": "1", "findings": "normal lungs", "impression": "no acute disease", "tags": "", "image": ""},
            {"uid": "2", "findings": "cardiomegaly", "impression": "enlarged heart", "tags": "", "image": ""},
        ]
        ds = IUXrayTextDataset(self._make_dir(tmp_path, rows))
        assert len(ds) == 2

    def test_getitem_returns_text_and_uid(self, tmp_path: Path) -> None:
        rows = [{"uid": "42", "findings": "clear lungs", "impression": "normal", "tags": "", "image": ""}]
        ds = IUXrayTextDataset(self._make_dir(tmp_path, rows))
        text, uid = ds[0]
        assert uid == "42"
        assert "Findings:" in text
        assert "Impression:" in text

    def test_findings_and_impression_concatenated(self, tmp_path: Path) -> None:
        rows = [{"uid": "1", "findings": "bilateral infiltrates", "impression": "pneumonia likely", "tags": "", "image": ""}]
        ds = IUXrayTextDataset(self._make_dir(tmp_path, rows))
        text, _ = ds[0]
        assert "bilateral infiltrates" in text
        assert "pneumonia likely" in text

    def test_empty_findings_uses_impression_only(self, tmp_path: Path) -> None:
        rows = [{"uid": "1", "findings": "", "impression": "normal study", "tags": "", "image": ""}]
        ds = IUXrayTextDataset(self._make_dir(tmp_path, rows))
        text, _ = ds[0]
        assert "normal study" in text
        assert "Findings:" not in text

    def test_both_empty_returns_fallback(self, tmp_path: Path) -> None:
        rows = [{"uid": "1", "findings": "", "impression": "", "tags": "", "image": ""}]
        ds = IUXrayTextDataset(self._make_dir(tmp_path, rows))
        text, _ = ds[0]
        assert text == "No clinical report available."

    def test_missing_csv_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="indiana_reports.csv"):
            IUXrayTextDataset(tmp_path)

    def test_empty_csv_gives_len_zero(self, tmp_path: Path) -> None:
        ds = IUXrayTextDataset(self._make_dir(tmp_path, []))
        assert len(ds) == 0
