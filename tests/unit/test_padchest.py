"""test_padchest.py — Tests for the PadChest adapter.

Mock-CSV + tiny-PNG based; no real data or GPU required. Exercises the report
loader, the image/text Datasets, and the PatientID group-key closure, plus the
PADCHEST_SPEC registration. The spec's factory callables (make_group_key_fn /
build_report_lookup) point at the real 99 MB CSV and are NOT invoked here.
"""

import csv

import pytest
from PIL import Image

from xai_datasets.padchest import (
    PadChestImageDataset,
    PadChestTextDataset,
    _load_padchest_reports,
    make_padchest_group_key,
)
from xai_datasets.spec import DATASETS, PADCHEST_SPEC, get_dataset

_COLS = ["ImageID", "PatientID", "Report"]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_png(path):
    Image.new("RGB", (4, 4)).save(path)


class TestLoadReports:
    def test_loads_all_when_no_image_dir(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            {"ImageID": "a.png", "PatientID": "1", "Report": "foo"},
            {"ImageID": "b.png", "PatientID": "2", "Report": "bar"},
        ])
        reports = _load_padchest_reports(csv_path)
        assert reports == {"a.png": "foo", "b.png": "bar"}

    def test_filters_to_on_disk_images(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            {"ImageID": "a.png", "PatientID": "1", "Report": "foo"},
            {"ImageID": "b.png", "PatientID": "2", "Report": "bar"},
        ])
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_png(img_dir / "a.png")  # only a.png present on disk
        reports = _load_padchest_reports(csv_path, img_dir)
        assert reports == {"a.png": "foo"}


class TestPadChestTextDataset:
    def test_yields_report_and_imageid_sorted(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            {"ImageID": "b.png", "PatientID": "2", "Report": "bar"},
            {"ImageID": "a.png", "PatientID": "1", "Report": "foo"},
        ])
        ds = PadChestTextDataset(csv_path)
        assert len(ds) == 2
        assert ds[0] == ("foo", "a.png")  # ImageID-sorted
        assert ds[1] == ("bar", "b.png")

    def test_empty_csv_raises(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [])  # header only
        with pytest.raises(FileNotFoundError):
            PadChestTextDataset(csv_path)


class TestPadChestImageDataset:
    def test_loads_pngs_sorted_basename(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_png(img_dir / "z.png")
        _make_png(img_dir / "a.png")
        ds = PadChestImageDataset(img_dir)
        assert len(ds) == 2
        img, path = ds[0]
        from pathlib import Path
        assert Path(path).name == "a.png"  # sorted
        assert img.mode == "RGB"


class TestPadChestGroupKey:
    def test_groups_by_patientid_column_not_filename(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [
            {"ImageID": "x_1.png", "PatientID": "P9", "Report": "a"},
            {"ImageID": "x_2.png", "PatientID": "P9", "Report": "b"},
            {"ImageID": "y_1.png", "PatientID": "P7", "Report": "c"},
        ])
        gk = make_padchest_group_key(csv_path)
        # PatientID is a column, NOT the filename prefix -> both x_* map to P9.
        assert gk("x_1.png") == "P9"
        assert gk("x_2.png") == "P9"
        assert gk("y_1.png") == "P7"
        assert gk("unknown.png") == "unknown.png"  # fallback => singleton group


class TestPadChestSpec:
    def test_registered_and_resolvable(self):
        assert DATASETS["padchest"] is PADCHEST_SPEC
        assert get_dataset("padchest") is PADCHEST_SPEC

    def test_fields(self):
        assert PADCHEST_SPEC.name == "padchest"
        assert PADCHEST_SPEC.language == "es"
        assert PADCHEST_SPEC.domain == "chest_xray"
        assert PADCHEST_SPEC.image_dataset_cls is PadChestImageDataset
        assert PADCHEST_SPEC.text_dataset_cls is PadChestTextDataset
