"""test_rocov2.py — Tests for the ROCOv2 adapter (P3.1, post-MeSH-pivot).

Mock-CSV + tiny-JPEG based; no real data or GPU. Exercises caption loading, the
image/caption Datasets, and the ROCOv2 spec wiring (now MeSH vocab, no CUIs).
"""

import csv
from pathlib import Path

import pytest
from PIL import Image

from xai_datasets.rocov2 import (
    ROCOImageDataset,
    ROCOCaptionDataset,
    ROCOV2_JUDGE_PROMPT,
    _load_rocov2_captions,
)
from xai_datasets.spec import DATASETS, ROCOV2_SPEC, get_dataset


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)


def _make_jpg(path: Path) -> None:
    Image.new("RGB", (4, 4)).save(path, format="JPEG")


class TestLoadCaptions:
    def test_skips_header_and_loads(self, tmp_path):
        cap_csv = tmp_path / "captions.csv"
        _write_rows(cap_csv, [
            ["image", "caption"],                       # header -> skipped
            ["a.jpg", "Head CT of parotiditis."],
            ["b.jpg", "Chest X-ray, cardiomegaly."],
        ])
        caps = _load_rocov2_captions(cap_csv)
        assert caps == {"a.jpg": "Head CT of parotiditis.", "b.jpg": "Chest X-ray, cardiomegaly."}

    def test_filters_to_on_disk(self, tmp_path):
        cap_csv = tmp_path / "captions.csv"
        _write_rows(cap_csv, [["a.jpg", "x"], ["b.jpg", "y"]])
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_jpg(img_dir / "a.jpg")  # only a.jpg on disk
        caps = _load_rocov2_captions(cap_csv, img_dir)
        assert caps == {"a.jpg": "x"}

    def test_matches_extensionless_ids_by_stem(self, tmp_path):
        """Real ROCOv2 format: caption IDs ship WITHOUT extension
        (``ROCOv2_2023_train_000001``), disk images WITH (``.jpg``). The loader
        must match by stem and key the result by the on-disk filename."""
        cap_csv = tmp_path / "captions.csv"
        _write_rows(cap_csv, [
            ["ID", "Caption"],  # header -> dropped (no matching stem on disk)
            ["ROCOv2_2023_train_000001", "Head CT of parotiditis."],
            ["ROCOv2_2023_train_000002", "Chest X-ray, cardiomegaly."],
            ["ROCOv2_2023_train_999999", "No matching image on disk."],
        ])
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_jpg(img_dir / "ROCOv2_2023_train_000001.jpg")
        _make_jpg(img_dir / "ROCOv2_2023_train_000002.jpg")
        caps = _load_rocov2_captions(cap_csv, img_dir)
        assert caps == {
            "ROCOv2_2023_train_000001.jpg": "Head CT of parotiditis.",
            "ROCOv2_2023_train_000002.jpg": "Chest X-ray, cardiomegaly.",
        }


class TestROCOCaptionDataset:
    def test_yields_caption_then_imagename_sorted(self, tmp_path):
        cap_csv = tmp_path / "captions.csv"
        _write_rows(cap_csv, [["b.jpg", "y"], ["a.jpg", "x"]])
        ds = ROCOCaptionDataset(cap_csv)
        assert len(ds) == 2
        assert ds[0] == ("x", "a.jpg")  # image_name-sorted; caption FIRST
        assert ds[1] == ("y", "b.jpg")

    def test_empty_raises(self, tmp_path):
        cap_csv = tmp_path / "captions.csv"
        _write_rows(cap_csv, [["image", "caption"]])  # header only
        with pytest.raises(FileNotFoundError):
            ROCOCaptionDataset(cap_csv)


class TestROCOImageDataset:
    def test_rglobs_jpgs_sorted(self, tmp_path):
        img_dir = tmp_path / "images"
        (img_dir / "train").mkdir(parents=True)
        (img_dir / "test").mkdir(parents=True)
        _make_jpg(img_dir / "train" / "z.jpg")
        _make_jpg(img_dir / "test" / "a.jpg")
        ds = ROCOImageDataset(img_dir)
        assert len(ds) == 2  # rglob across subdirs
        assert Path(ds[0][1]).name == "a.jpg"  # sorted


class TestROCOv2Spec:
    def test_registered_and_resolvable(self):
        assert DATASETS["rocov2"] is ROCOV2_SPEC
        assert get_dataset("rocov2") is ROCOV2_SPEC

    def test_fields(self):
        assert ROCOV2_SPEC.language == "en"
        assert ROCOV2_SPEC.domain == "multimodal_radiology"
        assert ROCOV2_SPEC.image_dataset_cls is ROCOImageDataset
        assert ROCOV2_SPEC.text_dataset_cls is ROCOCaptionDataset
        # Multimodal: external MeSH lexicon (no CUIs — CUI-matching was dropped).
        assert ROCOV2_SPEC.vocab_source == "mesh"
        assert ROCOV2_SPEC.mesh_file is not None
        # No grouping -> random split.
        assert ROCOV2_SPEC.make_group_key_fn() is None
        # Caption judge prompt is non-empty + has the placeholders.
        assert "{report}" in ROCOV2_SPEC.judge_prompt
        assert "{pseudo_report}" in ROCOV2_SPEC.judge_prompt
        assert ROCOV2_JUDGE_PROMPT  # the module constant is non-empty
