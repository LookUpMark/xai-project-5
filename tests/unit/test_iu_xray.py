"""test_iu_xray.py — Tests for the IU X-Ray dataset module + its DatasetSpec.

Covers the study-key parser (IU X-Ray filename convention) and the DatasetSpec
wiring that Phase 0 of the multi-dataset refactor introduces. The generic
group-aware split is tested separately in ``test_split.py``.
"""

from pathlib import Path

import pytest

from xai_datasets.iu_xray import (
    IUXrayImageDataset,
    IUXrayTextDataset,
    study_key_from_basename,
)
from xai_datasets.spec import DATASETS, IU_XRAY_SPEC, get_dataset


class TestStudyKeyFromBasename:
    def test_well_formed_names(self):
        assert study_key_from_basename("1_IM-0001-4001.dcm.png") == "1_IM-0001"
        assert study_key_from_basename("3222_IM-1522-2001.dcm.png") == "3222_IM-1522"
        assert study_key_from_basename("2_IM-0652-1001.dcm.png") == "2_IM-0652"

    def test_views_of_same_study_collapse(self):
        frontal = study_key_from_basename("1_IM-0001-4001.dcm.png")
        lateral = study_key_from_basename("1_IM-0001-3001.dcm.png")
        assert frontal == lateral == "1_IM-0001"

    def test_malformed_returns_input_unchanged(self):
        for bad in ["no_marker_here.png", "plain", ""]:
            assert study_key_from_basename(bad) == bad


class TestIUXraySpec:
    def test_registered_and_resolvable(self):
        assert DATASETS["iu_xray"] is IU_XRAY_SPEC
        assert get_dataset("iu_xray") is IU_XRAY_SPEC

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            get_dataset("does_not_exist")

    def test_fields_match_iu_xray(self):
        assert IU_XRAY_SPEC.name == "iu_xray"
        assert IU_XRAY_SPEC.language == "en"
        assert IU_XRAY_SPEC.domain == "chest_xray"
        assert IU_XRAY_SPEC.image_dataset_cls is IUXrayImageDataset
        assert IU_XRAY_SPEC.text_dataset_cls is IUXrayTextDataset
        assert isinstance(IU_XRAY_SPEC.image_dir, Path)
        assert isinstance(IU_XRAY_SPEC.text_source, Path)

    def test_group_key_factory_returns_study_key_fn(self):
        # The factory materializes the pure study-key fn (no setup needed for IU).
        group_fn = IU_XRAY_SPEC.make_group_key_fn()
        assert group_fn is study_key_from_basename
        assert group_fn("1_IM-0001-4001.dcm.png") == "1_IM-0001"
