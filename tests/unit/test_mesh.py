"""test_mesh.py — Tests for the MeSH vocabulary builder (P3.5, XML parser).

Mocks encode_texts / centroids / save (no BiomedCLIP, no writes) + exercises
load_and_filter_mesh on a mock MeSH **XML** file (tree-branch filtering), plus
the gzipped production path (``desc<year>.gz``).
"""

import gzip
from unittest.mock import MagicMock

from config import VLMConfig, VocabularyConfig
from vocabulary_building.mesh_support import load_and_filter_mesh
from vocabulary_building.mesh_vocab import build_mesh_vocabulary

# A tiny MeSH DescriptorRecordSet: one record per branch (A/C/E/D), plus a record
# with no tree numbers (excluded). Mirrors the real XML structure
# (<DescriptorRecordSet><DescriptorRecord><DescriptorName><String>... +
# <TreeNumberList><TreeNumber>...).
_MESH_SAMPLE = """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
<DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000001</DescriptorUI>
  <DescriptorName><String>Lung</String></DescriptorName>
  <TreeNumberList><TreeNumber>A04.411</TreeNumber></TreeNumberList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000002</DescriptorUI>
  <DescriptorName><String>Pneumonia</String></DescriptorName>
  <TreeNumberList><TreeNumber>C08.381</TreeNumber></TreeNumberList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000003</DescriptorUI>
  <DescriptorName><String>Aspirin</String></DescriptorName>
  <TreeNumberList><TreeNumber>D03.4</TreeNumber></TreeNumberList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000004</DescriptorUI>
  <DescriptorName><String>Computed Tomography</String></DescriptorName>
  <TreeNumberList><TreeNumber>E01.370</TreeNumber></TreeNumberList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000005</DescriptorUI>
  <DescriptorName><String>No Tree</String></DescriptorName>
</DescriptorRecord>
</DescriptorRecordSet>
"""


def _write_xml(tmp_path, name, text=_MESH_SAMPLE):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestLoadAndFilterMesh:
    def test_keeps_radiology_branches_excludes_drugs(self, tmp_path):
        mesh_file = _write_xml(tmp_path, "d.xml")
        terms = load_and_filter_mesh(mesh_file)  # default A/C/E
        assert "Lung" in terms                   # A
        assert "Pneumonia" in terms              # C
        assert "Computed Tomography" in terms    # E
        assert "Aspirin" not in terms            # D excluded
        assert "No Tree" not in terms            # no MN excluded

    def test_custom_categories(self, tmp_path):
        mesh_file = _write_xml(tmp_path, "d.xml")
        terms = load_and_filter_mesh(mesh_file, categories=("A",))  # anatomy only
        assert "Lung" in terms
        assert "Pneumonia" not in terms            # C excluded now
        assert "Computed Tomography" not in terms  # E excluded now

    def test_dedup(self, tmp_path):
        # Two records with the same name (different-case, different trees) -> deduped.
        mesh_file = _write_xml(
            tmp_path,
            "d.xml",
            """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
<DescriptorRecord><DescriptorName><String>Heart</String></DescriptorName>
  <TreeNumberList><TreeNumber>A07</TreeNumber></TreeNumberList></DescriptorRecord>
<DescriptorRecord><DescriptorName><String>heart</String></DescriptorName>
  <TreeNumberList><TreeNumber>C14</TreeNumber></TreeNumberList></DescriptorRecord>
</DescriptorRecordSet>
""",
        )
        terms = load_and_filter_mesh(mesh_file)
        assert terms.count("Heart") == 1  # case-insensitive dedup keeps first

    def test_parses_gzipped_xml(self, tmp_path):
        # Production path: desc<year>.gz. Gzip a real XML payload and confirm the
        # opener transparently gunzips it.
        gz = tmp_path / "desc2026.gz"
        with gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(_MESH_SAMPLE)
        terms = load_and_filter_mesh(gz)
        assert "Lung" in terms
        assert "Aspirin" not in terms

    def test_multi_tree_record_kept_if_any_matches(self, tmp_path):
        # A record spanning a kept (C) and an excluded (D) branch is kept.
        mesh_file = _write_xml(
            tmp_path,
            "d.xml",
            """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
<DescriptorRecord><DescriptorName><String>Iodine</String></DescriptorName>
  <TreeNumberList><TreeNumber>D03.4</TreeNumber><TreeNumber>C01.001</TreeNumber>
  </TreeNumberList></DescriptorRecord>
</DescriptorRecordSet>
""",
        )
        assert load_and_filter_mesh(mesh_file) == ["Iodine"]


class TestBuildMeshVocabulary:
    def test_builds_topk_with_mesh_source(self, monkeypatch, tmp_path):
        mesh_file = _write_xml(
            tmp_path,
            "d.xml",
            """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
<DescriptorRecord><DescriptorName><String>Lung</String></DescriptorName>
  <TreeNumberList><TreeNumber>A04</TreeNumber></TreeNumberList></DescriptorRecord>
<DescriptorRecord><DescriptorName><String>Heart</String></DescriptorName>
  <TreeNumberList><TreeNumber>A07</TreeNumber></TreeNumberList></DescriptorRecord>
</DescriptorRecordSet>
""",
        )

        captured = {}

        def fake_encode(terms, *a, **k):
            import torch
            captured["terms"] = list(terms)
            return torch.zeros((len(terms), 512))

        monkeypatch.setattr("vocabulary_building.mesh_vocab.encode_texts", fake_encode)
        monkeypatch.setattr(
            "vocabulary_building.mesh_vocab.compute_anchor_centroids", lambda *a, **k: MagicMock()
        )
        monkeypatch.setattr(
            "vocabulary_building.mesh_vocab.rank_terms_by_relevance",
            lambda terms, embs, cents: [(t, 0.5) for t in terms],
        )
        monkeypatch.setattr("vocabulary_building.mesh_vocab.save_vocabulary", lambda *a, **k: None)
        monkeypatch.setattr(
            "vocabulary_building.mesh_vocab.save_vocab_embeddings", lambda *a, **k: None
        )

        vocab = build_mesh_vocabulary(
            mesh_file, MagicMock(), MagicMock(),
            VLMConfig(device="cpu"), VocabularyConfig(), top_k=1,
        )
        assert len(vocab) == 1
        assert vocab[0]["source"] == "mesh"
        assert vocab[0]["term"] in ("Lung", "Heart")
        assert captured["terms"] == ["Lung", "Heart"]  # load_and_filter_mesh output

    def test_empty_mesh_raises(self, tmp_path):
        mesh_file = _write_xml(
            tmp_path,
            "d.xml",
            """<?xml version="1.0"?>
<DescriptorRecordSet LanguageCode="eng">
<DescriptorRecord><DescriptorName><String>Aspirin</String></DescriptorName>
  <TreeNumberList><TreeNumber>D03</TreeNumber></TreeNumberList></DescriptorRecord>
</DescriptorRecordSet>
""",
        )
        # D excluded -> no terms -> ValueError.
        import pytest
        with pytest.raises(ValueError):
            build_mesh_vocabulary(
                mesh_file, MagicMock(), MagicMock(),
                VLMConfig(device="cpu"), VocabularyConfig(),
            )
