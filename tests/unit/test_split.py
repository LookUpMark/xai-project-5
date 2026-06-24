"""
test_split.py — Tests for the group-aware train/test split.

  - study-level grouping so no radiograph study straddles train and test,
  - basename-derived study key that collapses multi-view / augmented copies,
  - the split is always recomputed and stale files are overwritten rather than reused,
  - deterministic partitioning (same source + seed -> same split).

No GPU or real model weights required; everything runs on synthetic tensors.
"""

import json

import torch

from utils import split_embeddings, study_key_from_basename


def _study_ids(n_studies, views_per_study=2, start=1):
    """Build image-id basenames: ``n_studies`` studies, each with N views.

    Each study ``s`` yields views ``{s}_IM-{s:04d}-{v:04d}.dcm.png`` so its study
    key (``{s}_IM-{s:04d}``) is shared across all views.
    """
    ids = []
    for s in range(start, start + n_studies):
        for v in range(1, views_per_study + 1):
            ids.append(f"{s}_IM-{s:04d}-{v:04d}.dcm.png")
    return ids


def _write_source(tmp_path, image_ids, dim=4, seed=0):
    """Write a synthetic source embeddings tensor + image-id sidecar."""
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(len(image_ids), dim, generator=g)
    src = tmp_path / "visual_embeddings.pt"
    sidecar = tmp_path / "visual_image_ids.json"
    torch.save(embeddings, src)
    with open(sidecar, "w") as f:
        json.dump(image_ids, f)
    return src, sidecar, embeddings


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


class TestGroupAwareSplit:
    def _run(self, tmp_path, ids):
        src, sidecar, _ = _write_source(tmp_path, ids)
        out = tmp_path / "out"
        out.mkdir()
        split_embeddings(
            source_path=src,
            train_path=out / "train.pt",
            test_path=out / "test.pt",
            train_ratio=0.8,
            seed=42,
            source_ids_path=sidecar,
            train_ids_path=out / "train_ids.json",
            test_ids_path=out / "test_ids.json",
        )
        with open(out / "train_ids.json") as f:
            train_ids = json.load(f)
        with open(out / "test_ids.json") as f:
            test_ids = json.load(f)
        return train_ids, test_ids

    def test_no_study_overlaps_partition(self, tmp_path):
        train_ids, test_ids = self._run(tmp_path, _study_ids(10, views_per_study=2))
        train_keys = {study_key_from_basename(i) for i in train_ids}
        test_keys = {study_key_from_basename(i) for i in test_ids}

        assert len(test_ids) > 0
        assert not (train_keys & test_keys), "a study straddles train and test"

    def test_all_views_of_a_study_stay_together(self, tmp_path):
        ids = _study_ids(10, views_per_study=3)
        train_ids, test_ids = self._run(tmp_path, ids)

        by_study = {}
        for i in ids:
            by_study.setdefault(study_key_from_basename(i), []).append(i)
        for study, members in by_study.items():
            in_train = all(m in train_ids for m in members)
            in_test = all(m in test_ids for m in members)
            assert in_train or in_test, f"study {study} split across partitions"

    def test_sidecar_stays_row_aligned(self, tmp_path):
        ids = _study_ids(10, views_per_study=2)
        train_ids, test_ids = self._run(tmp_path, ids)

        assert len(train_ids) + len(test_ids) == len(ids)
        assert set(train_ids) | set(test_ids) == set(ids)
        assert not (set(train_ids) & set(test_ids))  # no id duplicated

    def test_augmented_copies_stay_together(self, tmp_path):
        # 1 original + 2 augmented copies share a basename (same study key).
        copies = ["5_IM-0005-1001.dcm.png"] * 3
        ids = copies + _study_ids(10, views_per_study=1, start=10)
        train_ids, test_ids = self._run(tmp_path, ids)

        in_train = sum(1 for x in train_ids if x == "5_IM-0005-1001.dcm.png")
        in_test = sum(1 for x in test_ids if x == "5_IM-0005-1001.dcm.png")
        assert in_train + in_test == 3
        assert in_train == 0 or in_test == 0, "augmented copies split across partitions"


class TestNoCacheAlwaysRecomputed:
    def test_stale_files_are_overwritten(self, tmp_path):
        ids = _study_ids(10, views_per_study=2)
        src, sidecar, _ = _write_source(tmp_path, ids)
        train_p = tmp_path / "train.pt"
        test_p = tmp_path / "test.pt"
        # Pre-write STALE split files with sentinel content a fresh split cannot
        # produce. If the skip-if-exists cache were still present, these would be
        # returned unchanged.
        torch.save(torch.full((1, 4), -999.0), train_p)
        torch.save(torch.full((1, 4), -999.0), test_p)

        train_emb, test_emb = split_embeddings(
            source_path=src,
            train_path=train_p,
            test_path=test_p,
            train_ratio=0.8,
            seed=42,
            source_ids_path=sidecar,
            train_ids_path=tmp_path / "train_ids.json",
            test_ids_path=tmp_path / "test_ids.json",
        )

        assert train_emb.shape[0] > 1
        assert not torch.all(train_emb == -999.0)
        assert not torch.all(test_emb == -999.0)

    def test_deterministic_same_seed(self, tmp_path):
        ids = _study_ids(12, views_per_study=2)
        src, sidecar, _ = _write_source(tmp_path, ids)
        a_train, _ = split_embeddings(
            source_path=src,
            train_path=tmp_path / "a_tr.pt",
            test_path=tmp_path / "a_te.pt",
            train_ratio=0.8,
            seed=42,
            source_ids_path=sidecar,
            train_ids_path=tmp_path / "a_ti.json",
            test_ids_path=tmp_path / "a_te_ids.json",
        )
        b_train, _ = split_embeddings(
            source_path=src,
            train_path=tmp_path / "b_tr.pt",
            test_path=tmp_path / "b_te.pt",
            train_ratio=0.8,
            seed=42,
            source_ids_path=sidecar,
            train_ids_path=tmp_path / "b_ti.json",
            test_ids_path=tmp_path / "b_te_ids.json",
        )
        assert torch.equal(a_train, b_train)


class TestPrepareSplitDelegates:
    def test_calls_split_embeddings_with_config(self, tmp_path, monkeypatch):
        import config
        import utils
        from autoencoder.train_sae import prepare_split

        # Point the source at an existing tmp file so the FileNotFoundError guard passes.
        src = tmp_path / "visual_embeddings.pt"
        torch.save(torch.randn(4, 4), src)
        monkeypatch.setattr(config.paths, "visual_embeddings_path", src)

        captured = {}

        def fake_split(**kwargs):
            captured.update(kwargs)
            return torch.empty(0), torch.empty(0)

        monkeypatch.setattr(utils, "split_embeddings", fake_split)

        prepare_split()

        assert captured["source_path"] == src
        assert captured["train_path"] == config.paths.train_embeddings_path
        assert captured["test_path"] == config.paths.test_embeddings_path
        assert captured["train_ratio"] == config.training.train_split_ratio
        assert captured["seed"] == config.training.split_seed
        assert captured["source_ids_path"] == config.paths.visual_image_ids_path
