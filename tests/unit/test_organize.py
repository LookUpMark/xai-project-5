"""Unit tests for concept_discovery.organize."""
from __future__ import annotations

import sys
sys.path.insert(0, "src/")

import torch
from concept_discovery.organize import ConceptSet, ImageConcepts, from_spliece_explanations
from concept_discovery.organize import from_sae_explanations


class TestDataclasses:
    def test_concept_set_constructs(self):
        cs = ConceptSet(
            names=["a", "b"],
            embeddings=torch.randn(2, 512),
            name_to_idx={"a": 0, "b": 1},
            per_image=[ImageConcepts(image_id="x", activations={"a": 1.0})],
        )
        assert cs.names == ["a", "b"]
        assert cs.embeddings.shape == (2, 512)


class TestSpliCEAdapter:
    def _vocab(self):
        terms = [{"term": f"term_{i}"} for i in range(10)]
        emb = torch.eye(10, 512)  # 10 distinct atoms
        return terms, emb

    def test_builds_concept_set_from_explanations(self):
        terms, emb = self._vocab()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 0.5},
                {"feature_id": 7, "name": "term_7", "activation": 0.2},
            ],
            "pseudo_report": "x",
        }]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert set(cs.names) == {"term_3", "term_7"}
        assert cs.embeddings.shape == (2, 512)
        # rows match the vocab rows of the named terms
        assert torch.equal(cs.embeddings[cs.name_to_idx["term_3"]], emb[3])
        assert cs.per_image[0].image_id == "img0"
        assert cs.per_image[0].activations == {"term_3": 0.5, "term_7": 0.2}

    def test_drops_nonpositive_and_unresolved_names(self):
        terms, emb = self._vocab()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 1, "name": "term_1", "activation": 0.0},   # dropped (<=0)
                {"feature_id": 2, "name": "term_2", "activation": -1.0},  # dropped (<0)
                {"feature_id": 9, "name": "ghost", "activation": 0.9},    # dropped (not in vocab)
                {"feature_id": 4, "name": "term_4", "activation": 0.4},
            ],
        }]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert cs.names == ["term_4"]
        assert cs.per_image[0].activations == {"term_4": 0.4}

    def test_vocab_emb_count_mismatch_raises(self):
        terms = [{"term": f"t{i}"} for i in range(5)]
        emb = torch.randn(3, 512)  # mismatch
        import pytest
        with pytest.raises(ValueError, match="vocab_emb"):
            from_spliece_explanations([], terms, emb)

    def test_missing_image_id_falls_back_to_index(self):
        terms, emb = self._vocab()
        explanations = [{"top_k_concepts": [{"feature_id": 0, "name": "term_0", "activation": 1.0}]}]
        cs = from_spliece_explanations(explanations, terms, emb)
        assert cs.per_image[0].image_id == "img_0"


class TestSAEAdapter:
    def _inputs(self):
        terms = [{"term": f"term_{i}"} for i in range(10)]
        emb = torch.eye(10, 512)
        concept_names = {
            "3": {"name": "term_3", "score": 0.4, "is_dead": False},
            "7": {"name": "term_7", "score": 0.3, "is_dead": False},
            "9": {"name": "DEAD_FEATURE", "score": 0.0, "is_dead": True},
        }
        return terms, emb, concept_names

    def test_excludes_dead_features(self):
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.5},   # live
                {"feature_id": 9, "name": "DEAD_FEATURE", "activation": 9.9},  # dead -> dropped
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]
        assert cs.per_image[0].activations == {"term_3": 1.5}

    def test_fid_key_type_coercion(self):
        """concept_names keys are str; explanations feature_id is int -> coerce."""
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 7, "name": "term_7", "activation": 2.0},
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_7"]

    def test_unresolved_name_skipped(self):
        terms, emb, concept_names = self._inputs()
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.0},
                {"feature_id": 5, "name": "ghost", "activation": 5.0},  # not in vocab
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]

    def test_missing_concept_names_keeps_all_named(self):
        """If concept_names is empty, no feature is known-dead -> keep all resolvable."""
        terms, emb, _ = self._inputs()
        concept_names = {}
        explanations = [{
            "image_id": "img0",
            "top_k_concepts": [
                {"feature_id": 3, "name": "term_3", "activation": 1.0},
            ],
        }]
        cs = from_sae_explanations(explanations, concept_names, terms, emb)
        assert cs.names == ["term_3"]


from concept_discovery.organize import cluster_concepts


def _cs(names_embs, per_image=None):
    names, embs = names_embs
    return ConceptSet(
        names=names,
        embeddings=torch.tensor(embs, dtype=torch.float32),
        name_to_idx={n: i for i, n in enumerate(names)},
        per_image=per_image or [],
    )


class TestCluster:
    def test_two_well_separated_groups(self):
        # two clusters in 2D (padded to 512): {a,b} near (1,0), {c,d} near (0,1)
        z1 = [1.0] + [0.0] * 511      # a, b
        z2 = [0.0, 1.0] + [0.0] * 510  # c, d
        cs = _cs((["a", "b", "c", "d"], [z1, z1, z2, z2]))
        clusters = cluster_concepts(cs, n_clusters=2)
        assert len(clusters) == 2
        ids = {frozenset(c.members) for c in clusters}
        assert frozenset({"a", "b"}) in ids
        assert frozenset({"c", "d"}) in ids
        for c in clusters:
            assert c.medoid in c.members

    def test_deterministic_same_input_same_ids(self):
        import random
        rows = [[float(random.random())] + [0.0] * 511 for _ in range(8)]
        names = [f"n{i}" for i in range(8)]
        cs = _cs((names, rows))
        c1 = cluster_concepts(cs, n_clusters=3)
        c2 = cluster_concepts(cs, n_clusters=3)
        assert [c.members for c in c1] == [c.members for c in c2]
        assert [c.cluster_id for c in c1] == [0, 1, 2]

    def test_singleton_input(self):
        cs = _cs((["only"], [[0.0] * 512]))
        clusters = cluster_concepts(cs, n_clusters=1)
        assert len(clusters) == 1
        assert clusters[0].members == ["only"]
        assert clusters[0].medoid == "only"

    def test_empty_input(self):
        cs = ConceptSet(names=[], embeddings=torch.empty((0, 512)), name_to_idx={}, per_image=[])
        clusters = cluster_concepts(cs)
        assert clusters == []

    def test_distance_threshold_mode(self):
        z1 = [1.0] + [0.0] * 511
        z2 = [0.0, 1.0] + [0.0] * 510
        cs = _cs((["a", "b", "c"], [z1, z1, z2]))
        clusters = cluster_concepts(cs, distance_threshold=0.5, linkage="average")
        member_sets = {frozenset(c.members) for c in clusters}
        assert frozenset({"a", "b"}) in member_sets


from concept_discovery.organize import ancestor_rids, annotate_radlex, Cluster
from vocabulary_building.radlex_support import RadLexGraph


def _graph():
    """Tiny DAG: root <- mid <- leaf ; root <- sibling."""
    g = RadLexGraph()
    g.rid_to_label = {"RID0": "root", "RID1": "mid", "RID2": "leaf", "RID3": "sibling"}
    g.label_to_rids = {"root": ["RID0"], "mid": ["RID1"], "leaf": ["RID2"], "sibling": ["RID3"]}
    g.child_to_parents = {"RID2": ["RID1"], "RID1": ["RID0"], "RID3": ["RID0"]}
    return g


class TestAnnotate:
    def test_ancestor_rids_walks_parents(self):
        g = _graph()
        assert ancestor_rids(g, "RID2") == {"RID2", "RID1", "RID0"}
        assert ancestor_rids(g, "RID0") == {"RID0"}  # root has no parents

    def test_cluster_gets_specific_common_ancestor(self):
        g = _graph()
        # two members resolving to RID2 (leaf) and RID1 (mid) share RID1 + RID0;
        # most specific common = RID1 (not the root RID0).
        clusters = [Cluster(cluster_id=0, members=["leaf", "mid"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].radlex_label == "mid"
        assert annotated[0].radlex_rid == "RID1"
        assert annotated[0].n_resolved == 2
        assert annotated[0].n_members == 2

    def test_root_only_common_falls_back_to_none(self):
        g = _graph()
        # leaf (under mid under root) and sibling (under root) share ONLY root.
        # root is rejected as trivially uninformative -> radlex_label None.
        clusters = [Cluster(cluster_id=0, members=["leaf", "sibling"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].radlex_label is None
        assert annotated[0].radlex_rid is None

    def test_unresolved_members_skipped(self):
        g = _graph()
        clusters = [Cluster(cluster_id=0, members=["leaf", "ghost"], medoid="leaf")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].n_resolved == 1
        assert annotated[0].n_members == 2
        # single resolved member -> its own label is most specific common ancestor
        assert annotated[0].radlex_label == "leaf"

    def test_empty_cluster_no_crash(self):
        g = _graph()
        clusters: list[Cluster] = []
        assert annotate_radlex(clusters, g) == []

    def test_canopy_rejected_on_large_graph(self):
        """A generic 'canopy' node (>=1% of graph descendants) is rejected as a
        cluster label even when it is the only shared non-root ancestor. Builds a
        synthetic graph of 1002 RIDs so the subtree-size canopy activates."""
        g = RadLexGraph()
        g.rid_to_label = {"RIDroot": "root", "RIDcanopy": "canopy"}
        g.label_to_rids = {"root": ["RIDroot"], "canopy": ["RIDcanopy"]}
        g.child_to_parents = {"RIDcanopy": ["RIDroot"]}
        # 1000 leaves, all under RIDcanopy (so RIDcanopy has 1001 descendants,
        # well over 1% of the 1002-RID graph -> canopy -> rejected).
        for i in range(1000):
            rid = f"RIDt{i}"
            g.rid_to_label[rid] = f"term_{i}"
            g.label_to_rids[f"term_{i}"] = [rid]
            g.child_to_parents[rid] = ["RIDcanopy"]

        # cluster of 6 leaves: their only shared non-root ancestor is RIDcanopy,
        # which is canopy -> rejected -> no candidate -> radlex_label None.
        clusters = [Cluster(cluster_id=0, members=[f"term_{i}" for i in range(6)],
                            medoid="term_0")]
        annotated = annotate_radlex(clusters, g)
        assert annotated[0].radlex_label is None
        assert annotated[0].radlex_rid is None
        assert annotated[0].n_resolved == 6

        # Sanity: a leaf's own RID is NOT a candidate (must be shared by >=2 and
        # is a leaf-root anyway). Confirm medoid fallback used.
        assert annotated[0].display_label == "term_0"


from concept_discovery.organize import build_structured_explanations, AnnotatedCluster


class TestStructured:
    def test_family_aggregation_and_redundancy(self):
        clusters = [
            AnnotatedCluster(cluster_id=0, members=["a", "b"], medoid="a", radlex_label="famA"),
            AnnotatedCluster(cluster_id=1, members=["c"], medoid="c", radlex_label=None),
        ]
        cs = ConceptSet(
            names=["a", "b", "c"],
            embeddings=torch.zeros((3, 512)),
            name_to_idx={"a": 0, "b": 1, "c": 2},
            per_image=[ImageConcepts(image_id="img0", activations={"a": 1.0, "b": 2.0, "c": 4.0})],
        )
        out = build_structured_explanations(cs, clusters)
        assert len(out) == 1
        ex = out[0]
        assert ex["image_id"] == "img0"
        fam_by_label = {f["label"]: f for f in ex["families"]}
        assert set(fam_by_label) == {"famA", "c"}  # cluster1 radlex None -> medoid "c"
        assert fam_by_label["famA"]["aggregate_activation"] == 3.0
        assert fam_by_label["famA"]["intra_redundancy"] == 2
        assert len(fam_by_label["famA"]["concepts"]) == 2
        # 3 raw concepts / 2 families = 1.5
        assert ex["redundancy_score"] == 1.5

    def test_image_with_no_active_concepts(self):
        clusters = [AnnotatedCluster(cluster_id=0, members=["a"], medoid="a")]
        cs = ConceptSet(
            names=["a"],
            embeddings=torch.zeros((1, 512)),
            name_to_idx={"a": 0},
            per_image=[ImageConcepts(image_id="empty", activations={})],
        )
        out = build_structured_explanations(cs, clusters)
        assert out[0]["families"] == []
        assert out[0]["redundancy_score"] == 0


from concept_discovery.organize import compute_metrics


class TestMetrics:
    def test_basic_metrics_and_redundancy_reduction(self):
        clusters = [
            AnnotatedCluster(cluster_id=0, members=["a", "b"], medoid="a",
                             radlex_label="famA", n_resolved=2, n_members=2),
            AnnotatedCluster(cluster_id=1, members=["c"], medoid="c",
                             radlex_label=None, n_resolved=0, n_members=1),
        ]
        cs = ConceptSet(
            names=["a", "b", "c"],
            embeddings=torch.tensor([[1.0] + [0.0] * 511,
                                      [1.0] + [0.0] * 511,
                                      [0.0, 1.0] + [0.0] * 510], dtype=torch.float32),
            name_to_idx={"a": 0, "b": 1, "c": 2},
            per_image=[
                ImageConcepts(image_id="i0", activations={"a": 1.0, "b": 1.0, "c": 1.0}),
                ImageConcepts(image_id="i1", activations={"a": 1.0, "c": 1.0}),
            ],
        )
        structured = [
            {"image_id": "i0", "families": [{}, {}], "redundancy_score": 1.5},
            {"image_id": "i1", "families": [{}], "redundancy_score": 2.0},
        ]
        m = compute_metrics(cs, clusters, structured)
        assert m["n_concepts_active"] == 3
        assert m["n_clusters"] == 2
        assert m["mean_cluster_size"] == 1.5
        assert m["radlex_coverage_pct"] == (2 / 3) * 100
        assert m["n_empty_images"] == 0
        # mean raw = (3+2)/2 = 2.5 ; mean families = (2+1)/2 = 1.5 -> 2.5/1.5
        assert abs(m["redundancy_reduction"] - (2.5 / 1.5)) < 1e-9

    def test_silhouette_none_below_two_clusters(self):
        clusters = [AnnotatedCluster(cluster_id=0, members=["a"], medoid="a")]
        cs = ConceptSet(
            names=["a"], embeddings=torch.zeros((1, 512)),
            name_to_idx={"a": 0}, per_image=[],
        )
        m = compute_metrics(cs, clusters, [])
        assert m["silhouette_cosine"] is None


from dataclasses import replace
import config
from concept_discovery.organize import run as organize_run


class TestRun:
    def test_run_writes_three_output_files(self, tmp_path):
        terms = [{"term": f"t{i}"} for i in range(6)]
        emb = torch.eye(6, 512)
        explanations = [
            {"image_id": "i0", "top_k_concepts": [
                {"feature_id": 0, "name": "t0", "activation": 0.5},
                {"feature_id": 1, "name": "t1", "activation": 0.5}]},
            {"image_id": "i1", "top_k_concepts": [
                {"feature_id": 2, "name": "t2", "activation": 0.5},
                {"feature_id": 3, "name": "t3", "activation": 0.5}]},
            {"image_id": "i2", "top_k_concepts": [
                {"feature_id": 4, "name": "t4", "activation": 0.5},
                {"feature_id": 5, "name": "t5", "activation": 0.5}]},
        ]
        cs = from_spliece_explanations(explanations, terms, emb)
        cfg = replace(config.organize, n_clusters=2, output_dir=tmp_path, radlex_csv_path=tmp_path / "no.csv")
        metrics = organize_run(cfg, cs, graph=None)
        assert (tmp_path / "concept_clusters.json").exists()
        assert (tmp_path / "structured_explanations.json").exists()
        assert (tmp_path / "organization_metrics.json").exists()
        assert "n_clusters" in metrics
        import json
        clusters_json = json.loads((tmp_path / "concept_clusters.json").read_text())
        assert all("display_label" in c or "label" in c for c in clusters_json)
