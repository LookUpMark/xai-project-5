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
