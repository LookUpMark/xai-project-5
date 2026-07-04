"""Integration test: full organize pipeline on synthetic data."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src/")

import torch
import config
from concept_discovery.organize import (
    from_spliece_explanations, from_sae_explanations, run as organize_run,
)


def _write_inputs(tmp_path, source):
    V = 12
    torch.manual_seed(0)
    emb = torch.eye(V, 512)
    vocab_path = tmp_path / "vocabulary.json"
    vocab_path.write_text(json.dumps([{"term": f"t{i}"} for i in range(V)]))
    emb_path = tmp_path / "vocab_emb.pt"
    torch.save(emb, emb_path)
    explanations = [{
        "image_id": f"img_{i}",
        "top_k_concepts": [
            {"feature_id": (i % V), "name": f"t{i%V}", "activation": 1.0},
            {"feature_id": (i + 1) % V, "name": f"t{(i+1)%V}", "activation": 0.5},
        ],
    } for i in range(8)]
    expl_path = tmp_path / "sample_explanations.json"
    expl_path.write_text(json.dumps(explanations))
    cn_path = None
    if source.startswith("sae"):
        cn = {str(i): {"name": f"t{i}", "score": 0.5, "is_dead": False} for i in range(V)}
        cn_path = tmp_path / "concept_names.json"
        cn_path.write_text(json.dumps(cn))
    return vocab_path, emb_path, expl_path, cn_path


class TestPipeline:
    def test_spliece_end_to_end(self, tmp_path):
        vocab_path, emb_path, expl_path, _ = _write_inputs(tmp_path, "spliece")
        vocab_terms = json.loads(vocab_path.read_text())
        emb = torch.load(emb_path, weights_only=True)
        expl = json.loads(expl_path.read_text())
        cs = from_spliece_explanations(expl, vocab_terms, emb)
        cfg = replace(config.organize, n_clusters=3, output_dir=tmp_path,
                      radlex_csv_path=tmp_path / "no.csv")
        metrics = organize_run(cfg, cs, graph=None)
        assert (tmp_path / "concept_clusters.json").exists()
        assert (tmp_path / "structured_explanations.json").exists()
        assert (tmp_path / "organization_metrics.json").exists()
        assert metrics["n_clusters"] == 3
        # determinism: a second run with identical inputs yields identical cluster members
        import json as _json
        first = _json.loads((tmp_path / "concept_clusters.json").read_text())
        cs2 = from_spliece_explanations(expl, vocab_terms, emb)
        organize_run(cfg, cs2, graph=None)
        second = _json.loads((tmp_path / "concept_clusters.json").read_text())
        assert [c["members"] for c in first] == [c["members"] for c in second]

    def test_sae_end_to_end(self, tmp_path):
        vocab_path, emb_path, expl_path, cn_path = _write_inputs(tmp_path, "sae-hidden")
        vocab_terms = json.loads(vocab_path.read_text())
        emb = torch.load(emb_path, weights_only=True)
        expl = json.loads(expl_path.read_text())
        cn = json.loads(cn_path.read_text())
        cs = from_sae_explanations(expl, cn, vocab_terms, emb)
        cfg = replace(config.organize, n_clusters=3, output_dir=tmp_path,
                      radlex_csv_path=tmp_path / "no.csv")
        organize_run(cfg, cs, graph=None)
        assert (tmp_path / "structured_explanations.json").exists()
