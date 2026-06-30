"""
Tests for the vocabulary building pipeline in build_vocabulary.py.

- Unit tests use mocks to verify internal functions (encoding, centroid, ranking, filtering).
- The integration test loads the real BiomedCLIP model on GPU and verifies
  that the pipeline runs end-to-end correctly for a tiny list of terms.

---

### How to run tests
    
    # Only unit tests (without GPU):
    python -m pytest tests/unit/test_build_vocabulary.py -v -k "not Integration"

To run ALL tests including the integration ones:
    python -m pytest tests/unit/test_build_vocabulary.py -v
"""

import pytest
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import VLMConfig, VocabularyConfig
from vocabulary_building.build_vocabulary import (
    encode_texts,
    compute_anchor_centroids,
    rank_terms_by_relevance,
    build_final_vocabulary,
    build_vocabulary_pipeline,
)
from vocabulary_building.radlex_support import (
    filter_terms,
    load_and_filter_radlex,
    load_radlex_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_configs(tmp_path: Path):
    vlm_config = VLMConfig(
        batch_size=2,
        device="cpu",
    )
    vocab_config = VocabularyConfig(
        output_path=str(tmp_path / "vocabulary.json"),
        top_k=2,
        nih_seed_terms=["cardiomegaly", "effusion"],
        anchor_groups={"group1": ["chest radiograph finding"]},
    )
    return vlm_config, vocab_config


# =========================================================================
# Unit tests
# =========================================================================

class TestEncodeTexts:
    def test_embeddings_are_normalized_and_batched(self):
        vlm_config, _ = _make_configs(Path("dummy"))
        texts = ["term1", "term2", "term3"]

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Dummy processor output
        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        # Fake features from the model: return non-normalized
        def fake_features(**kwargs):
            # Infer batch size from the tokenizer call
            batch_size = len(mock_processor.tokenizer.call_args[1]["text"])
            # Return vectors that are NOT unit-norm
            return torch.full((batch_size, 512), 2.0)

        mock_model.get_text_features.side_effect = fake_features

        result = encode_texts(texts, mock_model, mock_processor, vlm_config)

        # 3 terms, batch_size=2 -> 2 batches
        assert mock_model.get_text_features.call_count == 2
        assert result.shape == (3, 512)

        # Check normalization
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5), "Embeddings are not L2-normalized"


class TestComputeAnchorCentroids:
    def test_centroid_computation(self):
        vlm_config, vocab_config = _make_configs(Path("dummy"))
        vocab_config.anchor_groups = {"group1": ["anchor1", "anchor2"]}

        mock_model = MagicMock()
        mock_processor = MagicMock()

        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        # The model returns some features
        mock_model.get_text_features.return_value = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        centroids = compute_anchor_centroids(mock_model, mock_processor, vlm_config, vocab_config)

        # It should process all anchors in batches
        assert mock_processor.tokenizer.call_args[1]["text"] == ["anchor1", "anchor2"]
        
        # We expect 1 group centroid of shape (1, 3)
        assert centroids.shape == (1, 3)

        # Centroid must be unit-norm
        assert torch.allclose(centroids[0].norm(dim=-1), torch.tensor(1.0)), "Centroid is not normalized"



    def test_centroid_computation_multiple_groups(self):
        vlm_config, vocab_config = _make_configs(Path("dummy"))
        vocab_config.anchor_groups = {
            "group1": ["anchor1"],
            "group2": ["anchor2", "anchor3"]
        }

        mock_model = MagicMock()
        mock_processor = MagicMock()
        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        mock_processor.tokenizer.return_value = proc_output

        # Mock feature return for each group
        # group1 has 1 item, group2 has 2 items
        mock_model.get_text_features.side_effect = [
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ]

        centroids = compute_anchor_centroids(mock_model, mock_processor, vlm_config, vocab_config)

        # Expected shape: 2 groups -> (2, 3)
        assert centroids.shape == (2, 3)
        assert mock_model.get_text_features.call_count == 2


class TestRankTermsByRelevance:
    def test_ranking_sorts_correctly(self):
        terms = ["apple", "banana", "cherry"]
        
        # Fake embeddings
        embeddings = torch.tensor([
            [1.0, 0.0],  # apple (sim: 0.707)
            [-1.0, 0.0],  # banana (sim: -0.707)
        ])
        embeddings = torch.cat([
            embeddings, 
            (torch.tensor([[1.0, 1.0]]) / torch.tensor(2.0).sqrt()) # cherry (sim: ~0.707)
        ], dim=0)
        
        # Fake centroid
        centroid = torch.tensor([[0.5, 0.5]])
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)

        ranked = rank_terms_by_relevance(terms, embeddings, centroid)

        assert len(ranked) == 3
        # Expected order: cherry, apple, banana
        assert ranked[0][0] == "cherry"
        assert ranked[1][0] == "apple"
        assert ranked[2][0] == "banana"
        assert ranked[0][1] > ranked[1][1] > ranked[2][1]



    def test_ranking_with_multiple_centroids(self):
        terms = ["apple", "banana", "cherry"]
        
        # 3 terms, 2 dimensions
        embeddings = torch.tensor([
            [1.0, 0.0],  # apple (matches centroid 1)
            [0.0, 1.0],  # banana (matches centroid 2)
            [-1.0, -1.0], # cherry (matches neither)
        ])
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # 2 centroids, 2 dimensions
        centroids = torch.tensor([
            [1.0, 0.0],  # Centroid 1
            [0.0, 1.0],  # Centroid 2
        ])
        centroids = centroids / centroids.norm(dim=-1, keepdim=True)

        ranked = rank_terms_by_relevance(terms, embeddings, centroids)

        # Apple matches C1 perfectly (score 1.0)
        # Banana matches C2 perfectly (score 1.0)
        # Cherry matches neither (score < 0)
        
        assert len(ranked) == 3
        
        # Apple and Banana should have score 1.0, Cherry < 0
        scores = {k: v for k, v in ranked}
        assert torch.isclose(torch.tensor(scores["apple"]), torch.tensor(1.0))
        assert torch.isclose(torch.tensor(scores["banana"]), torch.tensor(1.0))
        assert scores["cherry"] < 0.0


class TestBuildFinalVocabulary:
    def test_top_k_and_nih_seeds_inclusion(self):
        _, vocab_config = _make_configs(Path("dummy"))
        vocab_config.top_k = 2
        vocab_config.nih_seed_terms = ["seed1", "seed2"]

        # Ranked terms (already sorted)
        ranked_terms = [
            ("apple", 0.9),
            ("seed1", 0.8), # Seed naturally inside input
            ("banana", 0.7),
            ("cherry", 0.6),
            ("seed2", 0.1), # Seed not naturally in the top-k
        ]

        input_terms_set = {"apple", "banana", "cherry", "seed1"}

        result = build_final_vocabulary(ranked_terms, vocab_config, input_terms_set)

        # We expect:
        # Top 2 input terms: apple, seed1.
        # Plus any missing NIH seeds: seed2.
        # Total = 3
        assert len(result) == 3

        terms_only = [v["term"] for v in result]
        # It sorts by similarity score inside the function!
        assert terms_only == ["apple", "seed1", "seed2"]
        
        assert result[0]["source"] == "input_filtered"  # apple
        assert result[1]["source"] == "input_filtered"  # seed1
        assert result[2]["source"] == "nih_chestxray14_seed"  # seed2


class TestBuildVocabularyPipeline:
    @patch("vocabulary_building.build_vocabulary.load_and_filter_radlex")
    @patch("vocabulary_building.build_vocabulary.save_vocab_embeddings")
    @patch("vocabulary_building.build_vocabulary.save_vocabulary")
    @patch("vocabulary_building.build_vocabulary.build_final_vocabulary")
    @patch("vocabulary_building.build_vocabulary.rank_terms_by_relevance")
    @patch("vocabulary_building.build_vocabulary.compute_anchor_centroids")
    @patch("vocabulary_building.build_vocabulary.encode_texts")
    def test_pipeline_injects_seeds_and_calls_functions(
        self,
        mock_encode,
        mock_centroid,
        mock_rank,
        mock_build,
        mock_save_vocab,
        mock_save_embs,
        mock_load_filter,
    ):
        vlm_config, vocab_config = _make_configs(Path("dummy"))
        vocab_config.nih_seed_terms = ["seed1", "seed2"]

        # The pipeline derives terms from RadLex internally; mock the loader to
        # return a controlled list so the seed-injection assertions are deterministic.
        mock_load_filter.return_value = ["apple", "banana"]
        mock_encode.return_value = torch.zeros((4, 512))  # 2 input + 2 injected
        mock_centroid.return_value = torch.zeros(512)
        mock_rank.return_value = []
        mock_build.return_value = []

        build_vocabulary_pipeline(
            model=MagicMock(),
            processor=MagicMock(),
            vlm_config=vlm_config,
            vocab_config=vocab_config,
        )

        # The RadLex loader is called once (no all_terms parameter anymore).
        mock_load_filter.assert_called_once()

        # encode_texts receives the 2 input terms + the 2 injected NIH seeds.
        encoded_texts = mock_encode.call_args[1]["texts"]
        assert len(encoded_texts) == 4
        assert "seed1" in encoded_texts
        assert "seed2" in encoded_texts

        # Saves should be called
        mock_save_vocab.assert_called_once()
        mock_save_embs.assert_called_once()


class TestCxrFiltering:
    """Unit tests for the RadLex hierarchical chest filtering logic (mock CSV)."""

    def test_filter_terms(self, tmp_path):
        # Mock radlex.csv exercising every keep/drop branch of the calibrated filter:
        # thoracic anatomy (TARGET_ROOTS), clinical finding (FINDING_ROOTS),
        # device (DEVICE_ROOTS), anatomical-site, exclude keywords, safety-net.
        csv_content = (
            "Class ID,Preferred Label,Synonyms,Definitions,Obsolete,CUI,Semantic Types,Parents,"
            "http://www.radlex.org/RID/Anatomical_Site\n"
            "http://www.radlex.org/RID/RID1243,thorax,,,FALSE,,,,\n"
            "http://www.radlex.org/RID/RID1301,lung,,,FALSE,,,http://www.radlex.org/RID/RID1243,\n"
            "http://www.radlex.org/RID/RID1385,heart,,,FALSE,,,http://www.radlex.org/RID/RID1243,\n"
            "http://www.radlex.org/RID/RID34785,clinical finding,,,FALSE,,,,\n"
            "http://www.radlex.org/RID/RID5554,tube or catheter,,,FALSE,,,,\n"
            "http://www.radlex.org/RID/RID56,abdomen,,,FALSE,,,,\n"
            "http://www.radlex.org/RID/RID28530,opacity,,,FALSE,,,http://www.radlex.org/RID/RID34785,\n"
            "http://www.radlex.org/RID/RID5557,endotracheal tube,,,FALSE,,,http://www.radlex.org/RID/RID5554,\n"
            "http://www.radlex.org/RID/RID5103,autoimmune pancreatitis,,,FALSE,,,http://www.radlex.org/RID/RID56,\n"
            "http://www.radlex.org/RID/RID9998,pancreatic cyst,,,FALSE,,,http://www.radlex.org/RID/RID34785,\n"
            "http://www.radlex.org/RID/RID1000,mitral valve,,,FALSE,,,,http://www.radlex.org/RID/RID1385\n"
        )
        csv_file = tmp_path / "mock_radlex.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        input_terms = [
            "thorax",                   # TARGET_ROOT
            "lung",                     # descends from thorax
            "heart",                    # descends from thorax
            "opacity",                  # descends from clinical finding (RID34785)
            "endotracheal tube",        # descends from tube or catheter (RID5554)
            "mitral valve",             # anatomical site = heart -> kept
            "non-radlex term",          # not in CSV -> safety-net kept
            "autoimmune pancreatitis",  # descends from abdomen -> excluded
            "pancreatic cyst",          # finding, but label has 'pancrea' kw -> excluded
        ]

        filtered = filter_terms(load_radlex_graph(str(csv_file)), input_terms)

        for kept in ["thorax", "lung", "heart", "opacity", "endotracheal tube",
                     "mitral valve", "non-radlex term"]:
            assert kept in filtered, f"{kept!r} should be kept"
        for dropped in ["autoimmune pancreatitis", "pancreatic cyst"]:
            assert dropped not in filtered, f"{dropped!r} should be dropped"


# Path to the committed real RadLex CSV (git-tracked -> the test is portable).
_RADLEX_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "radlex.csv"


@pytest.mark.skipif(not _RADLEX_CSV.exists(), reason="data/radlex.csv not present")
class TestRadlexCsvFilter:
    """Regression test on the REAL RadLex CSV (data/radlex.csv).

    Locks in the calibration: CXR-critical findings + devices that the anatomy-only
    filter used to drop MUST be kept; an obviously non-CXR finding MUST be dropped.
    """

    def test_real_radlex_keeps_cxr_findings_and_devices(self):
        terms = [
            "pneumothorax", "consolidation", "granuloma",                       # clinical findings
            "endotracheal tube", "central venous catheter", "swan-ganz catheter",  # devices
        ]
        kept = set(filter_terms(load_radlex_graph(str(_RADLEX_CSV)), terms))
        for t in terms:
            assert t in kept, f"CXR-critical term {t!r} was dropped by the filter"

    def test_real_radlex_drops_non_cxr(self):
        kept = set(filter_terms(load_radlex_graph(str(_RADLEX_CSV)), ["autoimmune pancreatitis"]))
        assert "autoimmune pancreatitis" not in kept

    def test_real_radlex_reduces_vocabulary(self):
        # Sanity: load_and_filter_radlex must cut the ~46k RadLex terms down to a
        # small CXR set.
        kept = load_and_filter_radlex(str(_RADLEX_CSV))
        assert 500 <= len(kept) <= 6000, f"unexpected kept count: {len(kept)}"


# =========================================================================
# Integration test – real model
# =========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegrationVocabularyPipeline:
    """
    Integration test: loads the real BiomedCLIP model and builds a vocabulary
    for a small synthetic input list.
    """
    
    @pytest.fixture(scope="class")
    def model_and_processor(self):
        from utils import load_vlm
        config = VLMConfig()
        model, processor = load_vlm(config)
        yield model, processor
        del model
        torch.cuda.empty_cache()

    def test_pipeline_end_to_end(self, model_and_processor, tmp_path, monkeypatch):
        model, processor = model_and_processor

        vlm_config = VLMConfig(device="cuda", batch_size=4)
        vocab_config = VocabularyConfig(
            output_path=str(tmp_path / "medical_vocabulary.json"),
            top_k=2,
            nih_seed_terms=["pneumonia", "effusion"],
            anchor_groups={"group1": ["chest radiograph finding"]},
        )

        # The pipeline derives terms from RadLex internally; stub the loader to
        # keep this end-to-end test fast (3 terms instead of ~46k). encode/
        # centroid/ rank/ build still run on the REAL model.
        monkeypatch.setattr(
            "vocabulary_building.build_vocabulary.load_and_filter_radlex",
            lambda *args, **kwargs: ["broken bone", "cardiomegaly", "headache"],
        )

        result = build_vocabulary_pipeline(
            model, processor, vlm_config, vocab_config
        )

        # Pipeline steps verification
        # The input list has 3 terms.
        # Missing seeds are 2.
        # Expected encoded terms = 5 before top-k selection (3 input + 2 seeds).
        
        # Result vocabulary size check:
        # Top-2 input terms (probably broken bone + cardiomegaly, or headache)
        # Plus the 2 NIH seeds
        # Total = 4 terms
        assert len(result) == 4

        # Check JSON saving
        assert Path(vocab_config.output_path).exists()
        with open(vocab_config.output_path, "r") as f:
            saved_json = json.load(f)
        assert len(saved_json) == 4
        
        # Check embeddings saving
        assert Path(vocab_config.embeddings_output_path).exists()
        saved_emb = torch.load(vocab_config.embeddings_output_path, weights_only=True)
        # The saved embeddings tensor must match the final vocabulary size
        assert saved_emb.shape == (4, 512)
        assert torch.allclose(saved_emb.norm(dim=-1), torch.ones(4), atol=1e-4)

        # Check sources
        sources = [v["source"] for v in result]
        assert sources.count("input_filtered") == 2
        assert sources.count("nih_chestxray14_seed") == 2
