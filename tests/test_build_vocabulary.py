"""
Tests for the vocabulary building pipeline in build_vocabulary.py.

- Unit tests use mocks to verify internal functions (encoding, centroid, ranking, filtering).
- The integration test loads the real BiomedCLIP model on GPU and verifies
  that the pipeline runs end-to-end correctly for a tiny list of terms.

---

### How to run tests
    
    # Only unit tests (without GPU):
    python -m pytest tests/test_build_vocabulary.py -v -k "not Integration"

    # All tests (requires CUDA):
    python -m pytest tests/test_build_vocabulary.py -v
"""

import pytest
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import VLMConfig, VocabularyConfig
from build_vocabulary import (
    encode_texts,
    compute_anchor_centroid,
    rank_terms_by_relevance,
    build_final_vocabulary,
    build_vocabulary_pipeline,
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
        output_path=str(tmp_path / "medical_vocabulary.json"),
        embeddings_output_path=str(tmp_path / "vocab_embeddings.pt"),
        top_k=2,
        nih_seed_terms=["cardiomegaly", "effusion"],
        anchor_queries=["chest radiograph finding"],
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


class TestComputeAnchorCentroid:
    def test_centroid_computation(self):
        vlm_config, vocab_config = _make_configs(Path("dummy"))
        # We have 1 anchor query from the helper
        vocab_config.anchor_queries = ["anchor1", "anchor2"]

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

        centroid = compute_anchor_centroid(mock_model, mock_processor, vlm_config, vocab_config)

        # It should process all anchors in batches
        assert mock_processor.tokenizer.call_args[1]["text"] == ["anchor1", "anchor2"]
        
        # The centroid shape should match the feature dimension and keepdim=True -> (1, 3)
        assert centroid.shape == (1, 3)

        # Centroid must be unit-norm
        assert torch.allclose(centroid.norm(dim=-1), torch.tensor(1.0)), "Centroid is not normalized"


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
    @patch("build_vocabulary.save_vocab_embeddings")
    @patch("build_vocabulary.save_vocabulary")
    @patch("build_vocabulary.build_final_vocabulary")
    @patch("build_vocabulary.rank_terms_by_relevance")
    @patch("build_vocabulary.compute_anchor_centroid")
    @patch("build_vocabulary.encode_texts")
    def test_pipeline_injects_seeds_and_calls_functions(
        self,
        mock_encode,
        mock_centroid,
        mock_rank,
        mock_build,
        mock_save_vocab,
        mock_save_embs,
    ):
        vlm_config, vocab_config = _make_configs(Path("dummy"))
        vocab_config.nih_seed_terms = ["seed1", "seed2"]

        all_terms = ["apple", "banana"]

        mock_encode.return_value = torch.zeros((4, 512)) # 2 input + 2 injected
        mock_centroid.return_value = torch.zeros(512)
        mock_rank.return_value = []
        mock_build.return_value = []

        build_vocabulary_pipeline(
            model=MagicMock(),
            processor=MagicMock(),
            vlm_config=vlm_config,
            vocab_config=vocab_config,
            all_terms=all_terms,
        )

        # 1. Missing seeds should have been injected into all_terms
        assert len(all_terms) == 4
        assert "seed1" in all_terms
        assert "seed2" in all_terms

        # 2. encode_texts should be called with 4 terms
        assert len(mock_encode.call_args[1]["texts"]) == 4

        # 3. Saves should be called
        mock_save_vocab.assert_called_once()
        mock_save_embs.assert_called_once()


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

    def test_pipeline_end_to_end(self, model_and_processor, tmp_path):
        model, processor = model_and_processor

        vlm_config = VLMConfig(device="cuda", batch_size=4)
        vocab_config = VocabularyConfig(
            output_path=str(tmp_path / "medical_vocabulary.json"),
            embeddings_output_path=str(tmp_path / "vocab_embeddings.pt"),
            top_k=2,
            nih_seed_terms=["pneumonia", "effusion"],
            anchor_queries=["chest radiograph finding"],
        )

        all_terms = ["broken bone", "cardiomegaly", "headache"]

        result = build_vocabulary_pipeline(
            model, processor, vlm_config, vocab_config, all_terms
        )

        # Pipeline steps verification
        # The input list has 3 terms.
        # Missing seeds are 2.
        # Expected all_terms size = 5 before encoding.
        
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
