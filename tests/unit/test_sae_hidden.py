"""Unit tests for Path A (SAE on 768-d hidden state).

Guards the genuinely novel logic — the frozen-projection naming bridge — plus the
768-d training/encoding path through the reused SAEManager. Independent of the
conftest 4096-mock fixtures (synthetic 768-d tensors only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Mirror the repo's load-bearing sys.path hack (no package install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import config  # noqa: E402
from autoencoder.sae_module import SAEManager, matched_pair_stats  # noqa: E402
from sae_hidden.naming_hidden import (  # noqa: E402
    bridge_cosine_sims,
    dead_feature_mask,
    project_decoder_to_text,
)


# ── Config ────────────────────────────────────────────────────────────────────


def test_sae_hidden_config_defaults():
    cfg = config.sae_hidden
    assert cfg.activation_dim == 768
    assert cfg.dict_size == 2048
    assert cfg.k == 32
    assert cfg.steps == 8_000
    assert cfg.lr == 5e-5
    # Baseline config is untouched by Path A additions.
    assert config.sae.activation_dim == 512


def test_sae_hidden_config_rejects_bad_dict_size():
    with pytest.raises(ValueError):
        config.SAEHiddenConfig(dict_size=512)  # not > activation_dim


def test_hidden_paths_are_separate_from_baseline():
    p = config.paths
    assert p.hidden_embeddings_dir != p.embeddings_dir
    assert "standard_hidden" in str(p.hidden_embeddings_dir)
    assert p.hidden_models_dir != p.models_dir
    assert p.hidden_results_dir != p.results_dir


# ── Frozen-projection bridge math ─────────────────────────────────────────────


def test_project_decoder_to_text_shape():
    torch.manual_seed(0)
    W_proj = torch.randn(512, 768)
    W_dec = torch.randn(2048, 768)
    dec_512 = project_decoder_to_text(W_dec, W_proj)
    assert dec_512.shape == (2048, 512)


def test_project_decoder_to_text_known_vectors():
    """Projecting the i-th standard basis of 768-d yields the i-th column of W_proj."""
    torch.manual_seed(1)
    W_proj = torch.randn(512, 768)
    W_dec = torch.eye(768)[:5]  # first 5 basis vectors -> (5, 768)
    dec_512 = project_decoder_to_text(W_dec, W_proj)
    expected = W_proj[:, :5].T  # (5, 512)
    assert torch.allclose(dec_512, expected, atol=1e-6)


def test_project_decoder_to_text_gap_subtraction():
    torch.manual_seed(2)
    W_proj = torch.randn(512, 768)
    W_dec = torch.randn(4, 768)
    gap = torch.ones(512)
    no_gap = project_decoder_to_text(W_dec, W_proj)
    with_gap = project_decoder_to_text(W_dec, W_proj, gap=gap)
    assert torch.allclose(with_gap, no_gap - 1.0, atol=1e-6)


def test_dead_feature_mask_flags_zero_rows():
    W_dec = torch.randn(5, 768)
    W_dec[2] = 0.0
    mask = dead_feature_mask(W_dec)
    assert mask.shape == (5,)
    assert mask[2].item() is True
    assert mask.sum().item() == 1


def test_bridge_picks_matching_term():
    """When vocab terms ARE the projected decoder directions, argmax is the identity."""
    torch.manual_seed(3)
    W_proj = torch.randn(512, 768)
    W_dec = torch.randn(6, 768)
    dec_512 = project_decoder_to_text(W_dec, W_proj)      # (6, 512)
    vocab_emb = dec_512.clone()                            # term i == decoder i
    dead_mask = torch.zeros(6, dtype=torch.bool)
    sims = bridge_cosine_sims(dec_512, vocab_emb, dead_mask)  # (6, 6)
    best = sims.argmax(dim=1)
    assert torch.equal(best, torch.arange(6))
    # Each decoder matches itself with cosine ~1.
    diag = torch.diagonal(sims)
    assert (diag > 0.999).all()


def test_bridge_dead_row_never_wins():
    torch.manual_seed(4)
    W_proj = torch.randn(512, 768)
    W_dec = torch.randn(3, 768)
    W_dec[1] = 0.0  # dead
    dec_512 = project_decoder_to_text(W_dec, W_proj)
    vocab_emb = torch.randn(10, 512)
    dead_mask = dead_feature_mask(W_dec)
    sims = bridge_cosine_sims(dec_512, vocab_emb, dead_mask)
    assert dead_mask[1].item() is True
    assert torch.all(sims[1] == 0.0)  # dead row zeroed out


# ── 768-d path through the reused SAEManager ──────────────────────────────────


def test_sae_manager_trains_and_encodes_768(tmp_path):
    """Tiny end-to-end guard that SAEManager works at activation_dim=768."""
    torch.manual_seed(5)
    n, dim = 600, 768
    emb = torch.randn(n, dim)
    emb_path = tmp_path / "train_768.pt"
    torch.save(emb, emb_path)

    cfg = {
        "activation_dim": dim,
        "dict_size": 64,
        "k": 8,
        "lr": 5e-5,
        "steps": 50,
        "warmup_steps": 5,
        "batch_size": 64,
        "device": "cpu",
    }
    mgr = SAEManager(cfg)
    model_dir = mgr.train(emb_path, seed=0, save_dir=tmp_path, steps=50, batch_size=64)
    mgr.load(model_dir)

    assert mgr.get_decoder_weights().shape == (64, 768)
    sparse = mgr.encode(emb[:4])
    assert sparse.shape == (4, 64)
    # Top-K: exactly k non-zero entries per row.
    l0 = (sparse != 0).float().sum(dim=1)
    assert torch.all(l0 == 8)


# ── Permutation-invariant matched stability (F-001 fix) ────────────────────────


def _null_gen() -> torch.Generator:
    """Fresh deterministic generator for the isotropic null (keeps tests stable)."""
    return torch.Generator().manual_seed(7)


def test_matched_identity_is_one():
    """A decoder vs itself: every feature matches itself → cosine ~1.0, p≈0."""
    torch.manual_seed(0)
    W = torch.randn(40, 64)
    s = matched_pair_stats(W, W, n_perm=50, generator=_null_gen())
    assert s["mean_best_match_cosine"] > 0.999
    assert s["frac_matched_0.9"] == pytest.approx(1.0)
    assert s["p_value"] < 0.05  # isotropic null cannot reach ~1.0


def test_matched_permutation_invariant():
    """Permuting W_j's rows leaves the best-match distribution unchanged.

    This is the whole point: the observed metric is invariant to the arbitrary
    per-seed feature ordering — the property slot-wise index Jaccard lacks.
    """
    torch.manual_seed(1)
    W_i = torch.randn(30, 64)
    W_j = torch.randn(30, 64)
    perm = torch.randperm(30)
    base = matched_pair_stats(W_i, W_j, n_perm=20, generator=_null_gen())
    permuted = matched_pair_stats(W_i, W_j[perm], n_perm=20, generator=_null_gen())
    assert base["mean_best_match_cosine"] == pytest.approx(
        permuted["mean_best_match_cosine"], abs=1e-6
    )
    assert base["frac_matched_0.7"] == pytest.approx(
        permuted["frac_matched_0.7"], abs=1e-6
    )


def test_matched_random_is_at_null():
    """Two independent random decoders → observed ≈ isotropic null, low and not strong."""
    torch.manual_seed(2)
    W_i = torch.randn(80, 128)
    W_j = torch.randn(80, 128)
    s = matched_pair_stats(W_i, W_j, n_perm=100, generator=_null_gen())
    assert s["mean_best_match_cosine"] < 0.5  # well below any "matched" threshold
    assert abs(s["mean_best_match_cosine"] - s["null_mean"]) < 0.15  # consistent w/ null
    # Under a true null p is ~Uniform(0,1); assert it is not strongly significant.
    assert s["p_value"] > 0.01


def test_matched_dead_row_scores_zero():
    """A zero (dead) decoder row matches at ~0; only live rows contribute."""
    torch.manual_seed(3)
    D = 20
    W = torch.eye(D)  # orthonormal: each live row matches only itself
    W[-1] = 0.0       # kill the last row
    s = matched_pair_stats(W, W, n_perm=20, generator=_null_gen())
    # 19 live rows match at cosine 1.0; the dead row matches at 0.
    assert s["mean_best_match_cosine"] == pytest.approx((D - 1) / D, abs=1e-4)
    assert s["frac_matched_0.9"] == pytest.approx((D - 1) / D, abs=1e-4)
