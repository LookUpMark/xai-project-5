"""Relabeling control — isolates the slot-wise Jaccard artefact (H1).

The master symptom is cross-seed SAE Jaccard pinned at the analytical chance floor.
H1 (ML-AUDIT-2026-06-25 addendum) claims the slot-wise metric is floor-by-
construction: it compares raw feature-index sets with no permutation alignment, so
it cannot tell "two genuinely different SAEs" from "one SAE whose features have been
renamed by a random bijection."

This script is the decisive control: take ONE trained SAE, relabel its features by a
random permutation π (re-index the dict axis consistently across encoder rows,
encoder bias, and decoder columns), then measure slot-wise Jaccard AND the
permutation-invariant matched metric between the original and the relabeled copy.

Expected (H1 confirmed):
    slot-wise Jaccard  ≈ chance floor   (indistinguishable from real cross-seed)
    matched best-match ≈ 1.0, frac≥0.9 ≈ 1.0   (the SAME SAE — correctly identical)

If both hold, the slot-wise number measures feature *indexing*, not feature
*identity*, and "4 levers fail to move Jaccard" is a category error. The matched
metric is the sound cross-seed signal.

Run:
    python scripts/relabeling_control.py [--dataset rocov2] [--seed-model 42]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
import utils  # noqa: E402
from autoencoder.sae_module import SAEManager  # noqa: E402

logger = utils.setup_logging(__name__)

# Permutation applied to the dict axis. Fixed seed → reproducible π.
PERM_SEED = 12345


def relabel_state_dict(sd: dict, perm: torch.Tensor) -> dict:
    """Re-index the dict axis (D) consistently across encoder/decoder weights.

    Feature i is renamed π(i): encoder rows, encoder bias, and decoder columns are
    permuted together. b_dec (input bias) and scalars are untouched. The relabeled
    SAE computes an identical function to the original — only feature *ordering*
    changes.
    """
    out = {k: v for k, v in sd.items()}
    out["encoder.weight"] = sd["encoder.weight"][perm].clone()
    out["encoder.bias"] = sd["encoder.bias"][perm].clone()
    out["decoder.weight"] = sd["decoder.weight"][:, perm].clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="rocov2", choices=["rocov2", "iu_xray"])
    ap.add_argument("--seed-model", type=int, default=42, help="which trained seed to relabel")
    ap.add_argument("--perm-seed", type=int, default=PERM_SEED)
    args = ap.parse_args()

    config.select_dataset(args.dataset)
    model_dir = config.paths.models_dir / f"sae_seed{args.seed_model}"
    ae_path = model_dir / "trainer_0" / "ae.pt"
    if not ae_path.exists():
        ae_path = model_dir / "ae.pt"
    if not ae_path.exists():
        raise FileNotFoundError(f"No ae.pt under {model_dir}")

    # Original state dict + dict size.
    sd_orig = utils.load_state_dict(ae_path, device="cpu")
    D = sd_orig["encoder.weight"].shape[0]
    logger.info(f"Loaded {args.dataset} seed{args.seed_model}: dict_size={D}")

    # Random relabeling π.
    gen = torch.Generator().manual_seed(args.perm_seed)
    perm = torch.randperm(D, generator=gen)
    assert (perm.sort().values == torch.arange(D)).all(), "π must be a bijection"
    sd_rel = relabel_state_dict(sd_orig, perm)

    # Persist the relabeled copy to a tmp dir so SAEManager.load can read it.
    tmp_dir = config.paths.results_dir / f"_relabel_control_seed{args.seed_model}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    (tmp_dir / "trainer_0").mkdir(parents=True)
    torch.save(sd_rel, tmp_dir / "trainer_0" / "ae.pt")

    # Sanity: the relabeled SAE must reconstruct identically to the original.
    emb = utils.load_tensor(config.paths.test_embeddings_path)
    if config.training.stability_max_samples:
        emb = emb[: config.training.stability_max_samples]
    m_orig = SAEManager({"device": config.hardware.device}); m_orig.load(model_dir)
    m_rel = SAEManager({"device": config.hardware.device}); m_rel.load(tmp_dir)
    with torch.no_grad():
        cos_orig = m_orig.compute_cosine_reconstruction(emb)
        cos_rel = m_rel.compute_cosine_reconstruction(emb)
    logger.info(f"Reconstruction cosine sanity: orig={cos_orig:.6f} relabeled={cos_rel:.6f}")
    assert abs(cos_orig - cos_rel) < 1e-4, "Relabeling changed the function — permutation inconsistent"
    del m_orig, m_rel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dirs = [model_dir, tmp_dir]

    # 1. Slot-wise Jaccard (the artefact).
    sw = SAEManager.compute_stability(dirs, emb, config={"device": config.hardware.device})
    # 2. Matched permutation-invariant metric + subspace-conditioned null.
    mt = SAEManager.compute_stability_matched(dirs, config={"device": config.hardware.device})

    result = {
        "dataset": args.dataset,
        "seed_model": args.seed_model,
        "perm_seed": args.perm_seed,
        "dict_size": D,
        "n_test_samples": int(emb.shape[0]),
        "reconstruction_cosine_orig": cos_orig,
        "reconstruction_cosine_relabeled": cos_rel,
        "slot_wise_jaccard": {
            "mean": sw["mean_jaccard"],
            "note": "Floor-by-construction if H1 holds: a relabeled copy of ONE SAE scores ~chance floor.",
        },
        "matched": {
            "mean_best_match_cosine": mt["mean_best_match_cosine"],
            "mean_frac_matched_0.9": mt["mean_frac_matched_0.9"],
            "mean_frac_mutual_1to1": mt["mean_frac_mutual_1to1"],
            "mean_erank": mt["mean_erank"],
            "ratio_subspace": mt["ratio_subspace"],
            "note": "Should be ~1.0 (frac≥0.9 ≈ 1.0): the SAME SAE, correctly identified as identical.",
        },
    }

    out_path = config.paths.results_dir / "relabeling_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    mj = sw["mean_jaccard"]
    mc = mt["mean_best_match_cosine"]
    f9 = mt["mean_frac_matched_0.9"]
    print("\n=== Relabeling control (H1) ===")
    print(f"slot-wise Jaccard : {mj:.4f}   (chance floor ⇒ H1 confirmed)")
    print(f"matched best-match: {mc:.4f}   frac≥0.9 = {f9:.4f}   (≈1.0 ⇒ same SAE)")
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
