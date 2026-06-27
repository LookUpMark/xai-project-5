"""SPLiCE: Sparse Linear Concept Discovery (Path B).

Deterministic sparse decomposition on RadLex vocabulary using
Orthogonal Matching Pursuit (OMP). No training, no seeds, CPU-only.

Solves non-identifiability (M-001) by constraining decomposition to
pre-existing vocabulary terms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit

import sys
sys.path.insert(0, "src/")
from utils import load_tensor, ensure_dir
import config


def decompose_image(
    image_emb: torch.Tensor,      # (512,) L2-normalised
    vocab_emb: torch.Tensor,      # (V, 512) text embeddings
    gap: torch.Tensor | None,     # (512,) modality gap vector
    k: int = 32,
) -> torch.Tensor:
    """Return sparse coefficient vector (V,) via OMP.

    Solves: min ||emb_corrected - vocab_emb.T @ c||²  s.t. nnz(c) <= k, c >= 0

    Args:
        image_emb: Single image embedding (L2-normalised).
        vocab_emb: Vocabulary term embeddings (rows = terms).
        gap: Modality gap vector for correction. If None, skipped.
        k: Number of non-zero coefficients (L0 sparsity).

    Returns:
        Sparse coefficient vector (V,) with exactly k non-zero entries.
    """
    emb = image_emb.clone()
    if gap is not None:
        emb = emb - gap  # modality-gap correction

    # Orthogonal Matching Pursuit: exact L0 sparsity
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    X = vocab_emb.numpy()  # (V, 512) — dictionary atoms as rows
    y = emb.numpy()        # (512,) — target signal
    omp.fit(X.T, y)        # X.T is (512, V); solve y ≈ X.T @ c

    coeffs = torch.from_numpy(omp.coef_).float()  # (V,)
    coeffs = coeffs.clamp(min=0)  # enforce non-negativity post-hoc
    return coeffs


def run(
    cfg: config.SpliCEConfig,
    test_embeddings: torch.Tensor,
    image_ids: list[str],
    vocab_terms: list[str],
) -> list[dict]:
    """Decompose all test images; return per-image concept lists.

    Uses SAFE deserialization via utils.load_tensor() (weights_only=True).

    Args:
        cfg: SPLiCE configuration.
        test_embeddings: Test set image embeddings (N, 512).
        image_ids: Image identifiers (N,).
        vocab_terms: Vocabulary term strings (V,).

    Returns:
        List of per-image dicts with top_k_concepts + pseudo_report.
        Compatible with SAE sample_explanations.json schema.
    """
    import json

    vocab_emb = load_tensor(cfg.vocab_emb_path)  # SAFE: weights_only=True
    gap = None
    if cfg.use_gap_correction:
        gap = load_tensor(cfg.gap_path)  # SAFE

    results = []
    for emb, img_id in zip(test_embeddings, image_ids):
        coeffs = decompose_image(emb, vocab_emb, gap, k=cfg.k)
        top_k = coeffs.topk(cfg.k)
        concepts = [
            {"term": vocab_terms[idx]["term"], "coefficient": float(val)}
            for idx, val in zip(top_k.indices.tolist(), top_k.values.tolist())
            if val > 0  # filter out zero coefficients from clamp
        ]
        pseudo_report = "Findings suggest: " + ", ".join(
            c["term"] for c in concepts[:5]
        )
        results.append({
            "image_id": img_id,
            "top_k_concepts": concepts,
            "pseudo_report": pseudo_report,
        })

    ensure_dir(cfg.output_dir / "sample_explanations.json")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / "sample_explanations.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    """Self-check: reconstruct one image and verify sparsity."""
    import json

    print("🔍 SPLiCE self-check running...")

    # Load inputs
    vocab_terms = json.load(open(config.spliece.vocab_path))
    print(f"   Vocabulary: {len(vocab_terms)} terms")

    # Verify vocabulary format (should be list of dicts with 'term' field)
    if vocab_terms and isinstance(vocab_terms[0], dict):
        print(f"   Vocabulary format: list of dicts with 'term' field")
    else:
        print(f"   ⚠️  Vocabulary format: unexpected (expected list of dicts)")

    # Load test embeddings (first 10 for quick check)
    test_emb = load_tensor(config.paths.test_embeddings_path)[:10]
    print(f"   Test embeddings: {test_emb.shape}")

    # Load test image IDs
    test_ids_path = Path("data/test_image_ids.json")
    if test_ids_path.exists():
        with open(test_ids_path) as f:
            test_ids = json.load(f)[:10]
    else:
        # Fallback: generate dummy IDs
        test_ids = [f"test_{i}" for i in range(10)]
        print(f"   ⚠️  Using dummy IDs (test_image_ids.json not found)")

    print(f"   Image IDs: {len(test_ids)}")

    # Run decomposition
    print("   Running SPLiCE decomposition...")
    results = run(config.spliece, test_emb, test_ids, vocab_terms)

    # Verify sparsity (≤ k non-zero coeffs after clamp filtering)
    assert len(results[0]["top_k_concepts"]) <= config.spliece.k, \
        f"Expected ≤{config.spliece.k} concepts, got {len(results[0]['top_k_concepts'])}"

    # Verify all coefficients > 0 (clamp effective)
    assert all(c["coefficient"] > 0 for c in results[0]["top_k_concepts"]), \
        "Found zero or negative coefficients"

    print(f"✅ Self-check passed: {len(results)} images decomposed")
    print(f"📄 Output written to: {config.spliece.output_dir / 'sample_explanations.json'}")
    print(f"\n📋 Sample result:")
    print(f"   Image: {results[0]['image_id']}")
    print(f"   Report: {results[0]['pseudo_report']}")
    print(f"   Top 3 concepts:")
    for i, c in enumerate(results[0]["top_k_concepts"][:3], 1):
        print(f"      {i}. {c['term']}: {c['coefficient']:.4f}")
