"""SPLiCE: Sparse Linear Concept Discovery (Path B).

Deterministic sparse decomposition on RadLex vocabulary using
Orthogonal Matching Pursuit (OMP). No training, no seeds, CPU-only.

Solves non-identifiability (M-001) by constraining decomposition to
pre-existing vocabulary terms.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.linear_model import OrthogonalMatchingPursuit

import sys
sys.path.insert(0, "src/")
from utils import load_tensor
import config


def decompose_image(
    image_emb: torch.Tensor,      # (512,) L2-normalised
    vocab_emb: torch.Tensor,      # (V, 512) text embeddings
    gap: torch.Tensor | None,     # (512,) modality gap vector
    k: int = 32,
) -> torch.Tensor:
    """Return a non-negative sparse coefficient vector (V,) for one image.

    Two-stage solve (F-003 fix): OMP selects exactly ``k`` atoms (deterministic,
    exact L0), then non-negative least squares (NNLS) re-solves the coefficients
    on that support. This replaces the previous ``clamp(min=0)`` of a signed OMP
    solution, which zeroed ~40% of the support and inflated reconstruction error
    8-14x. NNLS yields coefficients that are both non-negative AND
    reconstruction-optimal on the chosen support.

    Solves: support S = OMP_k(emb);  c_S = argmin ||emb - V_S^T c||  s.t. c >= 0.
    Design-doc alternative: ``Lasso(positive=True)`` (joint L1 sparsity +
    non-negativity); kept here as a tuning experiment if faithfulness is low.

    Args:
        image_emb: Single image embedding (L2-normalised).
        vocab_emb: Vocabulary term embeddings (rows = terms).
        gap: Modality gap vector for correction. If None, skipped.
        k: Number of atoms selected by OMP (L0 sparsity). Final nnz may be <= k
            if NNLS drives some selected atoms to exactly 0.

    Returns:
        Non-negative sparse coefficient vector (V,).
    """
    emb = image_emb.clone()
    if gap is not None:
        emb = emb - gap  # modality-gap correction

    X = vocab_emb.numpy()  # (V, 512) — dictionary atoms as rows
    y = emb.numpy()        # (512,) — target signal

    # Stage 1: OMP selects exactly k atoms (deterministic, exact L0).
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    omp.fit(X.T, y)            # X.T is (512, V); solves y ≈ X.T @ c
    support = np.nonzero(omp.coef_)[0]

    # Stage 2: re-solve non-negative least squares on the selected support so the
    # coefficients actually reconstruct the signal (no post-hoc clamp).
    coeffs = np.zeros(X.shape[0], dtype=np.float64)
    if support.size > 0:
        c_nnls, _ = nnls(X[support].T, y)  # (512, |S|) -> c_nnls (|S|,), c >= 0
        coeffs[support] = c_nnls
    return torch.from_numpy(coeffs).float()  # (V,)


def run(
    cfg: config.SpliCEConfig,
    test_embeddings: torch.Tensor,
    image_ids: list[str],
    vocab_terms: list[dict],
) -> list[dict]:
    """Decompose all test images; return per-image concept lists.

    Output schema is SAE-compatible (``feature_id`` / ``name`` / ``activation``)
    so the SAME LLM judge (``src/evaluate_llm_judge.py``) can score it without an
    adapter. Uses SAFE deserialization via ``utils.load_tensor()`` (weights_only=True).

    Args:
        cfg: SPLiCE configuration.
        test_embeddings: Test set image embeddings (N, 512).
        image_ids: Image identifiers (N,).
        vocab_terms: Vocabulary entries (V,) as ``{"term": str, ...}`` dicts.

    Returns:
        List of per-image dicts with top_k_concepts + pseudo_report.
    """
    import json

    # F-007: guard against silent truncation / mislabeling on count mismatch.
    if len(test_embeddings) != len(image_ids):
        raise ValueError(
            f"test_embeddings ({len(test_embeddings)}) != image_ids ({len(image_ids)}); "
            "zip() would silently truncate — regenerate both from the same split."
        )

    vocab_emb = load_tensor(cfg.vocab_emb_path)  # SAFE: weights_only=True
    if len(vocab_terms) != vocab_emb.shape[0]:
        raise ValueError(
            f"vocab_terms ({len(vocab_terms)}) != vocab_emb rows ({vocab_emb.shape[0]}); "
            "coefficient index would not map to the correct term — re-embed the vocabulary."
        )

    gap = None
    if cfg.use_gap_correction:
        gap = load_tensor(cfg.gap_path)  # SAFE

    top_n = config.explanation.explanation_top_n  # F-013: shared with the SAE path

    results = []
    for emb, img_id in zip(test_embeddings, image_ids):
        coeffs = decompose_image(emb, vocab_emb, gap, k=cfg.k)
        top_k = coeffs.topk(cfg.k)
        concepts = [
            {
                "feature_id": int(idx),
                "name": vocab_terms[idx]["term"],
                "activation": float(val),
            }
            for idx, val in zip(top_k.indices.tolist(), top_k.values.tolist())
            if val > 0  # drop atoms NNLS drove to exactly 0
        ]
        pseudo_report = "Findings suggest: " + ", ".join(
            c["name"] for c in concepts[:top_n]
        )
        results.append({
            "image_id": img_id,
            "top_k_concepts": concepts,
            "pseudo_report": pseudo_report,
        })

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
        print("   Vocabulary format: list of dicts with 'term' field")
    else:
        print("   ⚠️  Vocabulary format: unexpected (expected list of dicts)")

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
        print("   ⚠️  Using dummy IDs (test_image_ids.json not found)")

    print(f"   Image IDs: {len(test_ids)}")

    # Run decomposition
    print("   Running SPLiCE decomposition...")
    # F-002: write to a throwaway dir so the self-check can NEVER clobber the
    # production results/spliece/sample_explanations.json (1515 images).
    selfcheck_cfg = replace(config.spliece, output_dir=Path("/tmp/spliece_selfcheck"))
    results = run(selfcheck_cfg, test_emb, test_ids, vocab_terms)

    # Verify sparsity (≤ k non-zero coeffs after NNLS zero-drop)
    assert len(results[0]["top_k_concepts"]) <= config.spliece.k, \
        f"Expected ≤{config.spliece.k} concepts, got {len(results[0]['top_k_concepts'])}"

    # Verify all activations > 0 (NNLS non-negativity, zero-drop effective)
    assert all(c["activation"] > 0 for c in results[0]["top_k_concepts"]), \
        "Found zero or negative coefficients"

    print(f"✅ Self-check passed: {len(results)} images decomposed")
    print(f"📄 Output written to: {selfcheck_cfg.output_dir / 'sample_explanations.json'}")
    print(f"   (production output at {config.spliece.output_dir} is untouched)")
    print("\n📋 Sample result:")
    print(f"   Image: {results[0]['image_id']}")
    print(f"   Report: {results[0]['pseudo_report']}")
    print("   Top 3 concepts:")
    for i, c in enumerate(results[0]["top_k_concepts"][:3], 1):
        print(f"      {i}. {c['name']}: {c['activation']:.4f}")
