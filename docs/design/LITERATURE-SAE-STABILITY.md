# Literature: Cross-Seed SAE Feature Stability / Universality

> Why our cross-seed stability metric exists, what it measures, and the references it is
> grounded in. Committed so the methodological basis is never "lost" between sessions.

## The problem we hit (ML-AUDIT-2026-06-26 F-001)

Our first stability metric was **slot-wise index Jaccard**: for each test sample, compare
the *set of active feature indices* (top-k) across two seeds, i.e. feature #342 vs #342.
The result was Jaccard 0.0083 ≈ analytical chance floor k/(2D−k) = 0.0079, and we
interpreted it as "the SAE is non-identifiable across seeds."

**That interpretation was wrong.** SAEs have no canonical feature ordering: two SAEs that
learned the *same* decomposition but with concepts in different slots score ~0 on this
metric. The chance floor already assumes "no slot correspondence," which is true even for
good SAEs. So slot-wise Jaccard measures **slot correspondence**, not feature identity — it
cannot establish non-identifiability. This is the same class of error as ML-AUDIT-2026-06-23
(contamination → upper bound): measuring one thing and concluding another.

## The literature-correct approach

### 1. Permutation-invariant feature matching (solve the ordering problem)

Pair each feature in seed A with its **most similar** feature in seed B before comparing.
The canonical pairing signal is **activation correlation** (Bricken et al. 2023): pass a
common input set through both SAEs, correlate activation vectors per feature pair, take the
max. We use **decoder-direction cosine** instead (see "Why decoder cosine" below).

### 2. A null that is NOT degenerate

A row-shuffle / permutation null is **degenerate** for a max-cosine metric: max-over-columns
is invariant to permuting the columns, so shuffling B's rows leaves the best-match unchanged.
The principled null is best-match against **independent random unit vectors** (isotropic);
the high-threshold matched-fraction (≥0.9 cosine) is the concentration-robust signal, since
random directions in 768-d cannot reach 0.9 (analytical E[max cosine] ≈ (1/√d)·√(2·ln D) ≈ 0.14).

### 3. Frame as weak vs strong universality

Leask et al. 2025 (*"SAEs do not find canonical units of analysis"*) and Lan et al. 2024 show
that cross-seed SAEs **do not** find identical features even at scale — they share at most a
**subspace** (weak universality). Not finding identical features is *expected*, not a bug.
The question is whether weak universality is present.

## Why decoder cosine, not activation correlation

Bricken's primary pairing is activation correlation, but on our data it is pathological:
TopK k=32 over ~1,515 test samples means each feature is active on ~2% of samples, so Pearson
correlation is dominated by shared zeros. **Decoder cosine** (the SAE feature *direction* —
what the feature "means") avoids this, and is explicitly validated by Lan et al. 2024 App. E.2
for the **same-model-different-seed** case (our exact case): *"we also use cosine similarity
instead of activation correlation … average cosine 0.9 … SVCCA 0.92."*

## Our result (Path A, 5 seeds, dict 2048 / k 32 / 768-d)

| metric | value |
|---|---|
| mean best-match cosine | 0.325 |
| isotropic null mean | 0.124 |
| observed / null | 2.6× |
| p-value (all pairs) | ≈0 |
| frac matched ≥0.9 | **0.0%** |
| frac matched ≥0.7 | 0.1% |
| frac mutual 1-to-1 | 0.32 (at mean cosine ~0.33) |

**Verdict: weak universality, no strong reproducibility.** The decoder subspaces share
structure well above chance (0.325 ≫ 0.124, p≈0), but **zero** features reproduce at ≥0.9
across seeds. This refines F-001's "non-identifiable" into "weakly universal, not strongly
reproducible" — consistent with Leask et al. 2025 and the M-002 data-scale limit (5,955
images ≪ the 10⁵–10⁶ of SAE/CLIP literature; Lan et al. found their 100M-token SAEs showed
"almost no similarity" vs 8.2B-token ones).

**Caveat:** the isotropic null does not control for data-manifold concentration, so the
0.325 vs 0.124 significance is a lower bound on evidence. The ≥0.9 fraction (0%) is the
concentration-robust read and it says: no strong shared features. A rotation null (R·W_j) is
the concentration-controlled variant, not run here for compute.

## References

- **Bricken et al. 2023** — *Towards Monosemanticity: Decomposing Language Models with
  Dictionary Learning*, Transformer Circuits Thread. The activation-correlation feature
  pairing. https://transformer-circuits.pub/2023/monosemantic-features/
- **Templeton et al. 2024** — *Scaling Monosemanticity*, Transformer Circuits Thread.
  Universality of SAE features at scale.
- **Gao et al. 2024 (OpenAI)** — *Scaling and Evaluating Sparse Autoencoders*,
  arXiv:2406.04093. k-sparse SAEs (our architecture); feature-quality metrics.
- **Lan et al. 2024** — *Sparse Autoencoders Reveal Universal Feature Spaces Across Large
  Language Models*, arXiv:2410.06981. **The direct precedent** — cross-seed/cross-model SAE
  comparison; §3 methodology, App. E.2 "different seeds", App. H weak/strong universality.
- **Leask et al. 2025** — *Sparse Autoencoders Do Not Find Canonical Units of Analysis*,
  arXiv:2502.04878. Cross-seed SAEs don't find identical features.
- **Rajamanoharan et al. 2024** — *Gemma Scope: Open Sparse Autoencoders*, arXiv:2408.05147.
- **Kriegeskorte et al. 2008** — *Representational Similarity Analysis* (RSA). Random-pairing
  null concept (note: the permutation null is for RSA/SVCCA global metrics, not max-cosine).
- **Raghu et al. 2017** — SVCCA. Rotation-invariant subspace similarity (the optional global
  layer beyond pairwise matching).

## Related repo docs

- Implementation: `src/autoencoder/sae_module.py` (`matched_pair_stats`,
  `SAEManager.compute_stability_matched`), `src/sae_hidden/stability_hidden.py` (`run_matched`).
- Tests: `tests/unit/test_sae_hidden.py` (identity / permutation-invariance / null / dead).
- Plan: `docs/plans/2026-06-26-sae-stability-matched.md`.
- Audit: `docs/audits/ML-AUDIT-2026-06-26.md` (F-001).
