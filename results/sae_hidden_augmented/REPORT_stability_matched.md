# Path A — Matched (Permutation-Invariant) Stability

_Generated: 2026-06-26_

## Summary

Mean best-match cosine = 0.3317 (isotropic null 0.1238, p=0.000). WEAK universality (p=0.000, obs/null=2.7x): decoder subspaces share structure well above chance, but only 0.0% match ≥0.9 (0.2% ≥0.7) — no strong feature-level reproducibility. Analytical random-anchor ≈ 0.141.  Cf. slot-wise Jaccard (0.0038-class) which cannot show this.

## Headline metrics

| metric | value |
| --- | --- |
| mean best-match cosine | 0.3317 |
| permutation null mean | 0.1238 |
| observed / null | 2.68x |
| p-value (P(null≥obs)) | 0.0000 |
| min p-value across pairs | 0.0000 |
| mean frac mutual 1-to-1 | 0.3230 |
| analytical random anchor | 0.1409 |
| dict_size / activation_dim | 2048 / 768 |

## Matched-fraction thresholds

| metric | value |
| --- | --- |
| mean frac matched ≥0.7 | 0.0022 |
| mean frac matched ≥0.9 | 0.0000 |

## Per-pair results

| pair | best-match | null | p | frac≥0.7 | frac≥0.9 | mutual1-1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0-1 | 0.3310 | 0.1238 | 0.000 | 0.001 | 0.000 | 0.311 |
| 0-2 | 0.3330 | 0.1238 | 0.000 | 0.001 | 0.000 | 0.319 |
| 0-3 | 0.3322 | 0.1238 | 0.000 | 0.003 | 0.000 | 0.335 |
| 0-4 | 0.3320 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.318 |
| 1-2 | 0.3318 | 0.1238 | 0.000 | 0.003 | 0.000 | 0.322 |
| 1-3 | 0.3320 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.320 |
| 1-4 | 0.3344 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.330 |
| 2-3 | 0.3284 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.321 |
| 2-4 | 0.3305 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.327 |
| 3-4 | 0.3321 | 0.1238 | 0.000 | 0.002 | 0.000 | 0.327 |

## Interpretation

Mean best-match cosine = 0.3317 (isotropic null 0.1238, p=0.000). WEAK universality (p=0.000, obs/null=2.7x): decoder subspaces share structure well above chance, but only 0.0% match ≥0.9 (0.2% ≥0.7) — no strong feature-level reproducibility. Analytical random-anchor ≈ 0.141.  Cf. slot-wise Jaccard (0.0038-class) which cannot show this.

**Metric**: decoder-cosine matching (each feature paired with its most similar decoder direction across seeds) + isotropic random-vector null. Unlike slot-wise index Jaccard, this is permutation-invariant in the matching step — the property the F-001 metric lacked. NB: a row-shuffle null is degenerate for max-cosine (max-over-columns is permutation-invariant), so the null uses independent random unit vectors; it does not control for data-manifold concentration, hence the ≥0.9 matched-fraction is the concentration-robust signal.

**Framing** (Lan et al. 2024; Leask et al. 2025): cross-seed SAEs are *expected* to share at most a subspace (weak universality), rarely identical features (strong universality). An observed/null ratio well above 1 with p<0.05 ⇒ weak universality present; at-null with p>0.05 ⇒ genuine non-identifiability at this data scale (M-002), now measured rather than asserted.

Refs: Bricken 2023; Lan 2024 (arXiv:2410.06981); Leask 2025 (arXiv:2502.04878); Kriegeskorte 2008. See `docs/design/LITERATURE-SAE-STABILITY.md`.

