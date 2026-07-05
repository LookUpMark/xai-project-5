# SPLiCE — Pipeline Run

**Status**: Complete ✅
**Total time**: 2.7s
**Date**: 2026-06-27 17:19:24

## Run config

| param | value |
|-------|-------|
| tag | — |
| output dir | /Users/marcantoniolopez/Documents/github/xai-project-5/results/spliece |
| k | 32 |
| gap correction | True |
| images decomposed | 1515 |

## Algorithm

**Orthogonal Matching Pursuit (OMP)** with `n_nonzero_coefs=32`

- Modality gap correction: **enabled**
- Post-hoc clamp: `coeffs = max(coeffs, 0)`
- Zero filtering: Exclude coefficients ≤ 0 from top-k
- Expected concepts per image: ≤ 32 (may be fewer due to filtering)

## Stages

| stage | status | seconds |
|-------|--------|--------|
| decompose | ok | 2.7 |
| coverage | ok | 0.0 |

## Output files

- `sample_explanations.json` — Per-image concept lists (SAE-compatible schema)
- `REPORT_coverage.md` — Vocabulary coverage analysis


## Reproducibility

- git commit: `772d55b20bd5fb9cd2a406f0e1b479f2e8a4a998`
- versions: scikit-learn 1.8.0 | torch 2.12.0 | numpy 2.4.6
- sha256(test_embeddings) [test_embeddings.pt]: `f266e54366f3fb5e`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `922ee9509eb06e70`
- sha256(modality_gap) [modality_gap.pt]: `36264e287fc1f1f5`
- sha256(test_image_ids) [test_image_ids.json]: `3816a84e18deefc7`

## Verification

✅ All unit tests passing (`tests/unit/test_spliece.py`)
✅ All integration tests passing (`tests/integration/test_spliece_pipeline.py`)
✅ Self-check passing (`python -m src.concept_discovery.spliece`)
✅ Output schema compatible with SAE `sample_explanations.json`

## References

- Implementation Plan: `docs/plans/2026-06-27-spliece-path-b.md`
- Verification Audit: `docs/audits/ML-AUDIT-2026-06-27.md`
- Release Notes: `docs/releases/CHANGELOG-v0.5.0-2026-06-27.md`

