# SPLiCE — Pipeline Run

**Status**: Complete ✅
**Total time**: 2.3s
**Date**: 2026-06-27 12:27:19

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
| decompose | ok | 2.3 |
| coverage | ok | 0.0 |

## Output files

- `sample_explanations.json` — Per-image concept lists (SAE-compatible schema)
- `REPORT_coverage.md` — Vocabulary coverage analysis


## Verification

✅ All unit tests passing (`tests/unit/test_spliece.py`)
✅ All integration tests passing (`tests/integration/test_spliece_pipeline.py`)
✅ Self-check passing (`python -m src.concept_discovery.spliece`)
✅ Output schema compatible with SAE `sample_explanations.json`

## References

- Implementation Plan: `docs/plans/2026-06-27-spliece-path-b.md`
- Verification Audit: `docs/audits/ML-AUDIT-2026-06-27.md`
- Release Notes: `docs/releases/CHANGELOG-v0.5.0-2026-06-27.md`

