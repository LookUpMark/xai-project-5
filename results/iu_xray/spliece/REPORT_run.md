# SPLiCE — Pipeline Run

**Status**: Complete ✅
**Total time**: 9.5s
**Date**: 2026-07-06 20:01:10

## Run config

| param | value |
|-------|-------|
| tag | — |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/spliece |
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
| decompose | ok | 9.5 |

## Output files

- `sample_explanations.json` — Per-image concept lists (SAE-compatible schema)


## Reproducibility

- git commit: `5b80bf250e0df95e6056843d0410d44db419b94e`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(test_embeddings) [test_embeddings.pt]: `d1b9081dbb5f0fe1`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `98c9cf7462d6181f`
- sha256(modality_gap) [modality_gap.pt]: `e6dda3a0a8ed454a`
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

