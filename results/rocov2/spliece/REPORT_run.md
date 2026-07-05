# SPLiCE — Pipeline Run

**Status**: Complete ✅
**Total time**: 101.0s
**Date**: 2026-07-05 19:34:18

## Run config

| param | value |
|-------|-------|
| tag | — |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/spliece |
| k | 32 |
| gap correction | True |
| images decomposed | 15958 |

## Algorithm

**Orthogonal Matching Pursuit (OMP)** with `n_nonzero_coefs=32`

- Modality gap correction: **enabled**
- Post-hoc clamp: `coeffs = max(coeffs, 0)`
- Zero filtering: Exclude coefficients ≤ 0 from top-k
- Expected concepts per image: ≤ 32 (may be fewer due to filtering)

## Stages

| stage | status | seconds |
|-------|--------|--------|
| decompose | ok | 101.0 |

## Output files

- `sample_explanations.json` — Per-image concept lists (SAE-compatible schema)


## Reproducibility

- git commit: `f2b44f422a3e68a945f857800309dff7a299f87c`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(test_embeddings) [test_embeddings.pt]: `909730509dc3cde2`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `846c2d086400dde2`
- sha256(modality_gap) [modality_gap.pt]: `1205a576aa512342`
- sha256(test_image_ids) [test_image_ids.json]: `560efb4b4319b40d`

## Verification

✅ All unit tests passing (`tests/unit/test_spliece.py`)
✅ All integration tests passing (`tests/integration/test_spliece_pipeline.py`)
✅ Self-check passing (`python -m src.concept_discovery.spliece`)
✅ Output schema compatible with SAE `sample_explanations.json`

## References

- Implementation Plan: `docs/plans/2026-06-27-spliece-path-b.md`
- Verification Audit: `docs/audits/ML-AUDIT-2026-06-27.md`
- Release Notes: `docs/releases/CHANGELOG-v0.5.0-2026-06-27.md`

