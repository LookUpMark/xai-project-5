# Concept Organization — Pipeline Run

**Status**: Complete ✅
**Total time**: 2.1s
**Date**: 2026-07-09 22:15:51

## Run config

| param | value |
|-------|-------|
| source | sae-baseline |
| tag | — |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/concept_organization |
| n_clusters | None |
| distance_threshold | None |
| linkage | average |
| radlex annotation | enabled |

## Metrics

| metric | value |
|-------|-------|
| n_concepts_active | 309 |
| n_clusters | 18 |
| mean_cluster_size | 17.17 |
| silhouette_cosine | 0.0970504954457283 |
| redundancy_reduction | 1.341 |
| radlex_coverage_pct | 14.2 |
| n_empty_images | 0 |

## Output files

- `concept_clusters.json` — clusters with RadLex ancestor labels
- `structured_explanations.json` — per-image concept families + redundancy
- `organization_metrics.json` — metrics snapshot

## Reproducibility

- git commit: `eb3d10f951cabf4fe9f8132efebe267935ee5142`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(explanations) [sample_explanations.json]: `e42414b479e70a64`
- sha256(vocab) [vocabulary.json]: `e7ea2f7fb019ac9f`
- sha256(vocab_emb) [text_vocab_embeddings.pt]: `846c2d086400dde2`
- sha256(radlex) [radlex.csv]: `dec82b28e1b9ecfe`
- sha256(concept_names) [concept_names.json]: `c9a18042948e3a81`

## References

- Spec: `docs/design/proposals/2026-07-03-concept-organization.md`
- Plan: `docs/plans/2026-07-03-concept-organization.md`

