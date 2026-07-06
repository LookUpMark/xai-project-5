# Concept Organization — Pipeline Run

**Status**: Complete ✅
**Total time**: 1.1s
**Date**: 2026-07-06 21:00:35

## Run config

| param | value |
|-------|-------|
| source | sae-hidden |
| tag | hidden |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/concept_organization_hidden |
| n_clusters | None |
| distance_threshold | None |
| linkage | average |
| radlex annotation | enabled |

## Metrics

| metric | value |
|-------|-------|
| n_concepts_active | 14 |
| n_clusters | 4 |
| mean_cluster_size | 3.50 |
| silhouette_cosine | 0.28415247797966003 |
| redundancy_reduction | 1.026 |
| radlex_coverage_pct | 100.0 |
| n_empty_images | 1260 |

## Output files

- `concept_clusters.json` — clusters with RadLex ancestor labels
- `structured_explanations.json` — per-image concept families + redundancy
- `organization_metrics.json` — metrics snapshot

## Reproducibility

- git commit: `59d2872ae9a2ed915414d8349a45dee7201ad900`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(explanations) [sample_explanations.json]: `5a04b75d882e8c48`
- sha256(vocab) [vocabulary.json]: `0f33f72d418db401`
- sha256(vocab_emb) [text_vocab_embeddings.pt]: `98c9cf7462d6181f`
- sha256(radlex) [radlex.csv]: `dec82b28e1b9ecfe`
- sha256(concept_names) [concept_names.json]: `a07716d529aef7ec`

## References

- Spec: `docs/design/proposals/2026-07-03-concept-organization.md`
- Plan: `docs/plans/2026-07-03-concept-organization.md`

