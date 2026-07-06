# Concept Organization — Pipeline Run

**Status**: Complete ✅
**Total time**: 1.4s
**Date**: 2026-07-06 21:00:28

## Run config

| param | value |
|-------|-------|
| source | spliece |
| tag | spliece |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/concept_organization_spliece |
| n_clusters | None |
| distance_threshold | None |
| linkage | average |
| radlex annotation | enabled |

## Metrics

| metric | value |
|-------|-------|
| n_concepts_active | 997 |
| n_clusters | 32 |
| mean_cluster_size | 31.16 |
| silhouette_cosine | 0.02034156769514084 |
| redundancy_reduction | 1.753 |
| radlex_coverage_pct | 99.5 |
| n_empty_images | 0 |

## Output files

- `concept_clusters.json` — clusters with RadLex ancestor labels
- `structured_explanations.json` — per-image concept families + redundancy
- `organization_metrics.json` — metrics snapshot

## Reproducibility

- git commit: `59d2872ae9a2ed915414d8349a45dee7201ad900`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(explanations) [sample_explanations.json]: `a482b52d1c2a81bb`
- sha256(vocab) [vocabulary.json]: `0f33f72d418db401`
- sha256(vocab_emb) [text_vocab_embeddings.pt]: `98c9cf7462d6181f`
- sha256(radlex) [radlex.csv]: `dec82b28e1b9ecfe`

## References

- Spec: `docs/design/proposals/2026-07-03-concept-organization.md`
- Plan: `docs/plans/2026-07-03-concept-organization.md`

