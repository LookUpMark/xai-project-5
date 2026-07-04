# Concept Organization — Pipeline Run

**Status**: Complete ✅
**Total time**: 1.5s
**Date**: 2026-07-04 00:47:06

## Run config

| param | value |
|-------|-------|
| source | spliece |
| tag | — |
| output dir | /home/marcantoniolopez/Documenti/github/xai-project-5/results/concept_organization |
| n_clusters | None |
| distance_threshold | None |
| linkage | average |
| radlex annotation | enabled |

## Metrics

| metric | value |
|-------|-------|
| n_concepts_active | 981 |
| n_clusters | 31 |
| mean_cluster_size | 31.65 |
| silhouette_cosine | 0.0947607010602951 |
| redundancy_reduction | 1.517 |
| radlex_coverage_pct | 99.5 |
| n_empty_images | 0 |

## Output files

- `concept_clusters.json` — clusters with RadLex ancestor labels
- `structured_explanations.json` — per-image concept families + redundancy
- `organization_metrics.json` — metrics snapshot

## Reproducibility

- git commit: `4ef1d42d6fe6e8aa18460e21b3602cc5d56d31db`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(explanations) [sample_explanations.json]: `1439f81d44f84997`
- sha256(vocab) [vocabulary.json]: `853bad633a866a1b`
- sha256(vocab_emb) [text_vocab_embeddings.pt]: `920c3e2e49ba0129`
- sha256(radlex) [radlex.csv]: `dec82b28e1b9ecfe`

## References

- Spec: `docs/design/proposals/2026-07-03-concept-organization.md`
- Plan: `docs/plans/2026-07-03-concept-organization.md`

