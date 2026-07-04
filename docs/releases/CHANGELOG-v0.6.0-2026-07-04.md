# Changelog — v0.6.0

**Date:** 2026-07-04
**Branch:** `feat/concept-organization`
**Scope:** Concept-organization extension (project-brief §3 extension; research gap #5).
**Spec:** `docs/design/proposals/2026-07-03-concept-organization.md`
**Plan:** `docs/plans/2026-07-03-concept-organization.md`

## Summary

Implemented the project-brief §3 "extension" — *filtraggio, clustering o organizzazione strutturata dei concetti scoperti* — as a **method-agnostic, dataset-portable** post-processing stage. It clusters discovered concepts (from SPLiCE, SAE-baseline, or SAE-hidden explanations) by RadLex text-embedding cosine similarity, annotates each cluster with a best-effort RadLex anatomical ancestor, and re-expresses per-image explanations as **concept families** with a redundancy metric. Output is standalone (does not feed the LLM judge).

This addresses Research Gap #5 (concepts treated as independent) — the one brief gap not covered by Methods A/B — and strengthens the *originalità/novità* and *methodology* rubric axes.

Developed test-driven (33 tests, red→green→commit per unit), with a final whole-branch review. Two real-data bugs in the RadLex ancestor selection were caught at the verification run and fixed (root-degeneracy → subtree-size canopy).

## Added

### New module: `src/concept_discovery/organize.py`
- **Dataclasses:** `ConceptSet` (normalized, method-agnostic), `ImageConcepts`, `Cluster`, `AnnotatedCluster` (with `display_label` property).
- **Adapters:** `from_spliece_explanations` and `from_sae_explanations` → both produce a `ConceptSet`. SPLiCE maps `feature_id`→vocab index directly; SAE excludes `is_dead`/`DEAD_FEATURE` features and coerces int `feature_id`→str for `concept_names.json` lookup. Both drop non-positive activations, skip names absent from the vocabulary (graceful), keep max activation on repeat, and enforce a vocab/vocab-emb count guard (mirrors spliece F-007).
- **Core (method-agnostic):**
  - `cluster_concepts` — `sklearn.AgglomerativeClustering(linkage='average', metric='cosine')`; `n_clusters` OR `distance_threshold` (compute_full_tree); default `n_clusters=max(2, round(√M))`; deterministic cluster ids via sorted-member-tuple tie-break; `_medoid` selection.
  - `annotate_radlex` — best-effort RadLex ancestor per cluster, with **two degeneracy guards** (see Fixes): leaf-root rejection + subtree-size canopy. Helpers `ancestor_rids` (cycle-safe), `_resolve_member_rid`, `_build_children_map`, `_descendant_count`.
  - `build_structured_explanations` — per-image concept families (`aggregate_activation`, `intra_redundancy`) + image-level `redundancy_score`.
  - `compute_metrics` — `n_concepts_active`, `n_clusters`, `mean_cluster_size`, `silhouette_cosine` (None when <2 clusters), `redundancy_reduction`, `mean_raw/families_per_image`, `radlex_coverage_pct`, `n_empty_images`.
  - `run` — orchestrates cluster→annotate→structured→metrics, writes 3 JSON outputs; emits `unresolved_terms` when a graph is given.
  - `__main__` self-check on a throwaway `/tmp/organize_selfcheck` dir (mirrors spliece F-002; never clobbers production).

### New driver: `scripts/run_concept_organization.py`
CLI mirroring `run_spliece.py`: `--source {spliece,sae-baseline,sae-hidden}`, `--tag`, `--n-clusters`, `--distance`, `--no-radlex`, plus full input overrides (`--explanations/--concept-names/--vocab/--vocab-emb/--radlex`) for dataset portability. Writes `REPORT_organization.md` with `_repro_info()` (git SHA + package versions + sha256 of inputs).

### New config: `OrganizeConfig` + `config.organize` singleton (`src/config.py`)
Frozen dataclass anchored to the `paths` singleton. `__post_init__` validates: `n_clusters`/`distance_threshold` mutual exclusivity, `linkage ∈ {average,complete,single}`, `metric='cosine'`.

### Tests (33 new, all green)
- `tests/unit/test_organize_config.py` (5)
- `tests/unit/test_organize.py` (26): dataclasses, both adapters, clustering (empty/singleton/distance-threshold/determinism), RadLex annotation (5 toy-graph + 1 subtree-size canopy regression), structured explanations, metrics, `run()`, unresolved_terms.
- `tests/integration/test_organize_pipeline.py` (2): end-to-end SPLiCE + SAE on synthetic data, with a determinism check against persisted JSON.

### Docs
- `docs/design/proposals/2026-07-03-concept-organization.md` (spec, §6.2 amended to track the canopy evolution).
- `docs/plans/2026-07-03-concept-organization.md` (TDD plan).
- `docs/CONCEPT-ORGANIZATION-REPORT.md` (implementation report + real-data results).

## Fixes Applied (during verification)

### Critical — RadLex ancestor root-degeneracy (2 iterations)
- **First attempt (active-set frequency canopy):** the original spec rejected only leaf-roots (RIDs with no parents). On the real 47k-RID RadLex graph this collapsed 41% of clusters to "RadLex entity". Amended to also reject ancestors supported by >50% of all active concepts. This caught the universal root but **mid-level generic terms still leaked** ("pathophysiologischer Befund" 4.7%, "anatomical entity" 81%, "organteil" 22%).
- **Final fix (subtree-size canopy, ontology-intrinsic):** reject candidate ancestors whose descendant count exceeds **1% of the graph** (~470 RIDs). Measured gap is clean: canopy-tier ≥2221 descendants vs specific ≤184. Active only for graphs ≥1000 RIDs (so unit-test toy graphs are unaffected). Result: **0 canopy terms** in real-data labels; clusters now show specific labels (lung imaging observation, tuberculosis, catheter, Lungenerkrankung, spine degeneration, implantable device, …) or honestly fall back to the medoid term (`None`) when no specific common ancestor exists.
- Also: candidate threshold changed to `max(2, ceil(0.5·n_resolved))` for n≥2 (a member's own RID alone never qualifies — it must be shared by ≥2 members).

### Final-review cleanups
- Emit `unresolved_terms` + `n_unresolved_terms` in metrics when a graph is given (spec §6.4; diagnostic for vocab/RadLex drift).
- Memoize `ancestor_rids` per `annotate_radlex` call (drop redundant recomputation).
- Remove dead `field` import; correct `_descendant_count` docstring (cycle-safe BFS); align spec §8 `vocab_emb_path` default with code.

## ML Pipeline Changes

- **Data Pipeline:** unchanged (extension consumes existing `sample_explanations.json` + `concept_names.json` + `vocabulary.json` + `text_vocab_embeddings.pt` read-only).
- **Model Architecture:** unchanged.
- **Evaluation:** standalone — does NOT feed the LLM judge. Judge can consume `structured_explanations.json` later if desired.

## Data note (working tree, not committed)

During verification, a subagent found `embeddings/standard/text_vocab_embeddings.pt` in a degraded state and regenerated the vocabulary (1030→1031) via `run_vocab_building_pipeline.py`. The regenerated vocab was *worse* (dropped CXR-relevant terms, added non-CXR terms). **Reverted `data/vocabulary.json` to the committed 1030-term version** and regenerated the standard embeddings to (1030, 512) for consistency. The `augmented/` variant remains broken (4, 512) — pre-existing, unrelated, not touched.

## Verification (real data)

| Source | Active concepts | Clusters | Resolved | Coverage | Redundancy reduction |
|---|---|---|---|---|---|
| SPLiCE | 981 | 31 | 10 (11 distinct labels) | 99.5% | 1.52 (12.89 → 8.50 families/img) |
| SAE baseline | — | 3 | 3 | 100% | 1.01 |
| SAE hidden | — | 4 | 3 | 100% | 1.03 |

Commands: `PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source spliece [--tag …]`.
