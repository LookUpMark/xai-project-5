# Concept Organization Extension — Implementation Report

> **Branch:** `feat/concept-organization` (18 commits, split from `dev` @ `fce2f0c`)
> **Spec:** `docs/design/proposals/2026-07-03-concept-organization.md`
> **Plan:** `docs/plans/2026-07-03-concept-organization.md`
> **Changelog:** `docs/releases/CHANGELOG-v0.6.0-2026-07-04.md`
> **Date:** 2026-07-04

## 1. What was built

A **method-agnostic, dataset-portable** post-processing stage that turns flat top-k concept lists into **structured concept families**, addressing project-brief §3 ("filtraggio, clustering o organizzazione strutturata dei concetti scoperti") and research gap #5 (concepts treated as independent).

It works on any concept source whose explanations expose the shared `{feature_id, name, activation}` schema — currently SPLiCE, SAE-baseline (512-d), and SAE-hidden (768-d) — and any dataset whose vocabulary terms are present in the supplied vocab embeddings.

### Pipeline

```
sample_explanations.json (+ concept_names.json for SAE)
        │   adapter (from_spliece_explanations | from_sae_explanations)
        ▼
   ConceptSet  ──cluster_concepts──▶  clusters
        │                                  │ annotate_radlex (subtree-size canopy)
        │                                  ▼
        │                       AnnotatedCluster[]
        ├─build_structured_explanations──▶ per-image families + redundancy
        └─compute_metrics + unresolved_terms──▶ organization_metrics.json
run() writes: concept_clusters.json, structured_explanations.json, organization_metrics.json
```

## 2. Components delivered

| Component | Location | Responsibility |
|---|---|---|
| `OrganizeConfig` + `config.organize` | `src/config.py` | Frozen config; validates n_clusters/distance exclusivity, linkage, metric. |
| Module | `src/concept_discovery/organize.py` | Dataclasses + adapters + core + `run()` + `__main__` self-check. |
| Driver | `scripts/run_concept_organization.py` | CLI (`--source`, `--tag`, `--n-clusters`, `--distance`, `--no-radlex`, input overrides); writes `REPORT_organization.md`. |
| Unit tests | `tests/unit/test_organize.py` (26), `tests/unit/test_organize_config.py` (5) | One test per unit, real tensors, no mocked system-under-test. |
| Integration tests | `tests/integration/test_organize_pipeline.py` (2) | End-to-end SPLiCE + SAE on synthetic data + determinism check. |
| Spec | `docs/design/proposals/2026-07-03-concept-organization.md` | Design (§6.2 amended to track the canopy evolution). |
| Plan | `docs/plans/2026-07-03-concept-organization.md` | 13-task TDD plan. |

**Total: 33 new tests, all green.**

## 3. Method-agnostic design

The core operates **only** on a normalized `ConceptSet` (`names`, `embeddings`, `name_to_idx`, `per_image`). Two thin adapters produce it:

- **SPLiCE adapter:** `feature_id` = vocab index; `name` = vocab term directly.
- **SAE adapter:** excludes `is_dead` / `DEAD_FEATURE` features (via `concept_names.json`); coerces int `feature_id` → str; otherwise identical normalization.

Both share `_build_term_to_idx` (count guard) and `_finalize_concept_set` (sorted active names → deterministic ids; empty-case handled). Adding a fourth concept source later = one new adapter, zero core changes.

## 4. The RadLex ancestor selection (the hard part)

**Goal:** label each cluster with a meaningful RadLex anatomical ancestor (not the trivial root).

**Two degeneracy guards** (the spec evolved through three iterations — see §5):

1. **Leaf-root rejection** — candidate RIDs with no parents are rejected.
2. **Subtree-size canopy (ontology-intrinsic)** — candidate RIDs whose descendant count exceeds **1% of the graph** are rejected as trivially generic. Active only for graphs ≥ 1000 RIDs (unit-test toy graphs unaffected).

Candidate selection: an ancestor must be supported by `≥ max(2, ceil(0.5·n_resolved))` members (≥2 sharing required, so a member's own RID alone never qualifies). Among survivors, pick the **most specific** (deepest in the candidate sub-DAG); tie-break by support desc, then shortest label. No survivor → `radlex_label = None` → cluster label falls back to the medoid term (honest).

Reuses `src/vocabulary_building/radlex_support.py` (`load_radlex_graph` → `RadLexGraph` with `child_to_parents`, `rid_to_label`, `label_to_rids`).

## 5. Bugs caught at verification (and fixed)

The unit tests use a 4–16 node toy graph and could not expose real-ontology failure modes. The real-data run (T13) caught two:

### Bug A — root collapse (41% of clusters → "RadLex entity")
The original "most frequent ancestor" rule always maximized at the universal root (every member's ancestor chain includes it). **Fix iteration 1:** majority + leaf-root stoplist. Synthetic tests passed, but…

### Bug B — mid-level canopy leakage ("pathophysiologischer Befund" ×5, "anatomical entity" ×3, …)
Leaf-root rejection only catches the single root. RadLex has a canopy of generic nodes *with parents* that still leaked. The active-set-frequency canopy (>50% of active concepts) only caught the universal root. **Fix iteration 2 (final):** subtree-size canopy. Measured descendant-count gap on the real graph:

| Label | Descendants | % graph | Verdict |
|---|---|---|---|
| RadLex entity | 45,958 | 98.0% | canopy (reject) |
| anatomical entity | 38,178 | 81.4% | canopy (reject) |
| anatomische Struktur | 34,207 | 73.0% | canopy (reject) |
| organteil | 10,224 | 21.8% | canopy (reject) |
| pathophysiologischer Befund | 2,221 | 4.7% | canopy (reject) |
| **implantable device** | 184 | 0.39% | **specific (keep)** |
| Lungenerkrankung | 47 | 0.10% | specific (keep) |
| lung imaging observation | 16 | 0.03% | specific (keep) |
| tuberculosis | 2 | 0.00% | specific (keep) |

A 1% threshold (~470 descendants) sits cleanly in the gap. **Result: 0 canopy terms in real-data labels.**

## 6. Real-data results

| Source | Active | Clusters | Resolved (specific label) | RadLex coverage | Redundancy reduction |
|---|---|---|---|---|---|
| SPLiCE | 981 | 31 | 10 / 31 (11 distinct labels) | 99.5% | **1.52** (12.89 → 8.50 families/img) |
| SAE baseline (512-d) | — | 3 | 3 / 3 | 100% | 1.01 |
| SAE hidden (768-d) | — | 4 | 3 / 4 | 100% | 1.03 |

SPLiCE resolved labels: *Lungenerkrankung, implantable device, spine degeneration, Mediastinaldrainage, Mechanische Erankung, Medizinprodukt, hangman fracture, imaging subspecialty, no posterior acoustic features, ruptured aneurysm.*

Clusters lacking a specific common ancestor honestly return `None` (medoid fallback) rather than a fake generic label — this is correct behavior, not a failure.

**Silhouette (SPLiCE) = 0.095** — low, indicating fuzzy clusters; a property of the data/clustering, not the annotation. Tuning `--n-clusters` (fewer, larger clusters) would yield more shared specific ancestors.

## 7. Process

- **Brainstorming → spec** (approved) → **writing-plans → 13-task TDD plan** → **subagent-driven-development**: one implementer subagent per task + spec/quality review; 2 whole-branch reviews.
- **TDD:** every unit written red → green → commit. 33 tests.
- **Determinism verified** against persisted JSON (integration test runs `run()` twice, asserts identical cluster members).
- **Convention adherence:** mirrors `spliece.py` / `run_spliece.py` (sys.path.insert + bare imports, frozen config + `replace`, `utils.load_tensor` weights_only=True, throwaway self-check dir, `_repro_info`).

## 8. Honest limitations

- Best-effort RadLex annotation: names match RadLex *preferred labels* (no synonym matching — YAGNI; 99.5% coverage on real data). The `unresolved_terms` metric surfaces drift.
- Silhouette is low on SPLiCE (0.095) — clusters are fuzzy; the 21/31 `None` clusters reflect members too diverse to share a specific ancestor.
- Does not feed the judge (M-007 still open on the judge side; out of scope here).

## 9. How to run

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source spliece
PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source sae-baseline --tag baseline
PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source sae-hidden --tag hidden
# dataset-portable overrides:
PYTHONPATH=src:. .venv/bin/python scripts/run_concept_organization.py --source spliece \
    --explanations <path> --vocab <path> --vocab-emb <path> --radlex <path>
```

Tests:
```bash
PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py tests/unit/test_organize_config.py tests/integration/test_organize_pipeline.py -q   # 33 passed
```
