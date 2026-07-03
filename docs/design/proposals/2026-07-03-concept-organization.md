# Concept Organization Extension — Design Spec

> **Date:** 2026-07-03
> **Status:** Approved design (pre-implementation)
> **Branch:** `feat/concept-organization`
> **Related:** `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md` §4.2, `docs/design/PROJECT-STRATEGY.md` §4.4, `docs/requirements/PROJECT-BRIEF.md` §3, `src/concept_discovery/spliece.py`, `src/vocabulary_building/radlex_support.py`
> **Spec location note:** written to `docs/design/proposals/` (repo convention) instead of the brainstorming skill's default `docs/superpowers/specs/`; implementation plan will live in `docs/plans/`.

---

## 1. Goal

Implement the project-brief §3 "extension" — *"filtraggio, clustering o organizzazione strutturata dei concetti scoperti"* — as a **method-agnostic, dataset-portable** post-processing stage that:

1. Clusters discovered concepts by semantic similarity in RadLex text-embedding space.
2. Annotates each cluster with a best-effort RadLex anatomical ancestor.
3. Re-expresses per-image explanations as **concept families** instead of flat top-k lists, and quantifies redundancy reduction.

**Rubric value:** addresses Research Gap #5 (concepts treated as independent) — the one brief gap not covered by Methods A/B; strengthens *originalità/novità* and *methodology* axes; provides structured (non-flat) explanations, a literally-requested extension.

## 2. Scope

- **Concept sources (universal):** SPLiCE (`results/spliece/`), SAE baseline 512-d (`results/baseline/`), SAE hidden 768-d (`results/sae_hidden/`). All three expose the same per-image schema (`{feature_id, name, activation}`); SAE additionally exposes `concept_names.json` (`{fid: {name, score, is_dead, candidates}}`).
- **Dataset-portable:** all input paths overridable via CLI; defaults resolve from `config.paths` / `results/<source>/`. Any dataset whose explanations reference terms present in the supplied vocabulary works unchanged.
- **Structure depth:** embedding-cosine agglomerative clustering (core, always works) + best-effort RadLex ancestor annotation (enrichment, graceful fallback).
- **Output:** standalone analysis artifacts + per-sample structured explanations. **Does not** feed or modify the LLM judge (other team; M-007 open).

## 3. Non-goals

- No new SAE, no retraining, no embedding extraction.
- No judge coupling / no `pseudo_report` restructuring for the judge.
- No requirement that every vocab term resolves to a RadLex RID (graceful skip).
- Not a notebook: deliverable is `scripts/run_concept_organization.py` (repo `run_*.py` convention).

## 4. Architecture (3 layers)

```
┌─ Adapter layer ──────────────────────────────────────────────┐
│ from_spliece_explanations(explanations, vocab, vocab_emb)     │
│ from_sae_explanations(explanations, concept_names, vocab, …)  │
│        └──▶ normalized ConceptSet                             │
├─ Core layer (method-agnostic) ────────────────────────────────┤
│ cluster_concepts(concept_set, cfg) ──▶ clusters               │
│ annotate_radlex(clusters, radlex_graph) ──▶ annotated clusters│
│ build_structured_explanations(concept_set, clusters) ──▶ per-image families + redundancy │
│ compute_metrics(...) ──▶ organization_metrics                 │
│ run(cfg, …) ──▶ orchestrator returning serializable dict      │
├─ Driver layer ────────────────────────────────────────────────┤
│ scripts/run_concept_organization.py (--source, --tag, …)      │
│        └──▶ results/concept_organization[_{tag}]/*.json + REPORT_organization.md │
└───────────────────────────────────────────────────────────────┘
```

Each unit has one purpose, a well-defined interface, and is independently testable.

## 5. Data model

### 5.1 Normalized `ConceptSet` (dataclass)

The core operates **only** on this; adapters produce it.

```python
@dataclass
class ConceptSet:
    names: list[str]                       # unique active concept names (vocab terms), len M
    embeddings: torch.Tensor               # (M, 512) text embeddings, row i ↔ names[i]
    name_to_idx: dict[str, int]            # name → row index
    per_image: list[ImageConcepts]         # per-image activation records

@dataclass
class ImageConcepts:
    image_id: str
    activations: dict[str, float]          # name → activation (>0); subset of names
```

**Invariants (enforced):**
- `embeddings.shape[0] == len(names) == len(name_to_idx)`.
- `names` unique; `name_to_idx[name] == index in names`.
- Every key in every `activations` dict ∈ `set(names)` (adapter guarantees).
- `embeddings` rows are the BiomedCLIP text embeddings of `names`, looked up from `vocab_emb` via the vocabulary term index.

### 5.2 Inputs (existing schemas — consumed read-only)

**SPLiCE / SAE `sample_explanations.json`:**
```json
[{"image_id": "...", "top_k_concepts": [{"feature_id": int, "name": str, "activation": float}], "pseudo_report": "..."}]
```
**SAE `concept_names.json`:**
```json
{"<fid>": {"name": str, "score": float, "is_dead": bool, "candidates": [...]}}
```
**`data/vocabulary.json`:** `[{"term": str, "similarity_score": float, "source": str}, …]`
**`text_vocab_embeddings.pt`:** `(V, 512)` tensor, row i ↔ `vocabulary[i]["term"]`.

### 5.3 Adapter semantics

- **`from_spliece_explanations`:** SPLiCE `feature_id` = vocab index; `name` = vocab term directly. Active set = union of names across all images' `top_k_concepts` (activation > 0). Build `ConceptSet` from those names + their `vocab_emb` rows.
- **`from_sae_explanations`:** `feature_id` = SAE feature index; `name` is the assigned vocab term. Active set = union of names across images, **excluding** names of `is_dead` features and the `DEAD_FEATURE` placeholder. Use `concept_names.json` to confirm liveness when available.
- **Shared normalization rule:** resolve each `name` to a vocab index via a `term → idx` map built from `vocabulary.json`. If a name is **not found** in the vocabulary (drift, placeholder, post-vocab-expansion mismatch), **skip it** (drop from that image's activations; do not crash). If skipping empties an image, keep the image with an empty `activations` dict and report it in metrics.

## 6. Core component contracts

### 6.1 `cluster_concepts(concept_set, cfg) -> list[Cluster]`

- **Algorithm:** `sklearn.cluster.AgglomerativeClustering(linkage='average', metric='cosine', ...)`.
- **Stopping rule:** `n_clusters` (default) **or** `distance_threshold` (if `--distance` passed); exactly one mode. Default `n_clusters` derived from `√M` heuristic when neither given (reported in metrics, overridable). No post-hoc cluster merging — agglomerative output is taken as-is (tiny clusters are real signal, not noise).
- **Determinism:** agglomerative is deterministic given fixed input; no RNG. Tie-breaking by sorted name order so cluster ids are stable across runs.
- **Output `Cluster`:** `{cluster_id: int, members: list[str], medoid: str}` — `medoid` = member whose embedding has max mean cosine to other members (label fallback).

### 6.2 `annotate_radlex(clusters, radlex_graph, label_to_rid) -> list[AnnotatedCluster]`

- **Reuse** `vocabulary_building.radlex_support.load_radlex_graph` → `RadLexGraph` (exposes `child_to_parents`) and `ancestor_labels(rid)`.
- **`label_to_rid` map:** built once from `radlex.csv` — `Preferred Label → RID` and each `Synonyms` (pipe/split) → RID, lowercased. Case-insensitive exact match.
- **Per cluster — ancestor selection rule (root-degeneracy guard):**
  1. For each resolved member, get `ancestor_labels(rid)` (set of ancestor labels).
  2. Compute **support(label)** = number of resolved members whose ancestor set contains it.
  3. Candidates = labels with `support ≥ ceil(0.5 × n_resolved)` (majority agreement).
  4. **Reject ontology roots** (a stoplist of top-level RadLex RIDs/labels, e.g. the universal root) as trivially uninformative — without this guard every cluster inherits the root and all labels collapse to one.
  5. Among remaining candidates pick the **most specific** = the candidate that is itself an ancestor of the fewest other candidates (deepest in the sub-DAG); tie-break by highest support, then shortest label.
  6. If no candidate survives (only the root qualifies, or zero resolved members) → `radlex_label = None`; the cluster label falls back to the medoid term.
- **Why not "most frequent":** every member's ancestor chain includes the root, so raw frequency always maximizes at the root → all clusters get the same useless label. Majority + specificity + root-stoplist avoids this.
- **Coverage tracked:** `n_resolved / n_members` per cluster and globally.

### 6.3 `build_structured_explanations(concept_set, clusters) -> list[StructuredExplanation]`

For each image:
- Group its active concepts by cluster → one **family** per cluster present.
- `StructuredExplanation = {image_id, families: [Family], redundancy_score}`.
- `Family = {cluster_id, label, radlex_label, concepts: [{name, activation}], aggregate_activation, intra_redundancy}`.
  - `aggregate_activation` = sum of member activations.
  - `intra_redundancy` = `len(members) / expected_unique` ≥ 1 (multiple members of same family = redundant).
- `redundancy_score` (image-level) = `n_raw_concepts / n_distinct_families` (≥ 1; higher = more redundant pre-organization).

### 6.4 `compute_metrics(...) -> dict`

- `n_concepts_active`, `n_clusters`, `mean_cluster_size`, `silhouette_cosine` (guard ≥ 2 clusters & cluster size > 1 else `None`),
- `redundancy_reduction` = `mean(raw_concepts_per_image) / mean(distinct_families_per_image)` (the headline gain),
- `radlex_coverage_pct` = resolved members / total active members,
- `unresolved_terms` (sample, first 20),
- `n_empty_images` (images left with no activatable concept after normalization).

### 6.5 `run(cfg, concept_set_inputs...) -> dict`

Orchestrates cluster → annotate → structured → metrics; writes all output files; returns the metrics dict for the driver's report. Pure-ish: I/O only at this boundary.

## 7. Output files (`results/concept_organization[_{tag}]/`)

| File | Schema |
|---|---|
| `concept_clusters.json` | `[{cluster_id, label, radlex_label, members:[str], size}]` |
| `structured_explanations.json` | `[{image_id, families:[Family], redundancy_score}]` |
| `organization_metrics.json` | metrics dict (§6.4) + run config snapshot |
| `REPORT_organization.md` | markdown report mirroring `REPORT_run.md` style + `_repro_info()` |

## 8. Config (`src/config.py`)

Add frozen `OrganizeConfig` (mirror `SpliCEConfig` style, anchored to `paths` singleton):

```python
@dataclass(frozen=True)
class OrganizeConfig:
    n_clusters: int | None = None          # None → √M heuristic
    distance_threshold: float | None = None  # mutually exclusive with n_clusters
    linkage: str = "average"
    metric: str = "cosine"
    radlex_csv_path: Path = field(default_factory=lambda: paths.data_dir / "radlex.csv")
    vocab_path: Path = field(default_factory=lambda: paths.vocab_labels_path)
    vocab_emb_path: Path = field(default_factory=lambda: spliece.vocab_emb_path)  # reuse
    output_dir: Path = field(default_factory=lambda: paths.results_dir / "concept_organization")
```

`__post_init__` validation: `n_clusters` and `distance_threshold` mutually exclusive; `linkage` ∈ {average, complete, single}; `metric='cosine'` (only supported value for this design).

Module-level singleton `config.organize`. Document the `paths`-ordering constraint (same pattern as existing configs).

## 9. Driver `scripts/run_concept_organization.py`

Mirror `run_spliece.py` structure exactly:
- `sys.path.insert` for `src/` + repo root; `from __future__ import annotations`.
- `parse_args()`: `--source {spliece,sae-baseline,sae-hidden}` (required), `--tag`, `--n-clusters`, `--distance`, `--no-radlex`, plus input overrides `--explanations`, `--concept-names`, `--vocab`, `--vocab-emb`, `--radlex`.
- `main()`: resolve default input paths from `config` by `--source`; build `ConceptSet` via the right adapter; call `run()`; write `REPORT_organization.md` (sections: run config, algorithm, stages, output files, reproducibility via `_repro_info`, metrics summary, references).
- `_repro_info()` mirrors spliece (git SHA, package versions, sha256 of inputs).
- `if __name__ == "__main__": main()`.

## 10. Guard rails (mirror `spliece.py` findings)

- **F-007-style:** hard `ValueError` on count mismatches (vocab vs vocab_emb rows; explanations vs image_ids).
- **F-010/F-011-style:** read inputs from canonical paths; no dummy fallbacks (missing input = hard error, never silent garbage).
- **Unresolved-name skip:** names absent from vocabulary are dropped, not crashed on (handle `DEAD_FEATURE`, drift).
- **RadLex graceful:** unresolved RIDs never raise; `radlex_label = None` fallback.
- **SAFE load:** `utils.load_tensor` (`weights_only=True`) for all `.pt`.
- **Determinism:** clustering deterministic; cluster ids stable via sorted-name tie-break.
- **Self-check (`__main__`):** runs on a throwaway dir (`/tmp/organize_selfcheck`) — never clobber production output (mirror spliece F-002).
- **Isolation:** `--tag` always writes to `results/concept_organization_{tag}/`.

## 11. TDD test plan

Unit (`tests/unit/test_organize.py`) — one test per unit, written **first** (red), then implementation (green):

1. `from_spliece_explanations`: builds correct `ConceptSet`; count guard raises on mismatch; unresolved name skipped.
2. `from_sae_explanations`: dead features & `DEAD_FEATURE` excluded; feature→name via `concept_names`; unresolved skipped.
3. `cluster_concepts`: deterministic (same input → same ids); `n_clusters` respected; singleton cluster when `M=1`.
4. `annotate_radlex`: known RID resolves to expected ancestor; unresolved → `None`; never raises on empty cluster.
5. `build_structured_explanations`: family aggregation correct; `redundancy_score` arithmetic; image with no active concepts handled.
6. `compute_metrics`: silhouette guarded (`None` when < 2 clusters); `redundancy_reduction` ratio correct.

Integration (`tests/integration/test_organize_pipeline.py`): end-to-end `run()` on mock vocab + embeddings + tiny explanations → asserts all 3 output files written with valid schemas, no crash, deterministic across two runs.

Run: `PYTHONPATH=src:. .venv/bin/pytest tests/unit/test_organize.py tests/integration/test_organize_pipeline.py -q`.

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Vocab terms don't resolve to RadLex RIDs (fuzzy) | Best-effort + `None` fallback; report coverage % honestly |
| Clustering non-identifiable at small active-set | Deterministic + medoid label; `min_cluster_size`; report silhouette |
| Concept name drift across methods/datasets | Unresolved-name skip; count guards |
| `concept_names.json` key type (str fid) vs explanations (int fid) | Adapter coerces both to str for lookup; tested |
| RadLex import path (`vocabulary_building.radlex_support`) under `PYTHONPATH=src` | Verified resolvable; test imports it |

## 13. Deliverable

- `src/concept_discovery/organize.py` (core + adapters + `run`)
- `scripts/run_concept_organization.py` (driver)
- `OrganizeConfig` + `config.organize` in `src/config.py`
- `tests/unit/test_organize.py`, `tests/integration/test_organize_pipeline.py`
- Output produced on real SPLiCE + SAE results after implementation; verified via run.

## 14. References

- `src/concept_discovery/spliece.py` — module pattern, output schema, guard-rail precedents.
- `scripts/run_spliece.py` — driver pattern, `_repro_info`, report style.
- `src/vocabulary_building/radlex_support.py` — `RadLexGraph`, `load_radlex_graph`, `ancestor_labels`.
- `data/radlex.csv` — ontology source (`Parents`, `Synonyms`, `Preferred Label` columns).
- `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md` §4.2 — original extension proposal.
- `docs/requirements/PROJECT-BRIEF.md` §3 — extension requirement.
