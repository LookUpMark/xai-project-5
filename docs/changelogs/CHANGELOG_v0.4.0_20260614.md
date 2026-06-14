# CHANGELOG v0.4.0 - 2026-06-14

## Summary

Reconcile the SAE concept-discovery **producer** (`generate_explanations.py`)
with the **LLM judge** (`evaluate_llm_judge.py`) data contract, and propagate a
real `image_id` (IU X-Ray PNG basename) end-to-end: extraction → train/test
split → `sample_explanations.json`. Previously the producer emitted a schema the
judge rejected (`findings` / `sample_idx` / `naming_confidence` → 3 `KeyError`),
and the bare `(N, 512)` embedding tensors carried no row identity for the judge
to join concepts back to `reports.csv`.

Secondary: fix the VLM extraction notebook to the local IU X-Ray layout so it
runs clean on Apple Silicon (MPS).

**Stats**: 15 files changed, +287 / -326 lines since v0.3.0.

**Companion audit**: `docs/audits/ML-AUDIT-2026-06-14.md`.

---

## Features

### `image_id` Propagation Chain

Identity is carried alongside the tensors via a human-readable sidecar JSON,
kept row-aligned by construction (same permutation at split time).

- **`src/extract_embeddings.py`**: `extract_visual_embeddings()` now captures
  `batch_paths` (previously discarded) and writes a `visual_image_ids.json`
  sidecar — the PNG basename per row — next to `visual_embeddings.pt`.
- **`src/config.py`**: `PathsConfig` gained three derived paths,
  `visual/train/test_image_ids_path` (all under `embeddings/`).
- **`src/utils.py`**: new `_split_ids()` helper; `split_embeddings()` gained
  optional `source_ids_path` / `train_ids_path` / `test_ids_path` and slices
  the id list with the **same** `train_idx` / `test_idx` used for the tensors.
  Guards: no-op when `source_ids_path` is None; `ValueError` if all three id
  paths aren't provided together; `ValueError` if the sidecar length ≠
  train+test rows (misalignment detector).
- **`src/autoencoder/train_sae.py`**: `prepare_split()` calls `utils._split_ids`
  when `config.paths.visual_image_ids_path` exists (split runs parallel to
  `split_embeddings`, same seed 42).

### Producer Contract Aligned to Judge

The judge + `docs/draft/plan.md:371-393` define the canonical schema; the
producer was moved to match (touches only the autoencoder side).

- **`src/autoencoder/generate_explanations.py`**: `generate_explanation()`
  returns `{top_k_concepts, pseudo_report}`; `run()` loads
  `test_image_ids.json` and stamps `image_id` per record (falls back to
  `sample_{idx}` if the sidecar is absent).
- **`src/autoencoder/contracts.py`**: `Finding` → `ConceptActivation`
  (`feature_id`, `name`, `activation`); `Explanation` →
  `{image_id, top_k_concepts, pseudo_report}`. Schema-doc only (stages still
  emit plain dicts; nothing imports these at runtime).
- **`src/autoencoder/__init__.py`**: re-exports `ConceptActivation`.

### New Schema (the contract)

```jsonc
// sample_explanations.json — one record per test image
{
  "image_id": "1000_IM-0003-1001.dcm.png",   // PNG basename = join key
  "top_k_concepts": [
    {"feature_id": 3303, "name": "infiltration", "activation": 1.234}
  ],
  "pseudo_report": "Findings suggest 'infiltration' ..."
}
```

---

## Bug Fixes

### VLM Extraction Notebook Pointed at Wrong Layout

`notebooks/vlm/extract_embeddings.ipynb` was configured for another machine's
Kaggle-extracted layout (`data/iu_xray/chest-xrays-indiana-university/`,
`indiana_reports.csv` in the root, device `cuda`). On the local machine the
staging cell raised `FileNotFoundError`. Repainted to the layout that matches
the default `EmbeddingConfig` (`data/iu_xray/images/images_normalized/`,
`data/iu_xray/reports/`), device auto-detected to `mps`. **Notebook-only fix**
— no VLM source files were modified.

---

## Documentation

- **`docs/audits/ML-AUDIT-2026-06-14.md`** — producer→judge contract audit:
  what changed, verification evidence (test slice, MPS notebook run, sidecar
  consistency), open issues.
- **`docs/changelogs/CHANGELOG_v0.4.0_20260614.md`** — this file.

---

## Testing

- **`tests/unit/test_generate_explanations.py`** — rewritten to the new schema
  (7 tests): asserts `top_k_concepts` / `pseudo_report` / `image_id` fields.
- **`tests/unit/test_extract_embeddings.py`** — added
  `test_saves_image_ids_sidecar` (sidecar written, aligned with the tensor).
- **`tests/integration/test_sae_pipeline.py`** — `test_full_explanation_flow`
  asserts the new schema.
- **`tests/integration/test_vlm_autoencoder.py`** — fixture repaired to the
  VLMConfig/EmbeddingConfig split (5-arg `extract_*`, output paths read from
  `embedding_config`, mock processor `max_length` kwarg) + explanation-schema
  assertions; 10 tests now pass (were setup-broken in v0.3.0).
- **`tests/unit/test_iu_xray_datasets.py`** — import fixed
  (`from xai_datasets.iu_xray import …`, repo-root on `sys.path`); 12 tests now
  collect & pass (were collection-broken since the `datasets/`→`xai_datasets/`
  rename).

---

## Validation

- **89 passed, 8 skipped** (`tests/` minus `test_llm_judge.py`; `test_iu_xray_datasets.py` +12 and `test_vlm_autoencoder.py` +10 now collect/pass — were collection/setup-broken in v0.3.0).
- **VLM notebook**: 8/8 cells run clean on MPS, no errors; produces
  `visual_embeddings.pt` (7470×512), `text_embeddings.pt`, and
  `visual_image_ids.json` (7470 basenames).
- **Sidecar consistency** (after propagating ids through the split):
  - counts match tensor rows — visual 7470, train 5976, test 1494;
  - `train ∩ test = ∅`, `train ∪ test = visual`;
  - no `sample_` fallback ids;
  - row-aligned: seed-42 split reproduces train/test from visual exactly, so
    the id lists are row-aligned with the tensors.
- **Embedding integrity**: all splits L2-normalized (mean norm 1.0000);
  `text_vocab_embeddings.pt` (310) matches `data/vocabulary.json` (310) →
  concept naming unblocked.

---

## Files Touched (15)

| File | Δ | Notes |
|------|---|-------|
| `.gitignore` | +1 | ignore `HANDOFF.md` |
| `notebooks/vlm/extract_embeddings.ipynb` | 144 | local IU X-Ray layout, MPS |
| `notebooks/autoencoder/pipeline.ipynb` | 105 | split propagates ids; explanation emits `image_id` |
| `notebooks/autoencoder/mock/pipeline_smoke_test.ipynb` | 105 | same two cells |
| `src/extract_embeddings.py` | 19 | capture `batch_paths`, write sidecar |
| `src/utils.py` | 59 | `_split_ids`, `split_embeddings` id args |
| `src/autoencoder/generate_explanations.py` | 53 | new schema + `image_id` |
| `src/autoencoder/contracts.py` | 23 | `ConceptActivation`, `Explanation` |
| `src/autoencoder/train_sae.py` | 13 | `prepare_split` splits sidecar |
| `src/config.py` | 9 | `*_image_ids_path` |
| `src/autoencoder/__init__.py` | 4 | re-export `ConceptActivation` |
| `tests/unit/test_generate_explanations.py` | 32 | new schema (7 tests) |
| `tests/unit/test_extract_embeddings.py` | 31 | sidecar test |
| `tests/integration/test_sae_pipeline.py` | 10 | schema assertion |
| `tests/integration/test_vlm_autoencoder.py` | 5 | schema assertion only |

**15 files changed, +287 / -326.**

---

## Known Limitations / Next Steps

- Judge still can't run end-to-end (environmental, not contract): `reports.csv`
  needs building (`image_id` = PNG basename), `langgraph` needs installing,
  MedGemma is gated and needs GPU. See audit "Open Issues".
- `test_vlm_autoencoder.py` and `test_iu_xray_datasets.py` are now fixed (were
  pre-existing breakage); only `test_llm_judge.py` stays uncollected (langgraph).
- Vocab filename mismatch reconciled in source: `VocabularyConfig` defaults now
  match `PathsConfig` (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt`).
  **Caveat:** the producer's *schema* still diverges — it writes
  `{term,similarity_score,source}` dicts while consumers read a string list, so
  re-running the builder would overwrite the committed `data/vocabulary.json`.
  Reconcile the schema separately before rebuilding.
