# CHANGELOG v0.5.0 — 2026-06-26

## Summary

Implements **Path A** of the project strategy: a Top-K SAE on BiomedCLIP's **768-d
pre-projection CLS hidden state** (the centrepiece method proposed to fix the 512-d baseline's
non-identifiability, per ML-AUDIT-2026-06-25). Additive — the 512-d baseline is untouched.

**Stats:** 11 files changed, +1409 lines since v0.4.0.

**Implementation plan:** `docs/plans/2026-06-26-sae-hidden-768.md`
**Verification audit:** `docs/audits/ML-AUDIT-2026-06-26.md`

---

## Features

### Path A — SAE on the 768-d hidden state (`src/sae_hidden/`)
New package with five runnable stages, each emitting a markdown REPORT under
`results/sae_hidden/`:

- **Extraction** — grabs `vision_model(...).last_hidden_state[:,0,:]` (the CLS token *before*
  the frozen `visual_projection`), raw (no per-sample L2 norm, residual-stream convention),
  group-aware train/test split (0 study leakage).
- **Training** — 5 seeds, audit-corrected hyperparameters (`dict_size=2048`, `steps=8000`,
  `lr=5e-5`, `k=32`; M-002/M-006).
- **Naming bridge** — projects 768-d decoder rows into the 512-d shared space via the **frozen**
  `visual_projection` (option (a) of audit M-001), then cosine-matches against RadLex.
- **Stability** — cross-seed Jaccard on the held-out test set vs the analytical chance floor.
- **Explanations** — per-sample pseudo-reports for the LLM judge.

### Config (`src/config.py`)
- New frozen `SAEHiddenConfig` (validated) + hidden paths in `PathsConfig`. Baseline `SAEConfig`
  and `EmbeddingConfig` unchanged.

### Data fix
- Regenerated `embeddings/standard/text_vocab_embeddings.pt` from the current 1030-term
  vocabulary (was stale at 508 rows, mismatching `data/vocabulary.json`). Old file backed up as
  `text_vocab_embeddings.bak508.pt`.

---

## Results (real, on IU X-Ray)

| Metric | Path A (768-d) | Baseline (512-d) | Verdict |
|---|---|---|---|
| Dead features (activation) | **12.9%** | 40–60% | ✅ improved |
| Naming mean cosine | **0.4711** | 0.395 (random 0.372) | ✅ improved |
| Cross-seed Jaccard | 0.0083 (floor 0.0079) | 0.0038 (floor 0.0039) | ❌ still at chance floor |

**Honest outcome:** two of three diagnostics improve; cross-seed stability does not — the SAE
remains non-identifiable at this data scale (~5955 imgs, audit M-002). See the audit for the full
analysis, including why the "2.2× Jaccard lift" is a dict-size artifact, not identifiability.

---

## Testing

- Unit tests: `tests/unit/test_sae_hidden.py` — **10/10 passing** (config validation, hidden
  paths, frozen-projection bridge math, 768-d train+encode path).
- End-to-end: all 5 stages run on real data; 5 REPORTs + `concept_names.json`,
  `stability_analysis.json`, `sample_explanations.json` produced.

---

## Files Touched (11)

| File | Δ | Notes |
|---|---|---|
| `src/config.py` | +80 | `SAEHiddenConfig` + hidden paths (modify) |
| `src/sae_hidden/__init__.py` | +11 | package |
| `src/sae_hidden/reports.py` | +46 | markdown report writer |
| `src/sae_hidden/extract_hidden.py` | +183 | 768-d extraction + split |
| `src/sae_hidden/train_hidden.py` | +144 | 5-seed training |
| `src/sae_hidden/naming_hidden.py` | +234 | frozen-projection naming bridge |
| `src/sae_hidden/stability_hidden.py` | +136 | cross-seed Jaccard |
| `src/sae_hidden/generate_explanations_hidden.py` | +124 | per-sample explanations |
| `tests/unit/test_sae_hidden.py` | +153 | unit tests |
| `docs/plans/2026-06-26-sae-hidden-768.md` | +146 | implementation plan |
| `docs/audits/ML-AUDIT-2026-06-26.md` | +152 | verification audit |

---

## Known limitations

- Cross-seed stability remains at the analytical chance floor (M-002 data-scale confound).
- Naming terms are broad RadLex anatomy; clinically loose for chest X-ray (dictionary ceiling).
- Extraction/training assume MPS/CUDA; CPU is untested (slow).
