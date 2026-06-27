# Changelog — v0.5.2

**Date:** 2026-06-27
**Audit Reference:** `docs/audits/ML-AUDIT-2026-06-27-170028.md`
**Scope:** SPLiCE (Path B) — fixes applied from the adversarial ML audit swarm (7 agents).

## Summary

Fixed 4 critical (RED) + 5 high (ORANGE) + 6 medium (YELLOW) findings in the SPLiCE
sparse-decomposition pipeline. The LLM-judge evaluation path for Path B was previously
**non-functional** (the judge could not parse SPLiCE output, the documented self-check
clobbered the production 1515-image artifact, and the `clamp(min=0)` step corrupted the
decomposition). All blockers are resolved; Member 3 can now run the judge end-to-end and
produce a valid, falsifiable head-to-head comparison (Baseline vs Path A vs SPLiCE vs null).

## Fixes Applied

### Critical (RED)
- **F-001 — SAE-compatible output schema** (`src/concept_discovery/spliece.py`):
  `run()` now emits `{feature_id, name, activation}` instead of `{term, coefficient}`,
  matching the keys the LLM judge reads (`evaluate_llm_judge.py:434`). Previously 100% of
  SPLiCE concepts were silently skipped ("Nothing to evaluate").
- **F-002 — Self-check no longer clobbers production output** (`src/concept_discovery/spliece.py`):
  `__main__` now writes to `/tmp/spliece_selfcheck` via `dataclasses.replace`. Verified:
  `results/spliece/sample_explanations.json` stays at 1515 images after the self-check.
  (The 10-image stub that had already clobbered the working tree was regenerated away.)
- **F-003 — Replaced `clamp(min=0)` with NNLS-on-OMP-support** (`src/concept_discovery/spliece.py`):
  OMP still selects exactly k atoms (deterministic, exact L0); coefficients are now re-solved
  with `scipy.optimize.nnls` on that support (non-negative + reconstruction-optimal).
  Reconstruction error dropped from **2.03 → 0.35** (mean over 10 images; signal norm 0.91).
- **F-004 — Guide `% Aligned` recipe corrected** (`docs/LLM-JUDGE-COMPLETE-GUIDE.md`):
  the recipe read a non-existent `aligned_score` column (KeyError); now uses
  `(df['verdict']=='Aligned').mean()`, with a pointer to the canonical
  `judge_scores.json["aligned_rate"]`.

### High (ORANGE)
- **F-005 — Chance-floor null added**: new `scripts/generate_null_explanations.py` emits a
  random-k null explanation set (SAE schema, seeded, per-image count matched to SPLiCE). The
  guide now instructs comparing **LIFT** (pct_method / pct_null), not raw `% Aligned`.
- **F-006 — `run()` signature corrected**: `vocab_terms: list[str]` → `list[dict]` (the body
  indexes dicts).
- **F-007 — Length guards** (`src/concept_discovery/spliece.py`): `run()` raises on
  `test_embeddings`/`image_ids` and `vocab_terms`/`vocab_emb` count mismatches (no silent
  truncation / mislabeling).
- **F-008 — Anchored SpliCEConfig paths to project_root** (`src/config.py`): path defaults
  now resolve via `config.paths.*` (absolute), so they work from any CWD.
- **F-009 — Schema integration test** (`tests/unit/test_spliece.py`): added
  `test_run_emits_sae_schema` + `test_run_length_guards` (synthetic fixtures, run anywhere);
  updated the real-fixture integration tests to assert the new schema and use isolated dirs.

### Medium (YELLOW)
- **F-010 — Canonical image-id path**: `_load_image_ids()` reads `config.paths.test_image_ids_path`
  (the sidecar written in lockstep with the embeddings), not a stale `data/` duplicate.
- **F-011 — No dummy-ID fallback**: a missing id file now raises `FileError` instead of
  emitting `test_{i}` garbage that the judge cannot match.
- **F-012 — Removed redundant `ensure_dir(FILE)`** (`src/concept_discovery/spliece.py`).
- **F-013 — `pseudo_report` reads `config.explanation.explanation_top_n`** instead of a
  hardcoded `[:5]` (shared with the SAE path).
- **F-014 / F-015 — Reproducibility metadata**: `REPORT_run.md` now records git SHA, package
  versions, and sha256 of the (gitignored) input tensors, so a future run can verify the
  inputs match the committed output.

## ML Pipeline Changes
- **Data Pipeline:** image-id source canonicalized; length guards added; no dummy fallback.
- **Model / Decomposition:** clamp removed; NNLS-on-OMP-support restores reconstruction
  fidelity while keeping exact-L0 determinism. Concepts/image now mean ~13 (was ~18 under
  the corrupted clamp).
- **Evaluation:** output schema is now judge-compatible; a random-k null + LIFT comparison
  makes the cross-method verdict falsifiable.

## Verification
- Test suite: **10 passed** (7 unit incl. 2 new, 3 integration) — `.venv/bin/python -m pytest
  --import-mode=importlib tests/unit/test_spliece.py tests/integration/test_spliece_pipeline.py -v`.
- Reconstruction error: **0.35 mean** (was 2.03 with clamp).
- Self-check: writes to `/tmp`; `results/spliece/sample_explanations.json` confirmed at 1515 images.
- Regenerated artifacts: `results/spliece/sample_explanations.json` (1515, SAE schema),
  `results/null/sample_explanations.json` (1515 null), `REPORT_run.md`, `REPORT_coverage.md`.
- Regressions: none within the SPLiCE scope.

## Notes for Member 3
- SPLiCE output is now directly consumable by `src/evaluate_llm_judge.py` — point
  `EXPLANATIONS_PATH` at `results/spliece/sample_explanations.json` and run.
- Compute `% Aligned` with `(df['verdict']=='Aligned').mean()`, and report **LIFT** over the
  null (`results/null/sample_explanations.json`) — see `docs/LLM-JUDGE-COMPLETE-GUIDE.md` FASE 3.
- Stale tracked test artifacts `results/spliece_test_gap/` and `results/spliece_test_no_gap/`
  are no longer regenerated (tests now use tmp dirs); safe to `git rm -r` if desired.
