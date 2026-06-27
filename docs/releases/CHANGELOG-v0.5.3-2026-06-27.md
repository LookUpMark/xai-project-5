# Changelog — v0.5.3

**Date:** 2026-06-27
**Audit Reference:** docs/audits/ML-AUDIT-2026-06-27-180529.md
**Scope:** Path A (768-d SAE) + Baseline (512-d SAE) judge-readiness.

## Summary

Fixed the blocker that prevented judging Path A and hardened the baseline-vs-Path-A comparison. A 9-agent audit found Path A's committed `sample_explanations.json` was stale/corrupt (3030 records = 1515 images × 2, non-reproducible). Regenerated it (→1515), made the baseline pipeline write to `results/baseline/` reproducibly, added a k=5 chance-floor null for the 5-concept methods, deduped judge eval pairs, and seeded generation for determinism.

## Fixes Applied

### Critical (RED)
- **F-001:** Regenerated `results/sae_hidden/sample_explanations.json` from the committed `sae_seed42` model — was 3030 duplicated records, now 1515 unique, reproduces from current code (F-011 `concept_names.json` regenerated in the same run). (`results/sae_hidden/`)

### High (ORANGE)
- **F-002:** Baseline stages (`concept_naming.py`, `generate_explanations.py`, `stability_analysis.py`) now resolve output paths lazily inside `run()`, so `baseline_variant`'s `results_dir` swap is honored → outputs land in `results/baseline/` (previously captured `results/` at import). (`src/autoencoder/`)
- **F-004:** Added `results/null_k5/sample_explanations.json` (1515 × 5 concepts) — a count-matched chance floor for the 5-concept Baseline/Path-A methods; the existing k≈13 null remains for SPLiCE. LIFT should be computed per-method against its own count-matched null.
- **F-005:** `evaluate_llm_judge.py` now dedups `(image_id, concept_name)` pairs on fresh runs, not only on `--resume` (`done_keys.add(key)`). Prevents inflated denominators from any repeated-key explanations file.
- **F-009:** `set_global_seed` added before generation in both `generate_explanations.py` / `generate_explanations_hidden.py` and before extraction in `extract_hidden.py` — closes the determinism gap that let the stale/doubled artifact ship undetected.

## ML Pipeline Changes

- **Data Pipeline:** seeded generation + extraction; baseline outputs now reproducibly routed to `results/baseline/`.
- **Model Architecture:** unchanged.
- **Evaluation:** added k=5 null; judge pair-dedup unified across fresh/resume runs.

## Verification

- Path A output: 1515 records, 1515 unique ids, 0 duplicates, schema `{feature_id,name,activation}` valid; first concept feat 876 @6.4122 reproduces from current code.
- All 4 judge inputs: 1515 unique each (baseline, path_a, null k≈13, null_k5).
- Tests: `tests/test_llm_judge.py` + `tests/unit/test_iu_xray_datasets.py` → **42 passed**. All 6 edited files compile.
- Note: `tests/unit/test_spliece.py` + integration fail to *collect* under pytest importlib mode (`cannot import name 'load_tensor' from 'utils'`) — pre-existing harness quirk, unrelated to these changes (SPLiCE untouched); `utils.load_tensor` exists and imports fine outside pytest.

## Round 2 — remaining correctness / polish (same session)

- **F-006:** Guide `% Aligned` recipe now filters `raw_response` `ERROR:` rows → denominator matches `judge_scores.json["aligned_rate"]` (valid-only). (`docs/LLM-JUDGE-COMPLETE-GUIDE.md`)
- **F-007/F-008:** Replaced the structurally-broken decoder-norm dead mask (always all-False after TopK column re-normalization) with an **activation-based** dead mask (`SAEManager.activation_dead_mask`, used by both naming stages). Dead features are now correctly flagged `DEAD_FEATURE`: baseline 322/2048 (15.7%), Path A 35/2048 (1.7%). This also mitigates F-008 (AuxK can't revive them, so they're filtered at naming). `concept_names.json` regenerated for both methods; `sample_explanations` unchanged (dead features never reach top-k). (`src/autoencoder/sae_module.py`, `concept_naming.py`, `src/sae_hidden/naming_hidden.py`)
- **F-010:** Added `utils.repro_info` (git SHA + versions + sha256 of inputs); both `run_path_a.py` and `run_baseline.py` now emit a **Reproducibility** section in `REPORT_run.md` (inputs are gitignored, so hashes let a future run verify identical inputs). (`src/utils.py`, `scripts/run_path_a.py`, `scripts/run_baseline.py`)
- **F-013:** `SAEConfig` now carries `dead_threshold` (mirrors `SAEHiddenConfig`; single source of truth). (`src/config.py`)
- **F-015:** `set_global_seed` now enables `torch.use_deterministic_algorithms(True, warn_only=True)` when `CUBLAS_WORKSPACE_CONFIG` is set. (`src/utils.py`)
- **F-017:** `generate_explanations_hidden.py` dict-merge reordered so the explicit device key wins. (`src/sae_hidden/generate_explanations_hidden.py`)
- **F-019:** Removed the dead `VLMConfig.device = "cuda"` line (overridden by the device ternary). (`src/config.py`)
- **Bonus:** Removed a stale `from autoencoder.tracking import …` in `train_sae.py` (the module was deleted in `6c53328`; the import was unused but broke `run_baseline.py`). `run_baseline.py` now runs end-to-end and writes to `results/baseline/` (validates F-002).

## Round 3 — F-003 training-budget parity

User chose to retrain the baseline at 8k steps to match Path A (was 50k, 6.25×).
- `config.sae.steps`: 50_000 → 8_000 (matches `config.sae_hidden`; fair baseline↔Path A budget).
- Retrained all 5 baseline seeds (8k each, ~16 min): frac_variance_explained 0→0.956; manifests now `steps=8000`.
- Regenerated `results/baseline/{concept_names,sample_explanations,stability_analysis,stability_matched}.json` + figures. Dead (activation) = 328/2048 (16.0%); `sample_explanations` 1515 unique, 0 dup.
- **Bonus fix:** `run_baseline.py` referenced a non-existent `train_sae.main` (pre-existing — the train stage never ran via this entrypoint); now loops `train_single` over seeds. The baseline pipeline now runs end-to-end.
- 42 judge/dataset tests still pass.

## Still open

- **F-012** (bootstrap CI on LIFT) — defer until Member 3's CSVs exist.
- **F-014** (REPORT_training dead % measured on a 256-subset) — only relevant on retrain; naming now reports the correct full-set activation-dead %.
- **F-016** (autocast dtype device-conditional) — optional reproducibility pin; no effect on committed MPS-float32 artifacts.
- **F-018** (7 images dropped, no report) — graceful, symmetric; documentation only.
- **Pre-existing, off the judge path:** `scripts/run_sae_training.py` still imports the deleted `autoencoder.tracking` (it genuinely used wandb hooks) and is broken until tracking is restored/stubbed. Not on the judge-readiness path; models are already trained.
