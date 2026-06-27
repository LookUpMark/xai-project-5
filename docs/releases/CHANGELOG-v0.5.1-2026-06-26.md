# CHANGELOG v0.5.1 - 2026-06-26

## Summary

Adds a **literature-aligned, permutation-invariant** cross-seed stability metric to Path A,
fixing the interpretation error in F-001 (slot-wise Jaccard cannot establish
non-identifiability). No retraining; reuses the 5 existing SAEs. Result reframes Path A from
"non-identifiable" to **weak universality with no strong feature-level reproducibility**.

**Stats:** 5 files changed (+~430 / −~0 loc) on top of v0.5.0.
**Implementation plan:** `docs/plans/2026-06-26-sae-stability-matched.md`

---

## Features

### Permutation-invariant matched stability
- `SAEManager.compute_stability_matched` (static) + pure `matched_pair_stats` in
  `src/autoencoder/sae_module.py`: decoder-cosine feature matching across seeds + isotropic
  random-vector null + p-value + matched fractions (≥0.7/≥0.9) + mutual-1-to-1. One model
  loaded at a time (mirrors `compute_stability`). The original slot-wise `compute_stability`
  is kept but its report is relabeled as non-permutation-invariant.
- `stability_hidden.run_matched()` writes `results/sae_hidden/stability_matched.json` +
  `REPORT_stability_matched.md`, framed as weak vs strong universality with a 3-way verdict.
- New `SAEHiddenConfig` fields: `n_perm`, `match_thresholds`.

### Result (Path A, 5 seeds, dict 2048 / k 32 / 768-d)
| metric | value |
|---|---|
| mean best-match cosine | 0.325 |
| isotropic null mean | 0.124 |
| obs / null | 2.6× (p≈0, all pairs) |
| frac matched ≥0.9 | 0.0% |
| frac matched ≥0.7 | 0.1% |
| frac mutual 1-to-1 | 0.32 |

**Verdict: weak universality** — shared subspace well above chance, but zero features reproduce
at ≥0.9. Consistent with Leask et al. 2025 and the M-002 data-scale limit, now measured.

---

## Why (literature)
Slot-wise index Jaccard compares feature #342 vs #342; SAEs have no canonical ordering, so it
is ~0 by construction and the chance floor already assumes no slot correspondence. Lan et al.
2024 (arXiv:2410.06981) — the direct cross-seed SAE precedent — pairs features by similarity
first. Decoder cosine is used (not activation correlation) because TopK k=32 over ~1.5k samples
makes Pearson correlation sparse-pathological; Lan App. E.2 validates decoder cosine for the
same-model-seed case. A permutation null is degenerate for max-cosine (max-over-columns is
permutation-invariant), so the null uses independent random unit vectors; ≥0.9 fraction is the
concentration-robust signal.

---

## Testing
- Unit: `tests/unit/test_sae_hidden.py` — **14/14 pass** (added 4: identity, permutation-
  invariance, null-sanity, dead-row). Synthetic 768-d tensors, no real models, <2 s.
- End-to-end: `stability_hidden.run_matched()` on the 5 real seeds; `compute_stability` relabel.

---

## Files Touched
| File | Δ | Notes |
|------|---|-------|
| `src/autoencoder/sae_module.py` | +`matched_pair_stats`, +`compute_stability_matched` | Core metric |
| `src/sae_hidden/stability_hidden.py` | +`run_matched`, relabel `run` | Wiring + report |
| `src/config.py` | +`n_perm`, +`match_thresholds` in `SAEHiddenConfig` | Config knobs |
| `tests/unit/test_sae_hidden.py` | +4 tests | Matching math |
| `docs/design/LITERATURE-SAE-STABILITY.md` | NEW | Committed references |
| `docs/audits/ML-AUDIT-2026-06-26.md` | F-001 update + exec-summary addendum | Reframe |
| `results/sae_hidden/stability_matched.json` + `REPORT_stability_matched.md` | NEW | Outputs (gitignored) |
