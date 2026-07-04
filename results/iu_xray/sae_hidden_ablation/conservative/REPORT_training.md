# Path A — 768-d SAE Training

_Generated: 2026-06-26_

## Summary

Trained 5 768-d Top-K SAEs (seeds (0, 42, 123, 456, 789)). Mean dead-feature rate 41.7% (baseline 512-d: 40-60%). dict_size=1024, steps=8000, lr=5e-05 (audit-corrected).

## Per-seed test metrics

| seed | test MSE | test cosine | dead % | util % | L0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.106001 | 0.9664 | 40.8 | 59.2 | 16.0 |
| 42 | 0.099675 | 0.9692 | 41.5 | 58.5 | 16.0 |
| 123 | 0.095724 | 0.9707 | 40.4 | 59.6 | 16.0 |
| 456 | 0.100414 | 0.9688 | 43.7 | 56.3 | 16.0 |
| 789 | 0.095428 | 0.9706 | 42.2 | 57.8 | 16.0 |

## Hyperparameters (audit-corrected)

| param | value | rationale |
| --- | --- | --- |
| activation_dim | 768 | pre-projection CLS (Paradigm B) |
| dict_size | 1024 | M-002: down from 4096 |
| k | 16 | Top-K |
| lr | 5e-05 | M-006: pinned low |
| steps | 8000 | M-006: down from 50k |
| input | raw | no per-sample L2 norm |

## Caveat

Reconstruction cosine is near-saturated for any overcomplete SAE on a low-dim manifold (M-004) — do NOT read it as evidence the SAE 'works'. Trust cross-seed Jaccard (see REPORT_stability.md) and naming instead.

