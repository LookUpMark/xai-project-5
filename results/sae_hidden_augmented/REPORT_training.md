# Path A — 768-d SAE Training

_Generated: 2026-06-26_

## Summary

Trained 5 768-d Top-K SAEs (seeds (0, 42, 123, 456, 789)). Mean dead-feature rate 15.6% (baseline 512-d: 40-60%). dict_size=2048, steps=8000, lr=5e-05 (audit-corrected).

## Per-seed test metrics

| seed | test MSE | test cosine | dead % | util % | L0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.079203 | 0.9763 | 15.5 | 84.5 | 32.0 |
| 42 | 0.081512 | 0.9754 | 15.1 | 84.9 | 32.0 |
| 123 | 0.078584 | 0.9765 | 15.6 | 84.4 | 32.0 |
| 456 | 0.076626 | 0.9770 | 16.3 | 83.7 | 32.0 |
| 789 | 0.081565 | 0.9754 | 15.6 | 84.4 | 32.0 |

## Hyperparameters (audit-corrected)

| param | value | rationale |
| --- | --- | --- |
| activation_dim | 768 | pre-projection CLS (Paradigm B) |
| dict_size | 2048 | M-002: down from 4096 |
| k | 32 | Top-K |
| lr | 5e-05 | M-006: pinned low |
| steps | 8000 | M-006: down from 50k |
| input | raw | no per-sample L2 norm |

## Caveat

Reconstruction cosine is near-saturated for any overcomplete SAE on a low-dim manifold (M-004) — do NOT read it as evidence the SAE 'works'. Trust cross-seed Jaccard (see REPORT_stability.md) and naming instead.

