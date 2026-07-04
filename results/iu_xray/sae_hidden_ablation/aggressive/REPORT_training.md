# Path A — 768-d SAE Training

_Generated: 2026-06-26_

## Summary

Trained 5 768-d Top-K SAEs (seeds (0, 42, 123, 456, 789)). Mean dead-feature rate 6.6% (baseline 512-d: 40-60%). dict_size=4096, steps=8000, lr=5e-05 (audit-corrected).

## Per-seed test metrics

| seed | test MSE | test cosine | dead % | util % | L0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.080500 | 0.9748 | 6.4 | 93.6 | 64.0 |
| 42 | 0.075480 | 0.9768 | 6.2 | 93.8 | 64.0 |
| 123 | 0.070429 | 0.9785 | 7.2 | 92.8 | 64.0 |
| 456 | 0.075839 | 0.9766 | 6.7 | 93.3 | 64.0 |
| 789 | 0.071209 | 0.9782 | 6.3 | 93.7 | 64.0 |

## Hyperparameters (audit-corrected)

| param | value | rationale |
| --- | --- | --- |
| activation_dim | 768 | pre-projection CLS (Paradigm B) |
| dict_size | 4096 | M-002: down from 4096 |
| k | 64 | Top-K |
| lr | 5e-05 | M-006: pinned low |
| steps | 8000 | M-006: down from 50k |
| input | raw | no per-sample L2 norm |

## Caveat

Reconstruction cosine is near-saturated for any overcomplete SAE on a low-dim manifold (M-004) — do NOT read it as evidence the SAE 'works'. Trust cross-seed Jaccard (see REPORT_stability.md) and naming instead.

