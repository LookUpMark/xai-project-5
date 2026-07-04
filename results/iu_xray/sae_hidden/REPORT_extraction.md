# Path A — 768-d Hidden-State Extraction

_Generated: 2026-06-26_

## Summary

Extracted 7470 raw 768-d CLS tokens (pre-projection) and split them at the radiograph-study level. CLS dim = 768 (confirms pre-projection path).

## Extraction

| property | value |
| --- | --- |
| shape | (7470, 768) |
| dtype | torch.float32 |
| activation_dim (config) | 768 |
| L2-norm mean / std | 35.5982 / 1.8573 |
| norm min / max | 28.8418 / 62.3666 |

## Train / test split (study-level, no leakage)

| split | samples | fraction |
| --- | --- | --- |
| train | 5955 | 79.7% |
| test | 1515 | 20.3% |
| total | 7470 | 100.0% |

## Outputs

| file | shape |
| --- | --- |
| visual_embeddings_768.pt | (7470, 768) |
| train_embeddings_768.pt | (5955, 768) |
| test_embeddings_768.pt | (1515, 768) |

## Notes

Raw activations (no per-sample L2 norm) — SAE-on-residual-stream convention. Group split enforced inside utils.split_embeddings (anti-leak assertion).

