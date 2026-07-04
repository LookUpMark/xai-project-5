# Path A — 768-d Hidden-State Extraction

_Generated: 2026-06-26_

## Summary

Extracted 14940 raw 768-d CLS tokens (pre-projection, augmented (x2)) and split them at the radiograph-study level. CLS dim = 768 (confirms pre-projection path).

## Extraction

| property | value |
| --- | --- |
| shape | (14940, 768) |
| mode | augmented (x2) |
| dtype | torch.float32 |
| activation_dim (config) | 768 |
| L2-norm mean / std | 36.1831 / 1.8764 |
| norm min / max | 28.9791 / 42.9885 |

## Train / test split (study-level, no leakage)

| split | samples | fraction |
| --- | --- | --- |
| train | 11910 | 79.7% |
| test | 3030 | 20.3% |
| total | 14940 | 100.0% |

## Outputs

| file | shape |
| --- | --- |
| visual_embeddings_768.pt | (14940, 768) |
| train_embeddings_768.pt | (11910, 768) |
| test_embeddings_768.pt | (3030, 768) |

## Notes

Raw activations (no per-sample L2 norm) — SAE-on-residual-stream convention. Group split enforced inside utils.split_embeddings (anti-leak assertion).

