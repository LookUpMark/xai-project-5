# Path A — Concept Naming (frozen-projection bridge)

_Generated: 2026-06-26_

## Summary

Named 1024/1024 live 768-d features via the frozen projection bridge. Mean naming cosine = 0.4834 (random 0.372, baseline 512-d SAE 0.395). Dead features: 0 (0.0%).

## Naming score vs references

| metric | value | reference |
| --- | --- | --- |
| mean (live) | 0.4834 | > random 0.372 |
| median (live) | 0.4870 |  |
| baseline 512-d SAE | 0.3950 | ML-AUDIT M-005 |
| random baseline | 0.3720 | ML-AUDIT M-005 |
| dead features | 0.0% | baseline 40-60% |

## Bridge check

| item | value |
| --- | --- |
| W_dec shape | (1024, 768) |
| W_proj shape | (512, 768) (bias=False) |
| dec_512 shape | (1024, 512) |
| gap applied | True |
| vocab size | 1030 |

## Top-10 most-confidently named live features

| feat_id | name | score |
| --- | --- | --- |
| 698 | drug delivery pump | 0.6198 |
| 422 | disc herniation | 0.6136 |
| 922 | lumbocostal ligament | 0.6022 |
| 10 | lumbocostal ligament | 0.6012 |
| 828 | superior surface of liver | 0.6006 |
| 722 | right lower lobe bronchus | 0.5997 |
| 551 | lumbocostal ligament | 0.5992 |
| 448 | ligamentum flavum | 0.5984 |
| 148 | lumbocostal ligament | 0.5975 |
| 30 | cyclops lesion | 0.5938 |

