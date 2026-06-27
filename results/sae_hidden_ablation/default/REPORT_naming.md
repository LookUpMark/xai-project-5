# Path A — Concept Naming (frozen-projection bridge)

_Generated: 2026-06-26_

## Summary

Named 2048/2048 live 768-d features via the frozen projection bridge. Mean naming cosine = 0.4711 (random 0.372, baseline 512-d SAE 0.395). Dead features: 0 (0.0%).

## Naming score vs references

| metric | value | reference |
| --- | --- | --- |
| mean (live) | 0.4711 | > random 0.372 |
| median (live) | 0.4709 |  |
| baseline 512-d SAE | 0.3950 | ML-AUDIT M-005 |
| random baseline | 0.3720 | ML-AUDIT M-005 |
| dead features | 0.0% | baseline 40-60% |

## Bridge check

| item | value |
| --- | --- |
| W_dec shape | (2048, 768) |
| W_proj shape | (512, 768) (bias=False) |
| dec_512 shape | (2048, 512) |
| gap applied | True |
| vocab size | 1030 |

## Top-10 most-confidently named live features

| feat_id | name | score |
| --- | --- | --- |
| 1968 | posterior rootlet of spinal nerve | 0.6185 |
| 780 | vena medullaris posteromediana | 0.6041 |
| 1564 | tunneled central venous catheter without port | 0.5979 |
| 1029 | t1 segment of cuneate fasciculus of spinal cord | 0.5958 |
| 1867 | large intestine | 0.5937 |
| 1237 | lumbocostal ligament | 0.5919 |
| 105 | C5 vertebral body | 0.5906 |
| 1572 | dorsal column of of spinal cord | 0.5880 |
| 20 | external surface of ureter proper | 0.5851 |
| 1228 | lumbocostal ligament | 0.5847 |

