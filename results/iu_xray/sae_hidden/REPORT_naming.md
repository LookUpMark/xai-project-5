# Path A — Concept Naming (frozen-projection bridge)

_Generated: 2026-06-27_

## Summary

Named 2013/2048 live 768-d features via the frozen projection bridge. Mean naming cosine = 0.4694 (random 0.372, baseline 512-d SAE 0.395). Dead features: 35 (1.7%).

## Naming score vs references

| metric | value | reference |
| --- | --- | --- |
| mean (live) | 0.4694 | > random 0.372 |
| median (live) | 0.4696 |  |
| baseline 512-d SAE | 0.3950 | ML-AUDIT M-005 |
| random baseline | 0.3720 | ML-AUDIT M-005 |
| dead features | 1.7% | baseline 40-60% |

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
| 1968 | posterior rootlet of spinal nerve | 0.6191 |
| 780 | vena medullaris posteromediana | 0.5993 |
| 1029 | t1 segment of cuneate fasciculus of spinal cord | 0.5970 |
| 105 | C5 vertebral body | 0.5919 |
| 1572 | dorsal column of of spinal cord | 0.5913 |
| 1237 | lumbocostal ligament | 0.5907 |
| 1564 | tunneled central venous catheter without port | 0.5904 |
| 388 | nucleus of spinal nerve | 0.5867 |
| 1867 | large intestine | 0.5867 |
| 1228 | lumbocostal ligament | 0.5849 |

