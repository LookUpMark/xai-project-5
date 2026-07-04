# Path A — Concept Naming (frozen-projection bridge)

_Generated: 2026-06-26_

## Summary

Named 2048/2048 live 768-d features via the frozen projection bridge. Mean naming cosine = 0.4706 (random 0.372, baseline 512-d SAE 0.395). Dead features: 0 (0.0%).

## Naming score vs references

| metric | value | reference |
| --- | --- | --- |
| mean (live) | 0.4706 | > random 0.372 |
| median (live) | 0.4704 |  |
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
| 687 | anterior ramus of spinal nerve | 0.6383 |
| 1106 | blind pouch | 0.5993 |
| 1042 | posterior rootlet of spinal nerve | 0.5956 |
| 63 | spinotectal tract of spinal cord | 0.5904 |
| 939 | region of diaphragmatic surface of liver | 0.5903 |
| 1464 | bone stimulator device | 0.5884 |
| 835 | cervical segment of cuneate fasciculus of spinal cord | 0.5866 |
| 1575 | t3 segment of left ventral gray column of spinal cord | 0.5802 |
| 669 | coccygeal segment of spinal cord | 0.5785 |
| 143 | ligamentum flavum | 0.5779 |

