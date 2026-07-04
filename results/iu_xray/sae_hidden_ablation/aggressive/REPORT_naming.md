# Path A — Concept Naming (frozen-projection bridge)

_Generated: 2026-06-26_

## Summary

Named 4096/4096 live 768-d features via the frozen projection bridge. Mean naming cosine = 0.4782 (random 0.372, baseline 512-d SAE 0.395). Dead features: 0 (0.0%).

## Naming score vs references

| metric | value | reference |
| --- | --- | --- |
| mean (live) | 0.4782 | > random 0.372 |
| median (live) | 0.4769 |  |
| baseline 512-d SAE | 0.3950 | ML-AUDIT M-005 |
| random baseline | 0.3720 | ML-AUDIT M-005 |
| dead features | 0.0% | baseline 40-60% |

## Bridge check

| item | value |
| --- | --- |
| W_dec shape | (4096, 768) |
| W_proj shape | (512, 768) (bias=False) |
| dec_512 shape | (4096, 512) |
| gap applied | True |
| vocab size | 1030 |

## Top-10 most-confidently named live features

| feat_id | name | score |
| --- | --- | --- |
| 1441 | coccygeal segment of spinal cord | 0.6333 |
| 4077 | branch of left coronary artery | 0.6184 |
| 3028 | disc herniation | 0.6143 |
| 3785 | lumbocostal ligament | 0.6097 |
| 2967 | left spinotectal tract of spinal cord | 0.6078 |
| 643 | muscle body of spinal part of deltoid | 0.5992 |
| 1558 | dorsal column of of spinal cord | 0.5983 |
| 2094 | c7 segment of cuneate fasciculus of spinal cord | 0.5971 |
| 170 | posterior root of spinal nerve | 0.5966 |
| 3234 | vena medullaris posteromediana | 0.5962 |

