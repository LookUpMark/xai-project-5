# Path A — Cross-seed Stability (768-d)

_Generated: 2026-06-26_

## Summary

Mean cross-seed Jaccard = 0.0080 (chance floor 0.0079, baseline 512-d 0.0038). near/at chance floor — features still seed-dependent. NOTE: this slot-wise index Jaccard is NOT permutation-invariant and cannot by itself establish non-identifiability (see REPORT_stability_matched.md). Lift over baseline: 2.1x.

## Stability vs references

| metric | value |
| --- | --- |
| mean Jaccard | 0.0080 |
| std Jaccard | 0.0004 |
| analytical chance floor (k/(2D-k)) | 0.0079 |
| baseline 512-d Jaccard | 0.0038 |
| lift over baseline | 2.1x |
| k / dict_size | 64 / 4096 |

## Per-seed-pair Jaccard

| seed pair | Jaccard |
| --- | --- |
| 0-42 | 0.0077 |
| 0-123 | 0.0082 |
| 0-456 | 0.0077 |
| 0-789 | 0.0083 |
| 42-123 | 0.0086 |
| 42-456 | 0.0074 |
| 42-789 | 0.0079 |
| 123-456 | 0.0086 |
| 123-789 | 0.0083 |
| 456-789 | 0.0077 |

## Interpretation

Mean Jaccard 0.0080 vs chance floor 0.0079. near/at chance floor — features still seed-dependent. NOTE: this slot-wise index Jaccard is NOT permutation-invariant and cannot by itself establish non-identifiability (see REPORT_stability_matched.md). If mean Jaccard sits near the floor, the 768-d SAE is still non-identifiable at this data scale (cf. M-002) — consider a smaller dict_size or more data, not more seeds.

