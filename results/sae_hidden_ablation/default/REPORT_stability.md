# Path A — Cross-seed Stability (768-d)

_Generated: 2026-06-26_

## Summary

Mean cross-seed Jaccard = 0.0083 (chance floor 0.0079, baseline 512-d 0.0038). near/at chance floor — features still seed-dependent. NOTE: this slot-wise index Jaccard is NOT permutation-invariant and cannot by itself establish non-identifiability (see REPORT_stability_matched.md). Lift over baseline: 2.2x.

## Stability vs references

| metric | value |
| --- | --- |
| mean Jaccard | 0.0083 |
| std Jaccard | 0.0006 |
| analytical chance floor (k/(2D-k)) | 0.0079 |
| baseline 512-d Jaccard | 0.0038 |
| lift over baseline | 2.2x |
| k / dict_size | 32 / 2048 |

## Per-seed-pair Jaccard

| seed pair | Jaccard |
| --- | --- |
| 0-42 | 0.0083 |
| 0-123 | 0.0088 |
| 0-456 | 0.0069 |
| 0-789 | 0.0080 |
| 42-123 | 0.0081 |
| 42-456 | 0.0088 |
| 42-789 | 0.0087 |
| 123-456 | 0.0086 |
| 123-789 | 0.0080 |
| 456-789 | 0.0088 |

## Interpretation

Mean Jaccard 0.0083 vs chance floor 0.0079. near/at chance floor — features still seed-dependent. NOTE: this slot-wise index Jaccard is NOT permutation-invariant and cannot by itself establish non-identifiability (see REPORT_stability_matched.md). If mean Jaccard sits near the floor, the 768-d SAE is still non-identifiable at this data scale (cf. M-002) — consider a smaller dict_size or more data, not more seeds.

