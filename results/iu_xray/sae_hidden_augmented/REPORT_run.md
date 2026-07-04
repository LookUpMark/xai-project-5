# Path A — Pipeline Run (augmented)

_Generated: 2026-06-26_

## Summary

Path A (augmented) run complete in 2167s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Each stage wrote its own REPORT_*.md under /Users/marcantoniolopez/Documents/github/xai-project-5/results/sae_hidden_augmented.

## Run config

| param | value |
| --- | --- |
| variant | augmented |
| tag | — |
| embeddings dir | /Users/marcantoniolopez/Documents/github/xai-project-5/embeddings/augmented_hidden |
| models dir | /Users/marcantoniolopez/Documents/github/xai-project-5/models/sae_hidden_augmented |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | mps |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| extract | ok | 888.1 |
| train | ok | 1224.4 |
| naming | ok | 5.5 |
| stability | ok | 41.5 |
| explain | ok | 7.7 |

## Stage reports

- `REPORT_explanations.md`
- `REPORT_extraction.md`
- `REPORT_naming.md`
- `REPORT_stability.md`
- `REPORT_stability_matched.md`
- `REPORT_training.md`

