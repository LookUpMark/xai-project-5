# Path A — Pipeline Run (standard)

_Generated: 2026-06-27_

## Summary

Path A (standard) run complete in 54s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Each stage wrote its own REPORT_*.md under /Users/marcantoniolopez/Documents/github/xai-project-5/results/sae_hidden.

## Run config

| param | value |
| --- | --- |
| variant | standard |
| tag | — |
| embeddings dir | /Users/marcantoniolopez/Documents/github/xai-project-5/embeddings/standard_hidden |
| models dir | /Users/marcantoniolopez/Documents/github/xai-project-5/models/sae_hidden |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | mps |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| naming | ok | 7.0 |
| stability | ok | 43.3 |
| explain | ok | 3.7 |

## Stage reports

- `REPORT_explanations.md`
- `REPORT_extraction.md`
- `REPORT_naming.md`
- `REPORT_stability.md`
- `REPORT_stability_matched.md`
- `REPORT_training.md`

## Reproducibility

- git commit: `4ba4451b91c6ba2788ef5cf14ebbe425d21a6bb7`
- versions: scikit-learn 1.8.0 | torch 2.12.0 | numpy 2.4.6
- sha256(train_embeddings) [train_embeddings_768.pt]: `59a298054528e6be`
- sha256(test_embeddings) [test_embeddings_768.pt]: `7a6169ffd030e2b6`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `922ee9509eb06e70`
- sha256(modality_gap) [modality_gap.pt]: `36264e287fc1f1f5`
- sha256(primary_model) [ae.pt]: `d374e87cb90ed9f6`

