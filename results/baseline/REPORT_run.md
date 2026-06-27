# Baseline (512-d) — Pipeline Run

_Generated: 2026-06-27_

## Summary

Baseline (512-d) run complete in 982s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Stages wrote JSON to /Users/marcantoniolopez/Documents/github/xai-project-5/results/baseline.

## Run config

| param | value |
| --- | --- |
| tag | — |
| models dir | /Users/marcantoniolopez/Documents/github/xai-project-5/models |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | mps |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| train | ok | 944.6 |
| naming | ok | 3.4 |
| stability | ok | 29.9 |
| explain | ok | 3.8 |

## Outputs

| artifact | path |
| --- | --- |
| concept names | /Users/marcantoniolopez/Documents/github/xai-project-5/results/baseline/concept_names.json |
| stability (jaccard) | /Users/marcantoniolopez/Documents/github/xai-project-5/results/baseline/stability_analysis.json |
| stability (matched) | /Users/marcantoniolopez/Documents/github/xai-project-5/results/baseline/stability_matched.json |
| explanations | /Users/marcantoniolopez/Documents/github/xai-project-5/results/baseline/sample_explanations.json |

## Reproducibility

- git commit: `4ba4451b91c6ba2788ef5cf14ebbe425d21a6bb7`
- versions: scikit-learn 1.8.0 | torch 2.12.0 | numpy 2.4.6
- sha256(train_embeddings) [train_embeddings.pt]: `46252db1b0ea2e5e`
- sha256(test_embeddings) [test_embeddings.pt]: `f266e54366f3fb5e`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `922ee9509eb06e70`
- sha256(modality_gap) [modality_gap.pt]: `36264e287fc1f1f5`
- sha256(primary_model) [ae.pt]: `05534a5b6a271e7d`

