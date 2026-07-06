# Baseline (512-d) — Pipeline Run

_Generated: 2026-07-06_

## Summary

Baseline (512-d) run complete in 48s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Stages wrote JSON to /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/baseline.

## Run config

| param | value |
| --- | --- |
| tag | — |
| models dir | /home/marcantoniolopez/Documenti/github/xai-project-5/models/rocov2 |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | cuda |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| modality_gap | ok | 0.3 |
| naming | ok | 2.0 |
| stability | ok | 43.1 |
| explain | ok | 2.4 |

## Outputs

| artifact | path |
| --- | --- |
| concept names | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/baseline/concept_names.json |
| stability (jaccard) | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/baseline/stability_analysis.json |
| stability (matched) | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/baseline/stability_matched.json |
| explanations | /home/marcantoniolopez/Documenti/github/xai-project-5/results/rocov2/baseline/sample_explanations.json |

## Reproducibility

- git commit: `bc3c76d8f8833388d650f752681106683c529fd5`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(train_embeddings) [train_embeddings.pt]: `9e47099c7a248f35`
- sha256(test_embeddings) [test_embeddings.pt]: `909730509dc3cde2`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `846c2d086400dde2`
- sha256(modality_gap) [modality_gap.pt]: `1205a576aa512342`
- sha256(primary_model) [ae.pt]: `365468922d2d62e7`

