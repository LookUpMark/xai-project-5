# Baseline (512-d) — Pipeline Run

_Generated: 2026-07-07_

## Summary

Baseline (512-d) run complete in 645s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Stages wrote JSON to C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\results\padchest\baseline.

## Run config

| param | value |
| --- | --- |
| tag | — |
| models dir | C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\models\padchest |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | cuda |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| train | ok | 555.5 |
| modality_gap | ok | 0.0 |
| naming | ok | 3.5 |
| stability | ok | 83.9 |
| explain | ok | 1.9 |

## Outputs

| artifact | path |
| --- | --- |
| concept names | C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\results\padchest\baseline/concept_names.json |
| stability (jaccard) | C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\results\padchest\baseline/stability_analysis.json |
| stability (matched) | C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\results\padchest\baseline/stability_matched.json |
| explanations | C:\Users\nico2\Desktop\XAI\xai_project\xai-project-5\results\padchest\baseline/sample_explanations.json |

## Reproducibility

- git commit: `bc3c76d8f8833388d650f752681106683c529fd5`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu126 | numpy 2.4.6
- sha256(train_embeddings) [train_embeddings.pt]: `b2c584709f0447fb`
- sha256(test_embeddings) [test_embeddings.pt]: `854ef3c302549375`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `3a74889e152327e8`
- sha256(modality_gap) [modality_gap.pt]: `5f8aa86af1b1ed2f`
- sha256(primary_model) [ae.pt]: `9fade03e32a7eeec`

