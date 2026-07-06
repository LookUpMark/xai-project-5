# Baseline (512-d) — Pipeline Run

_Generated: 2026-07-06_

## Summary

Baseline (512-d) run complete in 159s. dict_size=2048 k=32 steps=8000, seeds=[0, 42, 123, 456, 789]. Stages wrote JSON to /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/baseline.

## Run config

| param | value |
| --- | --- |
| tag | — |
| models dir | /home/marcantoniolopez/Documenti/github/xai-project-5/models/iu_xray |
| dict_size / k / steps | 2048 / 32 / 8000 |
| seeds | 0, 42, 123, 456, 789 |
| device | cuda |

## Stages

| stage | status | seconds |
| --- | --- | --- |
| train | ok | 110.5 |
| modality_gap | ok | 0.0 |
| naming | ok | 0.8 |
| stability | ok | 47.4 |
| explain | ok | 0.3 |

## Outputs

| artifact | path |
| --- | --- |
| concept names | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/baseline/concept_names.json |
| stability (jaccard) | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/baseline/stability_analysis.json |
| stability (matched) | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/baseline/stability_matched.json |
| explanations | /home/marcantoniolopez/Documenti/github/xai-project-5/results/iu_xray/baseline/sample_explanations.json |

## Reproducibility

- git commit: `5b80bf250e0df95e6056843d0410d44db419b94e`
- versions: scikit-learn 1.8.0 | torch 2.12.0+cu130 | numpy 2.4.6
- sha256(train_embeddings) [train_embeddings.pt]: `bf2207fa822c83ab`
- sha256(test_embeddings) [test_embeddings.pt]: `d1b9081dbb5f0fe1`
- sha256(text_vocab_embeddings) [text_vocab_embeddings.pt]: `98c9cf7462d6181f`
- sha256(modality_gap) [modality_gap.pt]: `e6dda3a0a8ed454a`
- sha256(primary_model) [ae.pt]: `c8bc3a0fc754e232`

