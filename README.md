# Unsupervised Concept Discovery for Medical Vision–Language Models

Unsupervised concept-discovery and evaluation pipeline for medical VLMs, after
the **MedConcept** framework. It decomposes BiomedCLIP image representations
into sparse concepts, names them against a RadLex radiology vocabulary, and
evaluates them with an LLM-as-judge (Aligned / Unaligned / Uncertain).

> Course project 5 — *Explainable and Trustworthy AI*, Politecnico di Torino.
> Full report: [`docs/latex/main.pdf`](docs/latex/main.pdf).

## What this repo does

Three concept-discovery methods, one evaluation, one organisational extension:

| Stage | Method | Script |
|---|---|---|
| Baseline | Top-K SAE on 512-d projected embeddings (documented failure case) | `02_baseline` |
| Path A | Top-K SAE on the 768-d pre-projection hidden state | `03_hidden` |
| Path B | SPLiCE — deterministic sparse decomposition on the RadLex dictionary | `04_spliece` |
| Extension | Concept organisation (clustering + RadLex families) | `05_concept_organization` |
| Evaluation | LLM judge (Aligned/Unaligned/Uncertain) + count-matched null | `06_generate_null`, `07_judge` |

**Headline finding:** the learned SAE decomposition is non-identifiable at this
data scale — cross-seed stability sits at the analytical chance floor (Jaccard
0.0084 vs floor 0.0079). Path A improves the dead-feature rate (16% → 1.7%) and
naming (0.42 → 0.47) but not stability (reframed as *weak universality*,
matched-cosine obs/null ≈ 2.6×). SPLiCE is deterministic with 95% vocabulary
coverage. See the [paper](docs/latex/main.pdf) for the full results.

## Setup

```bash
git clone <repo> && cd xai-project-5
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets (license-gated, gitignored — download locally):
- **IU X-Ray** (primary): `.venv/bin/python xai_datasets/download_iu_xray.py`
  (needs `kagglehub` + Kaggle credentials).
- **PadChest** (scale test): `.venv/bin/python xai_datasets/download_padchest.py --url <archive> ...`
  — pass one BIMCV archive URL via `--url` / `PADCHEST_DOWNLOAD_URL` / `.env`
  for a partial set (full dataset ≈ 1 TB).

## Reproduce all results

One command runs the full ordered pipeline:
```bash
./scripts/90_run_all.sh                # full pipeline
./scripts/90_run_all.sh --skip-train   # reuse cached SAE models
./scripts/90_run_all.sh --skip-judge   # skip the GPU-only judge stage
```

Or run stage by stage — scripts are numbered in execution order:
```
00_build_vocab.py         build RadLex vocabulary + text embeddings
01_extract_embeddings.py  BiomedCLIP 512-d image/text embeddings
02_baseline.py            baseline 512-d SAE (train → name → explain → stability)
03_hidden.py              Path A 768-d hidden-state SAE
04_spliece.py             SPLiCE (Path B) deterministic decomposition
05_concept_organization   cluster + RadLex families (per source)
06_generate_null.py       count-matched null explanations (k=5, k≈13)
07_judge.py               LLM judge (--input {baseline,hidden,spliece,null_k5,null_k13})
08_baseline_ablation.py   baseline dict_size×k sweep
09_hidden_ablation.py     Path A dict_size×k sweep
```
The judge (`07_judge.py`) requires a GPU + HuggingFace credentials. All results
are written under `results/iu_xray/`.

## Project layout

```
src/            pipeline modules: config, utils, autoencoder, sae_hidden,
                concept_discovery, vocabulary_building, embedding_extraction,
                augmentation, evaluate_llm_judge
xai_datasets/   dataset adapters + staging (iu_xray, padchest, spec)
scripts/        numbered pipeline drivers (00–09) + 90_run_all.sh
tests/          unit + integration tests
data/           vocabulary.json, radlex.csv (datasets gitignored)
docs/           requirements, design, audits, releases, paper (latex/)
results/iu_xray/  baseline, sae_hidden, spliece, concept_organization_*,
                  null*, sae_hidden_ablation, judge
```

## Reproducibility

```bash
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest --import-mode=importlib tests/
```

## License

See [LICENSE](LICENSE). Datasets are **not** redistributable (IU X-Ray, PadChest,
RadLex, BIMCV terms all forbid redistribution).
