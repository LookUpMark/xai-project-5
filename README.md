# xai-project-5

Unsupervised concept discovery & evaluation for medical vision-language models. We discover human-interpretable concepts in BiomedCLIP image embeddings via Sparse Autoencoders (Top-K), name them against a RadLex-derived vocabulary, generate per-image explanations, and judge faithfulness against radiology reports with an LLM.

The central finding is **methodological and negative**: under a permutation-invariant metric against a subspace-conditioned null, SAE concepts on the pooled CLS embedding are genuinely **non-identifiable** across seeds, and neither a 768-d hidden-state SAE (Path A) nor a 10× corpus (ROCOv2) lifts identifiability — the binding constraint is the *representation site*, not data scale. SPLiCE, a deterministic decomposition over a fixed dictionary, is the sole judge-ready alternative.

Full write-up: [`docs/latex/main.pdf`](docs/latex/main.pdf) (5 pages: body pp. 1–4 + references).

---

## Setup

Python 3.12 in a virtual environment. `requirements.txt` pulls PyTorch from the nightly CUDA 12.8 index.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

Datasets, embeddings, models, and results are **gitignored** and regenerated locally (only `embeddings/mock/` and `models/mock/` are committed).

---

## Pipeline

The pipeline is **CLI-driven end-to-end** — every stage is a `scripts/run_*.py` with a `main()`/`run()` that puts `src/` on `sys.path`. Centralized config in [`src/config.py`](src/config.py): all paths and hyperparameters flow from frozen dataclass singletons (`config.paths`, `config.sae`, `config.training`, …). Dataset-aware scripts call `config.select_dataset(ds)` before reading `config.paths`.

| Stage | Command | Output |
|---|---|---|
| Data download | `python xai_datasets/download_iu_xray.py` | `data/iu_xray/` |
| Embedding extraction | `python scripts/run_extraction.py [--dataset X --augmentation]` | `embeddings/standard/*.pt` + image-id sidecars |
| Vocabulary build | `python scripts/run_vocab_building_pipeline.py [--topk N --device cuda]` | `data/vocabulary.json`, `text_vocab_embeddings.pt` |
| SAE training (5 seeds) | `python src/autoencoder/train_sae.py` | `models/sae_seed*/`, `results/training_manifest.json` |
| Concept naming | `python src/autoencoder/concept_naming.py` | `results/concept_names.json` |
| Explanations | `python src/autoencoder/generate_explanations.py` | `results/sample_explanations.json` |
| Stability analysis | `python src/autoencoder/stability_analysis.py` | `results/stability_analysis.json` |
| LLM judge | `python src/evaluate_llm_judge.py [--resume]` | `results/aligned_scores.csv` |

Baselines and ablations are each a `scripts/run_*.py` writing a per-folder `REPORT.md` under `results/<dataset>/`:
- Baseline 512-d SAE — `scripts/run_baseline.py`
- Path A 768-d hidden-state SAE — `scripts/run_path_a.py` (+ ablation)
- SPLiCE (Path B) — `scripts/run_spliece.py`

**Gap analyses:**
- Gap 1 (cross-seed consensus stability) — `scripts/run_consensus.py` (pooled-decoder connected-components vs seed-tag shuffle-null; decoder-only, no GPU; `--tau 0.70`)
- Gap 2 (faithfulness) — `scripts/run_faithfulness.py` (point-biserial vs IU X-Ray MeSH/Problems)
- Gap 5 (concept organization) — `scripts/run_concept_organization.py --source {spliece,sae-baseline,sae-hidden}`

---

## Architecture

```
IU X-Ray images ──extract_embeddings──▶ visual_embeddings.pt (7470×512, L2-norm)
        │                                          │
        └─ reports ──▶ reports.csv (judge join)    ├─ prepare_split ──▶ train/test_*.pt (+ id sidecars, split in lockstep)
                                                   └─ modality_gap.pt
RadLex.csv ──build_vocabulary──▶ vocabulary.json + text_vocab_embeddings.pt
        │
        ▼
train_sae ──▶ SAEManager(AutoEncoderTopK) ──▶ models/sae_seed{0,42,123,456,789}/
        │
        ▼
concept_naming (decoder rows vs vocab cosine, modality-gap corrected) ──▶ concept_names.json
        ▼
generate_explanations ──▶ sample_explanations.json (image_id + top-k concepts)
        ▼
evaluate_llm_judge (MedGemma 4B / Llama-3.1-8B, LangGraph) ──▶ judge_scores_*.json
```

**Backbone:** BiomedCLIP (`chuhac/BiomedCLIP-vit-bert-hf`, 512-d shared contrastive space, 768-d pre-projection hidden state). SAE via the [`dictionary-learning`](https://github.com/saprmarks/dictionary_learning) library (`AutoEncoderTopK`).

**Dataset:** IU X-Ray (7,470 frontal/lateral chest X-rays across 3,852 studies) + ROCOv2 (~80k, 10× scale replication).

**Import convention:** `src/` modules use bare sibling imports (`import config`, `from autoencoder.sae_module import SAEManager`) after inserting `src/` on `sys.path` — not `from src.…`.

---

## Results (headline)

| Metric | IU X-Ray (1×) | ROCOv2 (10×) |
|---|---|---|
| Reconstruction cosine | 0.991 | 0.968 |
| Matched cosine (observed) | 0.299 | 0.327 |
| Subspace null ratio (honest) | 1.67× | 1.81× |
| Frac. matched ≥0.9 | 0.0% | 0.3% |
| Naming cosine (live, mean) | 0.40 | 0.48 |
| Slot-wise Jaccard (deprecated) | 0.0077 | 0.0077 |

**LLM-judge (run to convergence, 0 parse errors):** SPLiCE on IU X-Ray **81.6% Aligned** (MedGemma-4B) vs 3.3% (Llama-3.1-8B); SAE baseline on ROCOv2 **88.3%** vs 23.1%; Path A hidden source **76.3%** vs 0.5%. Random-k null explanations score comparably (79.6% MedGemma, 2.9% Llama), indicating weak discriminability. Judge Aligned measures *local* plausibility and is orthogonal to the cross-seed non-identifiability verdict.

---

## Tests

```bash
pytest tests/unit -q          # unit only (fast, no GPU/model)
pytest tests/integration -q   # integration (full pipeline on mocks)
```

> Run `tests/unit` and `tests/integration` **separately**, not `pytest tests/` as one — a name collision with a stale `test_llm_judge.py` breaks collection when combined. ~248 tests total.

---

## Project layout

```
src/                  pipeline source (config.py, autoencoder/, sae_hidden/, augmentation/, evaluate_llm_judge.py)
scripts/              run_*.py CLI stages + gap analyses (run_consensus.py, run_faithfulness.py, …)
xai_datasets/         dataset download + augmentation
tests/                unit/ + integration/
docs/latex/           paper source + main.pdf
docs/audits/          ML pipeline + judge audits (authoritative for caveats)
docs/wiki/            bilingual IT/EN sub-wiki (autoencoder internals)
docs/literature/      reference papers (PDFs gitignored)
results/              experiment outputs + REPORT.md per folder (tracked by convention)
```

Two authoritative ML audits document caveats (read before trusting any held-out metric): [`docs/audits/ML-AUDIT-2026-06-23.md`](docs/audits/ML-AUDIT-2026-06-23.md) (pipeline) and [`docs/audits/ML-AUDIT-2026-06-24.md`](docs/audits/ML-AUDIT-2026-06-24.md) (LLM judge).

## Authors

Marc'Antonio Lopez, Nicolò Colle, Carmine Francesco Benvenuto — Politecnico di Torino.
