# CHANGELOG v0.3.0 - 2026-05-31

## Summary

Major architectural overhaul: SAE pipeline restructured into a proper Python
package (`src/autoencoder/`), shared utilities extracted, VLM embedding
extraction implemented, comprehensive wiki documentation added, full test
coverage expanded from 40 to 61 tests, code quality enforced with ruff +
markdownlint, and BiomedCLIP model verified end-to-end on CPU.

**Stats**: 55 files changed, +10,243 / -808 lines since v0.2.0.

---

## Features

### SAE Pipeline Package (`src/autoencoder/`)

- Restructured flat scripts (`02a_train_sae.py`, `02b_concept_naming.py`,
  `02c_generate_explanations.py`, `02d_stability_analysis.py`) into proper
  package under `src/autoencoder/`
- New modules:
  - `sae_module.py` — `SAEManager` facade (train, load, encode/decode,
    metrics, stability) with 619 lines
  - `train_sae.py` — `prepare_split()` (sklearn 80/20), `train_single()`
  - `concept_naming.py` — cosine similarity naming with dead feature detection
  - `generate_explanations.py` — per-sample concept explanations
  - `stability_analysis.py` — cross-seed Jaccard, clustering, per-seed metrics
  - `contracts.py` — frozen dataclasses (`CandidateName`, `ConceptName`,
    `ConceptMap`, `Finding`, `Explanation`, `SeedMetrics`, `ClusteringResult`,
    `StabilityResult`)
  - `protocols.py` — `PipelineStage` and `TrackedStage` Protocol definitions
  - `tracking.py` — wandb wrapper (init, log_metrics, log_artifact; no-ops
    when disabled)
  - `visualization.py` — seaborn/matplotlib plots (Jaccard heatmaps, concept
    distributions, per-seed metrics, loss curves)
  - `__init__.py` — public API with `__all__`

### VLM Embedding Extraction (`src/extract_embeddings.py`)

- Full implementation of `extract_visual_embeddings()` and
  `extract_text_embeddings()` with DataLoader batching
- `load_vlm()` utility for BiomedCLIP loading
- Split from single function into two dedicated extractors

### Visualization & Notebooks

- `plot_loss_curve()` added to visualization module with W&B loss logging
- `notebooks/autoencoder/mock/pipeline_smoke_test.ipynb` — comprehensive
  mock pipeline notebook with visualizations and W&B integration
- Train/test loss curve visualization in smoke test

### Stability Analysis Enhancements

- Enhanced stability analysis with new clustering features
- Cross-seed Jaccard similarity computation
- Per-seed metrics aggregation

### BiomedCLIP Model Verification

- Downloaded and verified `chuhac/BiomedCLIP-vit-bert-hf` (195.9M params)
- End-to-end test: image embedding `[1, 512]`, text embedding `[1, 512]`
- Confirmed `utils.load_vlm()` works correctly on CPU

---

## Refactoring

### Shared Utilities Extraction (`src/utils.py`)

- New module with 6 functions extracted from scattered locations:
  - `load_vlm(config)` — load BiomedCLIP model + processor
  - `set_global_seed(seed)` — random/numpy/torch/cuda/cudnn seeds
  - `load_tensor(path)` — safe deserialization (`weights_only=True`)
  - `ensure_dir(path)` — mkdir with parents
  - `setup_logging(name)` — standardized logging config
  - `dataclass_to_dict(dc)` — recursive dataclass serialization
- All autoencoder modules now `import utils` instead of duplicating logic

### Configuration Expansion (`src/config.py`)

- Added `VLMConfig` dataclass: `model_name`, `processor_name`, `device`
  (auto-detected), `batch_size`, `num_workers`, image/reports dirs, output
  paths with `@property`
- Added `WandbConfig` dataclass for experiment tracking configuration
- `__post_init__` validation on all configs

### Mock Data Relocation

- Moved mock data from `data/mock/` to `embeddings/mock/` (closer to real
  embeddings path)
- Files: `visual_embeddings.pt`, `train_embeddings.pt`,
  `test_embeddings.pt`, `text_vocab_embeddings.pt`, `vocabulary.json`

### Dataset Module (`datasets/`)

- Added `datasets/__init__.py` and `datasets/iu_xray.py` for IU X-Ray
  dataset loading

---

## Bug Fixes

### Dead Features & Zero-Activation Edge Cases

- `name_concepts()` now detects zero-norm decoder vectors, assigns
  `"DEAD_FEATURE"` with `is_dead=True`
- `get_top_concepts()` filters out zero-activation entries
- Fixed `std_jaccard` NaN when ≤2 seeds (edge case in stability analysis)

### Transformers Compatibility

- Upgraded `transformers` from `4.38.2` to `4.44.2`
- v4.38.2 lacked `_valid_processor_keys` (needed by BiomedCLIP's custom
  processor code)
- v5.x had breaking change in `CLIPConfig` (positional args error)
- v4.44.2 is the sweet spot: supports BiomedCLIP custom code without 5.x
  breakage

---

## Documentation

### Wiki (`docs/wiki/`)

- **11 new wiki pages** covering every module:
  - `CONFIG.md` (804 lines) — all dataclass configs with examples
  - `autoencoder/SAE_MODULE.md` (1391 lines) — SAEManager API reference
  - `autoencoder/VISUALIZATION.md` (550 lines) — all plot functions
  - `autoencoder/CONTRACTS.md` (445 lines) — data contract schemas
  - `autoencoder/TRACKING.md` (399 lines) — wandb integration
  - `autoencoder/GENERATE_EXPLANATIONS.md` (335 lines)
  - `autoencoder/TRAIN_SAE.md` (322 lines)
  - `autoencoder/STABILITY_ANALYSIS.md` (320 lines)
  - `autoencoder/CONCEPT_NAMING.md` (291 lines)
  - `autoencoder/PROTOCOLS.md` (285 lines)
  - `autoencoder/UTILS.md` (152 lines)
- Contracts and protocols marked as aspirational schema documentation
- Wiki aligned with current code state after restructuring

### Other Documentation

- `docs/suggestions/SAE_TRAINING_SMALL_DATASET.md` — training tips for
  small datasets (~7400 samples)
- `CLAUDE.md` — comprehensive project guide for AI assistants
- Removed outdated `CONFIG.md` documentation

---

## Code Quality

### Linting & Formatting

- **ruff**: applied formatting to all `src/autoencoder/` and `tests/`; all
  checks passing (zero violations)
- **markdownlint**: configured `.markdownlint.json` (disabled MD013, MD024,
  MD060); all markdown files clean
- Removed section separator comments (`# ---`) from utils.py
- Fixed all unused imports across test files

### Docstrings & Type Annotations

- Google-style docstrings with tensor shape annotations added to all
  autoencoder modules
- Example: `embeddings (torch.Tensor): Input embeddings, shape (N, D).`

### Warning Suppression

- `pyproject.toml` added with `filterwarnings` for SWIG `DeprecationWarning`
- Per-file-ignores `E402` for tests (sys.path manipulation)

---

## Testing

### New Test Files

- `tests/test_extract_embeddings.py` — 469 lines, comprehensive extraction
  tests
- `tests/test_load_vlm.py` — 171 lines, model loading with mocks
- `tests/test_integration.py` — 140 lines, end-to-end pipeline tests
  (skipped without real data)

### Test Improvements

- Expanded from **40 tests** (v0.2.0) to **61 passed, 7 skipped**
- Dead-feature test added for concept naming edge case
- All tests CPU-only with mocked SAE models
- `conftest.py` updated with `sys.path.insert(0, "src/")` convention

---

## Infrastructure

### Dependencies (`requirements.txt`)

- Added: `wandb`, `nbformat`, `ollama`, `langchain-core>=0.2.0`,
  `langchain-ollama`, `dictionary-learning>=0.1.0`
- Changed: `transformers==4.38.2` → `transformers==4.44.2`
- Added PyTorch CUDA extra-index-url for GPU environments

### Git & CI

- `.gitignore`: added `.DS_Store`, `.claude/`
- PR #1 merged: `feat/vlm_embedding_extraction` → `dev`
- Branch `feat/autoencoder` fully pushed (13 commits ahead of `dev`)

### Project Config

- `pyproject.toml` created (pytest + ruff config)
- `.markdownlint.json` created

---

## Commits (chronological)

| Hash | Date | Message |
|------|------|---------|
| `9ae7ec2` | 2026-05-30 | refactor: SAE pipeline - DataLoader, dataclass config, code cleanup |
| `b8a15b2` | 2026-05-30 | docs: add SAEModule detailed wiki documentation |
| `a4e21af` | 2026-05-30 | Add project presentation and 2026 project requirements PDFs |
| `0d93b7c` | 2026-05-30 | docs: add SAE training suggestions for small datasets |
| `dec5d38` | 2026-05-30 | docs: rename suggestions file to uppercase |
| `2f02840` | 2026-05-30 | docs: add comprehensive documentation for SAEModule |
| `786b451` | 2026-05-30 | docs: add wiki documentation for 02a, 02b, 02c, 02d scripts |
| `cfb0fd5` | 2026-05-30 | docs: add wiki documentation for config.py |
| `4668701` | 2026-05-30 | refactor: move SAE scripts to src/autoencoder/ package |
| `4c89786` | 2026-05-30 | pin transformers version and add PyTorch CUDA support |
| `5f57158` | 2026-05-30 | docs: update wiki paths after src/autoencoder/ restructuring |
| `f226284` | 2026-05-30 | docs: rename wiki files (remove number prefixes) |
| `aa2b7fd` | 2026-05-30 | docs: add comprehensive documentation for config.py |
| `0a02117` | 2026-05-30 | docs: remove outdated CONFIG.md documentation |
| `39ad481` | 2026-05-31 | Merge pull request #1 from feat/vlm_embedding_extraction into dev |
| `dc2ee29` | 2026-05-31 | Add .DS_Store and .claude/ to gitignore |
| `429785c` | 2026-05-31 | feat: Enhance stability analysis and training pipeline |
| `404fbe4` | 2026-05-31 | feat: add mock pipeline notebook with visualizations and W&B |
| `f6ff7d1` | 2026-05-31 | feat: add train/test loss curve visualization |
| `d4f1626` | 2026-05-31 | feat: add plot_loss_curve to visualization module + W&B |
| `ce9c26b` | 2026-05-31 | fix: handle dead features + zero-activation edge cases |
| `57428cd` | 2026-05-31 | fix: resolve std_jaccard NaN with ≤2 seeds + dead-feature test |
| `14bc482` | 2026-05-31 | style: apply ruff formatting to src/autoencoder/ and tests/ |
| `25807e1` | 2026-05-31 | style: add Google-style docstrings with tensor shape annotations |
| `b7f1f1e` | 2026-05-31 | docs: align wiki with current code state |
| `4a64daa` | 2026-05-31 | chore: add wandb/nbformat to requirements, strip notebook metadata |
| `e0a4813` | 2026-05-31 | refactor: move mock data from data/mock/ to embeddings/mock/ |
| `3554d10` | 2026-05-31 | docs: mark contracts and protocols as aspirational schema docs |
| `802ba83` | 2026-05-31 | Merge remote-tracking branch 'origin/dev' into feat/autoencoder |
| `0ff0bab` | 2026-05-31 | refactor: extract shared utilities to src/utils.py |
| `b3c6cbb` | 2026-05-31 | docs: add UTILS wiki page, update CONFIG/SAE_MODULE/CLAUDE.md |
| `2e6a5d8` | 2026-05-31 | style: remove section separator comments from utils.py |
| `3acaf62` | 2026-05-31 | fix: resolve all markdown lint + suppress Python SWIG warnings |
| `99164ce` | 2026-05-31 | style: fix Python lint (unused imports, formatting) |
| `35c38de` | 2026-05-31 | fix: upgrade transformers to 4.44.2 for BiomedCLIP processor compat |

---

## Validation

- **61 tests passed**, 7 skipped (integration tests, require real data)
- **0 ruff violations** across entire codebase
- **0 markdownlint errors** across all markdown files
- **0 Python warnings** (SWIG deprecation suppressed in pyproject.toml)
- **BiomedCLIP verified**: 195.9M params, 512-dim embeddings, CPU inference OK
