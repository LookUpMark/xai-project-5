# Makefile — convenience targets for the concept-discovery pipeline.
PY := .venv/bin/python
export PYTHONPATH := src:.

.PHONY: help data embed baseline hidden spliece conceptorg null judge ablation all test paper lint clean

help: ## show available targets
	@awk 'BEGIN{FS":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

data: ## stage the IU X-Ray dataset
	$(PY) xai_datasets/download_iu_xray.py

embed: ## build vocabulary + extract BiomedCLIP embeddings
	$(PY) scripts/00_build_vocab.py
	$(PY) scripts/01_extract_embeddings.py

baseline: ## baseline 512-d SAE (train -> name -> explain -> stability)
	$(PY) scripts/02_baseline.py

hidden: ## Path A 768-d hidden-state SAE
	$(PY) scripts/03_hidden.py

spliece: ## SPLiCE (Path B) deterministic decomposition
	$(PY) scripts/04_spliece.py

conceptorg: ## concept organization for all three sources
	$(PY) scripts/05_concept_organization.py --source spliece
	$(PY) scripts/05_concept_organization.py --source sae-baseline
	$(PY) scripts/05_concept_organization.py --source sae-hidden

null: ## count-matched null explanations
	$(PY) scripts/06_generate_null.py --k 13
	$(PY) scripts/06_generate_null.py --k 5 --output results/iu_xray/null_k5

judge: ## LLM judge on the baseline explanations (needs GPU + HF creds)
	$(PY) scripts/07_judge.py --input baseline

ablation: ## dict_size x k sweeps (baseline + hidden)
	$(PY) scripts/08_baseline_ablation.py
	$(PY) scripts/09_hidden_ablation.py

all: ## run the full ordered pipeline
	./scripts/90_run_all.sh

test: ## run the test suite
	$(PY) -m pytest tests/

paper: ## build the LaTeX report
	cd docs/latex && latexmk -pdf main.tex

lint: ## run ruff
	ruff check .

clean: ## remove Python caches + LaTeX build artifacts
	find . -type d -name __pycache__ -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
	cd docs/latex && latexmk -C
