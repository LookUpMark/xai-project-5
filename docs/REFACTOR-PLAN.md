# Refactor Plan — Delivery-Ready Cleanup

**Branch:** `refactored` (off `dev`)
**Goal:** A clean, self-documenting repo where a single ordered pipeline of scripts
reproduces every result in the paper, with no dead code, no path mismatches, and
delivery-ready docs.
**Status:** PLAN — pending approval before execution.

## Guiding principles
1. **Correctness first.** The load-bearing `sys.path.insert(0, "src/")` + sibling-import
   convention is preserved unless an optional "deep rename" phase is approved.
2. **Lazy correctness for paths.** Data lives flat on disk but config expects namespaced
   `<dataset>/` paths. Fix by **moving files** (`mv`), not re-extracting (hours/GPU).
3. **One pipeline, numbered scripts.** Every paper result reproducible from
   `scripts/00_…` → `scripts/90_run_all.sh`.
4. **Delete freely what is verified dead; verify-then-delete the rest.**

---

## Phase 1 — Delete dead code & cruft (safe, no logic change)
- `datasets/` (top-level, only stale `__pycache__`; superseded by `xai_datasets/`)
- `report/` (empty `.gitkeep` only)
- `notebooks/autoencoder/mock/wandb/` (3 stale wandb run dirs; wandb is a no-op)
- `models/ablation_a2 2/` (macOS space-dup of `ablation_a2/`)
- all stale `__pycache__/` (incl. `tests/unit/test_fix_train_test_split.pyc`)
- `results/mock/`, loose `results/aligned_scores.csv` (all-ERROR placeholder)
- LaTeX build artifacts `docs/latex/main.{aux,bbl,blg,fdb_latexmk,fls,log,out,synctex.gz}`
  (already gitignored; remove from working tree)

## Phase 2 — Reconcile the path migration (the root-cause fix)
Config (`config._set_dataset_paths`) expects `embeddings/iu_xray/…` and `models/iu_xray/…`
but data sits flat. **Move, don't regenerate:**
- `embeddings/standard/` → `embeddings/iu_xray/standard/`
- `embeddings/standard_hidden/` → `embeddings/iu_xray/standard_hidden/`
- `embeddings/augmented_hidden/` → `embeddings/iu_xray/augmented_hidden/`
- `models/modality_gap.pt` → `models/iu_xray/modality_gap.pt`
- `models/sae_seed{N}/`, `models/sae_hidden/`, `models/sae_hidden_*/`, `models/ablation_*`,
  `models/loss_curve/` → under `models/iu_xray/`

**Namespace ALL results under `results/iu_xray/`:**
- move flat `results/{spliece,concept_organization,concept_organization_baseline,
  concept_organization_hidden,sae_hidden_augmented,sae_hidden_ablation,null,null_k5}/`
  → `results/iu_xray/<name>/`
- move `results/.judge_checkpoint_*.json`, `judge_checkpoint_*.json` → `results/iu_xray/judge/`

After Phase 2: **regenerate SPLiCE** (`run_spliece.py`) so the committed artifact reflects
the fixed OMP+NNLS solver and the stale "clamp" report is refreshed. (Unblocks the judge.)

## Phase 3 — Scripts: one numbered, documented pipeline
- Prefix scripts with ordered numbers reflecting execution order:
  `00_build_vocab.py` · `01_extract_embeddings.py` ·
  `10_baseline.py` (was `run_baseline.py`) · `11_hidden.py` (was `run_path_a.py`,
  incl. `--variant standard|augmented`) · `20_spliece.py` ·
  `30_concept_organization.py` · `40_generate_null.py` · `41_judge.py` (NEW wrapper around
  `src/evaluate_llm_judge.py`, with `--input {baseline,hidden,spliece,null_k5,null_k13}`) ·
  `50_ablation_baseline.py` · `51_ablation_hidden.py`.
- **Collapse** `run_sae_training.py` into `10_baseline.py` (train stage already lives there);
  delete `run_sae_training.py` and the no-op `src/autoencoder/tracking.py`.
- Each script: top docstring with purpose, inputs, outputs, prerequisites.
- **`scripts/90_run_all.sh`**: the single entry point — runs the full ordered pipeline
  end-to-end (with `--skip-train` / `--skip-extract` flags for fast re-runs).
- Verify/remove `src/autoencoder/visualization.py` if confirmed orphaned.

## Phase 4 — Notebooks → scripts  *(DECIDED: migrate A0–A5 to scripts)*
User intent: remove `notebooks/`. Most are already migrated, **except the ablation
notebooks** (`notebooks/autoencoder/ablation/00..05_*.ipynb`) whose logic (A0 consensus,
A1 dead%, A2 k-sweep, A3 KMeans naming, A4 TopK/BatchTopK/JumpReLU bake-off, A5
faithfulness) is **notebook-only** — `run_*_ablation.py` only cover dict×k sweeps.
**Decision: migrate A0–A5 to numbered scripts** (`50_ablation_consensus.py`,
`51_ablation_dead.py`, `52_ablation_ksweep.py`, `53_ablation_kmeans_naming.py`,
`54_ablation_activation_bakeoff.py`, `55_ablation_faithfulness.py`), each writing the
same `REPORT*.md` outputs the paper cites. Then delete `notebooks/` entirely.

## Phase 5 — Docs reorganization
- Consolidate the **4 overlapping guidance docs** to ONE canonical: keep `CLAUDE.md`
  (most complete), archive `AGENTS.md` + `.github/agents/*` + `HANDOFF.md` into
  `docs/archive/` (HANDOFF is transient session state).
- `docs/` cleanup:
  - collapse empty stub dirs (`implementation/`, `literature/`, `overview/` — README-only)
    into a single `docs/archive/` or remove;
  - dedupe `docs/audits/`: keep newest per day, archive same-day dupes
    (`-170028`, `-180529`);
  - reconcile `docs/wiki/CONFIG.md` (exists, in Italian, contradicts CLAUDE.md) —
    rewrite in English or delete; fix the missing `CONTRACTS` wiki reference;
  - rename `docs/latex/` → `docs/paper/` for clarity;
  - rewrite `docs/README.md` as a real reading-order index.
- Rewrite root **`README.md`** (currently a 16-byte stub): project overview, setup,
  one-command reproduction, link to the paper.

## Phase 6 — Standardization
- Fix `pyproject.toml`: list all `src/` packages, add `[tool.ruff]` config (line-length,
  import rules), add `[tool.pytest]` `--import-mode=importlib` default.
- Normalize imports in `src/augmentation/transforms.py` and `xai_datasets/augmentation.py`
  to the sibling style (`from config import …`) used everywhere else.
- Move `tests/test_llm_judge.py` → `tests/unit/`.
- Add a root `Makefile` mirroring `90_run_all.sh` targets (`make data`, `make embed`,
  `make baseline`, `make all`, `make paper`, `make test`).

## Phase 7 — (OPTIONAL) Deep rename  *(DECIDED: SKIP — keep package names)*
Renaming packages (`xai_datasets/` → `src/datasets/`, `embedding_extraction/`
→ `extraction/`, unify `autoencoder/`+`sae_hidden/` under `sae/`) is **high churn**
(every import + sys.path entry + test). Decision: **skip** — clean within existing
package names; correctness and stability win over cosmetic renaming.

---

## Decisions (LOCKED)
1. **Path strategy** → Phase 2 `mv` approach (cheap, no re-extraction).
2. **Ablation notebooks (Phase 4)** → migrate A0–A5 to numbered scripts, delete `notebooks/`.
3. **Deep package rename (Phase 7)** → skip; keep names.

## Risks & verification
- **Import breakage:** after every move/rename, run `.venv/bin/pytest --import-mode=importlib`
  (currently green except known mock-dict_size notes, which are fixed in conftest).
- **Path moves:** verify `config.paths.*` resolves to the new locations for both IU X-Ray
  and PadChest before deleting the flat originals.
- **Reproducibility gate:** `scripts/verify_byte_identical.py` must still pass after the
  extraction/split/vocab refactor surface is touched.
- **Order:** Phase 1 → 2 (regen SPLiCE) → 3 → 4 → 5 → 6 → (7). Each phase is a separate
  commit so it can be reviewed/reverted independently.
