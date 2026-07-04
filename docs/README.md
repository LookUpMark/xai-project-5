# Documentation Hub

Organised by purpose: requirements, design, implementation notes, reviews, and
historical material.

> **Status:** Baseline (512-d SAE), Path A (768-d hidden SAE), Path B (SPLiCE),
> and the concept-organisation extension are implemented and evaluated; the LLM
> judge is wired up (converged GPU run pending). See the report at
> `latex/main.pdf` and the active strategy at `design/PROJECT-STRATEGY.md`.

## Main folders

- `requirements/` — project brief, evaluation guidelines, and source material.
- `design/` — active strategy (`PROJECT-STRATEGY.md`), implementation plan/notes, and proposal documents.
- `plans/` — detailed implementation plans for specific tasks and features.
- `audits/` — methodological audit reports (the scientific record of fixes).
- `releases/` — changelogs and release history.
- `wiki/` — reference knowledge base and module documentation.
- `literature/` — reference papers (SAE monosemanticity, Top-K, stability, SPLiCE).
- `latex/` — the project report (ACM sigconf; sources + `main.pdf`).
- `archive/` — retired documentation kept only for traceability.

## Quick entry points

- **Project report (read this first):** `latex/main.pdf`
- **Active strategy:** `design/PROJECT-STRATEGY.md`
- **Why the reframe:** `audits/ML-AUDIT-2026-06-25.md` + `design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md`
- **Latest audit:** `audits/ML-AUDIT-2026-06-27-180529.md`
- **Ablation results:** `../results/iu_xray/ablation/`
