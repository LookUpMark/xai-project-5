# Docs Overview

This folder is the entry point for the documentation set used to hand off work between team members.

> **Current project status (2026-06-26):** The original strategy (SAE on 512-d projected
> embedding as primary method) has been superseded by a three-method reframe following
> methodological audit ML-AUDIT-2026-06-25. The active strategy is in
> `docs/design/PROJECT-STRATEGY.md` v2.0.

## How to navigate

- `requirements/` — project brief, evaluation guidelines, and source material.
- `design/` — active strategy, implementation plan, and proposal documents.
- `plans/` — detailed implementation plans for specific tasks and features.
- `audits/` — review and audit reports (most recent: ML-AUDIT-2026-06-25.md).
- `releases/` — changelogs and release history.
- `wiki/` — reference knowledge base and module documentation.
- `archive/` — retired or superseded documentation.

## Recommended reading order

### For understanding the current direction (start here)
1. `requirements/PROJECT-BRIEF.md` — what the project asks for.
2. `requirements/EVALUATION-GUIDELINES.md` — rubric, deadlines, deliverables.
3. `docs/audits/ML-AUDIT-2026-06-25.md` — methodological audit; explains WHY the reframe.
4. `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md` — the reframe proposal.
5. `docs/design/PROJECT-STRATEGY.md` — **active strategy v2.0**.
6. `docs/design/IMPLEMENTATION-PLAN.md` — **active implementation plan v3.0**.

### For understanding the existing codebase
7. `CLAUDE.md` — module map, entry points, conventions, known issues.
8. `HANDOFF.md` — session handoff notes; recent decisions and open issues.
9. `docs/wiki/autoencoder/SAE-MODULE.md` — SAEManager API reference.
10. `docs/wiki/CONFIG.md` — all configuration dataclasses.

### For historical context
- `docs/audits/ML-AUDIT-2026-06-23.md` and `ML-AUDIT-2026-06-24.md` — code bug audits.
- `docs/releases/` — changelogs v0.2.0 through v0.4.0.
- `docs/design/proposals/` — design proposals (CONCEPT-INSTABILITY-DIAGNOSIS, ADDITIONAL-ABLATION-STUDIES, etc.).
- `docs/archive/` — original v1 strategy and implementation plan.
- `docs/design/IMPLEMENTATION-NOTES.md` — historical notes (superseded; baseline reference only).

## Maintenance rule

When adding a new document, place it in the narrowest folder that matches its purpose.
If a document becomes obsolete, move it to `archive/` or add a deprecation notice at the top.
