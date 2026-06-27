# Project 5 — Revised Strategy: Unsupervised Concept Discovery for Medical VLMs

> **Version:** 2.0 (post-audit reframe)
> **Supersedes:** `docs/archive/PROJECT-STRATEGY-v1-original.md`
> **Key sources:** `docs/audits/ML-AUDIT-2026-06-25.md` (M-001..M-008), `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md`, `docs/requirements/PROJECT-BRIEF.md`
> **Status:** Active strategy — governs implementation.

---

## Executive Summary

The original strategy trained a Top-K SAE on BiomedCLIP's **512-d projected embedding**
as the primary method. Methodological audit `ML-AUDIT-2026-06-25.md` showed this
combination is structurally ill-suited: the projected space is anisotropic and
information-bottlenecked, sparse decomposition in it is non-identifiable at our data
scale, and the resulting cross-seed Jaccard (0.0038) sits at the analytical chance floor.

**The revised strategy reframes the existing code as a documented baseline (failure case)
and introduces two methodologically sound main methods**, maximising coverage of the
assessment rubric (16 pt):

| Role | Method | Space | Status |
|---|---|---|---|
| **Baseline / failure case** | SAE TopK on 512-d projected embedding | 512-d (shared) | done — re-narrated |
| **Main A** | SAE on 768-d pre-projection hidden state | 768-d (CLS token) | to implement |
| **Main B** | SPLiCE — sparse direct decomposition on RadLex dictionary | 512-d (shared) | to implement |
| **Extension** | Structured concept organisation: clustering + RadLex hierarchy | — | to implement |
| **Evaluation** | LLM judge (MedGemma) — Aligned / Unaligned / Uncertain | — | pending M-007 fixes |

---

## 1. Scoring Rationale

The course rubric allocates 16 points across six axes. The revised strategy maximises each one:

| Rubric axis | How the reframe addresses it |
|---|---|
| **Literature review** | Two verified paradigms: SPLiCE (projected embedding) + SAE-interpretability (hidden state). Broader and more accurate than SAE-only. |
| **Research gaps** | Gap #1 (concept instability) is *addressed*, not merely cited. The baseline manifests it; the main methods resolve or bypass it. |
| **Methodology + assessment** | Three methods compared (A / B / baseline) + quantitative judge on all. Richer methodology than a single run. |
| **Originality / novelty** | Systematic comparison of SAE-on-projected vs SAE-on-hidden vs SPLiCE on a real clinical dataset = original methodological contribution. |
| **Discussion and analysis** | `ML-AUDIT-2026-06-25.md` is the deep critical analysis with root-causes, verified hypotheses, and refutations. |
| **Failure cases** | Baseline 512-d is a documented failure case with root-cause (M-001): spurious concepts, chance-floor instability, naming ≈ random. |
| **Clarity** | Explicit roles (baseline / main / extension) yield a linear narrative. |

---

## 2. Project Context

### 2.1 Reference framework — MedConcept

**MedConcept** (Haque et al., arXiv 2604.11868, 2026) is the primary reference.
Pipeline: (1) extract sparse activations from pretrained VLM; (2) align to medical
vocabulary; (3) generate pseudo-reports; (4) evaluate with frozen LLM judge.
Deviations declared in `notebooks/autoencoder/baseline/REPORT.md §3.1`.

### 2.2 The two literature paradigms (verified)

| Paradigm | Operates on | Naming | Canonical method |
|---|---|---|---|
| **A — SAE interpretability** | Intermediate hidden state / residual stream (768-d) | Separate naming bridge | Steering CLIP ViT with SAEs (arXiv 2504.08729); OpenAI Scaling SAEs |
| **B — Concept decomposition** | Final projected embedding (512-d, shared space) | Intrinsic (coefficients = concepts) | SPLiCE (Bhalla et al., NeurIPS 2024) |

The current baseline mixes the tool of Paradigm A (learned SAE) with the location of
Paradigm B (512-d projected). This is the root cause of the structural failure (M-001).

### 2.3 Dataset and backbone

- **Backbone:** `chuhac/BiomedCLIP-vit-bert-hf` — ViT-B/16, hidden dim 768, projected to 512.
- **Dataset:** IU X-Ray — 7,470 images; ~5,976 train / ~1,494 test (group-aware split).
- **Scale note:** 5,976 train images is 2–3 orders of magnitude below what a *learned* SAE
  normally requires — makes Path B more data-frugal by construction.

### 2.4 Vocabulary

- `data/vocabulary.json` — 508 RadLex terms + 14 NIH seeds. Format: `{term, similarity_score, source}`.
- `embeddings/standard/text_vocab_embeddings.pt` — 508 × 512 text embeddings.
- **Coverage risk:** see `docs/design/proposals/VOCAB-BUILDING-ALTERNATIVES.md`.

---

## 3. Research Gaps Addressed

| Gap | Description | Addressed by |
|---|---|---|
| **#1 Concept instability** | No stability guarantees across seeds/runs | Main A (Jaccard should improve); Main B (deterministic) |
| **#2 Representation mismatch** | SAE-interp. literature uses hidden states, not projected embeddings | Main A |
| **#3 Small-data robustness** | Learned SAEs require 10⁵–10⁶ samples; we have ~6k | Main B (SPLiCE, fixed dictionary) |
| **#4 LLM judge bias** | Position bias, verbosity bias, hallucination unresolved | Documented; M-007 fixes applied |
| **#5 Concept independence** | Flat top-k lists; no structure across concepts | Extension (clustering + hierarchy) |
| **#6 Incomplete clinical reports** | Findings/Impression may omit anomalies → spurious Unaligned | Documented in failure analysis |

---

## 4. Methods

### 4.1 Baseline — SAE on 512-d projected embedding

**Role:** documented failure case that motivates the main methods.
**Status:** done and characterised.

Key results: Jaccard = 0.0038 ≈ analytical null; dead features 40–60%; naming 0.395 ≈ random.
Root cause (M-001): non-identifiable sparse factorisation in anisotropic projected space.

Narrative: *"We faithfully implement MedConcept's SAE on the shared projected embedding.
The result isolates a structural limitation. This failure case motivates Methods A and B."*

### 4.2 Main B — SPLiCE (implement first — safety net)

**Status:** to implement (`src/concept_discovery/spliece.py`).

Decomposes the 512-d image embedding (gap-corrected) into a sparse non-negative linear
combination of RadLex text embeddings. No learned SAE; no seed; naming is intrinsic.

```
image_emb (512,) → subtract modality_gap → emb_corr
solve: min ||emb_corr - vocab_emb.T @ c||  s.t. c >= 0
  via OrthogonalMatchingPursuit(k) or Lasso(positive=True)
top-k concepts = argsort(c, descending)[:k]
```

Properties: deterministic, data-frugal, positive result guaranteed.
Risk: RadLex coverage limits expressiveness.

### 4.3 Main A — SAE on 768-d hidden state (centrepiece)

**Status:** to implement.

Extracts the CLS token of BiomedCLIP's last hidden state (768-d, pre-projection):

```python
out = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
hidden = out.last_hidden_state[:, 0, :]  # (B, 768)
```

Naming bridge (768→512): use the **frozen projection** `W_proj` of BiomedCLIP:
```python
dec_projected = W_dec_768 @ W_proj.T  # (dict_size, 512) → cosine vs RadLex
```

Config: `SAEConfig.activation_dim=768`; persist to `embeddings/standard_hidden/`;
apply Path C hygiene (steps 5k–10k, lr=5e-5 explicit).

Verification: Jaccard lifts above analytical null; dead-feature rate drops; naming > random.

### 4.4 Extension — Structured concept organisation

**Status:** to implement (`src/concept_discovery/organize.py`).

1. Cluster concept directions/coefficients by cosine similarity in RadLex text space.
2. Map clusters to RadLex anatomical hierarchy.
3. Per-sample: report which concept families activate, not just a flat top-k list.
4. Notebook: `notebooks/autoencoder/06_concept_organization.ipynb`.

### 4.5 LLM Judge

Model: `unsloth/medgemma-4b-it`. Requires GPU + HuggingFace auth.
Prerequisites: M-007 fixes (F-001..F-003, F-007) in `src/evaluate_llm_judge.py`.

---

## 5. Recommended Sequencing

1. **Fix judge M-007** — prerequisite for faithfulness numbers (independent, parallel).
2. **Implement Path B (SPLiCE)** — safety net first; positive result guaranteed.
3. **Implement extension** on top of B — immediate rubric value, low cost.
4. **Implement Path A (SAE 768-d)** — centrepiece; higher cost; schedule after B stable.
5. **Apply Path C hygiene** to A (steps reduction, explicit lr).
6. **Consolidation:** recap, slides, repo.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Path A does not lift Jaccard | Medium | High | B as safety net |
| Naming bridge 768→512 distorts | Medium | Medium | Frozen projection (recommended); fallback to learned probe |
| SPLiCE limited by RadLex coverage | Medium | Medium | Extended vocabulary (`VOCAB-BUILDING-ALTERNATIVES.md`) |
| Time/compute: new extraction + 5-seed retrain | High | Medium | B first; A scheduled after |
| Judge not fixed in time | Medium | High | M-007 as prerequisite |

---

## 7. Deliverables

| Required output | Content |
|---|---|
| Critical literature review | Two paradigms (SAE-interp. + SPLiCE) + MedConcept, primary sources |
| Motivated gaps | Instability (gap #1) addressed, not merely cited |
| Working pipeline | A + B end-to-end |
| Metrics + failure cases | Judge on A/B + baseline as documented failure case |
| Original contribution | Comparison A vs B vs baseline on real clinical dataset |
| Recap 2–3 pages | 6-section structure |
| Slides 15 min | Same structure; live demo of B (deterministic) |
| GitHub repo | Existing modules + `src/concept_discovery/{spliece,organize}.py` |

---

## 8. References

- **MedConcept** — Haque et al., arXiv 2604.11868 (2026)
- **SPLiCE** — Bhalla et al., NeurIPS 2024, arXiv 2402.10376
- **Steering CLIP's ViT with SAEs** — arXiv 2504.08729
- **clip-topk-sae** — HuggingFace lasgroup
- **Scaling and Evaluating SAEs** — OpenAI
- Internal: `docs/audits/ML-AUDIT-2026-06-25.md`, `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md`,
  `docs/design/proposals/CONCEPT-INSTABILITY-DIAGNOSIS.md`, `docs/design/proposals/ADDITIONAL-ABLATION-STUDIES.md`
