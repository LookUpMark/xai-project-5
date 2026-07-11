# Speaker Script

**Project:** *Unsupervised Concept Discovery for Medical Vision–Language Models:
A Rigorous Characterization of Sparse-Autoencoder Failure and Deterministic Alternatives*
**Course:** Explainable & Trustworthy AI — Politecnico di Torino
**Paper:** project paper (`docs/latex/`) + implementations in `src/`
**Authors / members:** Marc'Antonio Lopez · Nicolò Colle · Carmine Francesco Benvenuto

---

## How to use this script

- **28 slides.** Part 1 — Nicolò Colle (01–07), Part 2 — Marc'Antonio Lopez (08–18), Part 3 — Carmine Francesco Benvenuto (19–28).
- **~30 s/slide → ~14 min total** (under 15). Part 2 is trimmed tight (~25 s/slide).
- Each slide also carries a section tag (1 Introduction · 2 Related work · 3 Research gaps · 4 Methodology · 5 Experiments and analysis) in the eyebrow above its title.
- The deck shows the essentials; **this script is what you actually say** (more detail, plain English).
- Numbers are real — from the project paper + `results/*.json`. If you round, say "about".
- Each block: **slide label**, the spoken text, timing. Press **`N`** in the deck for on-side notes.

---

## Paper coverage map (every section touched)

| Paper section | Slide |
|---|---|
| §1 Introduction | 01, 02 |
| §2 Related work (CBM, TCAV, SAE, SPLiCE, BiomedCLIP, MedConcept, RadLex/MeSH) | 03 |
| §3 Research gaps (1→6, in order) | 04 |
| §4.1 Backbone / vocabulary / split | 05, 06 |
| §4.2 Method (3 paths, one SAE config) | 07 |
| §4.6 Evaluation metrics (matched + subspace null) | 08, 09 |
| §5.1 Baseline non-identifiability (Tab.1) | 10, 11 |
| §5.2 Relabeling control | 12 |
| §4.3 + §5.3 Path A (768-d) + Tab.3 ablation | 13, 14 |
| §5.3 Data scale | 15 |
| §4.4 + §5.4 Path B (SPLiCE) | 16, 17 |
| §4.5 + §5.5 Concept organisation + Tab.4 | 18, 24 |
| §5.7 LLM-judge (models, matrix, limitations, scale) | 19, 20, 21, 22, 23, 26 |
| §5.6 Faithfulness | 25 |
| §6 Conclusions | 27 |
| §6 Limitations + future work | 28 |

---

# PART 1 — Nicolò Colle · Problem, background, setup, method overview  (slides 01–07, ~4 min)

## 01 · Cover
> Good morning. Our project re-implements the **MedConcept** pipeline for discovering interpretable
> concepts inside a medical vision–language model, **BiomedCLIP**, on two scales — IU X-Ray, about
> 7 thousand chest X-rays, and ROCOv2, about 80 thousand radiology images, ten times larger.
>
> The central finding is **methodological and negative**. Sparse-autoencoder concepts on the pooled CLS
> embedding are **non-identifiable across seeds**, and neither a deeper representation nor a ten-times
> larger corpus lifts identifiability. The binding constraint is the representation site.
>
> **Timing:** ~40 s

## 02 · Problem
> Medical VLMs are accurate but opaque. They perform well on pathology classification, segmentation and
> report generation — but their internal representations are high-dimensional, polysemic and unreadable.
>
> In safety-critical clinical use, a model right for the wrong reasons is as dangerous as one that is
> simply wrong. Classical post-hoc tools show *where* the model looks, not *what* it knows.
>
> **Timing:** ~35 s

## 03 · Approach
> Two families of concept-based XAI. **Supervised** — Concept Bottleneck Models (Koh 2020) and TCAV
> (Kim 2018) — need a pre-defined labelled concept set: exactly the limitation unsupervised discovery
> targets.
>
> **Sparse Autoencoders** decompose activations into a sparse overcomplete dictionary; the Top-K SAE
> (Gao 2024) sets the sparsity exactly. **MedConcept** (Haque 2026) instantiates the full pipeline —
> SAE, vocabulary alignment, LLM-judge — which we re-implement and stress-test.
>
> **Timing:** ~40 s

## 04 · Research gaps
> Six gaps, in the paper's order. **1 instability:** sparse factorisations aren't reproducible across
> seeds. **2 clinical validity:** a cosine-assigned name is cosmetic unless it tracks pathology.
> **3 representation-location mismatch:** the SAE literature uses raw hidden states, reference pipelines
> use the projected CLIP space. **4 small data:** SAEs need 10⁵–10⁶ activations; medical corpora are
> 2–3 orders smaller. **5 flat concepts:** top-k lists ignore hierarchy. **6 non-reproducible eval:**
> no chance floor; slot-wise overlap is degenerate.
>
> **Timing:** ~40 s

## 05 · Setup
> Our backbone is **BiomedCLIP**. Its image encoder exposes a **768-d hidden state** that a frozen
> projection compresses into the **512-d** contrastive space. We name features against **domain-matched**
> vocabularies — **RadLex** for IU X-Ray, **MeSH** for ROCOv2 — and correct the **modality gap** first.
>
> **Timing:** ~35 s

## 06 · Data
> Two scales. **IU X-Ray** — about 7,470 chest X-rays across 3,852 studies, split **by study** so no
> study leaks across train/test. **ROCOv2** — about 80 thousand images, split by image. ROCOv2 tests one
> hypothesis: does more data cure non-identifiability? It does not.
>
> **Timing:** ~35 s

## 07 · Method
> Three decompositions, **one SAE config**. **Baseline** — Top-K SAE on the 512-d projected space, five
> seeds. **Path A** — same SAE on the 768-d hidden state before projection. **Path B** — SPLiCE, a fixed
> dictionary, no learning, deterministic. One config: D=2048, k=32, 8k steps, five seeds. Hand off to
> Marc'Antonio for the metrics and the core result.
>
> **Timing:** ~30 s

---

# PART 2 — Marc'Antonio Lopez · Metrics + non-identifiability + alternatives  (slides 08–18, ~5 min)

## 08 · Evaluation — matched metric
> Stability must be **permutation-invariant**: for each decoder row of one seed, take its best cosine
> match across all rows of the other, then average. Order doesn't matter. The old slot-wise Jaccard
> isn't — degenerate, proved on slide 12.
>
> **Timing:** ~25 s

## 09 · Evaluation — subspace null
> The null is conditioned on **effective rank**. Real decoder directions live in a subspace of rank ~357
> (IU) / ~363 (ROCOv2) — not the ambient 512. We draw nulls inside that subspace; an isotropic null
> would inflate the ratio.
>
> **Timing:** ~22 s

## 10 · Headline result
> The headline: reconstruction is near-perfect, but cross-seed **feature identity is absent** — matched
> cosine just **1.81×** the null on ROCOv2 (**1.67×** on IU). Genuine non-identifiability, not a bug.
>
> **Timing:** ~20 s

## 11 · Baseline results (Tab.1)
> Reconstruction **0.99 / 0.97** — but matched-over-null only **1.67× / 1.81×**, almost nothing above
> cosine 0.9, naming **0.40 / 0.48**. A cosine-assigned name is cosmetic.
>
> **Timing:** ~22 s

## 12 · Relabeling control
> Why is slot-wise degenerate? **Permutation.** Relabel one SAE's features randomly — same network, same
> reconstruction — and slot-wise Jaccard hits the **0.0077** floor; matched sees identity. Contribution 1;
> from here we use matched.
>
> **Timing:** ~25 s

## 13 · Path A — concept
> Path A moves the SAE **before** the projection — healthier: dead features **1.7% vs 16%**, naming
> **0.47 vs 0.42**. But the matched verdict is unchanged. **Weak universality**: a shared subspace, not
> identical features. Still the pooled CLS token.
>
> **Timing:** ~25 s

## 14 · Path A — ablation (Tab.3)
> The ablation confirms it. From D=1024 to 4096, matched-over-null falls **2.78× → 2.00×** — toward the
> null. No setting escapes. The constraint is the **representation site**, not capacity.
>
> **Timing:** ~20 s

## 15 · Scale refuted
> Does 10× more data help? **No.** ROCOv2 improves training health (dead **16% → 0.6%**) but
> identifiability stays flat: **0.299 → 0.327**, ratio **1.67× → 1.81×**. Constraint is the
> representation site. Contribution 3.
>
> **Timing:** ~22 s

## 16 · SPLiCE — concept
> SPLiCE is the deterministic alternative — sparse non-negative combination over a **fixed dictionary**
> (RadLex / MeSH). Never estimated, so deterministic, zero cross-seed instability. Same schema as the
> SAE → same judge scores it.
>
> **Timing:** ~25 s

## 17 · SPLiCE — coverage + gap correction
> It scales: IU **1,515** images in **9.5 s**, ROCOv2 **15,958** in **101 s**. And gap correction is the
> decisive fix — max atom–image cosine jumps **0.49 → 0.85** on IU. The gap was binding, not the solver.
>
> **Timing:** ~22 s

## 18 · Organisation
> Organisation: cluster into **concept families** via ontology ancestors plus two degeneracy guards.
> SPLiCE cuts concepts-per-image **1.75×**, but silhouette is weak (~0.02) — the noisy vocabulary poisons
> every cluster. Hand off to Carmine for evaluation and conclusions.
>
> **Timing:** ~25 s

---

# PART 3 — Carmine Francesco Benvenuto · LLM-judge evaluation, conclusions  (slides 19–28, ~5 min)

## 19 · Judge models
> We score concept quality with three LLM judges — **MedGemma-4B**, **Llama-3.1-8B** and **Gemma-4-26B**.
> None works as a standalone metric. MedGemma approves even random noise at about 80%. Llama is extremely
> uncertain. Gemma rejects almost everything. Each has a different bias.
>
> **Timing:** ~30 s

## 20 · Results matrix
> The results matrix tells the story. MedGemma scores the **null control at 81%** — the same as the real
> baseline. Llama scores everything low, with SPLiCE slightly ahead. Neither judge discriminates real
> explanations from the null.
>
> **Timing:** ~30 s

## 21 · Judge limitations
> Why they fail. MedGemma is a "yes-man" for medical text — it approves random mappings and fails our null
> control. Llama-3.1 is the opposite: overly cautious, "Uncertain" 96% of the time on IU X-Ray. It
> improves on the ten-times-larger ROCOv2.
>
> **Timing:** ~30 s

## 22 · Gemma 4 26B
> Gemma-4-26B we ran locally via LM Studio, but the hardware cost was too high, so we limited it to the IU
> baseline. Even one run shows an **80-point disagreement**: Gemma rejects 87% while MedGemma approves 81%.
> Llama stays in between.
>
> **Timing:** ~30 s

## 23 · Judge scale effects
> Across data scales: MedGemma is insensitive to explanation quality — it just says yes. Llama-3.1 jumps
> from **0.9% to 23%** with scale. More data brings better naming and fewer dead features, and Llama
> rewards that — but MedGemma does not.
>
> **Timing:** ~30 s

## 24 · Organisation (recap)
> Organisation does its job — SPLiCE cuts concepts-per-image **1.75×** — but separation is weak
> (silhouette ~0.02) because the IU vocabulary is noisy. Tab.4: SPLiCE 32 families, baseline 9, Path A
> hidden 4.
>
> **Timing:** ~25 s

## 25 · Faithfulness
> Do any features track real pathology? A small upper tail is genuinely faithful — **17.8% ± 0.9%** beat
> the shuffle null; the strongest has |r| = **0.459** and looks like a *mass*. But those same features are
> **unstable across seeds** — partially faithful, entirely non-reproducible.
>
> **Timing:** ~35 s

## 26 · LLM judge
> The judge verdict: SPLiCE scores **81.6%** aligned under MedGemma but only **3.3%** under Llama.
> Random-k nulls score **79.6%**. So percent-aligned is a property of the **judging model** as much as the
> explanation — orthogonal to identifiability.
>
> **Timing:** ~30 s

## 27 · Conclusions
> Three lessons. **One:** calibrate interpretability claims against analytical nulls and a
> permutation-invariant statistic — slot-wise is degenerate. **Two:** data scale is the least likely
> culprit. **Three:** an honest, partially-negative evaluation beats a curated one; the defect is
> localised to the representation site. SPLiCE is the sole deterministic, judge-ready alternative.
>
> **Timing:** ~35 s

## 28 · Future work
> The diagnosis is predictive. Next: an SAE on the BiomedCLIP **patch-token residual stream** — a mid
> layer, not the pooled CLS — should restore identifiability. Three limitations: the judge is
> model-dependent; the predicted mitigation is the patch-token SAE; and PadChest is the clean scale test,
> begun but unfinished. Thank you — questions?
>
> **Timing:** ~35 s

---

**Total timing:** ~14 minutes — Part 1 ~4 min · Part 2 ~5 min · Part 3 ~5 min (under 15).

---

## References cited in the deck

Slides that refer to the literature show a **footer citation** (`.citations`, author + year + short title).
Say the paper aloud when you reach it. Grounded verbatim in `index.html`.

| Slide | Footer citation |
|---|---|
| 03 · Approach | Koh et al. 2020 — Concept Bottleneck Models |
| 05 · Setup | Zhang et al. 2025 — BiomedCLIP |
| 06 · Data | Demner-Fushman et al. 2016 — IU X-Ray Dataset |
| 12 · Relabeling control | Contribution (1) — Relabeling Control (This Work) |
| 13 · Path A · concept | Lan et al. 2024 — Feature Space Universality via SAEs |
| 16 · SPLiCE · concept | Bhalla et al. 2024 — Sparse Linear Concept Embeddings (SpLiCE) |
| 19 · Judge models | Sellergren et al. 2025 — MedGemma-4B |
| 26 · LLM judge | Sellergren et al. 2025 — MedGemma-4B |
| 28 · Future work | Bricken et al. 2023 — SAEs on Residual Streams |

Additional authors **named in the slide body** (say them if asked):

- **03** — Kim 2018 (TCAV); Bricken 2023 (dictionary learning); Gao 2024 (Top-K SAE); Haque 2026 (MedConcept / Merlin 3D CT); Blankemeier 2024 (Merlin).
- **05** — Langlotz 2006 (RadLex); NLM 2026 (MeSH); Liang 2022 (modality gap).
- **06** — Rückert 2024 (ROCOv2).
- **13** — Leask 2025 (SAEs do not find canonical units).

All entries cross-check against the project's `docs/latex/biblio.bib`.

---

## Regenerating `deck.pdf`

`deck.pdf` is a 28-page 16:9 render of `index.html`, one slide per page, driven by the `@media print`
block (1920×1080 `@page`, enlarged body text). Regenerate headless with the bundled Chromium:

```bash
chrome --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=deck.pdf "file://$PWD/index.html"
```

(or run `node generate-pdf.js` if Puppeteer is installed; or open `index.html` → Ctrl+P → Save as PDF).
Last render: 28 pages, 1440×810 pt (16:9).
