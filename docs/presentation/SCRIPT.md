# Speaker Script

**Project:** *Unsupervised Concept Discovery for Medical Vision–Language Models:
A Rigorous Characterization of Sparse-Autoencoder Failure and Deterministic Alternatives*
**Course:** Explainable & Trustworthy AI — Politecnico di Torino
**Paper:** docs/latex/main_extended.tex (extended version)
**Authors / members:** Marc'Antonio Lopez · Nicolò Colle · Carmine Francesco Benvenuto

---

## How to use this script

- **22 slides.** Part 1 — Nicolò Colle (01–06), Part 2 — Marc'Antonio Lopez (07–18), Part 3 — Carmine Francesco Benvenuto (19–22).
- **~35–40 s per slide → ~13.5–14 min total** (under 15). Member 2 is the methodological core, so it carries more slides at lighter density (12 vs 6 / 4).
- The deck shows the essentials; **this script is what you actually say** (more detail, plain English).
- Numbers are real — from `docs/latex/main_extended.tex`, cross-checked on `results/*.json`. If you round, say "about".
- Each block: **slide label**, the spoken text, timing. Press **`N`** in the deck for on-side notes.

---

## Paper coverage map (every section touched)

| Paper (`main_extended.tex`) | Slide |
|---|---|
| Abstract + §1 Introduction | 01, 02 |
| §2 Related work (CBM, TCAV, SAE, SPLiCE, BiomedCLIP, MedConcept, RadLex/MeSH) | 03 |
| §3 Research gaps (1→6, in order) | 04 |
| §4.1 Backbone / vocabulary / split | 05, 06 |
| §4.2 Baseline Top-K SAE (config) | 07 |
| §4.6 Evaluation metrics (matched + subspace null) | 08, 09 |
| §5.1 Baseline non-identifiability (Tab.1) | 10, 11 |
| §5.2 Relabeling control | 12 |
| §4.3 + §5.3 Path A (768-d) + Tab.3 ablation | 13, 14 |
| §5.3 Data scale + why 2D differs from 3D | 15 |
| §4.4 + §5.4 Path B (SPLiCE) | 16, 17 |
| §4.5 + §5.5 Concept organisation + Tab.4 | 18 |
| §5.6 Faithfulness + upper tail stats | 19 |
| §5.7 LLM-judge + Path A hidden + random-k | 20 |
| §6 Conclusions (3 lessons, 4 contributions) | 21 |
| §6 Limitations + future work (3 limitations + PadChest) | 22 |

---

# PART 1 — Nicolò Colle · Problem, background, setup  (slides 01–06, ~3.5 min)

## 01 · Cover
> Good morning. Our project re-implements the **MedConcept** pipeline for discovering interpretable
> concepts inside a medical vision–language model, **BiomedCLIP**, on two scales — IU X-Ray, about
> 7 thousand frontal and lateral chest X-rays, and ROCOv2, about 80 thousand radiology images, ten times
> larger.
>
> We adapted this paradigm from the original MedConcept paper, which targets **3D volumetric**
> abdominal CT with the Merlin foundation model. Our 2D adaptation trades resolution for accessibility:
> it runs on a single consumer GPU.
>
> The central finding is **methodological and negative**. Sparse-autoencoder concepts on the pooled CLS
> embedding are genuinely **non-identifiable across seeds**, and neither a deeper representation nor a
> ten-times-larger corpus lifts identifiability. The binding constraint is the representation site.
>
> **Timing:** ~40 s

## 02 · Problem
> Why does this matter? Medical VLMs are accurate but opaque. They perform well on pathology
> classification, segmentation, and report generation — but their internal representations are
> high-dimensional, polysemic, and unreadable.
>
> In safety-critical clinical use, a model right for the wrong reasons is as dangerous as one that is
> simply wrong. Classical post-hoc tools show *where* the model looks, not *what* it knows.
>
> **Timing:** ~35 s

## 03 · Approach
> Two families of concept-based XAI. **Supervised** — Concept Bottleneck Models (Koh 2020) and TCAV
> (Kim 2018) — need a pre-defined, labelled concept set: exactly the limitation unsupervised discovery
> targets.
>
> **Sparse Autoencoders** decompose activations into a sparse overcomplete dictionary. Bricken et al.
> showed this yields monosemantic latents; Gao et al. scaled it with the Top-K SAE, which we adopt.
>
> **MedConcept** (Haque 2026) instantiates the full pipeline: SAE extraction, alignment to a radiology
> vocabulary, and LLM-judge evaluation. This is the template we re-implement and stress-test.
>
> **Timing:** ~40 s

## 04 · Research gaps
> From the literature we identify **six gaps**, in the paper's order.
>
> **Gap 1 — instability:** sparse factorisations are not reproducible across seeds; no prior medical-VLM
> work reports seed-stability against a null. **Gap 2 — clinical validity:** a cosine-assigned name is
> cosmetic unless it tracks pathology. **Gap 3 — representation-location mismatch:** the SAE literature
> uses raw hidden states, but reference pipelines fit on the projected, gap-bearing CLIP space.
> **Gap 4 — small data:** SAEs need 10⁵–10⁶ activations; medical corpora are 2–3 orders smaller.
> **Gap 5 — flat concepts:** top-k lists ignore hierarchy. **Gap 6 — non-reproducible evaluation:**
> single-metric claims never report a chance floor, and slot-wise overlap is degenerate.
>
> **Timing:** ~40 s

## 05 · Setup
> Our backbone is **BiomedCLIP**. Its image encoder exposes a **768-d hidden state** that a frozen
> projection compresses into the **512-d** contrastive space. We name features against **domain-matched**
> vocabularies — **RadLex** for IU X-Ray (chest-specific), **MeSH** for ROCOv2 (multimodal) — and we
> correct the **modality gap** first.
>
> **Timing:** ~35 s

## 06 · Data
> Two scales. **IU X-Ray** — about 7,470 frontal and lateral chest X-rays across 3,852 studies, split
> **by study** so no study appears in both train and test (verified zero overlap). **ROCOv2** — about
> 80 thousand images, split by image. ROCOv2 exists to test one hypothesis: does more data cure
> non-identifiability? Spoiler — it does not. Hand off to Marc'Antonio: method and the core result.
>
> **Timing:** ~35 s

---

# PART 2 — Marc'Antonio Lopez · Method + non-identifiability + deterministic alternatives  (slides 07–18, ~7.5 min)

## 07 · Method
> Three decompositions, **one SAE config**. The **baseline** is a faithful MedConcept re-implementation —
> Top-K SAE on the 512-d projected space, trained across **five seeds** so a stability metric is possible.
> **Path A** runs the same SAE on the **768-d hidden state before** the projection, bridged by the frozen
> projection matrix. **Path B** is **SPLiCE** — decomposition over a fixed domain-matched dictionary,
> no learning, deterministic.
>
> One config for clean comparisons: dictionary D = 2048, top-k = 32, 8,000 steps, seeds 0/42/123/456/789.
> Each path targets tagged gaps.
>
> **Timing:** ~40 s

## 08 · Evaluation — matched metric
> Stability must be measured **permutation-invariantly**. For each decoder row of one seed, take its
> **best cosine match** across *all* rows of the other seed, then average. Feature order does not matter —
> that is what makes it a real stability measure.
>
> The old slot-wise Jaccard is *not* permutation-invariant, so it is degenerate. We prove that on slide 12.
>
> **Timing:** ~35 s

## 09 · Evaluation — subspace null
> The null is the honest part. Real decoder directions concentrate in a **subspace** of effective rank
> about **357** on IU, **363** on ROCOv2 — not the ambient 512. We draw null vectors *inside* that
> subspace. An isotropic full-512 null would inflate the ratio; we report it only as a lower bound.
>
> **Timing:** ~35 s

## 10 · Headline result
> The headline. The baseline reconstructs its inputs almost perfectly — but cross-seed **feature identity
> is absent**. Matched cosine sits at just **1.67×** the conditioned null on IU, **1.81×** on ROCOv2.
>
> This is genuine non-identifiability, not a training bug: decoders are full-rank, initializations distinct,
> no collapse. It is intrinsic to decomposing a pooled, L2-normalized, contrastive-shaped global vector.
>
> **Timing:** ~30 s

## 11 · Baseline results (Tab.1)
> The numbers. Reconstruction is **0.99** on IU, **0.97** on ROCOv2 — looks great. But matched-over-null
> is only **1.67× / 1.81×**, raw matched cosine **0.299 / 0.327**, and almost nothing matches strongly —
> **0.0% / 0.3%** above cosine 0.9. Naming stays at **0.40 / 0.48**: a cosine-assigned name is cosmetic.
>
> **Timing:** ~35 s

## 12 · Relabeling control
> Why is slot-wise degenerate? **Permutation invariance.** Take one trained SAE and rename its features
> with a random permutation. The network is mathematically identical — reconstruction unchanged — yet
> slot-wise Jaccard collapses to the **0.0077** floor, indistinguishable from two genuinely different SAEs.
> The matched metric correctly sees identity.
>
> Slot-wise is permutation noise. This is **contribution 1** of the paper. From here, every stability claim
> uses the matched metric.
>
> **Timing:** ~40 s

## 13 · Path A — concept
> Path A moves the SAE to the hidden state **before** the projection. It is genuinely healthier — dead
> features **1.7% vs 16%**, naming **0.47 vs 0.42**. But the matched verdict does **not** change: still
> almost nothing matches strongly.
>
> This is **weak universality** — a shared subspace exists, but not canonical, reproducible features. Path A
> still operates on the pooled CLS token, not patch tokens.
>
> **Timing:** ~40 s

## 14 · Path A — ablation (Tab.3)
> The ablation confirms it. Pushing overcompleteness from D=1024 to 4096, matched-over-null falls
> **2.78× → 2.63× → 2.00×** — toward the null, never away from it. No setting escapes weak universality.
>
> The binding constraint is the **representation site**, not capacity.
>
> **Timing:** ~35 s

## 15 · Scale refuted
> Does ten-times more data cure non-identifiability? **No.** Training on ROCOv2 — 80 thousand images —
> improves training health dramatically (dead features **16% → 0.6%**). But identifiability stays flat:
> matched **0.299 → 0.327**, ratio **1.67× → 1.81×**, fraction ≥0.9 **0% → 0.3%**.
>
> The binding constraint is the representation site, not data volume. This is **contribution 3**.
>
> **Timing:** ~35 s

## 16 · SPLiCE — concept
> SPLiCE is the deterministic alternative. It decomposes each embedding into a sparse, non-negative
> combination over a **fixed domain-matched dictionary** — RadLex for IU, MeSH for ROCOv2. The dictionary
> is never estimated, so the result is **deterministic by construction**: zero cross-seed instability.
>
> Same output schema as the SAE, so the same judge can score it. Honest caveat: frequent terms are
> "mixed" — a property of the supplied dictionary, not SPLiCE's method.
>
> **Timing:** ~40 s

## 17 · SPLiCE — coverage + gap correction
> SPLiCE scales to both corpora. IU X-Ray: **1,515** images in **9.5 s**, using **997** of 1,031 RadLex
> terms. ROCOv2: **15,958** images in **101 s**, all **1,024** MeSH terms.
>
> And correcting the modality gap is the decisive fix — max atom–image cosine jumps **0.49 → 0.85** on IU,
> **0.54 → 0.86** on ROCOv2. The gap was binding, not the solver.
>
> **Timing:** ~35 s

## 18 · Organisation
> Finally, organisation. We cluster the active vocabulary into **concept families** using ontology ancestors
> plus two degeneracy guards — leaf-root rejection, and a subtree canopy that rejects overly generic
> ancestors like "anatomical entity". SPLiCE, the densest, cuts concepts-per-image about **1.75×**.
>
> But silhouette is weak — **0.020** SPLiCE, **0.094** baseline, **0.284** Path A hidden (mostly empty):
> the noisy IU vocabulary makes every cluster inherit a noisy ancestor. ROCOv2 SPLiCE replicates
> non-degenerately (1,024 concepts, 32 families, silhouette 0.066).
>
> End of Part 2. Hand off to Carmine: evaluation + conclusions.
>
> **Timing:** ~40 s

---

# PART 3 — Carmine Francesco Benvenuto · Evaluation, conclusions  (slides 19–22, ~2.5 min)

## 19 · Faithfulness
> Do **any** features track real pathology? We correlate activations against ground-truth labels with a
> shuffle null. A small upper tail is genuinely faithful — about **17.8% ± 0.9%** of live features exceed
> the 95th percentile of the null; the strongest has |r| = **0.459** and looks like a *mass*.
>
> But those same features stay **unstable across seeds** — zero cluster at cosine ≥0.90. Partially faithful,
> entirely non-reproducible.
>
> **Timing:** ~40 s

## 20 · LLM judge
> The judge scores each concept as Aligned, Unaligned, or Uncertain. SPLiCE on IU scores **81.6% Aligned**
> under MedGemma-4B — but only **3.3%** under Llama-3.1-8B. Random-k null explanations score comparably
> (**79.6%** under MedGemma).
>
> So %Aligned is a property of the **judging model** as much as of the explanation — and this local
> plausibility is **orthogonal** to the global identifiability verdict.
>
> **Timing:** ~40 s

## 21 · Conclusions
> Three lessons. **One:** interpretability claims about discovered concepts must be calibrated against
> analytical nulls **and** a permutation-invariant statistic — slot-wise overlap is degenerate. **Two:**
> for a factorisation with limited identifiability, data scale is the least likely culprit. **Three:** an
> honest, partially-negative evaluation is more useful than a curated one; the defect is localised to the
> representation site.
>
> SPLiCE is the sole deterministic, judge-ready alternative; a small upper tail is genuinely faithful.
>
> **Timing:** ~40 s

## 22 · Future work
> The diagnosis is predictive. The next experiment is an SAE on the BiomedCLIP **patch-token residual
> stream** — a mid-layer representation, not the pooled CLS token. This should restore identifiability.
>
> Three limitations: **(i)** the LLM-judge is model-dependent, with weak discriminability; **(ii)** the
> predicted mitigation is the patch-token SAE; **(iii)** our 10× test also broadens the domain — the clean
> scale test is PadChest (~160k chest X-rays), begun but unfinished.
>
> Thank you. Questions?
>
> **Timing:** ~40 s

---

**Total timing:** ~13.5–14 minutes (under 15).

---

## References cited in the deck

Each slide that refers to the literature shows a **footer citation** in the deck
(`.citations` block, author + year + short title). When you reach that slide,
name the paper aloud. The list below is grounded verbatim in `index.html`.

| Slide | Footer citation |
|---|---|
| 03 · Approach | Koh et al. 2020 — Concept Bottleneck Models |
| 05 · Setup | Zhang et al. 2025 — BiomedCLIP |
| 06 · Data | Demner-Fushman et al. 2016 — IU X-Ray Dataset |
| 12 · Relabeling control | Contribution (1) — Relabeling Control (This Work) |
| 13 · Path A · concept | Lan et al. 2024 — Quantifying Feature Space Universality via SAEs |
| 16 · SPLiCE · concept | Bhalla et al. 2024 — Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE) |
| 20 · LLM judge | Sellergren et al. 2025 — MedGemma-4B |
| 22 · Future work | Bricken et al. 2023 — SAEs on Residual Streams |

Additional authors **named in the slide body** (no separate footer, but say them
if asked for the source):

- **03** — Kim 2018 (TCAV); Bricken 2023 (dictionary learning → monosemantic
  latents); Gao 2024 (Top-K SAE); Haque 2026 (Merlin / MedConcept paradigm on
  3D CT); Blankemeier 2024 (Merlin 3D CT foundation model).
- **05** — Langlotz 2006 (RadLex radiology lexicon); NLM 2026 (MeSH); Liang 2022
  (modality gap, "Mind the Gap").
- **06** — Rückert 2024 (ROCOv2 dataset).
- **13** — Leask 2025 (co-cited with Lan 2024 on SAE universality).

All entries cross-check against `docs/latex/biblio.bib` (author, year, title).

---

## Regenerating `deck.pdf`

`deck.pdf` is a 22-page 16:9 render of `index.html`, one slide per page, driven
by the `@media print` block (1920×1080 `@page`, enlarged body text). Regenerate
headless with the bundled Chromium:

```bash
chrome --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=deck.pdf "file://$PWD/index.html"
```

(or run `node generate-pdf.js` if Puppeteer is installed; or open `index.html`
in a browser → Ctrl+P → Save as PDF). Last render: 22 pages, 1440×810 pt (16:9).
