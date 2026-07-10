# Speaker Script

**Project:** *Unsupervised Concept Discovery for Medical Vision–Language Models:
A Rigorous Characterization of Sparse-Autoencoder Failure and Deterministic Alternatives*
**Course:** Explainable & Trustworthy AI — Politecnico di Torino
**Authors / members:** Marc'Antonio Lopez · Nicolò Colle · Carmine Francesco Benvenuto

---

## How to use this script

- **18 slides = 3 parts × 6**, one member per part. **~45–50 s per slide → ~13.5–14 min total** (under 15).
- The deck shows the essentials; **this script is what you actually say** (more detail, plain English).
- Numbers are real — from `docs/latex/main.tex`, cross-checked on `results/*.json`. If you round, say "about".
- Each block: **slide label**, the spoken text, timing. Press **`N`** in the deck for on-slide notes.

Member → part: **Part 1 — Nicolò Colle**, **Part 2 — Marc'Antonio Lopez**, **Part 3 — Carmine Francesco Benvenuto** (swappable).

---

## Paper coverage map (every section touched)

| Paper (`main.tex`) | Slide |
|---|---|
| Abstract + §1 Introduction | 01, 02 |
| §2 Related work (CBM, TCAV, SAE, SPLiCE, BiomedCLIP, MedConcept, RadLex) | 03 |
| §3 Research gaps (1→6, in order) | 04 |
| §4.1 Backbone / vocabulary / split | 05, 06 |
| §4.2 Baseline Top-K SAE | 07, 09 |
| §4.3 Path A (768-d) | 11 |
| §4.4 Path B (SPLiCE) | 13 |
| §4.5 Concept organisation | 14 |
| §4.6 Evaluation metrics | 08 |
| §5.1 Baseline non-identifiability | 09 |
| §5.2 Relabeling control | 10 |
| §5.3 Path A + data scale (+ Tab.3 D/k sweep) | 11, 12 |
| §5.4 SPLiCE results | 13 |
| §5.5 Organisation results | 14 |
| §5.6 Faithfulness | 15 |
| §5.7 LLM-judge | 16 |
| §6 Conclusions (3 lessons, 4 contributions) | 17 |
| §6 Future work (patch-token residual stream) | 18 |

---

# PART 1 — Nicolò Colle · Problem, background, setup  (slides 01–06, ~4.5 min)

## 01 · Cover
> Good morning. Our project re-implements the **MedConcept** pipeline for discovering interpretable
> concepts inside a medical vision–language model, **BiomedCLIP**, on two scales — IU X-Ray, about
> 7 thousand images, and ROCOv2, about 80 thousand, a ten-times scale-up. The punchline up front: we
> did **not** discover clean, reproducible concepts. Instead we produced a **rigorous characterisation
> of why** sparse-autoencoder discovery fails — and what a deterministic alternative can still do.
> I'm Nicolò Colle; with Marc'Antonio Lopez and Carmine Benvenuto we'll walk you through it in three
> equal parts.

*(~45 s. Name the three members.)*

## 02 · Problem — Medical VLMs are accurate but opaque
> Start with the problem. Medical vision–language models are now genuinely good — at pathology
> classification, segmentation, even report generation. But their internal representations are
> **high-dimensional, polysemic, and unreadable**. In a clinical setting that is a real safety concern.
> And the usual tools — saliency maps, attention — tell you **where** the model looks, but almost never
> resolve into **semantic structure** you can audit. That gap is what concept-based explainability
> tries to close.

*(~40 s. One idea: accuracy ≠ understanding.)*

## 03 · Approach — from supervised to unsupervised discovery
> Concept-based XAI has two flavours. **Supervised** methods — Concept Bottleneck Models and TCAV —
> need a pre-defined, labelled concept set; that labelling is the bottleneck. So recent work tries
> **unsupervised** discovery: decompose a frozen model into sparse features, name each against a
> clinical vocabulary, validate. **MedConcept** packages that into three stages — extraction,
> alignment, LLM-judge. We re-implement it faithfully and stress-test it. On the right, sparse
> autoencoders: Bricken showed monosemantic latents; we use the **Top-K** variant from Gao. And
> **SPLiCE** — a deterministic third way — comes back in Part 3.

*(~55 s. Three cards: supervised → SAE → MedConcept.)*

## 04 · Research gaps — six gaps, in the paper's order
> Here is our roadmap: six gaps, exactly in the order the paper lists them. **Gap 1**: sparse
> factorisations aren't reproducible across seeds, and no prior medical-VLM work reports stability
> against a null. **Gap 2**: a cosine-assigned name is cosmetic unless it tracks real pathology.
> **Gap 3**: the SAE literature operates on raw hidden states, but reference pipelines fit on the
> already-projected CLIP space. **Gap 4**: SAEs want hundreds of thousands of activations; medical
> corpora are orders smaller, so you get dead features. **Gap 5**: explanations are flat top-k lists
> with no anatomical structure. **Gap 6**: claims rest on metrics whose chance floor is never reported.
> Every later slide is tagged with the gap it answers.

*(~55 s. Read the six in order; flag that each result slide carries its gap tag.)*

## 05 · Setup — BiomedCLIP + RadLex
> Two ingredients. The backbone is **BiomedCLIP** — a ViT-B/16 image encoder paired with PubMedBERT.
> Its image encoder exposes a **768-dimensional hidden state** that a frozen projection compresses
> into the **512-d** shared contrastive space we actually decompose. That 768-to-512 detail matters a
> lot in Part 2. The vocabulary is **RadLex** — about a thousand radiology terms, ranked by relevance
> using 39 anchor queries across 13 sub-domains, plus 14 ChestX-ray14 seeds. And we handle the
> **modality gap** — the systematic offset between the image and text cones — by estimating it and
> subtracting it before any naming or decomposition. Keep that in mind; in Part 3 correcting it turns
> out to be the decisive fix.

*(~55 s. Land "768 → 512" and "modality gap".)*

## 06 · Data — IU X-Ray vs ROCOv2
> Two datasets, deliberately at different scales. **IU X-Ray** is the primary, small-scale setting.
> We split it **by radiographic study** — the key is patient-and-study — so no study leaks between
> train and test; verified **zero** overlap, recomputed deterministically every run. **ROCOv2** is
> roughly 80 thousand images — a **ten-times** replication — split by image, since it has no shared
> patient structure. The point of ROCOv2 is not "more data is better"; it is a direct test of the
> **scale hypothesis**. That ends Part 1 — over to Marc'Antonio for method and the core result.

*(~55 s. Hand off to Marc'Antonio Lopez.)*

---

# PART 2 — Marc'Antonio Lopez · Method + non-identifiability  (slides 07–12, ~4.5 min)

## 07 · Method — three decompositions, one SAE config
> Thanks. We attack the problem with **three decompositions**. The **baseline** is a faithful
> re-implementation of MedConcept: a Top-K sparse autoencoder on the 512-d space. **Path A** moves the
> SAE onto the 768-d hidden state *before* the projection, to test the representation site. **Path B**
> is **SPLiCE** — decomposition over a *fixed* RadLex dictionary, no learning, so deterministic. One
> SAE config for clean comparisons: dictionary size 2048, k equals 32, eight thousand steps, and
> **five seeds** — 0, 42, 123, 456, 789. The multi-seed setup is what makes a stability metric
> possible at all. Each path targets explicit gaps, tagged on the slide.

*(~55 s. One config; each path → a gap.)*

## 08 · Evaluation metric — matched, against a subspace null
> Before any result, the metric — because it is the heart of our contribution. Stability must be
> **permutation-invariant**: for each decoder row of one seed, take the best cosine match across all
> rows of the other seed. The null is conditioned on the decoder's **effective rank** — about 357, not
> the ambient 512 — so it controls for how concentrated the data manifold really is; an isotropic null
> would inflate the ratio. The old **slot-wise Jaccard** is not permutation-invariant, so it is
> degenerate. We prove that on the next slides.

*(~50 s. erank null is the honest detail.)*

## 09 · Headline result — the SAE is non-identifiable
> Here is the headline. The baseline SAE **reconstructs** its inputs well — so as an autoencoder it
> works — but cross-seed **feature identity is absent**: the best-match is only **1.67 times** the
> conditioned null on IU X-Ray, **1.81 times** on ROCOv2. Almost **zero percent** of features match at
> cosine ≥0.9, and naming stays low, capped around 0.4 to 0.48. And this is not a training bug — the
> decoders are full-rank, the inits are distinct, there is no collapse. It is **genuine
> non-identifiability**.

*(~50 s. Hammer: reconstruction good ≠ features canonical.)*

## 10 · Relabeling control — the slot-wise metric is floor-by-construction
> Before that result, we had to fix the metric people used. The standard slot-wise cross-seed Jaccard
> is **floor-by-construction**. We prove it: take one trained SAE and just **rename** its features with
> a random permutation — same network, reconstruction unchanged — and the slot-wise Jaccard collapses
> to **0.0077**, indistinguishable from two genuinely different SAEs. The matched metric correctly
> reports identity, 1.0. The figure is the cross-seed matrix on IU X-Ray — every cell sits on the
> floor. So slot-wise is permutation noise. From here every stability claim uses the matched metric.
> This is contribution **one** of the paper.

*(~55 s. Point at the matrix.)*

## 11 · Path A vs baseline — moving the site is not enough
> So is it the representation site? Path A trains the SAE on the 768-d hidden state *before* the
> projection. It is genuinely healthier — dead features drop from 16 percent to under 2, naming
> improves from 0.42 to 0.47 — but the permutation-invariant verdict does **not** change: still around
> zero percent of features match at ≥0.9. This is **weak universality** — a shared subspace, not
> identical features. A dictionary/k sweep confirms it: matched-over-null falls from 2.78× to 2.00× as
> the dictionary grows, but **no setting escapes**. And Path A still operates on the **pooled CLS
> token**, not patch tokens. The naming histogram on the right shows why a cosine name is cosmetic:
> most features sit well below 0.5.

*(~55 s. Local win, global verdict unchanged; mention the D/k sweep.)*

## 12 · Scale refuted — 10× does not help
> The other tempting explanation is just **more data**. We test it head-on with ROCOv2, ten times the
> corpus. It works where you'd expect — training health improves, dead features fall from 16 percent
> to under 1 — but identifiability stays **flat**: matched cosine 0.299 to 0.327, subspace ratio 1.67×
> to 1.81×, fraction ≥0.9 zero to 0.3 percent, naming still 0.40 to 0.48. Ten times the data lifts
> **none** of these out of the non-identifiability regime. So the binding constraint is the
> **representation site**: a sparse autoencoder on the CLS vector factors a single, compressed global
> vector whose effective rank, about 357, is far below the 2048-way dictionary. That closes Part 2 —
> over to Carmine for what still works.

*(~55 s. Contribution 3. Hand off to Carmine Benvenuto.)*

---

# PART 3 — Carmine Francesco Benvenuto · Alternatives, evaluation, conclusions  (slides 13–18, ~4.5 min)

## 13 · SPLiCE — deterministic, judge-ready
> Thanks. If learned factorisation is non-identifiable, the clean fix is to **not learn the
> dictionary**. That is **SPLiCE**. For each image we decompose the embedding into a sparse,
> non-negative combination over the **fixed RadLex** dictionary: Orthogonal Matching Pursuit picks 32
> atoms, then non-negative least squares re-solves the coefficients. No estimation from data, so
> **deterministic by construction**, zero cross-seed instability. It scales to both corpora, using
> essentially the full vocabulary. And remember the modality gap? Correcting it lifts the maximum
> atom–image cosine from 0.49 to 0.85 on IU — so the gap, not the solver, was the real bottleneck. The
> output schema is SAE-compatible, so the same judge scores it. Honest caveat: frequent terms are
> "mixed" — a property of the supplied dictionary, not of SPLiCE.

*(~60 s. Determinism + the gap-correction jump.)*

## 14 · Organisation — dense, but weakly separated
> Addressing Gap 5, we cluster the active vocabulary into **concept families** with RadLex ancestors
> and two degeneracy guards — leaf-root rejection, and a subtree canopy that rejects overly generic
> ancestors like "anatomical entity", so labels stay honest. SPLiCE, the densest source, groups its
> concepts into 32 families and cuts concepts-per-image by about **1.75×** — a real redundancy
> reduction. But the silhouette is weak, around 0.02: the IU vocabulary is noisy, so every cluster
> inherits a noisy ancestor. It's a standalone stage; it does not feed the judge.

*(~50 s.)*

## 15 · Faithfulness — a small upper tail is genuinely faithful
> So is any of this real? We correlated feature activations against the ground-truth MeSH and Problems
> labels, with a per-feature shuffle null. The honest answer: a **small upper tail** is genuinely
> faithful — about **18 percent** of live features beat the null's 95th percentile, and the strongest,
> at correlation 0.459, looks like a **mass**, which is clinically plausible. But those same features
> stay **unstable across seeds** — zero cluster at cosine ≥0.90. So concepts are **partially faithful,
> entirely non-reproducible**. That is exactly the partially-negative result worth reporting honestly.

*(~50 s. Two-sided: real signal, but small and unstable.)*

## 16 · LLM judge — local plausibility, model-dependent
> Finally, evaluation with an **LLM-as-judge**. Each concept is scored Aligned, Unaligned or Uncertain
> against the reference report, run to convergence at temperature zero. SPLiCE on IU X-Ray scores
> **81.6 percent Aligned** under **MedGemma-4B** — 1230 of 1508 — but only **3.3 percent** under
> **Llama-3.1-8B**, a general model, which is mostly Uncertain. On ROCOv2 the baseline hits 88.3 versus
> 23.1. The takeaway: the Aligned rate is a property of the **judging model** as much as of the
> explanation. And this **local** plausibility is **orthogonal** to global identifiability — a feature
> can align with one report and still fail to recur across seeds.

*(~55 s. 81.6 vs 3.3 + "orthogonal axes".)*

## 17 · Conclusions — three lessons that generalise
> Three lessons we think generalise. **One**: interpretability claims must be calibrated against
> analytical nulls **and** a permutation-invariant statistic — slot-wise overlap is degenerate. **Two**:
> for a failing factorisation, **data scale is the last suspect** — 10× changed training health, not
> identifiability. **Three**: an honest, **partially-negative** evaluation beats a curated success
> story; the defect is localised in the representation site, not global. In total, four contributions:
> the relabeling control, the matched metric with its null, the scale refutation, and a
> partially-negative evaluation.

*(~50 s.)*

## 18 · Future work + close
> And the diagnosis is **predictive** about what to do next. Everything points at the same fix: train
> the sparse autoencoder on the BiomedCLIP **patch-token residual stream** — a mid layer, **not** the
> pooled CLS token. That is the experiment our characterisation says should restore identifiability.
> Recap: Part 1 set up the problem, Part 2 showed the SAE is non-identifiable and why, Part 3 showed
> the deterministic alternative and the honest evaluation. Thank you — happy to take questions.

*(~45 s. Decisive close + thanks.)*

---

## Timing summary

| Part | Presenter | Slides | Target |
|---|---|---|---|
| 1 | Nicolò Colle | 01–06 | ~4.5 min |
| 2 | Marc'Antonio Lopez | 07–12 | ~4.5 min |
| 3 | Carmine Francesco Benvenuto | 13–18 | ~4.5 min |
| **Total** | | **18** | **~13.5–14 min** (+ Q&A) |

If you run long: trim the caveats on slides 13 and 15 first — easiest to shorten without losing the argument.
