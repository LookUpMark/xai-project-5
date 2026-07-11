# Speaker Script

**Project:** *Unsupervised Concept Discovery for Medical Vision–Language Models:
A Rigorous Characterization of Sparse-Autoencoder Failure and Deterministic Alternatives*
**Course:** Explainable & Trustworthy AI — Politecnico di Torino
**Paper:** docs/latex/main_extended.tex (extended version)
**Authors / members:** Marc'Antonio Lopez · Nicolò Colle · Carmine Francesco Benvenuto

---

## How to use this script

- **18 slides = 3 parts × 6**, one member per part. **~45–50 s per slide → ~13.5–14 min total** (under 15).
- The deck shows the essentials; **this script is what you actually say** (more detail, plain English).
- Numbers are real — from `docs/latex/main_extended.tex`, cross-checked on `results/*.json`. If you round, say "about".
- Each block: **slide label**, the spoken text, timing. Press **`N`** in the deck for on-side notes.

Member → part: **Part 1 — Nicolò Colle**, **Part 2 — Marc'Antonio Lopez**, **Part 3 — Carmine Francesco Benvenuto** (swappable).

---

## Paper coverage map (every section touched)

| Paper (`main_extended.tex`) | Slide |
|---|---|
| Abstract + §1 Introduction | 01, 02 |
| §2 Related work (CBM, TCAV, SAE, SPLiCE, BiomedCLIP, MedConcept, RadLex) | 03 |
| §3 Research gaps (1→6, in order) | 04 |
| §4.1 Backbone / vocabulary / split | 05, 06 |
| §4.2 Baseline Top-K SAE | 07, 09 |
| §4.3 Path A (768-d) + Tab.3 ablation | 11 |
| §4.4 Path B (SPLiCE) | 13 |
| §4.5 Concept organisation + Tab.4 | 14 |
| §4.6 Evaluation metrics | 08 |
| §5.1 Baseline non-identifiability | 09 |
| §5.2 Relabeling control | 10 |
| §5.3 Path A + data scale + why 2D differs from 3D | 11, 12 |
| §5.4 SPLiCE results | 13 |
| §5.5 Organisation results | 14 |
| §5.6 Faithfulness + upper tail stats | 15 |
| §5.7 LLM-judge + Path A hidden + random-k | 16 |
| §6 Conclusions (3 lessons, 4 contributions) | 17 |
| §6 Limitations + future work (3 limitations + PadChest) | 18 |

---

# PART 1 — Nicolò Colle · Problem, background, setup  (slides 01–06, ~4.5 min)

## 01 · Cover
> Good morning. Our project re-implements the **MedConcept** pipeline for discovering interpretable
> concepts inside a medical vision–language model, **BiomedCLIP**, on two scales — IU X-Ray, about
> 7 thousand frontal and lateral chest X-rays, and ROCOv2, about 80 thousand radiology images, ten times
> larger.
>
> We adapted this paradigm from the original MedConcept paper, which targets **3D volumetric**
> abdominal CT with the Merlin foundation model — a compute-heavy setting. Our 2D adaptation trades
> some resolution for accessibility: it runs on a single consumer GPU.
>
> The central finding is **methodological and negative**. We show that sparse autoencoder concepts on the
> pooled CLS embedding are genuinely **non-identifiable across seeds**, and that neither a deeper
> representation nor a ten-times-larger corpus lifts identifiability. The binding constraint is the
> representation site — the price of decomposing a 2D compressed, contrastive-shaped global vector.
>
> We divided the work equally: **three members, three parts, six slides each**. Part 1 sets up the problem
> and data. Part 2 explains the method and the core non-identifiability result. Part 3 covers
> deterministic alternatives, evaluation, and conclusions.
>
> **Timing:** ~50 s

## 02 · Problem
> Why does this matter? Medical VLMs are accurate but opaque. They perform well on pathology
> classification, segmentation, and report generation — but their internal representations are
> high-dimensional, polysemic, and unreadable.
>
> In safety-critical clinical use, that is a serious concern. A model that is right for the wrong reasons
> is as dangerous as one that is simply wrong. Classical post-hoc tools like saliency maps and attention
> weights show *where* the model looks, but not *what* it knows in human-readable form.
>
> Concept-based explainability reframes explanations in terms of human-interpretable units — concepts
> like "cardiomegaly" or "pneumothorax" rather than pixel-level heatmaps.
>
> **Timing:** ~45 s

## 03 · Approach
> There are two main families of concept-based XAI. **Supervised approaches** like Concept Bottleneck
> Models and TCAV need a pre-defined, labelled concept set — exactly the limitation unsupervised discovery
> targets.
>
> **Sparse Autoencoders** decompose a model's activations into a sparse overcomplete dictionary.
> Bricken et al. showed this yields monosemantic latents. Gao et al. scaled this with the Top-K SAE,
> which sets the sparsity level exactly — we adopt this architecture.
>
> **MedConcept** instantiates the full pipeline: SAE extraction, alignment to a radiology vocabulary,
> and LLM-judge evaluation with Aligned/Unaligned/Uncertain metrics. This is the template we
> re-implement and stress-test rigorously.
>
> **Timing:** ~50 s

## 04 · Research gaps
> From the literature and the project brief we identify **six gaps** our work addresses, in the paper's
> order.
>
> **Gap 1: Instability / non-identifiability.** Sparse factorisations are not reproducible across seeds,
> yet no prior medical-VLM work reports seed-stability against a null.
>
> **Gap 2: Clinical validity.** A cosine-assigned name is cosmetic unless it tracks real pathology.
>
> **Gap 3: Representation-location mismatch.** The SAE literature operates on raw hidden states,
> but reference pipelines fit on the already-projected, gap-bearing CLIP space.
>
> **Gap 4: Small-data robustness.** SAEs need 10⁵–10⁶ activations; medical corpora are 2–3 orders
> smaller, yielding dead features.
>
> **Gap 5: Flat concepts.** Explanations are top-k lists that ignore anatomical or hierarchical structure.
>
> **Gap 6: Non-reproducible evaluation.** Single-metric claims never report a chance floor; slot-wise
> overlap is degenerate under per-seed permutation.
>
> **Timing:** ~50 s

## 05 · Setup
> Let's talk about our backbone and vocabulary. We use **BiomedCLIP** — a domain-specific contrastive
> VLM tuned on biomedical text-image pairs. Its image encoder exposes a 768-d hidden state that a frozen
> projection compresses into the 512-d shared contrastive space we actually decompose.
>
> We name features against a curated **RadLex** radiology vocabulary — filtered to chest terms for IU
> X-Ray, and MeSH for ROCOv2's broader multimodal domain.
>
> Crucially, we correct the **modality gap** — the systematic offset between the image and text cones —
> before any naming or decomposition. This correction is decisive: it lifts the maximum atom–image
> cosine from 0.49 to 0.85.
>
> **Timing:** ~45 s

## 06 · Data
> We use two corpora at different scales.
>
> **IU X-Ray** is the primary, small-scale setting — about 7,470 frontal and lateral chest X-rays across
> 3,852 radiographic studies. We split **by radiographic study**, so no study appears in both train and
> test — verified zero overlap, recomputed deterministically every run.
>
> **ROCOv2** is a ten-times-larger replication — about 80 thousand radiology images. It exists to test
> one hypothesis head-on: does more data cure non-identifiability?
>
> Spoiler: it does not.
>
> **Timing:** ~40 s

---

# PART 2 — Marc'Antonio Lopez · Method + non-identifiability  (slides 07–12, ~4.5 min)

## 07 · Method
> We train **three decompositions** with one SAE configuration.
>
> **01 — Baseline (512-d).** Faithful MedConcept re-implementation on the projected space. Trained
> across five seeds — that is what makes a stability metric possible. Addresses Gap 1.
>
> **02 — Path A (768-d).** We extract the CLS hidden state *before* the projection and train an identical
> SAE on these raw hidden states. We bridge the two spaces with the frozen projection matrix. Tests the
> representation site. Addresses Gap 3.
>
> **03 — Path B (SPLiCE).** Decomposition over a fixed RadLex dictionary — no learning, so deterministic
> by construction. Addresses Gaps 1 and 4.
>
> One SAE config for clean comparisons: D equals 2048, k equals 32, 8 thousand steps, five seeds.
>
> **Timing:** ~50 s

## 08 · Evaluation metric
> Stability must be measured **permutation-invariantly**. For each decoder row of one seed, we take the
> best cosine match across all rows of the other seed.
>
> The null is **subspace-conditioned**. Real decoder directions concentrate in a subspace of effective
> rank about 357 to 363 — not the ambient 512. We draw null vectors inside each decoder's top effective-rank
> right-singular subspace and compare. The isotropic full-512-d ratio is a reported lower bound.
>
> The old slot-wise Jaccard is not permutation-invariant, so it is degenerate — we prove that next.
>
> **Timing:** ~45 s

## 09 · Headline result
> The baseline SAE reconstructs its inputs well — cosine 0.99 on IU, 0.97 on ROCOv2. But cross-seed
> feature identity is **absent**.
>
> Matched cosine is only 0.30 to 0.33 — just 1.67× to 1.81× the subspace-conditioned null. Almost no
> features match strongly at cosine ≥0.9 — 0% on IU, 0.3% on ROCOv2. Naming stays low — 0.40 to 0.48.
>
> This is genuine non-identifiability — decoders are full-rank, initializations are distinct, no collapse.
> It is intrinsic to decomposing a pooled, L2-normalized, contrastive-shaped global vector.
>
> The binding constraint is the **representation site**, not data volume or training bugs.
>
> **Timing:** ~45 s

## 10 · Relabeling control
> Why is the slot-wise metric degenerate? **Permutation invariance.**
>
> Take one trained ROCOv2 SAE and rename its features with a random permutation. The network is
> mathematically identical — reconstruction unchanged to 10⁻⁴ — but slot-wise Jaccard collapses to the 0.0077
> floor, indistinguishable from two genuinely different SAEs.
>
> The matched metric correctly reports identity — matched cosine 1.0, 100% at ≥0.9.
>
> Slot-wise signal is pure permutation noise — a binary identical-permutation check, not a stability metric.
> From here, every stability claim uses the matched metric.
>
> **Timing:** ~50 s

## 11 · Path A vs baseline
> Path A trains the SAE on the hidden state **before** the projection. It is genuinely healthier — far fewer
> dead features (1.7% vs 16%), better naming (0.47 vs 0.42).
>
> But the permutation-invariant verdict does **not** change. Matched cosine over null is still about 2× — almost
> no features match strongly at ≥0.9.
>
> This is **weak universality** — a shared subspace exists, but not canonical, reproducible features. Path A still
> operates on the pooled CLS token, not patch tokens.
>
> An ablation sweep (Tab.3) confirms this: as we increase overcompleteness from D=1024 to 4096, matched
> obs/null falls from 2.78× through 2.63× to 2.00×. No setting escapes weak universality.
>
> **Timing:** ~50 s

## 12 · Scale refuted
> What about data volume? Does 10× more data cure non-identifiability?
>
> Training the baseline on ROCOv2 — 80 thousand images, ten times IU X-Ray — improves training health
> dramatically (dead features 16% → 0.6%). But identifiability stays flat: matched cosine 0.299 → 0.327,
> subspace ratio 1.67× → 1.81×, fraction ≥0.9 0% → 0.3%.
>
> A ten-times corpus lifts **none** of these above the non-identifiability regime. The binding constraint is the
> representation site — the 2D compressed, contrastive-shaped global vector — not data volume.
>
> **Timing:** ~40 s

---

# PART 3 — Carmine Francesco Benvenuto · Alternatives, evaluation, conclusions  (slides 13–18, ~4.5 min)

## 13 · SPLiCE
> SPLiCE decomposes each embedding into a sparse, non-negative combination over the **fixed RadLex**
> dictionary. The dictionary is never estimated, so the result is deterministic by construction — zero
> cross-seed instability.
>
> It scales to both corpora. On IU X-Ray it covers all 1,515 test images in 9.5 seconds using 997 of 1,031
> RadLex terms. On ROCOv2 it covers all 15,958 images in 101 seconds using all 1,024 MeSH terms.
>
> Gap correction raises the maximum atom–image cosine sharply — from 0.49 to 0.85 on IU, 0.54 to 0.86
> on ROCOv2 — confirming the gap, not the solver, was binding.
>
> Frequent terms are "mixed" — a property of the supplied dictionaries, not SPLiCE's method.
>
> **Timing:** ~50 s

## 14 · Organisation
> Addressing Gap 5, we cluster the active vocabulary into **concept families** using RadLex ancestors
> with two degeneracy guards: leaf-root rejection, and a subtree-size canopy that rejects overly generic
> ancestors like "anatomical entity."
>
> SPLiCE, the densest, cuts concepts-per-image by about 1.75× — from 14.18 to 8.09. But silhouette is
> weak — about 0.02 on SPLiCE, 0.094 on the SAE baseline, 0.284 on Path A hidden (but 1,260 of 1,515
> images empty).
>
> The noisy IU vocabulary makes every cluster inherit a noisy ancestor. ROCOv2 SPLiCE replicates
> non-degenerately (1,024 concepts, 32 families, silhouette 0.066) — density scales with corpus size,
> separation stays weak.
>
> This stage is standalone — it does not feed the judge.
>
> **Timing:** ~45 s

## 15 · Faithfulness
> Do **any** features track real pathology? We correlate activations against ground-truth MeSH/Problems
> labels from IU X-Ray, with a shuffle null.
>
> A small upper tail is genuinely faithful. About **17.8% ± 0.9%** of live features exceed the 95th percentile
> of the shuffle null. Only about **1.1%** have correlation |r| greater than 0.30. The strongest feature
> has |r| = 0.459 ± 0.057 — it looks like a "mass", clinically plausible.
>
> But those same features stay **unstable across seeds**. Zero features cluster across seeds at cosine ≥0.90,
> not above a seed-tag shuffle null (p = 0.17). Partially faithful, entirely non-reproducible.
>
> **Timing:** ~45 s

## 16 · LLM judge
> The judge scores each concept against the reference report as Aligned, Unaligned, or Uncertain.
>
> SPLiCE on IU X-Ray scores **81.6% Aligned** under MedGemma-4B — a medical-tuned model — but only **3.3%**
> under Llama-3.1-8B, a general model. Path A hidden source scores 76.3% vs 0.5%. The SAE baseline on
> ROCOv2 scores 88.3% vs 23.1%.
>
> Random-k null explanations score comparably: 79.6% under MedGemma, 2.9% under Llama.
>
> This shows **%Aligned is a property of the judging model** as much as of the explanation. And this local
> plausibility is orthogonal to the global identifiability verdict.
>
> **Timing:** ~50 s

## 17 · Conclusions
> Three lessons that generalize.
>
> **One:** Interpretability claims about discovered concepts must be calibrated against analytical nulls
> **and** a permutation-invariant statistic. Slot-wise overlap is degenerate.
>
> **Two:** For a factorisation with limited identifiability, data scale is the least likely culprit. Ten times more data improved
> training health, not identifiability.
>
> **Three:** An honest, partially-negative evaluation is more useful than a curated one. The defect is
> localised to the representation site, not global.
>
> SPLiCE is the sole deterministic, judge-ready alternative. A small upper tail of features is genuinely
> faithful. Organisation is weakly separated.
>
> **Timing:** ~45 s

## 18 · Future work
> The diagnosis is predictive. The next experiment is an SAE on the BiomedCLIP **patch-token residual
> stream** — a mid-layer representation, not the pooled CLS token. This should restore identifiability.
>
> Three limitations from the extended paper:
>
> **(i)** The LLM-judge returns model-dependent verdicts, orthogonal to identifiability. Random-k null
> explanations score comparably, indicating weak discriminability.
>
> **(ii)** The predicted mitigation is the patch-token SAE — testing whether the 2D limit is intrinsic to the
> representation site.
>
> **(iii)** Our 10× scale test also broadens the domain. The clean test is PadChest — about 160 thousand
> chest X-rays, same IU X-Ray domain — begun but unfinished.
>
> Thank you. Questions?
>
> **Timing:** ~45 s

---

**Total timing:** ~13.5–14 minutes (under 15).
