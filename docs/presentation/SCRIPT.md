# Speaker Script

**Project:** *Unsupervised Concept Discovery for Medical Vision–Language Models:
A Rigorous Characterization of Sparse-Autoencoder Failure and Deterministic Alternatives*
**Course:** Explainable & Trustworthy AI — Politecnico di Torino
**Authors / members:** Marc'Antonio Lopez · Nicolò Colle · Carmine Francesco Benvenuto

---

## How to use this script

- 15 slides = 3 parts × 5, one member per part. **~50–55 s per slide → ~14 min total** (under 15).
- The deck shows the essentials; **this script is what you actually say** (more detail, plain English).
- Numbers are real — from `docs/latex/main.tex`, cross-checked on `results/*.json`. Don't paraphrase them loosely; if you round, say "about".
- Each block: **slide label**, the spoken text, and the key numbers to land. Press **`N`** during the deck to see on-slide notes too.

Member → part assignment: **Part 1 — Nicolò Colle**, **Part 2 — Marc'Antonio Lopez**, **Part 3 — Carmine Francesco Benvenuto**. Swap if you prefer; content stays the same.

---

# PART 1 — Nicolò Colle · Problem, background, setup  (slides 1–5, ~4.5 min)

## 01 · Cover
> Good morning. Our project re-implements the **MedConcept** pipeline for discovering interpretable
> concepts inside a medical vision–language model, **BiomedCLIP**, and tests it on two scales —
> IU X-Ray, about 7 thousand images, and ROCOv2, about 80 thousand, a ten-times scale-up.
> The punchline up front: we did **not** manage to discover clean, reproducible concepts. Instead we
> produced a **rigorous characterisation of why** sparse-autoencoder discovery fails — and what a
> deterministic alternative can still do. I'm [name], and with [name] and [name] we'll walk you through
> it in three equal parts.

*(~45 s. Names of the three members.)*

## 02 · Problem — Medical VLMs are accurate but opaque
> Start with the problem. Medical vision–language models are now genuinely good — at pathology
> classification, segmentation, even report generation. But their internal representations are
> **high-dimensional, polysemic, and unreadable**. In a clinical setting that is a real safety concern.
> And the tools people usually reach for — saliency maps, attention — tell you **where** the model is
> looking, but almost never resolve into **semantic structure** you can audit. That gap is what
> concept-based explainability tries to close.

*(~40 s. One idea: accuracy ≠ understanding.)*

## 03 · Approach — from supervised to unsupervised discovery
> Concept-based XAI comes in two flavours. **Supervised** methods — Concept Bottleneck Models and TCAV —
> need a pre-defined, labelled concept set. That labelling is exactly the bottleneck. So recent work
> tries **unsupervised** discovery: decompose a frozen model's representations into sparse features,
> name each feature against a clinical vocabulary, and validate the result. **MedConcept** packages that
> into three stages — sparse extraction, vocabulary alignment, and an LLM-judge. We re-implement that
> pipeline **faithfully**, and then stress-test it. On the right, sparse autoencoders: Bricken showed
> they find monosemantic latents; we use the **Top-K** variant from Gao, which fixes sparsity exactly.
> And **SPLiCE**, which we'll come back to in Part 3, is a deterministic third way.

*(~60 s. Three cards left→right: supervised, SAE, MedConcept.)*

## 04 · Setup — BiomedCLIP + RadLex
> Two ingredients. The **backbone** is **BiomedCLIP** — a ViT-B/16 image encoder paired with PubMedBERT,
> pretrained on PMC-15M. Crucially, its image encoder exposes a **768-dimensional hidden state**, which
> a frozen projection layer then compresses into a **512-dimensional** shared contrastive space. That
> 768-to-512 detail matters a lot in Part 2. The **vocabulary** is **RadLex** — about a thousand
> radiology terms, ranked by relevance using 39 anchor queries across 13 sub-domains, plus 14 seeds from
> ChestX-ray14. And there is a subtlety we have to handle: the **modality gap** — images and text live in
> slightly offset cones. We estimate that offset and subtract it before naming or decomposition. Keep that
> in mind; in Part 3 correcting it turns out to be the decisive fix.

*(~70 s. Land "768 → 512" and "modality gap".)*

## 05 · Data — IU X-Ray vs ROCOv2
> Two datasets, deliberately at different scales. **IU X-Ray** is our primary, small-scale setting:
> about 7,500 images across roughly 3,850 studies. We split it **by study** — the key is
> patient-and-study — so no study leaks between train and test. That gives 5,955 train and 1,515 test
> images, with verified **zero** group overlap, recomputed deterministically every run. **ROCOv2** is
> roughly 80,000 images — a **ten-times** replication — and we split it by image, since it has no shared
> patient structure. The point of ROCOv2 is not "more data is better"; it is a direct test of the
> **scale hypothesis**. That's the end of Part 1 — over to Marc'Antonio Lopez for the method and the core result.

*(~70 s. Hand off to Marc'Antonio Lopez.)*

---

# PART 2 — Marc'Antonio Lopez · Method + non-identifiability  (slides 6–10, ~4.5 min)

## 06 · Method — three paths + an organisation extension
> Thanks. We attack the problem with **three paths plus one extension**. The **baseline** is a faithful
> re-implementation of MedConcept: a Top-K sparse autoencoder on the 512-d space — dictionary size 2048,
> k equals 32, eight thousand steps, and **five seeds**: 0, 42, 123, 456, 789. The multi-seed setup is
> what makes a stability metric possible at all. **Path A** moves the SAE onto the 768-d hidden state
> *before* the projection, to test whether the representation site is the problem. **Path B** is **SPLiCE**
> — decomposition over a *fixed* RadLex dictionary, no learning, so deterministic by construction. And
> the **organisation** extension clusters concepts into families with RadLex ancestors. One detail on
> evaluation, because it is the heart of our contribution: we measure stability with a **matched**,
> permutation-invariant statistic — best cosine match across the other seed's decoder — against a null
> conditioned on the decoder's **effective rank**, about 357 to 363, not the full 512. We'll see on the
> next slides why that matters.

*(~75 s. One SAE config; matched metric + erank null.)*

## 07 · Headline result — the SAE is non-identifiable
> Here is the headline. The baseline SAE **reconstructs** its inputs almost perfectly — cosine 0.97 to
> 0.99 — so as an autoencoder it works. But cross-seed **feature identity** is essentially absent: the
> best-match cosine is only **1.67 times** the conditioned null on IU X-Ray, and **1.81 times** on ROCOv2.
> Roughly **zero percent** of features match at cosine ≥0.9, and naming is capped at **0.40 to 0.48**,
> with nothing above 0.7. And this is not a training bug — the decoders are full-rank, the
> initialisations are distinct, there is no collapse. It is **genuine non-identifiability**: the features
> are not canonical, reproducible units.

*(~60 s. Hammer: reconstruction good ≠ features canonical.)*

## 08 · Relabeling control — the slot-wise metric is floor-by-construction
> Before that result, we had to fix the metric people were using. The standard "slot-wise cross-seed
> Jaccard" is **floor-by-construction** — it is not invariant to permutations. We prove it directly:
> take one trained SAE and just **rename** its features with a random permutation — it is mathematically
> the same network, reconstruction unchanged to 10⁻⁴ — and the slot-wise Jaccard collapses to **0.0077**,
> indistinguishable from two genuinely different SAEs. The matched metric, by contrast, correctly reports
> identity, 1.0. The figure is the cross-seed Jaccard matrix on IU X-Ray — every cell sits on the floor.
> So slot-wise is permutation noise, and from here on every stability claim uses the matched metric.

*(~65 s. This is contribution (1). Point at the matrix.)*

## 09 · Path A vs baseline — moving the site is not enough
> So is it the representation site? Path A trains the SAE on the 768-d hidden state *before* the
> projection. It is genuinely healthier — dead features drop from **16 percent to 1.7**, and naming
> improves from **0.42 to 0.47**. But the permutation-invariant verdict does **not** change: matched
> cosine over null is still around **1.97× and 2.63×**, and still **≈0%** of features match at ≥0.9. This
> is **weak universality** — a shared subspace, not identical features. And the reason: Path A still
> operates on the **pooled CLS token** — the contrastive global summary — not on the patch tokens. The
> naming-score histogram on the right shows why a cosine-assigned name is cosmetic: most features' best
> term sits well below 0.5.

*(~70 s. Path A = local win, global verdict unchanged. Point at naming histogram.)*

## 10 · Scale refuted — 10× does not help
> The other tempting explanation is just **more data**. We test it head-on with ROCOv2, a ten-times
> corpus. It works exactly where you'd expect — training health improves, dead features fall from 16% to
> 0.6%. But identifiability stays **flat**: matched cosine 0.299 → 0.327, subspace ratio 1.67× → 1.81×,
> fraction ≥0.9 zero to 0.3%, naming still 0.40–0.48. Ten times the data does not lift a single one of
> these out of the non-identifiability regime. So the binding constraint is the **representation site**:
> a sparse autoencoder on the CLS vector is factoring a single, already-compressed global vector whose
> effective rank — about 357 — is far below the 2048-way dictionary. That closes Part 2 — over to
> Carmine Francesco Benvenuto for what still works.

*(~75 s. Contribution (3). Hand off to Carmine Francesco Benvenuto.)*

---

# PART 3 — Carmine Francesco Benvenuto · Deterministic alternatives, evaluation, conclusions  (slides 11–15, ~4.5 min)

## 11 · SPLiCE — deterministic, judge-ready
> Thanks. If learned factorisation is non-identifiable, the clean fix is to **not learn the dictionary**.
> That is **SPLiCE**, our Path B. For each image we decompose the gap-corrected 512-d embedding into a
> sparse, non-negative combination over the **fixed RadLex** dictionary: Orthogonal Matching Pursuit
> picks 32 atoms, then non-negative least squares re-solves the coefficients. No estimation from data →
> **deterministic by construction**, zero cross-seed instability. It scales well — all 1,515 IU test
> images in 9.5 seconds, and all 15,958 ROCOv2 images in 101 seconds, using essentially the full
> vocabulary. And remember the modality gap from Part 1? Correcting it lifts the maximum atom–image
> cosine from **0.49 to 0.85** on IU, 0.54 to 0.86 on ROCOv2 — so the gap, not the solver, was the real
> bottleneck. Honest caveat: the most frequent terms are "mixed", not clean findings — that's a property
> of the supplied dictionary, not of SPLiCE.

*(~80 s. Land determinism + the gap-correction jump.)*

## 12 · Faithfulness — a small upper tail is genuinely faithful
> So is any of this real? We ran a **faithfulness** ablation — point-biserial correlation between
> feature activations and the real IU X-Ray MeSH/Problems labels, against a per-feature shuffle null.
> The honest answer is: a **small upper tail** is genuinely faithful. About **17.8 percent**, plus or
> minus 0.9, of live features beat the null's 95th percentile; **1.1 percent** reach a correlation above
> 0.30 — the strongest is **0.459**, on a feature that looks like *mass*, which is clinically plausible.
> But — and this matters — those same features are still **unstable cross-seed**: zero cluster across the
> five seeds at cosine ≥0.90. So concepts are simultaneously partially faithful and entirely
> non-reproducible. That is exactly the partially-negative result we want to report honestly.

*(~70 s. Two-sided: real signal, but small and unstable.)*

## 13 · LLM judge — local plausibility, model-dependent
> Finally, evaluation with an **LLM-as-judge**. We score each concept Aligned, Unaligned, or Uncertain
> against the reference report, run to convergence at temperature zero, zero parse errors. SPLiCE on IU
> X-Ray scores **81.6 percent Aligned** under **MedGemma-4B** — 1,230 out of 1,508. But under
> **Llama-3.1-8B**, a general model, it scores only **3.3 percent**, with 73.9 percent Uncertain. On
> ROCOv2 the baseline hits 88.3% vs 23.1%. The take-away: the Aligned rate is a property of the **judging
> model** as much as of the explanation. And crucially, this **local** plausibility is **orthogonal** to
> the global identifiability verdict — a feature can align with one report and still fail to recur across
> seeds. The two measure different things.

*(~70 s. Land 81.6 vs 3.3 and "orthogonal axes".)*

## 14 · Conclusions — three lessons that generalise
> To wrap up, three lessons we think generalise beyond this project. **One**: interpretability claims
> about discovered concepts must be calibrated against analytical nulls **and** a permutation-invariant
> statistic — slot-wise overlap is degenerate. **Two**: for a failing factorisation, **data scale is the
> last suspect** — ten-times changed training health, not identifiability. **Three**: an honest,
> **partially-negative** evaluation is more useful than a curated success story; the defect here is
> localised — it's the representation site — not a global failure of the idea.

*(~55 s. Three cards, one breath each.)*

## 15 · Future work + close
> And the diagnosis is actually **predictive** about what to do next. Everything points at the same fix:
> train the sparse autoencoder on the BiomedCLIP **patch-token residual stream** — a mid layer, **not**
> the pooled CLS token. That is the experiment our characterisation says should restore identifiability.
> To recap: Part 1 set up the problem, Part 2 showed the SAE is non-identifiable and why, Part 3 showed
> the deterministic alternative and the honest evaluation. Thank you — we're happy to take questions.

*(~50 s. Decisive close, then thanks + Q&A.)*

---

## Timing summary

| Part | Presenter | Slides | Target |
|---|---|---|---|
| 1 | Nicolò Colle | 01–05 | ~4.5 min |
| 2 | Marc'Antonio Lopez | 06–10 | ~4.5 min |
| 3 | Carmine Francesco Benvenuto | 11–15 | ~4.5 min |
| **Total** | | **15** | **~13.5–14 min** (+ Q&A) |

If you run long: trim the caveats on slides 11 and 12 first — they're the easiest to shorten without losing the argument.
