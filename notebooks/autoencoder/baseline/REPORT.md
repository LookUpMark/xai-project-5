# REPORT — Autoencoder notebook run (`pipeline.ipynb`)

**Run date:** 2026-06-21
**Machine:** Linux / NVIDIA RTX 5070 Laptop, **CUDA device** (auto-detected)
**Input:** BiomedCLIP embeddings (512-d) of IU X-Ray — `train_embeddings.pt` (5976), `test_embeddings.pt` (1494), RadLex vocabulary **508 terms** (`text_vocab_embeddings.pt` + `data/vocabulary.json`)
**SAE config:** Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4), `batch_size=256`, **5 seeds** = `(0, 42, 123, 456, 789)`, `primary_seed=42`

**Companion:** `../ablation/REPORT.md` extends this run with the ablation program (00–05). The cross-seed instability observed here (Jaccard 0.0039) is the central question of that program; its interpretation as the "chance floor" is established by Ablation 03, and the clinical faithfulness of the concepts by Ablation 05.

**Table of contents**
- [Executive summary](#executive-summary)
- [Glossary](#glossary)
- [Metrics: formal definitions](#metrics-formal-definitions)
- [1. What each stage produced](#1-what-each-stage-produced)
- [2. Results in detail](#2-results-in-detail)
- [3. Overall assessment](#3-overall-assessment)
- [4. Next directions (already covered by the ablations)](#4-next-directions-already-covered-by-the-ablations)
- [5. Reproducibility notes & status](#5-reproducibility-notes--status)

---

## Executive summary

The pipeline transforms radiographs into interpretable concepts. BiomedCLIP converts each image into an opaque 512-dimensional vector; the **Sparse Autoencoder (SAE)** decomposes each vector into `k=32` features (concepts) drawn from a dictionary of 4096. The goal is to replace an illegible dense vector with a short list of medical concepts (e.g. "endotracheal tube", "vertebral anatomy").

Three evaluation axes:

| Axis | Result |
|---|---|
| **Does it reconstruct well?** (technical quality) | ✅ Cosine 0.988 with only 32 concepts out of 4096 — nearly perfect |
| **Do the concepts make medical sense?** (interpretability) | ✅ After the modality gap fix: RadLex alignment mean 0.40 / max 0.55 (was 0.12 / 0.29) |
| **Are they reproducible?** (robustness) | ❌ 5 runs discover almost completely different concepts (Jaccard 0.004). A structural limitation, not a bug |

**The novelty of this run — modality gap corrected (Solution 1).** BiomedCLIP has a modality gap: the space of visual vectors and the space of text vectors do not coincide; they are "shifted" relative to each other (img↔text cosine ~0.27 vs intra-modal ~0.79). Without correction, decoder columns (visual space) were being compared with vocabulary embeddings (text space) — like measuring the distance between two cities on maps with offset coordinates. The correction `W_dec -= (visual_centroid − text_centroid)` realigns the maps before the comparison. **Result: naming score from mean 0.117 / max 0.29 to mean 0.395 / max 0.55** (~3.4× on average).

Historical note: the previous models were **100-step toy models** (dead 97.6%, cosine 0.14), later replaced with **5 real 50k-step models**. The 5 current SAEs are valid: the modality gap correction does not require retraining (it is a local shift on `W_dec` inside `name_concepts`, it does not touch the saved weights). Reconstruction and stability are identical pre/post fix.

**On the instability (Jaccard 0.004).** This report documents it as the main limitation but does not resolve it: it is material for the subsequent ablations. The key result, established in `../ablation/REPORT.md` (Ablation 03), is that 0.0039 **is not an SAE failure** but the mathematical chance floor — two large, independent dictionaries always overlap by ~0.004 by pure probability (Random@4096 = 0.0037 ≈ SAE). And Ablation 05 shows that the concepts, although unstable, are moderately **faithful** to real clinical labels (~10% of live features above the null). Instability does not equate to uselessness.

---

## Glossary

- **SAE (Sparse Autoencoder)** — reconstructs an embedding `x` as `x̂ = W_dec·z + b_dec`, where `z` is a **sparse** code (few non-zero entries). TopK forces exactly `k` non-zero entries. The columns of `W_dec` are the concept "directions".
- **Cosine reconstruction** — `cos(x, x̂)`: how parallel the reconstruction is to the original. 0.988 = minimal loss.
- **L0** — number of active (non-zero) features per image. Here exactly `k=32` by construction (TopK).
- **Dead feature** — two diverging definitions (see Metrics):
  - *naming dead* = decoder column with zero norm. **0** here (the library normalizes each column at every step).
  - *activation dead* = feature never active on the test set. **~44%** here (dictionary oversized for ~7400 images).
- **Cross-seed Jaccard** — overlap between the active index sets of two SAEs: `|A∩B|/|A∪B|`. 0.004 = almost completely different concepts across seeds. Sensitive to index permutation (Ablation 00 verifies it in the direction space).
- **Concept naming** — for each feature, the most similar RadLex term (cosine between the feature direction and the term embedding). High score = concept anchored to a real term.
- **Modality gap** — systematic geometric offset between image space and text space in contrastive models. Corrected post-hoc with `W_dec -= (visual_centroid − text_centroid)`. Full analysis in `docs/suggestions/concept_naming_analysis.md`.
- **Explanations (pseudo-report)** — for each test image, its active top-k concepts assembled into a textual description. It is the input that the LLM judge (MedGemma) will evaluate.

---

## Metrics: formal definitions

**Reconstruction (TopK SAE).** `x̂ = W_dec·z + b_dec`, with `z = topk(ReLU(W_enc·(x − b_enc) + b_enc), k)`. TopK zeros out all but the top `k` values.
- Cosine: `cos(x, x̂) = ⟨x, x̂⟩/(‖x‖·‖x̂‖)`.
- Variance Explained: `VE = 1 − ‖x − x̂‖²/‖x − b_dec‖²` (~99.3%).
- L0: `‖z‖₀ = k = 32` (rigid, guaranteed by TopK).
- Entropy: `H(p) = −Σᵢ pᵢ log pᵢ` over the feature usage distribution on the test set (~6.3 nats → ~540 features used).

**Dead feature.**
- naming dead: `‖w_i‖ ≈ 0` (0 here, unit-norm columns post-training).
- activation dead: `∀ s : zᵢ(s) = 0` on the test set (~44% here).

**Cross-seed Jaccard.** `J(A,B) = |A∩B|/|A∪B|` where `A,B` = active index sets of two SAEs on a sample. Mean over the 10 seed pairs. Analytical null `k/(2D−k)` = 0.0039 at D=4096, k=32 → ratio ~1 (on the chance floor).

**Modality gap (naming gap-corrected).** `gap = mean(train_emb, 0) − mean(vocab_emb, 0)`; `W_dec ← W_dec − gap`, then `F.normalize` rows + cosine with `F.normalize(vocab_emb)`. Corresponds to *Solution 1* of `docs/suggestions/concept_naming_analysis.md`.

---

## 1. What each stage produced

| Stage | Output | Status |
|---|---|---|
| Train/test split | `train/test_embeddings.pt` | ⏭️ skipped (already current) |
| **SAE training** | `models/sae_seed{0,42,123,456,789}/` (50k step) | ✅ 5 models (reused, 06-05) |
| Modality gap | `models/modality_gap.pt` (512-d) | ✅ pre-computed, reused |
| Loss curve | `figures/loss_curve.png` + 50 checkpoint | ✅ regenerated (17:41) |
| Concept naming | `results/concept_names.json` (4096 feature) | ✅ gap-corrected |
| Explanations | `results/sample_explanations.json` (1494 record) | ✅ ⚠️ see image_id |
| Stability | `results/stability_analysis.json` (Jaccard 5×5 + per-seed) | ✅ |
| Figure | `concept_scores_dist`, `per_seed_metrics`, `jaccard_heatmap`, `sparsity_summary`, `concept_activations_heatmap`, `loss_curve` | ✅ regenerated |

---

## 2. Results in detail

### 2.1 Reconstruction quality — ✅ EXCELLENT

Reconstruction measures whether, by recombining the 32 active concepts for an image, the original vector is obtained again. A cosine of 0.988 means the reconstructed vector is almost parallel to the original: the decomposition loses very little. This metric is **gap-independent** (it uses the encoder path, it does not touch `W_dec`).

| Seed | MSE (raw) | Cosine | Dead % | Dict util % | Entropy |
|------|-----------|--------|--------|-------------|---------|
| 0 | 4.6e-5 | 0.9882 | 41.5 | 58.5 | 6.3955 |
| 42 | 4.4e-5 | **0.9888** | 44.3 | 55.7 | 6.3219 |
| 123 | 4.6e-5 | 0.9882 | 44.9 | 55.1 | 6.3660 |
| 456 | 4.4e-5 | 0.9887 | 45.7 | 54.3 | 6.3529 |
| 789 | 4.5e-5 | 0.9886 | 43.0 | 57.0 | 6.3258 |

- **Mean cosine ~0.988** → reconstructed vector almost parallel to the original.
- **L0 = 32.0** across all seeds = exactly `k`: the Top-K constraint is perfectly respected (each image uses exactly 32 concepts, never one more).
- The 5 seeds are mutually consistent (MSE 4.4–4.6e-5, cosine 0.9882–0.9888) → reproducible at the *quality* level.

*(The "loss curve" cell reports MSE on a normalized activation scale, not comparable with the raw MSE; the correct convergence signal is cosine ~0.988 + plateau of the curve.)*

### 2.2 Sparsity — ✅ CORRECT

Sparsity is the point of an SAE: a dense vector (all 512 numbers) is illegible, 32 concepts are interpretable. L0=32 and entropy ~6.3 confirm that sparsity works as intended.

- **Mean L0 = 32.0** = `k`. Top-K constraint respected.
- **Entropy ~6.3 nats** → activations are spread across ~`e^6.3 ≈ 540` distinct features in the test set (widespread usage, not concentrated on a few features).

### 2.3 Dead features — ⚠️ MODERATE

"Dead" features never activate, on any image: they are waste. ~44% means that out of 4096 features, ~1800 are unused. This happens because the dictionary (4096) is oversized relative to the number of images (~7400). Expected for small datasets — not fatal, but it is waste.

- **~41–46% of the 4096 features never activate** on the test set (dict utilization ~54–59%).
- ~1800 "dead" features → dictionary (4096) oversized for ~7400 images.
- ⚠️ **Two diverging definitions of "dead"** (as per CLAUDE.md) — important not to confuse them:
  - *Naming dead* (zero-norm decoder, in `concept_names.json`): **0** (none — the `dictionary_learning` library normalizes each column to unit-norm at every step, so zero-norm columns do not exist post-training).
  - *Activation dead* (never active on the test set, in stability): **~44%**.

### 2.4 Cross-seed stability — ❌ THE MAIN CRITICAL ISSUE

This is the most important metric for the trustworthiness of the concepts. By training the same model 5 times (changing only the seed, i.e. the random starting point), do the 5 results find the same concepts? The answer is almost not at all: they share <0.4% of the active features. In practice, the extracted "concepts" depend heavily on which of the 5 runs is chosen. The primary seed 42 is arbitrary; changing it, naming and explanations change. This is the structural limitation of the project, openly declared.

**Mean Jaccard = 0.0039** (5×5 matrix, off-diagonal ~0.002–0.009). **Gap-independent.**

| | 0 | 42 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.004 | 0.009 | 0.003 | 0.003 |
| 42 | 0.004 | 1.00 | 0.004 | 0.003 | 0.004 |
| 123 | 0.009 | 0.004 | 1.00 | 0.003 | 0.002 |
| 456 | 0.003 | 0.003 | 0.003 | 1.00 | 0.003 |
| 789 | 0.003 | 0.004 | 0.002 | 0.003 | 1.00 |

- The 5 SAEs reconstruct equally well but with almost completely different features. They share <0.4% of the active features.
- → The discovered "concepts" are not robust/reproducible: they depend heavily on the seed.
- This is the open problem "poor concept robustness" that the project explicitly cites — an expected but significant result to discuss.
- Ablations 00–05 investigate whether this 0.004 is a real failure or the mathematical "chance floor". **Spoiler: it is on the chance floor** (Ablation 03: Random@4096 = 0.0037 ≈ SAE), and the existing concepts are nonetheless clinically faithful (Ablation 05). See `../ablation/REPORT.md`.

### 2.5 Concept naming — ✅ IMPROVED (gap-corrected)

Naming assigns each feature the most similar RadLex medical term (cosine between the feature direction and the term embedding). High score = concept anchored to a real term → interpretable. Before the fix it was ~0.12 (very weak, names were almost random); after the fix ~0.40 average with peaks at 0.55. The top concepts are clinically plausible: endocavitary tubes/devices, vertebral anatomy, spinal pathways.

Headline of this run: the **modality gap** correction solved the main limitation of naming.

| Metric | Pre-fix (stale) | **Post-fix (this run)** | Δ |
|---|---|---|---|
| Mean score | 0.117 | **0.3949** | ×3.4 |
| Max score | 0.291 | **0.5457** | ×1.9 |
| Min score | −0.063 | **0.2815** | — |

- 4096 features named, **0 marked `DEAD_FEATURE`**.
- **Mean score 0.395, max 0.55** → decoder↔vocabulary alignment is now solid (before, 0.29 was weakly-grounded). The correction `W_dec -= gap` brings the decoder columns into the text space before the cosine.
- **Top-8 by score (clinically plausible and now strongly anchored):**

  | Feat | Name | Score |
  |---|---|---|
  | 690 | cricothyroid tube | 0.5457 |
  | 1090 | ligamentum flavum | 0.5230 |
  | 1806 | moderate central spinal stenosis | 0.5192 |
  | 3824 | sacral segment of spinal epidural space | 0.5172 |
  | 1239 | right spinotectal tract of spinal cord | 0.5159 |
  | 2059 | endotracheal tube | 0.5117 |
  | 2977 | Foramina vertebralia | 0.5114 |
  | 1172 | brachytherapy catheter | 0.5114 |

- The top concepts are mutually coherent (endocavitary tubes/devices, vertebral anatomy, spinal pathways) → the dictionary captures real patterns of the embedding space.
- **How it is implemented:** `train_sae.compute_and_save_modality_gap()` precomputes `gap = train_emb.mean(0) − vocab_emb.mean(0)` → `models/modality_gap.pt`; `sae_module.name_concepts(..., modality_gap=gap)` does `W_dec = W_dec − gap` then `F.normalize` + cosine. Corresponds to *Solution 1* of `docs/suggestions/concept_naming_analysis.md`.
- **Collateral fix of this run:** `name_concepts` now coerces the label dict → string (`_vocab_term`), so consumers that load `vocabulary.json` as a list of dicts (baseline notebook) still get a string `name` instead of crashing `generate_explanation`.
- **Cross-check with faithfulness (Ablation 05):** RadLex naming is *off-distribution* and sometimes noisy. Ablation 05 shows that a feature can be faithful to "implanted medical device" while carrying the RadLex name "anterior segment of upper lobe" — the behavior is more informative than the name. See `../ablation/REPORT.md` §Ablation 05.

### 2.6 Explanations — ✅ STRUCTURALLY CORRECT, ✅ image_id RESTORED

For each test image, its active top-k concepts are taken and assembled into a pseudo-report (a textual description generated from the concepts). It is the final output that the LLM judge (MedGemma) will evaluate: "is this pseudo-report aligned with the real clinical report of the image?". Now ready: 1494 records, correct schema, real image_id (no more placeholders), joined reports.

- **1494 records** (one per test image). Judge contract schema verified: `{image_id, top_k_concepts[].{feature_id,name,activation}, pseudo_report}`.
- Activations: mean 0.1423, max 0.3865 (consistent with the gap-corrected naming).
- **`image_id`: 1494/1494 real basenames** (e.g. `3222_IM-1522-2001.dcm.png`) — sidecar `embeddings/{visual,train,test}_image_ids.json` rebuilt (see §5). Previously they were all fallback `sample_N` (sidecars missing from the 06-05 run); now joinable via `indiana_projections.csv` (filename→uid) → `indiana_reports.csv` (uid→findings).
  - ✅ **Judge-ready:** `data/iu_xray/reports.csv` generated (7466 rows, columns `image_id`+`combined_text`, join filename→uid→findings). End-to-end lookup verified: **1493/1494** test image_id have a non-empty report (1 orphan PNG with no entry in `indiana_projections.csv` → the judge skips it).
- Example (`3222_IM-1522-2001.dcm.png`): `intervertebral foramen`, `progressive massive fibrosis`, `left coronary artery`, `ligamentum flavum`… — template-based pseudo-report.

---

## 3. Overall assessment

A sensible result, and clearly improved over the pre-fix run for naming.

| Objective | Outcome |
|---|---|
| Does the SAE learn to decompose the embeddings? | ✅ Yes, reconstructs at cosine ~0.988 with k=32 |
| Are the concepts *sparse and monosemantic*? | ✅ Sparse (L0=32); monosemanticity is now more plausible (naming mean 0.395) |
| Are the concepts *robust*? | ❌ No — Jaccard 0.004, they depend on the seed (not resolved by the gap fix; but it is the chance floor, see Ablation 03) |
| Are the concepts *clinically anchored*? | ✅ Improved — RadLex alignment from max 0.29 to max 0.55 (mean 0.117→0.395) |
| Does the pipeline produce judge-ready output? | ✅ Correct schema + real image_id + reports.csv generated (1493/1494 covered) |

Key points for discussion:
1. The **modality gap was the culprit** behind weak naming: corrected, +3.4× on average.
2. The SAE *works technically* but concept discovery is not stable cross-seed (Jaccard 0.004) — material for "failure cases / limitations", not a bug. The ablation program has since clarified that it is the chance floor (03) and that the existing concepts are clinically faithful (05).
3. **Operational blocker:** restore the image-id sidecars before relaunching the judge.

### 3.1 Relationship to MedConcept (declared deviations)

The pipeline is a **MedConcept-inspired** instance, not a faithful replica (Haque et al., arXiv:2604.11868). The skeleton is faithful — SAE on a medical VLM → sparse activations → grounding in a clinical vocabulary → pseudo-report → Aligned/Unaligned/Uncertain LLM judge — but with three material deviations on the method, declared for honesty:

- **D1 — TopK SAE instead of ReLU+L1.** MedConcept (Eq. 2) imposes sparsity with an L1 penalty (λ₁=2e-3); the pipeline uses TopK (hard top-k selection in the encoder, no L1, auxk for the dead). Different sparsity mechanism (fixed k vs data-dependent), same idea of sparse decomposition.
- **D2 — `dict_size` decoupled from the vocabulary.** MedConcept ties `k = |vocabulary|` (one neuron = one concept); the pipeline uses `dict_size=4096` against a 508-term RadLex vocabulary → many-to-one naming, ~44% of neurons never active. MedConcept's 1:1 constraint is absent.
- **D3 — modality gap correction.** MedConcept (Eq. 3) uses pure cosine and accepts the vision-text gap as a limitation; the pipeline subtracts a modality gap vector (visual_centroid − text_centroid) from the decoder rows before the cosine (Mind the Gap, Liang et al.) — an extra step that raises the naming mean from 0.117 to 0.395.

> **Reporting nuance:** the "mean naming score" (0.395) is averaged over all 4096 features, including the ~44% *dead-by-activation* (never active on the test set), which receive poorly meaningful labels and drag down the average. It does not corrupt the explanations (the judge filters `activation>0`) but should be declared when citing the mean score.

---

## 4. Next directions (already covered by the ablations)

The improvement hypotheses listed below have been **verified by the ablation program** (`../ablation/REPORT.md`). The verdicts are reported for closure.

- **Reduce `dict_size`** (4096 → 2048/1024): Ablation 01 → reduces the dead (40.9 → 30.7%) but does **NOT** increase cross-seed stability (non-monotonic ratio).
- **Cross-seed aggregation / consensus**: Ablation 00 → consensus in the direction space is ~0 (no shared direction across ≥4 seeds out of 5); aggregation requires validation on much lower τ values.
- **Lower LR (`5e-5`)**: remains untested as a single lever; the ablations use a pinned lr to control confounds.
- **Variation of `k`**: Ablation 02 → weak sweet spot at k=16 (ratio 1.30, the only one above the null), but absolute agreement remains tiny; k=32 (baseline) is on the chance floor.
- **Alternative activation family**: Ablation 04 → BatchTopK reduces the dead (4.8%) but consensus 0 for all three families (TopK/BatchTopK/JumpReLU).
- **Naming beyond cosine (SPLiCE)** and **larger/curated vocab**: future work (`docs/suggestions/VOCAB_BUILDING_ALTERNATIVES.md`).
- **Qualitative validation / faithfulness**: Ablation 05 → ~10% of live features are faithful to real clinical labels above the null (implants, effusion, emphysema).

The instability is a **structural limitation** of the method on this dataset (few samples + non-uniqueness of the sparse decomposition on projected CLIP embeddings). The full causal diagnosis is in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

---

## 5. Reproducibility notes & status

- **CUDA (RTX 5070)**: the metrics of this run (cosine 0.988, dead ~44%) are consistent with the reference MPS run (06-15) → the device change did not alter the results (cross-device reproducible).
- **Modality gap is cached** (`models/modality_gap.pt`): `compute_and_save_modality_gap()` has a skip-if-exists guard without a content check. If the embeddings are regenerated with a different split, the gap must be **deleted manually** (`rm models/modality_gap.pt`) before relaunching, otherwise it stays stale.
- **`vocabulary.json` = 508 dicts** `{"term","similarity_score","source"}` (output of the multi-centroid builder). The consumers (CLI `concept_naming.py` and notebook) normalize to a `term`-string; `name_concepts` additionally coerces via `_vocab_term` as a safety net. The old "open issue #7" is resolved.
- **Image-id sidecars rebuilt** (18:12): `visual/train/test_image_ids.json` regenerated from `sorted(glob("*.png"))` of the 7470 real PNGs (`chest-xrays-indiana-university/images/images_normalized/`) + split `sklearn(random_state=42, ratio=0.8)`; row alignment verified via `torch.equal` against the existing tensors (exact match on train + test). `sample_explanations.json` now has 1494/1494 real basenames. No re-extraction nor retraining.
- **`data/iu_xray/reports.csv` generated**: join `indiana_projections`(filename→uid) + `indiana_reports`(uid→findings+impression), columns `image_id`+`combined_text` (schema required by `evaluate_llm_judge.py`: `zip(image_id, combined_text)`). Judge lookup verified: **1493/1494** test covered (1 orphan PNG with no entry in projections). The judge is now executable end-to-end.
- **5 SAEs not retrained**: the modality gap correction is a local shift on `W_dec` in `name_concepts`, not persisted in the weights. The 06-05 models remain valid.
