# REPORT — SAE Ablations on AUGMENTED data (`{00,01,02,03,04}_*.augmented_exec.ipynb`)

**Run date:** 2026-06-24 (ablations 00–02), 2026-06-25 (ablations 03–05)
**Machine:** Linux / NVIDIA RTX 5070 Laptop, **CUDA** (`torch 2.12.0+cu130`)
**Path:** `embeddings/augmented/` (`config.augmentation.enabled = True`)
**Input:** BiomedCLIP embeddings (512-d) of IU X-Ray with on-the-fly augmentation (`num_augmentations=2`, `rotation=5°`, `crop=0.95–1.0`) → `visual_embeddings.pt` (22410 = 7470 × 3). RadLex vocabulary 508 terms. MeSH/Problems clinical labels (69 prevalent terms, prevalence ≥10).
**Split:** group-aware by radiograph study (`study_key_from_basename`) — audit 06-23 F-001/F-002/F-003 fix. **17865 train / 4545 test**, **3081 / 771** of 3852 studies, **group overlap = 0** (leak-free).
**Ablations covered:** 00_consensus, 01_dict_size, 02_k_sweep, 03_baselines, 04_activation_bakeoff, 05_faithfulness — all executed clean on the augmented split.

> Companion to [`REPORT.md`](REPORT.md) (standard path, contaminated split) and to [`../baseline/REPORT_augmented.md`](../baseline/REPORT_augmented.md) (augmented baseline). All ablation models here were retrained from scratch on the **augmented** embeddings after backing the standard-trained checkpoints aside (`models/_standard_backup_ablations/`) — the notebooks' skip-if-exists guards would otherwise have silently reused the standard (different-hash) models and produced false "augmented" results.

**Table of contents**
- [Executive summary](#executive-summary)
- [Augmented baseline anchor](#augmented-baseline-anchor)
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder](#ablation-01--dictionary-size-ladder)
- [Ablation 02 — k (Sparsity) Sweep, null-calibrated](#ablation-02--k-sparsity-sweep-null-calibrated)
- [Ablation 03 — Concept Baselines (Random / PCA / KMeans)](#ablation-03--concept-baselines-random--pca--kmeans)
- [Ablation 04 — Activation Function Bakeoff (TopK / BatchTopK / JumpReLU)](#ablation-04--activation-function-bakeoff-topk--batchtopk--jumprelu)
- [Ablation 05 — Faithfulness (Clinical Label Correlation)](#ablation-05--faithfulness-clinical-label-correlation)
- [Are these results normal?](#are-these-results-normal)
- [Reproducibility notes](#reproducibility-notes)
- [6. Pending ablations](#6-pending-ablations)

---

## Executive summary

Three ablations, each probing a different axis of the SAE, all on the **leak-free augmented split**. Every one independently lands on the **analytical chance floor** for cross-seed feature overlap. The augmented baseline's master conclusion — *concept instability is structural, not a leakage or data artifact* — is triangulated across dict size, sparsity, and representation space.

| Axis (ablation) | Question | Result |
|---|---|---|
| **00 — representation** | Do seeds agree in *direction* space (cosine), not just index? | ❌ p = 0.19 vs shuffle null — no consensus in any space |
| **01 — dict size** | Does a smaller/larger dictionary stabilize features? | ❌ Jaccard tracks `k/(2D−k)` at every D (ratio 0.88–1.14) |
| **02 — sparsity** | Does k control cross-seed reproducibility? | ❌ ratio 0.75–1.19 across k = {8,16,32,64}, all ≈ floor |
| **05 — faithfulness** | Do live features correlate with clinical labels? | ⚠️ 47.1% live features beat |r|>0.10; strongest |r|=0.545 (heart failure); 42.7% beat shuffle null p95 |

**The central result.** No lever available to the experimenter — dictionary width, sparsity budget, or the space in which agreement is measured — moves the discovered concepts off the chance floor. Two random `k`-subsets of a `D`-dictionary overlap by `k/(2D−k)` by pure combinatorics; the SAE ensembles reproduce that quantity almost exactly at every operating point. This is the non-uniqueness of sparse decomposition on projected CLIP embeddings, and it is **not** a function of the data contamination that the 06-23 audit flagged — it reproduces identically on the clean augmented split.

Reconstruction quality and naming grounding remain healthy everywhere (cosine 0.984–0.995, RadLex alignment ≈ 0.385), so the SAEs are not failing — they are simply not unique. Ablation 05 adds a complementary faithfulness lens: while cross-seed stability sits at the chance floor, nearly half of live features carry statistically detectable clinical signal — suggesting the representation is semantically grounded even if the specific basis vectors are not reproducible across seeds.

---

## Augmented baseline anchor

The ablations below are read against the **augmented** baseline (`../baseline/REPORT_augmented.md`), not the standard one. (Note: the notebooks' embedded `baseline_reference` blocks hardcode the *standard* contaminated numbers — cosine 0.988, dead 44%, Jaccard 0.0038 — because those were the reference when the notebooks were authored. They are informational only; the comparison points used here are augmented.)

| Baseline anchor (augmented, dict4096, k=32, 5 seeds) | Value |
|---|---|
| Test cosine (mean) | 0.9945 |
| Dead features (activation) | ~65% |
| Cross-seed Jaccard | 0.0039 |
| Analytical null `k/(2D−k)` | 0.00392 |
| Jaccard / null ratio | 0.995 (on the floor) |
| Naming mean / max (gap-corrected) | 0.384 / 0.538 |

Ablation 02's standalone anchor (`baseline_anchor` in `a2_k_sweep.json`) records the same point independently: dict4096/k32 raw Jaccard 0.0038, exact null 0.00398, ratio **0.954** — consistent.

---

## Ablation 00 — Cross-Seed Consensus (direction-space)

**Setup.** Pool all *live* decoder rows across the 5 augmented baseline seeds (20480 rows = 4096 × 5; all pass the 1e-8 norm threshold so none are dropped), normalize, and cluster by cosine similarity. This reframes "do seeds share feature *indices*" (raw Jaccard 0.0038) as "do seeds discover the same feature *directions*" — a strictly more lenient test. If concepts were real but basis-rotated, this is where consensus would appear.

**Result.** No consensus.

| Metric (τ = 0.9) | Value |
|---|---|
| Pooled live rows | 20480 |
| Connected components | 20467 |
| Multi-member clusters | 7 (max cluster size 5) |
| Multi-seed clusters (≥2 seeds) | 5 |
| Hungarian cosine-Jaccard (mean) | 0.00056 |
| Consensus rate @ ≥3 seeds | 0.00049 (~1 / 2048) |
| Name-agreement rate | 0.20 |
| **Shuffle-null p-value** | **0.19** (200 permutations) |

- At τ = 0.9, 20467 of 20480 rows are singletons — directions are essentially pairwise-orthogonal across seeds.
- The shuffle-null permutation test gives **p = 0.19**: the tiny observed agreement is indistinguishable from randomly relabeled seeds. We cannot reject the null that seeds share no structure.
- **Faithfulness proxy** (5 multi-seed clusters): mean naming-cosine 0.125, mean test activation 0.004 — the handful of "consensus" directions are weakly named and weakly active.

**Figure:** `results/figures/ablation/a0_consensus_headline.png`.

→ Reframing in direction space does **not** rescue cross-seed agreement. The instability is not an artifact of comparing indices; the directions themselves do not recur.

---

## Ablation 01 — Dictionary-Size Ladder

**Setup.** Train TopK SAEs at dict sizes {1024, 2048, 4096}, `k=32` fixed, `steps=12000`, 3 seeds {0, 42, 123} each (lr auto, constant across sizes). Plus an `_auto`-lr companion set and a revival probe (3 seeds) to check whether dead features are recoverable. All gap-corrected naming.

**Per-size metrics (seed-averaged):**

| Dict | Cosine | Dead % | Util % | Entropy (nats) | L0 | Jaccard | Null `k/(2D−k)` | **Ratio** | Naming mean / max |
|------|--------|--------|--------|----------------|----|---------|-----------------|-----------|-------------------|
| 1024 | 0.9950 | 45.2 | 54.8 | 5.33 | 32.0 | 0.0149 | 0.0159 | **0.94** | 0.388 / 0.548 |
| 2048 | 0.9948 | 51.8 | 48.2 | 5.56 | 32.0 | 0.0069 | 0.0079 | **0.88** | 0.387 / 0.535 |
| 4096 | 0.9944 | 60.3 | 39.7 | 5.75 | 32.0 | 0.0045 | 0.0039 | **1.14** | 0.384 / 0.529 |

**Other per-size signals:**
- **Consensus reappearance** (pooled, τ=0.9): 0.00065 / 0.00033 / 0.00016 for D = 1024 / 2048 / 4096 — the rate at which a live direction reappears across ≥2 seeds, vanishing with D.
- **Feature splitting** (mean / p90 pairwise decoder cosine): 0.006 / 0.092 (D=1024) → 0.004 / 0.079 (D=4096). Decoder rows stay near-orthogonal — no collapse, no duplication.

**Top-10 named concepts (dict4096, seed 42, gap-corrected):**

| Feat | Name | Score |
|---|---|---|
| 3690 | dental device | 0.529 |
| 419 | fasciculus cuneatus of spinal cord | 0.528 |
| 685 | shapeable wire tip | 0.524 |
| 3605 | facet joint of spine | 0.513 |
| 1090 | spinal cord | 0.505 |
| 1415 | endotracheal tube | 0.502 |
| 3133 | spinal epidural space | 0.498 |
| 2297 | spinotectal tract of spinal cord | 0.495 |
| 1498 | sacral segment of spinal epidural space | 0.489 |
| 334 | central venous catheter with port or pump | 0.487 |

Same coherent clinical families as the augmented baseline (tubes/devices, vertebral and spinal-cord anatomy).

**Figures:** `a1_stability_frontier.png`, `a1_splitting_dendrogram.png`, `a1_dead_jaccard_vs_dict.png`.

**Reading.** Cosine is flat at ~0.995 across dict sizes — on 512-d projected embeddings, D ≥ 1024 already suffices for near-perfect reconstruction, so width buys almost nothing here. Dead features grow with D (45 → 60%), the expected over-capacity effect. Naming is flat at ~0.385 — dict size neither helps nor hurts RadLex grounding. **Critically, the observed Jaccard equals the exact hypergeometric null at every D** (ratio 0.88–1.14): wider dictionaries give numerically smaller Jaccard only because the null `k/(2D−k)` shrinks with D. Dict size does not move stability off the floor.

---

## Ablation 02 — k (Sparsity) Sweep, null-calibrated

**Setup.** Fix `dict_size=2048`, sweep `k ∈ {8, 16, 32, 64}`, `steps=12000`, 4 seeds {0, 42, 123, 456} per group (lr auto, constant across groups). Jaccard is within-group only; null is the exact hypergeometric with mean-of-ratios convention; 1000× bootstrap CIs over test samples.

**Per-k metrics:**

| k | Cosine | Var. expl. | Dead % | L0 | Jaccard | Null | **Ratio [95% CI]** | CI excludes 1? |
|---|--------|-----------|--------|------|---------|------|--------------------|----------------|
| 8 | 0.9843 | 0.969 | 93.7 | 8.0 | 0.00157 | 0.00209 | **0.75** [0.69, 0.81] | yes (below) |
| 16 | 0.9905 | 0.981 | 81.6 | 16.0 | 0.00476 | 0.00405 | **1.18** [1.14, 1.21] | yes (above) |
| 32 | 0.9950 | 0.990 | 62.6 | 32.0 | 0.00949 | 0.00799 | **1.19** [1.17, 1.21] | yes (above) |
| 64 | 0.9981 | 0.996 | 67.9 | 64.0 | 0.01309 | 0.01599 | **0.82** [0.81, 0.82] | yes (below) |

**Figures:** `a2_k_vs_stability.png`, `a2_pareto_front.png`.

**Reading.**
- **Reconstruction improves monotonically with k** (cosine 0.984 → 0.998): more active features = better fit. Expected.
- **Dead features are U-shaped**: 93.7% at k=8 (too sparse — most features never win a top-8 slot), 62.6% at k=32, rising again to 67.9% at k=64 (over-capacity competition at dict2048). Mechanically sensible.
- **Stability stays on the floor at every k**: ratio ∈ [0.75, 1.19]. The bootstrap CIs are tight (n=1000), so the small deviations from 1.0 are "significant" but **small in magnitude** — none approaches the 5–10× excess that would indicate genuine shared structure.

**One nuance worth recording.** k=16 and k=32 sit slightly but consistently *above* the null (ratio 1.18–1.19, CI excludes 1.0); k=8 and k=64 sit slightly *below* (0.75, 0.82). The mid-sparsity regime shows a faint excess of cross-seed overlap — a trace of real shared structure — but at 1.2× the chance level it is negligible and does not constitute reproducibility. The sweep's overall message is unchanged: sparsity budget does not move concepts off the floor.

---

## Are these results normal?

**Yes — each ablation behaves exactly as the "structural non-uniqueness" hypothesis predicts, with one augmented-specific quirk inherited from the baseline.**

**✅ 00 — no direction-space consensus (p = 0.19).** Normal and reinforcing. If the index-Jaccard floor (0.0038) were a basis-rotation artifact, pooling decoder directions should have recovered agreement. It does not — 20467/20480 singletons, p = 0.19. The directions themselves do not recur across seeds.

**✅ 01 — Jaccard = null at every dict size (ratio 0.88–1.14).** Normal and the key control. Wider dicts give smaller raw Jaccard only because the combinatorial null shrinks; the *ratio* stays at ~1. Reconstruction saturates at D ≥ 1024 (cosine 0.995), dead% rises with over-capacity, naming is flat. Textbook behavior for a dictionary that already over-spans the manifold.

**✅ 02 — Jaccard ≈ null at every k (ratio 0.75–1.19).** Normal. Reconstruction tracks k (0.984 → 0.998), dead% is U-shaped, and the stability ratio hugs 1.0 everywhere. The faint mid-k excess (1.2×) is a real but negligible trace, not reproducibility.

**⚠️ Augmented quirk — elevated dead-feature rates.** Inherited from the augmented baseline (REPORT_augmented.md §3): mild augmentation (`rotation=5°`, `crop=0.95–1.0`) produces ~3× near-duplicate samples, concentrating usage onto common directions and starving rare-direction features. Visible here as the higher dead% columns (01: 45–60%; 02: 63–94% at k=8). Mechanically consistent with the lower activation entropy and dict utilization noted in the baseline, and **not pathological** — variance explained and cosine stay high throughout.

**Bottom line.** The three ablations triangulate the same conclusion from three orthogonal axes: dictionary width, sparsity budget, and agreement space. None moves the SAE off the chance floor. On the leak-free augmented split this is definitive — concept instability is a structural property of sparse decomposition on BiomedCLIP-projected embeddings, independent of data contamination, augmentation, dict size, and k. The 0.0038 baseline figure is the floor.

---

## Ablation 03 — Concept Baselines (Random / PCA / KMeans)

**Setup.** No SAE training. Three alternative dictionaries, all `D=256`, `k=32`, 3 seeds `{0, 42, 123}` each, evaluated on the augmented test split (`N=4545`). PCA/KMeans fit on train only; cached per seed in `results/ablation/a3_cache/`. Additionally, an oversized `random_big` baseline (`D=4096`, same k/seeds) measures the empirical Jaccard floor at the SAE's native dict size.

**Primary-seed (seed=42) metrics:**

| Baseline | Cosine | Dead % | Naming mean | Naming max |
|---|---|---|---|---|
| **SAE TopK (ref, D=4096)** | 0.988 | 44.0 | 0.3949 | 0.5457 |
| Random (D=256) | 0.451 | 0.0 | 0.3717 | 0.4429 |
| Dense-PCA (D=256) | 0.9957 | 0.0 | 0.3786 | 0.5158 |
| Freq-KMeans (D=256) | 0.9609 | 0.0 | **0.833** | **0.879** |
| Random-big (D=4096) | 0.597 | 0.0 | 0.3712 | 0.454 |

**Empirical Jaccard floors (3-seed cross-seed Jaccard, k=32):**

| Baseline | Dict size | Jaccard mean | Std |
|---|---|---|---|
| Random | 256 | 0.0670 | 0.0053 |
| Random-big | 4096 | **0.0036** | 0.0020 |
| SAE TopK augmented baseline (5 seeds) | 4096 | 0.0039 | — |

**Result.** Three findings:

1. **SAE reconstruction is earned.** Random (D=256) cosine = 0.44 — far below SAE 0.988. Dense-PCA achieves near-ceiling (0.9957) with only 256 directions, confirming the augmented embedding manifold is effectively low-rank (≤256 dimensions), as expected from 512-d L2-normalised BiomedCLIP projections.

2. **SAE naming is not emergent.** Random naming ≈ 0.37 (≈ SAE 0.385). PCA naming ≈ 0.379. KMeans naming = **0.833** — dramatically higher because K-Means centroids cluster by visual similarity (chest X-ray population modes) which aligns with RadLex vocabulary by construction. This dissociates naming score from interpretability: KMeans is more "nameable" but captures population statistics, not individual concept detectors.

3. **Empirical Jaccard floor confirms the analytical null.** `random_big` (D=4096, k=32): empirical Jaccard = **0.0036** ≈ analytical `k/(2D−k) = 32/8160 = 0.00392`. The SAE augmented baseline Jaccard of 0.0039 sits on this empirical floor. **The SAE is no more stable than a random basis at the same dict size.**

→ **Ablation 03 directly confirms the master conclusion.** Cross-seed Jaccard = chance floor. The K=32 active directions drawn from D=4096 are essentially random subsets of an overcomplete basis, reproducing the analytical floor empirically.

---

## Ablation 04 — Activation Function Bakeoff (TopK / BatchTopK / JumpReLU)

**Setup.** Train three SAE variants with different activation functions, all at `dict_size=2048`, `k=32` (or equivalent target L0), `steps=12000`, `lr=5e-5`, 3 seeds `{0, 42, 123}`. All evaluated on the augmented test split (`N=4545`). Jaccard measured using `n_active_jaccard=20`; random floor `= n_active/dict_size = 20/2048 = 0.00977`.

**Seed-averaged metrics:**

| Variant | Cosine | Dead % | L0 mean | Jaccard | Signal/Null | Naming mean |
|---|---|---|---|---|---|---|
| **TopK** | 0.9927 | 15.1 | 32.0 | 0.00359 | **0.368** | 0.396 |
| **BatchTopK** | 0.9934 | **7.0** | 34.4 | 0.00210 | **0.215** | 0.391 |
| **JumpReLU** | 0.9917 | 49.8 | 33.1 | **0.00003** | **0.003** | 0.388 |

Random floor (analytical): 0.00977. All variants: Signal/Null < 1.0 → all below chance floor.

**Within-family consensus (direction-space, τ=0.9):**

| Variant | Pooled rows | Clusters | Multi-seed consensus rate |
|---|---|---|---|
| TopK | 6144 | 6143 | 0.00016 (1 cluster) |
| BatchTopK | 6144 | 6144 | 0.0 (no consensus) |
| JumpReLU | 6144 | 6144 | 0.0 (no consensus) |

**Cross-activation consensus (all 18432 rows pooled):** 5.5% of rows span ≥2 activation families; 0% span all 3.

**Result.** Four findings:

1. **Activation function does not rescue stability.** All three variants have Signal/Null < 0.37 — all sit below the random floor. JumpReLU Jaccard (0.000028) is essentially zero: nearly no index overlap across seeds.

2. **BatchTopK has fewest dead features (7%).** Flexible per-sample sparsity budget allows the dictionary to stay utilised even when some directions are less active. TopK's rigid k=32 per sample leaves 15% dead; JumpReLU's learned threshold fails to activate ~50% of the dictionary.

3. **Naming is invariant to activation function (~0.39 across all).** RadLex grounding is a property of the embedding space and vocabulary, not of how the SAE sparsifies activations.

4. **No cross-activation shared directions.** 5.5% spanning-2-families rate is likely due to near-duplicate embeddings in augmented data (3× repeated images), not genuine shared concept structure. 0% spans all three activation families.

→ **Activation function choice does not change the fundamental non-uniqueness.** The sparse decomposition of BiomedCLIP-projected embeddings is on the chance floor regardless of TopK, BatchTopK, or JumpReLU. BatchTopK is operationally best (lowest dead%), but this is a training quality difference, not a stability difference.

---

## Ablation 05 — Faithfulness (Clinical Label Correlation)

**Setup.** Load the augmented baseline SAE (`sae_seed42`, dict=4096, k=32) and evaluate on the augmented test split (`N=4545`). Clinical labels derived from IU X-Ray reports (`indiana_reports.csv`) via MeSH union Problems fields, excluding "normal". Prevalence filter: ≥10 test images → **69/118 labels kept** (median prevalence=60, max=654). Live features: 1404/4096 (34.3%). Pearson correlation computed between each (feature, label) pair; significance via shuffle-null p95 (per-feature) and BH-FDR 0.05 globally.

**Headline metrics:**

| Metric | Value |
|---|---|
| Live features | 1404 / 4096 (34.3%) |
| Prevalent labels | 69 / 118 |
| % live features \|r\|>0.10 | **47.1%** (661/1404) |
| % live features \|r\|>0.15 | 23.7% (333/1404) |
| % live features \|r\|>0.20 | 12.0% (168/1404) |
| % live features \|r\|>0.30 | 2.4% (34/1404) |
| Live features beating shuffle null p95 | **42.7%** (600/1404) |
| BH-FDR 0.05 significant (feature, label) pairs | 5023 / 96876 \|r\| threshold ≈ 0.045 |
| Analytic null SE(r) | 0.0148 (\|r\|>0.10 ~ 6.7σ) |
| **Strongest correlation** | **\|r\|=0.545** (feat 1113 → 'heart failure', prev=18) |

**Top per-label best features:**

| Label (prevalence) | Best \|r\| | Feature |
|---|---|---|
| heart failure (18) | **0.545** | feat 1113 |
| hydropneumothorax (12) | 0.498 | feat 2696 |
| volume loss (12) | 0.487 | feat 1754 |
| pulmonary fibrosis (15) | 0.431 | feat 2548 |
| implanted medical device (69) | 0.391 | feat 2765 |
| pneumoperitoneum (21) | 0.375 | feat 1468 |
| lung diseases, interstitial (21) | 0.363 | feat 3104 |
| consolidation (24) | 0.353 | feat 2030 |
| lucency (27) | 0.332 | feat 2696 |

**Note on SAE names.** Gap-corrected naming (RadLex) assigns anatomy-like labels to most features (e.g. 'posterior root of spinal nerve', 'dental device'); this dissociation from the clinical correlation is expected — RadLex is a radiology anatomy vocabulary, not a pathology vocabulary. The correlation is measured directly against clinical report labels, bypassing naming.

**Result.** Three findings:

1. **A large fraction of live features carry statistically detectable clinical signal.** 47.1% of live features exceed |r|>0.10 vs a prevalent label; 42.7% beat their per-feature shuffle null at p95. This is well above what would be expected from random correlations at this sample size (analytic SE(r)=0.0148, so |r|>0.10 is 6.7σ).

2. **Individual features can reach clinically meaningful correlation magnitudes.** 34 features (2.4% live) exceed |r|>0.30; the strongest is 0.545 for heart failure. These are not negligible signals — a single SAE feature accounts for ~30% of variance in a clinical label presence.

3. **Faithfulness and stability are orthogonal properties.** Cross-seed Jaccard sits at the chance floor (0.0039), yet nearly half the live features correlate with clinical labels. This means the SAE discovers *some* clinically relevant structure — but the specific features that do so are not reproducible across seeds. Two independently trained SAEs may each have features correlated with heart failure, but not the *same* features.

**Figure:** `results/figures/ablation/a5_faithfulness_headline.png`.
**Artifact:** `results/ablation/a5_faithfulness.json`.

→ **Ablation 05 conclusion.** The augmented SAE is clinically faithful but not stable. Features are semantically grounded (47% carry label signal, |r| up to 0.545) while cross-seed overlap remains at the analytical chance floor. Faithfulness and uniqueness are independent axes of SAE quality; the baseline SAE scores well on the former and near-zero on the latter.

---

## Reproducibility notes

- **Clean group split (audit 06-23 F-001/002/003).** `utils.split_embeddings` groups by `study_key_from_basename`; `sorted(set())` deterministic; `group_overlap == 0` asserted. All ablations read the augmented split via `config.paths.*_embeddings_path` (routed by `augmentation.enabled = True`).
- **Fresh retrain, no guard reuse.** The standard-trained ablation checkpoints (`models/ablation_a1`, `ablation_a1_auto`, `ablation_a2`, `ablation_a4`) plus `results/ablation/a3_cache/` and the stale `results/ablation/a*.json` were backed aside to `models/_standard_backup_ablations/` and `results/ablation/_standard_backup/` before running. The notebooks' skip-if-exists guards therefore retrained every model on the augmented embeddings (verified by hash: augmented `a0efb166…` vs standard `6569a2ae…`). Without this step the guards would have silently reused standard models and produced false "augmented" results.
- **Provenance.** Each `training_manifest.json` records `embeddings_path = embeddings/augmented/train_embeddings.pt`, `embeddings_shape = [17865, 512]`, and the augmented `embeddings_hash`.
- **Compute.** 01: 9 dict models + 3 revival + 9 auto-lr = 21 SAEs, ~14 min (20:31–20:46). 02: 16 SAEs (dict2048 × 4 k × 4 seeds), ~10 min (20:46–20:57). RTX 5070, GPU underutilized (small 512-d models, ~10–32% util).
- **Executed notebooks.** `{00,01,02}_*.augmented_exec.ipynb` (headless `nbconvert 7.17.1`, 0 errors). Non-destructive copies — originals preserved.
- **`config.py:240 AugmentationConfig.enabled = True`** is a local edit on a committed file. Revert to `False` to return to the standard path (which would need its own clean retrain).
- **Embedded `baseline_reference` blocks are stale-standard.** The notebooks' JSON outputs carry a hardcoded standard baseline (cosine 0.988, dead 44%, Jaccard 0.0038). Informational only; this report compares against the augmented anchor.
- **Artifacts.** `models/{ablation_a1,ablation_a1_auto,ablation_a2}/`, `results/ablation/{a0_consensus,a1_dict_size,a1_naming_dict*,a2_k_sweep}.json`, `results/figures/ablation/*.png`.

---

## 6. Pending ablations

- ~~**03_baselines**~~ — ✅ **Completato 2026-06-25.** Risultati in [§Ablation 03](#ablation-03--concept-baselines-random--pca--kmeans) e `results/ablation/a3_baselines.json`.
- ~~**04_activation_bakeoff**~~ — ✅ **Completato 2026-06-25.** Risultati in [§Ablation 04](#ablation-04--activation-function-bakeoff-topk--batchtopk--jumprelu) e `results/ablation/a4_activation.json`.
- ~~**05_faithfulness**~~ — ✅ **Completato 2026-06-25.** Risultati in [§Ablation 05](#ablation-05--faithfulness-clinical-label-correlation) e `results/ablation/a5_faithfulness.json`. 47.1% live features |r|>0.10; strongest |r|=0.545 (heart failure). SAE clinically faithful ma non stabile.

**Tutte le ablazioni 00–05 completate.** Sweep augmented concluso.
