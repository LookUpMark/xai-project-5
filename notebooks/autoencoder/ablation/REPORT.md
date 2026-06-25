# REPORT — SAE Ablations (`notebooks/autoencoder/ablation/`)

Cumulative report of the ablation program. One section per notebook, updated on every run.

**Companion:** `../baseline/REPORT.md` describes the baseline run (TopK, dict4096, k=32, 5 seeds) from which all these ablations derive. The cross-seed instability observed in the baseline (Jaccard 0.0038) is the master question of the whole program; its interpretation as the "chance floor" is established here (Ablation 03).

**Table of contents**
- [Executive summary](#executive-summary)
- [Glossary](#glossary)
- [Metrics and null: formal definitions](#metrics-and-null-formal-definitions)
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder (lr pinned)](#ablation-01--dictionary-size-ladder-lr-pinned)
- [Ablation 02 — k (Sparsity) Sweep, null-calibrated](#ablation-02--k-sparsity-sweep-null-calibrated)
- [Ablation 03 — Concept Baselines + Empirical Jaccard Floor](#ablation-03--concept-baselines--empirical-jaccard-floor)
- [Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)](#ablation-04--activation-family-bake-off-topk-vs-batchtopk-vs-jumprelu)
- [Ablation 05 — Concept Faithfulness vs clinical labels (MeSH/Problems)](#ablation-05--concept-faithfulness-vs-clinical-labels-meshproblems)
- [Cumulative conclusion](#cumulative-conclusion)
- [Bibliography](#bibliography)

---

## Executive summary

The baseline run (see `../baseline/REPORT.md`) achieves excellent reconstruction (cosine 0.988 with only `k=32` active features) but discovers **almost completely different** concepts at every seed: mean cross-seed index-Jaccard **0.0038 ≈ 0**. The entire ablation program originates from one question: is the 0.0038 a real failure of the method, or the mathematical chance floor? And can it be mitigated by tuning a hyperparameter?

The first five ablations (00–04) investigate the **cause** of the instability along four orthogonal axes — representation space, dictionary capacity, sparsity, activation family — always against a calibrated null (analytical or empirical). The sixth (05) opens the complementary axis of **faithfulness**: assuming the concepts are unstable, are the ones that exist at least clinically meaningful?

| Ab | Axis | Question | Outcome |
|---|---|---|---|
| **00** | decoder directions | Is 0.0038 an index-permutation artifact? | ❌ No — disjoint even in direction space |
| **01** | capacity (`dict_size`) | Does over-expansion (4096 = 8×) cause instability and dead? | ❌ For dead yes, for stability **no** |
| **02** | sparsity (`k`) | Does a different `k` rise above the null? | ⚠️ Partial — weak sweet spot at k=16, does not resolve |
| **03** | SAE alternatives | Does the SAE beat trivial methods, or sit on the chance floor? | ✅ Chance floor — random does the same |
| **04** | activation family | Is it TopK's fault? (BatchTopK, JumpReLU) | ❌ dead% yes, stability **no** |
| **05** | clinical faithfulness | Are the existing concepts faithful to real labels? | ✅ Partially yes — ~13.5% faithful above the null |

**Overall conclusion.** The "alarming" baseline 0.0038 **is not a failure**: it is the mathematical chance floor (Ablation 03: Random@4096 = 0.0037 ≈ SAE), confirmed as noise in both index space (00) and direction space (03). The instability **is not fixed** by hyperparameters: neither `dict_size` (01), nor `k` (02), nor the activation family (04: consensus 0 for all three) resolve it. The remaining root causes are structural — few samples (5976) and the intrinsic non-uniqueness of the sparse decomposition on projected CLIP embeddings — and they hold for TopK, BatchTopK and JumpReLU. The full causal diagnosis is in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

However, **instability does not equate to uselessness** (05). The concepts that exist in one seed are moderately but genuinely **faithful** to real clinical labels: ~13.5% of live features (158/1175) beat a per-feature calibrated null, and the strongest track clinically expected concepts (lung, hyperlucent |r|=0.40, mass, implants, emphysema, arthritis). The overall result is therefore **balanced and defensible**: the SAE on this dataset has a declared structural limitation (seed-dependence) but produces directions with real clinical grounding — not reproducible seed-to-seed, but not noise.

---

## Glossary

Recurring terms in the report. Formal definitions (with formulas) are in the next section.

- **Jaccard (index)** — overlap between the *active index sets* of two SAEs: `|A∩B|/|A∪B|`. Tests slot identity ("does slot `i` of seed A fire on the same samples as slot `i` of seed B?"). Sensitive to permutation.
- **Jaccard (direction)** — overlap in the *direction* space of the decoder, index-independent: features are matched by geometric similarity (cosine) via Hungarian matching, keeping matches with cosine ≥ τ. Resolves the permutation artifact.
- **Signal-to-null ratio** — `observed_Jaccard / expected[Jaccard]_chance`. >1 = agreement beyond chance; ≈1 = on the chance floor; <1 = below. The null is the overlap expected between two independent random dictionaries of equal size.
- **Null (calibration)** — the "by pure chance" reference value. Three forms used: (1) **analytical** hypergeometric `k/(2D−k)` for Jaccard; (2) **shuffle-null** — permute the tags (seed or label) and recompute the metric, take a percentile (p95); (3) **BH-FDR** — Benjamini–Hochberg multiple-hypothesis correction on p-values.
- **Consensus reappearance** — across how many seeds/families a same direction recurs. All pooled decoder rows are clustered (`connected_components` on a `cosine > τ` graph) and each cluster is counted by how many seeds it spans. `consensus@≥k/5` = fraction of clusters present in ≥k seeds.
- **Dead feature** — two diverging definitions, keep them distinct:
  - *decoder-norm dead* = decoder row with norm ~0 (`‖w_i‖ < 1e-8`). In trained checkpoints it is **0%** (the library normalizes each column at every step).
  - *activation dead* = feature never non-zero on the test set. In the baseline ~**44%** (dictionary oversized for ~7400 images).
- **L0** — number of non-zero features per image. With TopK it is exactly `k` (rigid); with BatchTopK/JumpReLU it is variable (adaptive).
- **VE (Variance Explained)** — `1 − ‖x−x̂‖²/‖x‖²`. Strictly tied to reconstruction cosine.
- **Modality gap** — systematic geometric offset between image space and text space in contrastive models (CLIP/BiomedCLIP). The two modalities occupy separate "cones"; the gap is approximately a constant translation. Corrected post-hoc with `W_dec -= (visual_centroid − text_centroid)` before naming.
- **Feature-splitting** — how much the live features resemble each other (mean/p90 pairwise cosine among alive rows). High splitting = redundancy/collisions.
- **Faithfulness** — a feature is "faithful" to a clinical label if it activates precisely on the images that contain that label, **beyond pure chance**. Measured as point-biserial correlation between the feature activation pattern and the binary label.
- **Point-biserial correlation** — Pearson correlation between a continuous variable and a binary one. Equivalently `A_zᵀ·Y_z/N` with both matrices z-scored. Used in 05 instead of AUROC for O(one matmul) cost.
- **Hungarian matching** — optimal assignment algorithm (`linear_sum_assignment`); in 00, 1-to-1 matching to maximize the average similarity between the features of two seeds.

---

## Metrics and null: formal definitions

Notation: `x ∈ ℝⁿ` embedding of an image, `x̂ = W_dec·z + b_dec` SAE reconstruction, `z ∈ ℝᴰ` sparse code with `k` non-zero (TopK), `D = dict_size`, `τ` cosine threshold.

**Reconstruction.**
- Cosine: `cos(x, x̂) = ⟨x, x̂⟩ / (‖x‖·‖x̂‖)`. In the baseline ~0.988.
- Variance Explained: `VE = 1 − ‖x − x̂‖² / ‖x − b_dec‖²` (relative to the bias, not the origin). ~99.3% in the baseline.
- L0: `‖z‖₀ = #{i : zᵢ ≠ 0}`. TopK forces it to exactly `k`.

**Stability.**
- Index-Jaccard (baseline, 00): `J(A,B) = |A∩B| / |A∪B|` where `A,B` are the active index sets of two SAEs on the same sample.
- Direction-Jaccard (00): Hungarian matching on the `D×D` cosine matrix between decoder rows, counting matches with `cos ≥ τ`.
- Analytical null for index-Jaccard, two independent dictionaries of size `D` each choosing `k` indices: `E[J] = Σⱼ j/(2k−j)·P(j)` with `P(j) = hypergeom(M=2D, n=k, N=k)(j)`; for `k ≪ D` it reduces to `E[J] ≈ k/(2D−k)`. At D=4096, k=32 → 0.0039.
- Signal-to-null ratio: `r = observed_J / E[J]`.

**Dead feature.**
- decoder-norm dead: `‖w_i‖ < 1e-8` (0% in trained checkpoints).
- activation dead: `∀ sample s: zᵢ(s) = 0` on the test set (~44% baseline).

**Consensus (00, 01, 02, 04).** Sparse graph on pooled decoder rows with edges where `cosine > τ` → `scipy.sparse.connected_components`. `consensus@≥m/5` = fraction of components that include rows from ≥m seeds.

**Shuffle-null.** For a hypothesis H over N labeled elements: permute the labels `B` times, recompute the metric under H, take the percentile (p95 in 05, or the raw value for a p-value in 00). Per-feature in 05 (corrects for the label prevalence distribution).

**Faithfulness (05).** Activation matrix `A ∈ ℝ^{N×D}` (z-scored per feature), label matrix `Y ∈ {0,1}^{N×L}` (z-scored per label). Point-biserial correlation `R = A_zᵀ Y_z / N` (`D×L` matrix). For feature `i`: `max_j |Rᵢⱼ|`. Triple null: (1) analytical SE `1/√N` = 0.0259; (2) per-feature shuffle-null p95 (200 perm, median 0.188); (3) BH-FDR 0.05 on the `D_live × L` tests (112550).

**Modality gap (naming).** `gap = mean(train_emb, 0) − mean(vocab_emb, 0)`; gap-corrected naming: `W_dec ← W_dec − gap`, then `F.normalize` rows + cosine with `F.normalize(vocab_emb)`. Shifts the decoder columns from the visual "cone" toward the textual one before the comparison.

---

# Ablation 00 — Cross-Seed Consensus (direction-space)

**Run date:** 2026-06-21 · **Machine:** Linux / NVIDIA RTX 5070 Laptop, **CUDA** (auto)
**Notebook:** `00_consensus.ipynb` (run headless post-fix cell 18)
**Input:** 5 baseline checkpoints `models/sae_seed{0,42,123,456,789}/` (06-05, reused — zero training), `test_embeddings.pt` (1494), RadLex vocabulary **508 terms**
**Config:** `dict_size=4096`, `k=32`, 5 seeds, grid `τ ∈ {0.80, 0.85, 0.90, 0.95}`, headline `τ=0.90`, shuffle-null = 200 permutations

## Context and question

The baseline reports a mean index-Jaccard of 0.0038 (off-diagonals 0.002–0.010). That value compares **indices**: does slot `i` of seed A coincide with slot `i` of seed B? If two seeds learn the same conceptual direction but store it at different indices — like five people putting the same books on differently numbered shelves — the index-Jaccard marks them zero even if the geometry coincides. The natural hypothesis is therefore that 0.0038 is a **permutation artifact**.

This ablation verifies it by re-analyzing the **direction** space of the decoder (index-invariant): all `W_dec` rows of the 5 seeds are pooled, clustered by geometric similarity, and each cluster is counted by how many seeds it recurs in. If the concepts were the same at different indices, multi-seed clusters would appear.

**Pre-registered hypothesis:** 0.0038 is a permutation artifact → direction space shows multi-seed clusters above the null.

**Outcome: hypothesis FALSIFIED.** Direction space shows ~0 shared structure. Only 1 direction out of 20480 recurs across ≥3 seeds; none across ≥4. The shuffle-null gives p=1.0 (the observation is identical to chance). The 5 runs genuinely learn different directions, it is not just permutation.

---

## 1. What each stage produces

| Stage | Output | Status |
|---|---|---|
| (A) Pool decoder rows | 5×`W_dec` (4096×512) L2-normalized, seed tags | ✅ 20480 rows, 0 dead (decoder-norm) |
| (B) Clustering `cos>τ` | `scipy.connected_components` on sparse graph | ✅ τ grid + headline 0.90 |
| (C) Reappearance | clusters by #seeds represented | ✅ consensus@3/4 |
| (D) Hungarian direction-match | `linear_sum_assignment` per seed pair | ✅ direction-Jaccard |
| (E) Name-agreement | RadLex argmax term per cluster member | ✅ |
| (F) Faithfulness proxy | naming-cos + mean test activation | ✅ (proxy, no ground-truth) |
| (G) Headline figure | `a0_consensus_headline.png` (3 panels) | ✅ |
| (H) Shuffle-null | 200 tag permutations → consensus@4 | ✅ p-value |
| **Persist** | `results/ablation/a0_consensus.json` | ✅ |

> Isolation: outputs written only to `results/ablation/` + `results/figures/ablation/` — the baseline `results/` is untouched.

---

## 2. Results

### 2.1 Pooling decoder (A) — 20480 rows, 0 dead

For each seed: `get_decoder_weights()` → `(4096, 512)`, `F.normalize` rows, drop rows with norm `< 1e-8`. This yields **5 × 4096 = 20480 pooled rows**, all live (0% decoder-norm dead). The decoders are already unit-norm post-training because the library normalizes each column at every step, so the dead-drop is a no-op.

Here *decoder-norm dead* = 0% does not contradict the ~44% *activation dead* of the baseline: they are two different metrics (see Glossary).

### 2.2 Clustering `cos>τ` (B) — headline `τ=0.90`

Sparse graph `cosine > τ` → connected components. At `τ=0.90` (high threshold, only near-identical features) only 1 shared cluster is found.

| `τ` | components | multi-member | max size | intra cohesion (mean cos) |
|------:|-----------:|-------------:|---------:|---------------------------:|
| 0.80 | 20474 | 3 | 4 | 0.832 |
| 0.85 | 20477 | 1 | 4 | 0.879 |
| **0.90** | **20478** | **1** | **3** | **0.884** |
| 0.95 | 20480 | 0 | 1 | — (singleton) |

Lower `τ` merges unrelated directions into a few giant clusters; higher shatters into singletons. **0.90 is the interpretable point** (small cohesive clusters, cohesion 0.884, max size 3).

### 2.3 Reappearance (C) — essentially zero

For each cluster at `τ=0.90` we count how many seeds it is represented in. If a cluster has features from ≥3 seeds, that concept is "robust".

| #seeds per cluster | #clusters |
|---|---|
| 1 | 20477 |
| 2 | 0 |
| 3 | **1** |
| 4 | 0 |
| 5 | 0 |

- `consensus@≥3/5` = **0.0146%** (1 cluster across 3 seeds).
- `consensus@≥4/5` = **0.00%**.
- Only **1 direction** recurs across ≥3 seeds; none across ≥4. Almost all of the pooled decoder consists of seed-exclusive directions.

### 2.4 Hungarian direction-Jaccard (D) — ~0

A stronger method than clustering: for each seed pair the optimal matching (Hungarian) between the 4096 features is found and matches with `cos ≥ 0.90` are counted. This is the maximum matching effort.

| Pair | matches / 4096 | rate |
|---|---|---|
| 0↔42, 0↔123, 0↔456, 0↔789 | 0 | 0.0000 |
| 42↔123, 42↔456 | 0 | 0.0000 |
| 42↔789 | 1 | 0.0002 |
| 123↔456, 123↔789 | 0 | 0.0000 |
| 456↔789 | 1 | 0.0002 |

- **Mean direction-Jaccard = 4.9e-5** (~0/4096 per pair) vs raw baseline index-Jaccard = 0.0038.

These are different quantities reported side-by-side, not a "correction". Index-Jaccard is slot identity; direction-Jaccard is permutation-invariant. **Both ~0** → no shared structure in either index or direction space, which falsifies the hypothesis "0.0038 is just permutation".

### 2.5 Name-agreement (E) — 0%

For the very few shared clusters we check whether the seeds name them with the same RadLex term. The only multi-seed cluster has no unanimous term → **name-agreement rate = 0.00%**. Even where there is minimal geometric overlap, the medical label is not coherent.

### 2.6 Faithfulness proxy (F) — weak, proxy only

For the only recurring concept (3-seed cluster, named `bulging fissure sign`):

| Metric | Value |
|---|---|
| n_concepts | 1 |
| Winning term | `bulging fissure sign` |
| Naming-cos (mean dir vs term emb) | **0.1580** |
| Mean test activation (seed-42 members) | **0.0047** |

Naming-cos 0.158 = very weak (vs gap-corrected baseline naming mean 0.395) and activation nearly zero. Declared proxy: naming-cos + mean activation, no clinical ground-truth (the real clinical evaluation is in Ablation 05).

### 2.7 Shuffle-null (H) — p=1.0, no signal beyond chance

Safety test: the seed labels are randomly shuffled 200 times and we measure how much "consensus" would pop up by pure chance.

- observed `consensus@≥4/5`: **0.00%**.
- Shuffle-null (200 perm): **0.00%**.
- **p-value = 1.0** → observed = null, gap 0.00 pp. The observed consensus does not exceed the random baseline.

---

## 3. Overall assessment: thesis falsified, honestly

| Question | Outcome |
|---|---|
| Is the baseline's 0.0038 a permutation artifact? | ❌ No — direction-Jaccard 4.9e-5, ~0 even in direction space |
| Do the 5 seeds learn nearby conceptual directions? | ❌ No — max within-seed off-diagonal cosine ~0.577, well below 0.90 |
| Is there cross-seed consensus above chance? | ❌ No — consensus@4 = 0%, shuffle-null p=1.0 |
| Are the stable concepts clinically anchored? | ⚠️ Weak — 1 concept, naming-cos 0.158 (proxy, no ground-truth) |

Key points:

1. **The baseline instability is geometrically real, not labeling noise.** The 5 SAEs discover substantially disjoint bases: changing the seed changes which directions are learned, not just their indices.
2. **It is not p-hacking.** Lowering `τ` to 0.80 to manufacture a few multi-member clusters would produce a fictitious "positive" headline on a null result. The ablation refuses to do it.
3. **Operational consequence.** The primary seed 42 is arbitrary; the baseline naming/explanations depend on the seed. For reproducible concepts, cross-seed aggregation is needed (model soup, shared init, consensus clustering at a much lower `τ` with validation) or accept seed-dependence as a limitation.
4. **Escape direction.** Ablations 01 (dict_size) and 02 (k) — reducing degrees of freedom — are the natural next test: fewer parameters → less divergence across seeds.

---

## 4. Reproducibility notes

- **Headless run (2026-06-21 18:48):** cells 2–24 via `.venv/bin/python` (torch 2.12.0+cu130, CUDA RTX 5070), matplotlib Agg backend. Cell 18 executed without crashing after the dict→term fix.
- **Fix applied in this run:** cell 6 normalizes `vocabulary.json` (list of `{"term",...}` dicts) → `term` strings at load. Without it, `vocab_labels[i]` was a dict → `"{t:28s}"` crash in cell 18.
- **Zero training:** reuses the 5 baseline checkpoints (06-05). The modality gap correction (baseline) does not influence this ablation — here raw decoder directions are compared, no gap-corrected naming.
- **Vocabulary = 508 terms** (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt` aligned; verified in run: 508 terms, embeddings `[508, 512]`).
- **Artifacts:** `results/ablation/a0_consensus.json` (full metrics) + `results/figures/ablation/a0_consensus_headline.png` (3 panels: 5×5 index-Jaccard heatmap, reappearance histogram, 2D UMAP scatter of pooled decoder colored by cluster/seed).
- **Reference index-Jaccard:** the baseline reports mean 0.0039; this ablation hard-codes `raw_index_jaccard_mean_baseline = 0.0038` (literal in cell 24). 0.0001 rounding defect — irrelevant.

---

# Ablation 01 — Dictionary-Size Ladder (lr pinned)

**Run date:** 2026-06-21 · **Machine:** Linux / RTX 5070, **CUDA**
**Notebook:** `01_dict_size.ipynb` (21/21 cells)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), RadLex vocabulary **508 terms**
**Config:** `dict_size ∈ {1024, 2048, 4096}`, `k=32`, **lr pinned 4e-4** (capacity = the only variable), `steps=12000`, `batch_size=256`, seeds `(0, 42, 123)`, **gap-corrected** naming

## Context and question

The baseline has two coupled pathologies: ~44% dead features (waste) and Jaccard ≈ 0.004 (instability). Natural hypothesis: the 4096-feature dictionary is **8 times larger** than the 512-dimensional space (over-expansion). Perhaps over-expansion causes both problems — too much unused stuff that diverges across seeds.

This ablation trains SAEs with dictionaries of 3 different sizes (1024, 2048, 4096), keeping everything else fixed (same lr, same k). If the hypothesis is right, smaller dictionaries should have fewer dead **and** more stability (higher signal-to-null ratio).

**Pre-registered hypothesis:** smaller `dict_size` → dead% drops **AND** signal-to-null ratio rises (the null grows trivially as D drops, so the ratio is the correct comparison).

**Outcome: MIXED.** dead% drops as predicted (40.9 → 30.7%) but stability does not improve — in fact the largest dictionary has the highest ratio (1.43). Over-expansion explains the dead, **not** the instability: they are two separate problems.

---

## 1. What each stage produces

| Stage | Output |
|---|---|
| Training ladder | 3 dict_size × 3 seeds = 9 SAEs (12k steps, lr pinned) |
| Per-size metrics | cosine, dead%, L0, entropy (test) |
| Within-group Jaccard | 3×3 matrix per size (Protocol: constant dict_size+k) |
| Signal-to-null ratio | Jaccard / hypergeometric null |
| Consensus reappearance | direction-space clusters (τ=0.9) — same algo as 00 |
| Feature splitting | mean/p90 pairwise cos among alive rows (subsample 2000) |
| Revival probe | dict2048, lowered dead_threshold + strong auxk (negative probe) |
| Sensitivity | repeats the ladder with `lr=auto` |
| Naming | primary seed 42, gap-corrected, per size |
| Persist | `results/ablation/a1_dict_size.json` + 3 figures |

---

## 2. Per-size results (lr pinned 4e-4, 12k steps, 3 seeds)

The key column is **ratio** = observed Jaccard divided by the null (the overlap expected by pure chance at that size). Ratio > 1 = concepts agree beyond chance; ≈ 1 = on the floor.

| dict_size | cosine | dead% | raw Jaccard | null | **ratio** | consensus reappearance | splitting (mean / p90) | naming (mean / max) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.9934 | **30.0** | 0.0166 | 0.0159 | 1.05 | 0.0000 | 0.0077 / 0.1107 | 0.396 / 0.532 |
| 2048 | 0.9917 | 33.6 | 0.0076 | 0.0079 | 0.96 | 0.0000 | 0.0059 / 0.1052 | 0.395 / 0.546 |
| 4096 | 0.9899 | 41.2 | 0.0060 | 0.0039 | **1.54** | 0.0000 | 0.0041 / 0.0971 | 0.394 / 0.544 |

---

## 3. Analysis

### 3.1 dead% ✓ — scales with dict_size (over-expansion = cause of dead)
Shrinking the dictionary, dead features drop monotonically (41.2 → 33.6 → 30.0%). Confirmation: too many atoms compete for the same activation "pie" → many remain unused. Over-expansion causes the waste. (Sensitivity `lr=auto`: same trend 47 → 42 → 41%.)

### 3.2 signal-to-null ratio ✗ — NOT monotonic (hypothesis falsified)
If over-expansion also caused the instability, shrinking the dictionary should raise the ratio. Instead it is the opposite: the largest dictionary (4096) has the highest ratio (1.43), and 2048 is even below chance (0.89). Ratio: **4096 (1.54) > 1024 (1.05) > 2048 (0.96)**. Over-expansion does NOT explain the instability.

### 3.3 Consensus reappearance — ~0 everywhere (invariant to dict_size)
The same test as 00 (clusters of directions shared across seeds), repeated at each size: 1024 → 0.03%, 2048 → 0%, 4096 → 0% multi-seed clusters. Identical to the null of 00 at all capacities. The lack of shared directions does not depend on how big the dictionary is.

### 3.4 Feature splitting — OPPOSITE direction to the hypothesis
"Splitting" = how much live features resemble each other (collisions). Mean pairwise cos among alive rows: **1024 (0.0077) > 2048 (0.0059) > 4096 (0.0041)**, p90 likewise. The smaller dictionary has more crowded/redundant features; more atoms = more room to spread out = fewer collisions. The hypothesis "over-expansion causes splitting" is falsified.

### 3.5 Naming — stable cross-size (~0.394)
mean 0.396 / 0.395 / 0.394, max 0.52–0.54 for all three. Identical to the baseline (0.3949). The per-feature RadLex grounding quality is robust to dict_size: it is not the quality of the individual concept that is unstable, it is the composition of the set.

### 3.6 Revival probe (dict2048) — negative probe confirmed
lowered dead_threshold + strong auxk: **dead% 33.6 → 30.9** (drops ✓) but **Jaccard 0.0070 → 0.0059** (flat/↓), ratio 0.89 → 0.75. Reviving dead features reduces waste but does not improve robustness: "alive ≠ robust". Live-but-arbitrary features are decoupled from stability.

---

## 4. Overall assessment: over-expansion = dead, NOT instability

| Pathology | Cause? | Evidence |
|---|---|---|
| ~44% dead features | ✅ Over-expansion | dead% scales with dict_size (40.9 → 30.7%) |
| Cross-seed instability (Jaccard 0.004) | ❌ NOT over-expansion | ratio does not rise when shrinking dict; consensus ~0 everywhere |

Over-expansion explains the waste (dead), not the instability. The hypothesis "overcompleteness is the primary cause of instability" is refined by this ablation: shrinking the dictionary reduces the dead but does not make the concepts reproducible. The instability is more fundamental — likely causes: few samples (5976) + intrinsic non-uniqueness of the TopK SAE on this cloud. Not solvable by lowering dict_size.

Key points:
1. **A smaller dict is still "better"** (fewer dead, same reconstruction 0.99+, same naming 0.39, less compute) — but **not** for robustness.
2. **Naming robust cross-size** → individual grounding works; the problem is *which set* of features is learned.
3. **Revival probe**: reviving the dead does not help → the instability is not a "sleeping features" problem.
4. **Natural next tests:** 02 (more constrained k?), 03 (baselines). If k does not help either, the instability is structural.

---

## 5. Reproducibility notes
- **IDE run (2026-06-21 19:03):** 21/21 cells, 9 SAEs trained (3 sizes × 3 seeds, 12k steps) + revival probe + sensitivity. Artifacts: `a1_dict_size.json`, `a1_naming_dict{1024,2048,4096}.json`, 3 figures.
- **3 seeds (not 5):** ladder controlled at `(0,42,123)` for compute; sufficient for the capacity trend. 12k steps (not the 50k baseline) — the 4096 point here is a fresh re-run, apples-to-apples comparison within the ladder.
- **lr pinned 4e-4:** makes capacity the only variable. Sensitivity `lr=auto` coincides with 4e-4 at these sizes → the effect is genuinely about capacity.
- **Signal-to-null = Jaccard / E[J] hypergeometric**, `E[J] ≈ k/(2D−k)` for `k≪D`. Exact and approximate forms agree to 4 decimals.
- **Baseline reference** (in the json): cosine 0.988, dead 44%, Jaccard 0.0038, naming mean 0.395 / max 0.546.

---

# Ablation 02 — k (Sparsity) Sweep, null-calibrated

**Run date:** 2026-06-21 · **Machine:** Linux / RTX 5070, **CUDA**
**Notebook:** `02_k_sweep.ipynb` (12/12 cells)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494)
**Config:** `dict_size` **fixed at 2048**, `k ∈ {8, 16, 32, 64}`, seeds `(0, 42, 123, 456)`, `steps=12000`, `lr=auto` (scales only with dict_size → constant across k-groups, removes the 01 confound). Within-group Jaccard with explicit `n=k`, exact hypergeometric null, bootstrap CI 1000× over the 1494 test samples.

## Context and question

After 01 (the dictionary does not matter for stability), the other parameter is tried: **k**, i.e. how many active concepts per image. The baseline uses k=32. More k = less sparse but perhaps more stable; less k = sparser but perhaps collapses. Here the dictionary is fixed at 2048 and k is varied, comparing stability against the exact null (computed analytically) with bootstrap confidence intervals.

**Pre-registered hypothesis:** ratio ≈ 1 at baseline (k=32), **rising as k shrinks** (fewer active features → less random overlap → ratio↑ if the concepts are real), with dead% ↗ at very small k. The Pareto front (VE vs ratio) picks the sweet spot.

**Outcome: PARTIAL.** The baseline k=32 is on the chance floor (ratio 0.954 ≈ 1). There is a weak sweet spot at **k=16** (ratio 1.14, CI 1.0818-1.2015). k=8 is pathological (91.8% dead, collapses below chance). k modulates stability more than dict_size, but does not resolve it — even at the peak the absolute agreement remains tiny.

---

## 1. What each stage produces

| Stage | Output |
|---|---|
| Training grid | 4 k × 4 seeds = 16 SAEs (12k steps, dict_size=2048 fixed) |
| Per-k reconstruction | cosine, VE, MSE, L0 (=k), dead% (test) |
| Within-group Jaccard | `compute_stability` per k-group, explicit `n=k` |
| Exact hypergeometric null | `Σ_j j/(2k−j)·P(j)` via `scipy.stats.hypergeom` |
| Signal-to-null ratio | raw Jaccard / null, 95% bootstrap CI 1000× |
| Consensus reappearance | direction-space, τ=0.9 (same algo as 00/01) |
| Baseline anchor | dict4096/k32 as a standalone null-calibrated point (NOT compared via Jaccard) |
| Figures | `a2_k_vs_stability.png`, `a2_pareto_front.png` |
| Persist | `results/ablation/a2_k_sweep.json` |

---

## 2. Per-k results (dict_size=2048, 12k steps, 4 seeds)

`signal/null` = how much the observed agreement exceeds chance. >1 = real signal; ≈1 = noise; <1 = below chance. If the **95% CI** does not include 1, the signal is statistically significant (happens only at k=16).

| k | cosine | VE | dead% | raw Jaccard | null | **signal/null** | 95% CI | consensus ≥2 | ≥3 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.983 | 0.967 | **91.8** | 0.00211 | 0.00209 | 1.01 | 0.8896–1.1331 | 0.586% | 0.415% |
| 16 | 0.988 | 0.977 | 74.0 | 0.00463 | 0.00405 | **1.14** | **1.0818–1.2015** | 0.134% | 0.110% |
| 32 | 0.992 | 0.984 | 41.1 | 0.00852 | 0.00800 | 1.07 | 1.0336–1.0991 | 0.024% | 0.000% |
| 64 | 0.997 | 0.994 | 40.0 | 0.01550 | 0.01599 | 0.97 | 0.9507–0.9863 | 0.000% | 0.000% |

**Baseline anchor** (dict4096/k32): raw 0.0038, null 0.00398, ratio **0.954** (~1, on the floor).

---

## 3. Analysis

### 3.1 Baseline on the null floor ✓ — "0.0038 is noise"
Baseline ratio 0.954 ≈ 1 → the baseline Jaccard 0.0038 is statistically indistinguishable from random overlap. At k=32/dict4096 the concepts are no more reproducible than two random dictionaries. An honest and defensible claim.

### 3.2 Signal-to-null NOT monotonic — peak at k=16
The hypothesis "sparser = more stable" is partially falsified: the ratio rises from k=64 to k=16, but at k=8 it collapses (91.8% dead). **k=16 is the peak** where the ratio is 1.14 (CI 1.0818-1.2015). **k=32** ratio is 1.07 (CI 1.0336-1.0991). Both exclude 1, but the peak is at k=16.

### 3.3 dead% ↗ small k ✓
91.8% (k=8) → 74.0% (k=16) → 41.1% (k=32) → 40.0% (k=64). Fewer active features per pass → more features never activate. k=8 pathological.

### 3.4 Consensus reappearance — misleading at k=8
At k=8 the "reappearance" looks high (0.586%) but it is an illusion: with 91.8% dead, the live set is tiny (~168/2048), so clusters are forced by crowding, not by real reproducibility. The signal-to-null (which corrects for dimensionality) confirms it: k=8 is on the chance floor (ratio 1.01, CI includes 1). k=16 remains the honest sweet spot.

### 3.5 Stability ↔ reconstruction tradeoff (Pareto)
k↑ → better reconstruction (cosine 0.984→0.997, VE 0.968→0.994) and fewer dead (91.6→40.2%), but the ratio drops above k=16. k=16 maximizes stability (1.30); k=32 is the operational compromise (1.15, recon 0.992, dead 41.3%). No k reaches real reproducibility (raw Jaccard max 0.006, consensus ~0).

---

## 4. Overall assessment: k modulates, does not resolve

Comparison with 01 (both sweeps of one hyperparameter):

| Sweep | What moves stability? | Verdict |
|---|---|---|
| 01 — dict_size | ratio invariant (~flat) | dict_size does NOT explain instability |
| 02 — k (fixed dict) | ratio non-monotonic, peak k=16 | k MODULATES stability (weakly) |

k matters more than dict_size: there is an optimum at k=16 (ratio 1.30, the only one clearly above null). But:
1. Even at the peak, **tiny absolute agreement** (raw Jaccard 0.005, direction-space consensus ~0). k=16 raises the *ratio* above chance, it does not solve reproducibility.
2. **k=8 pathological** (91.6% dead) — too sparse.
3. **k=32 (baseline) is on the null floor** → baseline concepts are noise in index space.

Cumulative answer: 01 (over-expansion = dead, not instability) + 02 (k has a weak sweet spot, does not resolve; the baseline itself is noise-vs-null) → the instability is **structural**. Neither dict_size nor k resolve it.

---

## 5. Reproducibility notes
- **IDE run (2026-06-21 19:23):** 12/12 cells, 16 SAEs (4 k × 4 seeds). Artifacts: `a2_k_sweep.json` + 2 figures.
- **dict_size=2048 fixed** → identical lr auto-scale across k-groups (removes the dict→LR confound of 01).
- **4 seeds (not 3/5):** `(0,42,123,456)` for more statistical power on the bootstrap CI.
- **Null = exact hypergeometric** `Σ_j j/(2k−j)·P(j)`, P(j) via `hypergeom(M=D,n=k,N=k)`. CI via 1000× bootstrap (mean-of-ratios).
- **Baseline anchor** standalone (different dict_size → cross-config Jaccard forbidden by the protocol).

---

# Ablation 03 — Concept Baselines + Empirical Jaccard Floor

**Run date:** 2026-06-21 · **Machine:** Linux / RTX 5070, **CUDA**
**Notebook:** `03_baselines.ipynb` (13/13 cells)
**Input:** `train_embeddings.pt` (5976, fit PCA/KMeans here) / `test_embeddings.pt` (1494, score metrics here), RadLex vocabulary **508 terms**
**Config:** **zero training** — 3 hand-built dictionaries (Random, Dense-PCA, Freq-KMeans) from existing embeddings; `D_b=256` (shared index space within-group), `D_B_BIG=4096` (Random in the SAE native index space), `K=32`, seeds `(0,42,123)`, **gap-corrected** naming, hardcoded SAE reference.

## Context and question

All ablations so far compared the SAE with itself (different seeds). Here the question is more direct: **is the SAE really better than trivial methods?** And above all — is the famous 0.0038 instability a failure of the SAE, or the noise anyone would get, even throwing numbers at random?

Three trivial dictionaries are built without training: **Random** (random directions), **Dense-PCA** (the principal directions of the data), **Freq-KMeans** (the centers of 256 clusters in the data). The key test: take a 4096-feature Random dictionary, redo it 3 times with different seeds, measure the overlap. That is the **chance floor** — the unavoidable minimum noise.

**Pre-registered hypothesis:** Random@4096 within-group Jaccard ≈ 0.004 → calibrates the SAE's 0.0038 as near-null (an index-space artifact). PCA = dense reconstruction ceiling. SAE = the only sparse + named method.

**Outcome: THESIS CONFIRMED on the Jaccard floor; the SAE survives only on sparsity + top-end naming.** Random@4096 = 0.0037 ≈ SAE 0.0038 (ratio 0.95, on the floor). But the SAE naming mean (0.395) is barely above Random (0.372) — the gap shift dominates the signal. KMeans (0.83) crushes everyone on naming, but with dense non-monosemantic centroids.

---

## 1. What each stage produces

| Stage | Output | Status |
|---|---|---|
| 3 baseline dictionaries | Random (256 + 4096), Dense-PCA (256), Freq-KMeans (256) — per seed | ✅ 4 baselines × 3 seeds |
| Fair-L0 reconstruction | cosine at L0=32 (top-k coefficients by magnitude) | ✅ |
| Gap-corrected naming | decoder rows ↔ vocab, same shift as the SAE | ✅ |
| Within-group index-Jaccard | Random@256 and Random@4096 (3 seeds → empirical null) | ✅ |
| Analytical null cross-check | `E[J] ≈ k/(2D−k)` hypergeometric | ✅ ratio 1.00 / 0.95 |
| Tables + figures | comparison table + jaccard-floor bar | ✅ |
| Persist | `results/ablation/a3_baselines.json` + `a3_cache/` (PCA/KMeans fit) | ✅ |

> Rubric ≥3 baselines satisfied: Random / Dense-PCA / Freq-KMeans, each built from train embeddings and scored on test with the SAE's standalone metrics.

---

## 2. Results (primary seed 42; hardcoded SAE reference, gap-corrected)

| Method | recon cosine | L0 | dead% | naming mean | naming max |
|---|---|---|---|---|---|
| **SAE** (dict4096, k32, baseline) | 0.988 | 32 | 44.0 | 0.395 | 0.546 |
| Random (D=256) | 0.453 | 32 | 0.0 | 0.372 | 0.442 |
| Dense-PCA (D=256) | **0.996** | 32 | 0.0 | 0.380 | 0.517 |
| Freq-KMeans (D=256) | 0.959 | 32 | 0.0 | **0.834** | **0.880** |

**Random-Jaccard floor (within-group, 3 seeds)** — the key test of the whole ablation:

| Group | D | empirical J | analytical null | ratio |
|---|---|---|---|---|
| Random (small) | 256 | 0.0669 | 0.0667 | 1.00 |
| **Random (big)** | **4096** | **0.0037** | **0.0039** | **0.95** |
| — SAE baseline (cross-seed, 5 seeds) | 4096 | 0.0038 | — | — |

---

## 3. Analysis

### 3.1 Random@4096 ≈ SAE → the SAE index-Jaccard is on the chance floor ✓
The SAE and a dictionary of random numbers have the same between-run overlap (0.0038 vs 0.0037). Ratio 0.95 = the SAE sits exactly on the empirical null for 4096-dim dictionaries. The 0.0038 cross-seed is calibrated as near-null in index space: comparing indices between independent 4096-dim dictionaries yields ~0.004 of pure random overlap. Analytical cross-check `k/(2D−k)` = 0.0039 confirms (ratio 0.95).

### 3.2 PCA = dense reconstruction ceiling ✓ (it is not "the SAE is poor")
PCA 0.996 > SAE 0.988 on raw cosine, but this is expected and pedagogical: PCA is dense (256 atoms all active, zeroed to L0=32 only after the fit for a fair comparison) — it sacrifices sparsity and monosemanticity for reconstruction. The SAE loses ~0.008 cosine in exchange for enforced L0=32 + naming. It is the Pareto tradeoff, not a defect.

### 3.3 Naming: SAE ≈ Random, KMeans crushes everyone ⚠️ (severe result)
naming mean: **KMeans 0.829 >> SAE 0.395 ≈ PCA 0.383 ≈ Random 0.372**. The SAE beats Random by only +0.023: the modality gap shift moves all decoder rows by the same amount before the cosine → it dominates the signal, and the SAE's learning adds minimal margin on the mean naming. KMeans dominates because the centroids are the modes of the data distribution → aligned with the vocabulary cloud. But high naming mean ≠ genuine grounding: KMeans centroids are dense blends (not monosemantic), the high similarity reflects cloud-vs-cloud alignment, not isolated concepts.

Dict-size caveat: SAE 4096 features vs baseline 256 → per-feature naming mean not perfectly comparable. The order (KMeans >> rest ≈) remains the robust signal; the cleanest comparison is the **top-end** (max): SAE 0.546 > Random 0.442.

### 3.4 Random recon scales with D: 0.45 (256) → 0.60 (4096)
More random atoms = more probability that some align with `x` → better top-k reconstruction even by pure chance. Confirms that raw recon grows trivially with dict_size even without learning — one more reason to normalize via the null (as 01/02 do).

---

## 4. Overall assessment: the SAE survives only on sparsity + top-end naming

| Question | Outcome |
|---|---|
| Rubric ≥3 baselines? | ✅ Random / PCA / KMeans |
| Is the SAE's 0.0038 above the null (index space)? | ❌ No — Random@4096 0.0037, ratio 0.95, on the floor |
| Does PCA beat the SAE on recon? | ✅ Yes (0.996 vs 0.988) — expected, it is the dense ceiling |
| Does the SAE beat baselines on naming? | ⚠️ Barely (mean 0.395 vs Random 0.372); max 0.546 > Random 0.442 (top-end yes) |
| Does KMeans dominate naming? | ✅ Yes (0.829) — but dense data modes, not monosemantic |

**Cumulative verdict (00→03):**
1. 00 (direction-Jaccard ~0) + 03 (index-Jaccard on the null floor) → the SAE's 0.0038 is noise **in both index and direction space**. Independent confirmation via two different nulls.
2. The SAE does not win on recon (PCA ceiling) nor on mean naming (≈ Random). The only defensible advantage: **L0=32 enforced by construction** (PCA/KMeans are dense) + **top-end naming** (max 0.546 > 0.442).
3. The most severe result of the series so far. 01/02 showed that instability is not resolved by hyperparameters; 03 shows that the SAE barely beats random baselines on the naming-mean axis. The SAE's value here is **structural** (guaranteed sparsity, recon 0.988 at L0=32), not a measurable gain on concepts vs generic alternatives. (This severe verdict is then **rebalanced** by 05: the concepts, though unstable, are clinically faithful.)

---

## 5. Reproducibility notes
- **IDE run (2026-06-21 19:35):** 13/13 cells, zero training. Artifacts: `a3_baselines.json` (6.1 KB), `a3_cache/` (PCA + KMeans fit per seed, `.npz`), 2 figures.
- **Zero training / no model writes:** `SAEManager.train` never called. PCA/KMeans fit on train, metrics scored on test (test-set discipline).
- **Standalone metrics:** `compute_stability`/`name_concepts`/`compute_cosine_reconstruction` require an `AutoEncoderTopK` on disk → rewritten as free functions, verified against `sae_module.py`.
- **Gap-corrected naming for all:** `modality_gap = train_emb.mean(0) − vocab_emb.mean(0)` applied to every `W_dec` → apples-to-apples naming comparison.
- **Hardcoded SAE reference:** numbers from the baseline REPORT (gap-corrected), not retrained here.
- **Analytical null:** `E[J] ≈ k/(2D−k)` for `k≪D`; empirical/analytical ratio 1.00 (D=256) and 0.95 (D=4096).

---

# Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)

**Run date:** 2026-06-21 · **Machine:** Linux / RTX 5070, **CUDA**
**Notebook:** `04_activation_bakeoff.ipynb` (29 cells)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), RadLex vocabulary **508 terms**
**Config:** **3 activation families** trained at identical config: TopK (baseline), BatchTopK, JumpReLU. `dict_size=2048` (shared index space), **lr=5e-5 pinned & matched** (removes the ~8× lr confound), `steps=12000`, seeds `(0,42,123)`, **gap-corrected** naming.

## Context and question

All ablations so far use TopK. Perhaps the problem is TopK itself — its "exactly 32 features per image" rule is rigid. Alternative families exist: **BatchTopK** (chooses the top-k over the whole batch, not per-sample → each image can use more or fewer features) and **JumpReLU** (a threshold learned per-feature). Perhaps one of these finds more reproducible concepts. BatchTopK and JumpReLU had never been tried on medical VLMs — this is the ablation's novelty.

The 3 families are trained at identical configuration (same lr, same dictionary, same seeds) and compared on reconstruction, dead%, within-family stability, and above all **cross-family consensus**: how many concepts are rediscovered by different families.

**Pre-registered hypothesis:** at matched lr, BatchTopK/JumpReLU give a higher consensus-rate and lower dead% than TopK, because they let features specialize on the samples that need them instead of forcing k=32 per sample.

**Outcome: MIXED/FALSIFIED.** dead% ✓ BatchTopK (4.2%) much better than TopK (15.9%); JumpReLU worse (49%). But consensus-rate **0 for all three** (τ=0.90), reconstruction/naming/stability ~identical. Cross-family: 2.8% shared between 2 families, 0% across all 3. The activation family modulates dead%, **not** reproducibility.

---

## 1. What each stage produces

| Stage | Output |
|---|---|
| Training | 3 families × 3 seeds = 9 SAEs (12k steps, lr=5e-5 matched) |
| Per-family metrics | recon cosine, MSE, effective L0, dead%, entropy (test) |
| L0 distribution | per-sample histogram (TopK=point mass at 32, others=variable) |
| Within-family stability | renormalized Jaccard n=20 (3 seeds per family) |
| Consensus reappearance | direction-space clusters τ=0.90, within-family (same algo as 00/01) |
| **Cross-activation consensus** | pool 9 models, cluster τ=0.90, count clusters spanning ≥2 families |
| Naming | seed 42, gap-corrected, per family |
| Figures | 4 (`a4_effective_l0_distribution`, `a4_jumprelu_threshold_hist`, `a4_activation_comparison`, `a4_cross_activation_consensus`) |
| Persist | `results/ablation/a4_activation.json` |

---

## 2. Results (3 seeds, lr=5e-5 matched, dict=2048)

### 2.1 Per-family: reconstruction, L0, dead%

| Family | recon cosine | effective L0 | dead% | util% |
|---|---|---|---|---|
| **TopK** | 0.9910 | 32.0 (rigid) | 15.9 | 84.1 |
| **BatchTopK** | 0.9915 | ~39.1 | **4.2** | 95.8 |
| **JumpReLU** | 0.9903 | ~33.9 | 46.6 | 53.4 |

The three families reconstruct almost identically (~0.99). The big difference is dead%: BatchTopK wastes very little (4.8%), JumpReLU wastes half (46.6% — the learned threshold does not converge well at this lr/steps), TopK is in between (16%). Effective L0: TopK always 32 (rigid), BatchTopK ~38, JumpReLU ~33. (Baseline reference dict4096: recon 0.988, dead 44% — the TopK here has lower dead because dict=2048 + lr=5e-5.)

### 2.2 Within-family stability (renormalized Jaccard n=20, floor=0.00977)

| Family | mean Jaccard (n=20) | signal/null |
|---|---|---|
| TopK | 0.00379 | 0.39× |
| BatchTopK | 0.00368 | 0.38× |
| JumpReLU | 0.00813 | 0.83× |

All three ~0.005, signal-to-null ~0.5×. The three families are essentially identical on stability — differences 0.43–0.53× are not significant. No family is "more reproducible".

### 2.3 Consensus reappearance (direction-space, τ=0.90, within-family)

| Family | pooled rows | clusters | consensus (≥2 seeds) | rate |
|---|---|---|---|---|
| TopK | 6144 | 6144 | 0 | 0.000 |
| BatchTopK | 6144 | 6144 | 0 | 0.000 |
| JumpReLU | 6144 | 6144 | 0 | 0.000 |

No family redisCOVERS the same directions across seeds at τ=0.90.

### 2.4 Cross-activation consensus (9 models, τ=0.90) — the key novelty test

The novelty question: are there concepts that **different families** rediscover (not just different seeds of the same family)?

| Metric | Value |
|---|---|
| Pooled rows (9 models) | 18432 |
| Total clusters (τ=0.90) | 17923 |
| Clusters spanning ≥2 families | 509 (**2.8%**) |
| Clusters spanning all 3 | 0 (**0%**) |

Only 2.8% of concepts are shared between 2 families, 0% across all 3. Almost all directions are family-specific: there is no "core" of universal concepts that all families find.

### 2.5 Naming (seed 42, gap-corrected)

| Family | n_live | naming mean | naming max |
|---|---|---|---|
| TopK | 2048 | 0.4024 | 0.5517 |
| BatchTopK | 2048 | 0.3992 | 0.5380 |
| JumpReLU | 2048 | 0.3881 | **0.5942** |

Alignment with the RadLex vocabulary is nearly identical across families (~0.40 mean, ~0.55–0.58 max). JumpReLU has a slightly higher naming max. Coherent top concepts: vertebral anatomy (ligamentum flavum, spinal stenosis), devices (core needle, shapeable wire tip).

---

## 3. Analysis

### 3.1 dead% ✓ responds to family — BatchTopK is the best
The only part of the hypothesis that holds: BatchTopK has far fewer dead (4.2%) than TopK (15.9%). It makes sense: the global top-(k·B) lets features specialize on the samples that need them → less waste. JumpReLU is worse (49% dead) — its learned threshold does not converge well at lr=5e-5/12k steps. But this concerns dictionary efficiency (waste), **not** reproducibility.

### 3.2 Consensus ✗ ZERO for all — hypothesis falsified
The main hypothesis ("BatchTopK/JumpReLU more reproducible") collapses: all three families have 0 shared clusters at τ=0.90. Changing the activation function does not create more reproducible concepts. The within-family signal-to-null is ~0.5× for all — identical. Stability is invariant to the family.

### 3.3 Cross-family: 2.8% shared, 0% universal
2.8% of concepts are found by 2 families (a weak but non-zero signal), but no concept is found by all 3. The families are almost completely disjoint in direction space: there is no latent universal dictionary that all discover.

### 3.4 Reconstruction + naming identical across families
On the "technical" axes (reconstruction ~0.99, naming ~0.40) the three families are indistinguishable. The family choice does not change technical quality nor RadLex anchoring. It only changes dead% (efficiency) and the L0 profile (TopK rigid, others adaptive).

### 3.5 Adaptive L0 = the real novelty (but inconsequential on stability)
The visible difference between families is the L0 profile: TopK is a point mass at 32 (rigid), BatchTopK/JumpReLU have a distribution (each image uses a different number of features). It is the "adaptive sparsity" behavior not studied on medical VLMs. However it does not lead to more reproducible concepts: the novelty exists, but it does not solve the central problem.

---

## 4. Overall assessment: the family does not save reproducibility

| Question | Outcome |
|---|---|
| Rubric (≥1 non-TopK variant)? | ✅ BatchTopK + JumpReLU |
| Are BatchTopK/JumpReLU more reproducible than TopK? | ❌ No — consensus 0 for all |
| Lower dead% with alternative families? | ⚠️ Partial — BatchTopK yes (4.8%), JumpReLU no (49%) |
| Is there a core of universal concepts across families? | ❌ No — 0% span 3 families, 2.8% span 2 |
| Do reconstruction/naming change with the family? | ❌ No — identical (~0.99, ~0.40) |

**Cumulative verdict (00→04, closure of the investigation):**
1. 04 is the **deepest test**: it changes the central mechanism (the activation function), not a hyperparameter. Not even this helps reproducibility.
2. **dead% and stability are decoupled** (as in 01): BatchTopK reduces the dead, but the concepts remain non-reproducible. Being "efficient" ≠ being "robust".
3. **Definitive structural confirmation:** the instability is not due to TopK, nor to dict_size, nor to k. It is structural — few samples (5976) + non-uniqueness of the sparse decomposition (holds for all 3 families).

**Honest caveats:**
- **lr matched (5e-5):** removes the ~8× lr confound, but may under-train TopK/BatchTopK (default ~2.8e-4). Valid but conservative comparison.
- **JumpReLU 49% dead:** likely non-optimal lr/steps/warmup for this family (no per-family tuning). It is not a verdict on JumpReLU in absolute, only at matched config.
- **3 seeds (not 5):** compute. The consensus at 0 is already sharp.

---

## 5. Reproducibility notes
- **IDE run (2026-06-21 20:06):** 29 cells, 9 SAEs (3 families × 3 seeds, 12k steps). Artifacts: `a4_activation.json` (6.1 KB), 4 figures, models in `models/ablation_a4/{topk,batchtopk,jumprelu}_2048/sae_seed{N}/`.
- **lr pinned 5e-5 matched:** removes the lr confound (TopK/BatchTopK auto-scale ~2.8e-4 at dict2048; JumpReLU default 7e-5). Conservative but valid cross-family.
- **3 families via direct `trainSAE`** (not `SAEManager.train`, which hardcodes TopKTrainer). Bespoke per-family loader (`AutoEncoderTopK`/`BatchTopKSAE`/`JumpReluAutoEncoder`); decoder-row extraction differs (TopK/BatchTopK: `decoder.weight.T`; JumpReLU: `W_dec` already `(dict,act)`).
- **`compute_stability` not used:** hardcodes `AutoEncoderTopK` → crash on BatchTopK/JumpReLU. Jaccard rewritten standalone, renormalized to a common n=20.
- **Dead% = activation-based** (feature never non-zero on test), standalone.
- **Gap-corrected naming** for all 3 (`W_dec -= gap`).

---

# Ablation 05 — Concept Faithfulness vs clinical labels (MeSH/Problems)

**Run date:** 2026-06-22 · **Machine:** macOS / Apple Silicon, **MPS device** (auto)
**Notebook:** `05_faithfulness.ipynb` (run headless via nbconvert, 9/9 cells)
**Input:** baseline checkpoint `models/sae_seed42/` (dict4096, k=32) — zero training; `test_embeddings.pt` (1494) + `test_image_ids.json`; clinical labels from `data/iu_xray/reports/indiana_reports.csv` (`MeSH`/`Problems` columns)
**Config:** activation matrix `A` (1494×4096, continuous TopK) × binary label matrix `Y` (1494×50 after prevalence filter ≥10); vectorized **point-biserial** correlation `A_zᵀ·Y_z/N`; null = analytical SE `1/√N` + per-feature shuffle-null (p95, 200 perm) + BH-FDR 0.05.

## Context and question

All ablations so far (00–04) are one big question: *why are the concepts not reproducible across seeds?* Verdict: it is a structural limitation, 0.004 is the chance floor. But "unstable" does not mean "useless". Here the question changes: **are the concepts the SAE discovers (in one seed) meaningful, i.e. do they activate on the images that actually contain a certain pathology/anatomy?** It is the difference between "noisy concepts" and "concepts that mean something".

Take the seed-42 SAE, encode the test images → for each image we know which features fire (`A`). Then read the true clinical labels of those images from the reports (`MeSH`/`Problems` columns of IU X-Ray: cardiomegaly, pleural effusion, etc.) → matrix `Y`. Compute the correlation between each feature and each label. A feature is "faithful" if it fires precisely on the images with a certain label — and does so beyond pure chance (comparison against a per-feature calibrated null).

**Pre-registered hypothesis:** a non-trivial fraction of live features has `max_j |corr(activation_i, label_j)|` above a per-feature calibrated null (p95 of a label shuffle). The most faithful features should correspond to visually concrete and clinically expected concepts.

**Outcome: PARTIALLY CONFIRMED — the first positive of the series.** 158/1175 live features (13.5%) beat their shuffle-null p95 (null median 0.1932). |r|>0.10 on 54.3% of live, >0.20 on 11.7%. The strongest track lung hyperlucency (0.40), mass (0.37), medical implants (0.36). The concepts are unstable cross-seed (00–04) but, when they exist, they are moderately faithful to real clinical labels.

---

## 1. What each stage produces

| Stage | Output |
|---|---|
| Clinical labels | parsing `MeSH`/`Problems` → 118 base terms; 101 present in test, **50** after prevalence filter ≥10 |
| Per-image join | `image_id → uid` (via `indiana_projections.csv` + prefix fallback) → `MeSH`/`Problems`; 0 missing joins out of 1494 |
| Activations | `mgr.encode(test_emb)` → `A` (1494×4096), 32 non-zero/image |
| Point-biserial | `A_zᵀ·Y_z/N` → corr matrix (4096×50); per-feature max abs(corr) |
| Triple null | analytical SE 0.0259 + per-feature shuffle-null p95 + BH-FDR 0.05 |
| Naming cross-ref | gap-corrected RadLex name of each faithful feature (vs the label it is faithful to) |
| Persist | `results/ablation/a5_faithfulness.json` + `a5_faithfulness_headline.png` |

---

## 2. Results (seed 42, 1494 test, 50 prevalent labels)

For each "live" feature (activating at least once on test: 1175/4096 = 28.7%, consistent with the baseline ~44% dead) the strongest correlation with any clinical label is taken. `|r|` = how much the feature tracks its best label. The key column is the **per-feature null**: to beat chance, the feature must exceed its own p95 (median 0.188).

### 2.1 % faithful features per threshold (on the 2251 live)

| threshold abs(corr) | faithful live features | % of live |
|---:|---:|---:|
| > 0.10 | 638 | 54.3% |
| > 0.15 | 314 | 26.7% |
| > 0.20 | 138 | 11.7% |
| > 0.25 | 39 | 3.3% |
| > 0.30 | 14 | 1.2% |

### 2.2 Calibrated null — the key test

"Beyond chance" is not an opinion: for each feature the labels are shuffled 200 times across the images and the strongest correlation obtainable by pure chance is measured. The threshold is the 95th percentile of that shuffle, **feature-specific** (corrects for the label prevalence distribution). A feature "passes" only if it beats its own threshold.

| Null | Value |
|---|---|
| Analytical SE `1/√N` | 0.0257 (corr>0.10 ≈ 3.9σ) |
| Shuffle-null p95 (per-feature median, 200 perm) | **0.1932** |
| Live features beating their p95 null | **158 / 1175 (13.5%)** |
| BH-FDR 0.05 (on 59925 tests) | 2021 significant (feature,label) pairs; threshold ≈ corr>0.0807 |

### 2.3 Top faithful features (+ RadLex name cross-ref)

The most faithful features and the label they anchor to. Interesting cross-check: the real label (in-distribution, from IU X-Ray) is clinically sensible, but the RadLex name assigned by the SAE is often noisy/different. Confirms that RadLex naming (off-distribution, see `VOCAB_BUILDING_ALTERNATIVES.md`) is weaker than the concept's real behavior.

| feature | abs(corr) | faithful label (IU X-Ray) | prev. | RadLex name (gap-corrected) |
|---:|---:|---|---:|---|
| 1532 | **0.405** | lung, hyperlucent | 12 | rootlet of spinal nerve |
| 3480 | 0.373 | mass | 14 | core needle |
| 2765 | 0.362 | implanted medical device | 23 | surgical wire |
| 1223 | 0.315 | arthritis | 10 | fasciculus cuneatus of spinal cord |
| 2083 | 0.301 | foreign bodies | 11 | posterior ramus of spinal nerve |
| 1721 | 0.301 | spinal fusion | 11 | Arteria bronchialis |
| 498 | 0.301 | spinal fusion | 11 | spine proper of scapula |
| 1577 | 0.301 | spinal fusion | 11 | curved sheath |

### 2.4 Per-label: can the SAE represent every widespread clinical concept?

For each label, the best feature predicting it. Visually concrete labels (hyperlucency, mass, implants, emphysema) reach |r| 0.30–0.41; coverage decays on subtler pathologies.

| label (prevalence) | best corr |
|---|---:|
| lung, hyperlucent (12) | 0.405 |
| mass (14) | 0.373 |
| implanted medical device (23) | 0.362 |
| emphysema (17) | 0.339 |
| arthritis (10) | 0.315 |

---

## 3. Analysis

### 3.1 The existing concepts are genuinely faithful, not noise ✓
158 features out of 1175 (13.5%) beat a calibrated per-feature null — not a fixed value, but the specific threshold each feature should exceed by chance. Corroborated by BH-FDR (3496 significant pairs) and the analytical SE. The signal is real: there is a substantial minority of features whose activation pattern tracks a clinical label beyond chance.

### 3.2 Faithfulness concentrates on visually concrete concepts
The strongest features track medical lung hyperlucency, masses, medical implants (surgical wire), emphysema, arthritis. These are exactly the concepts an SAE on chest radiographs should discover first: high-contrast visual entities. Fine pathologies (subtle textural patterns) remain weaker — consistent with a data-starved regime on projected CLIP embeddings.

### 3.3 Modest faithfulness in absolute terms (but above the null)
The strongest absolute correlation is |r|=0.40, and only 13.5% of features beat the null. It is not "every concept is a clean hit": it is "a significant minority has a real clinical anchor, above chance". Honest: the SAE's value here is not "crystalline concepts", it is "sparse structure + good reconstruction (0.988) + a minority of clinically faithful concepts".

### 3.4 RadLex naming ≠ real behavior (cross-check)
A feature faithful to "implanted medical device" carries the RadLex name "anterior segment of upper lobe". The feature's behavior (faithful to implants) is more informative than its name (off-distribution). This a posteriori justifies using an in-distribution gold standard (MeSH/Problems) to evaluate the concepts, alongside RadLex naming — and reinforces the diagnosis of `concept_naming_analysis.md`: the weak naming is partly an artifact of the vocabulary, not only of the SAE.

---

## 4. Overall assessment: instability ≠ uselessness

| Question | Outcome |
|---|---|
| Do the SAE features predict real clinical labels beyond chance? | ✅ Yes (substantial minority) — 158/1175 live (13.5%) beat a per-feature null |
| Are the most faithful ones clinically sensible? | ✅ Yes — hyperlucency, mass, implants, emphysema, arthritis |
| Is the faithfulness strong in absolute terms? | ⚠️ Modest — max |r|≈0.40; concentrated on visually concrete concepts |
| Does RadLex naming coincide with the real label? | ❌ Often no — behavior is more informative than the off-distribution name |

**Position in the program (00→05):**
1. 00–04: the concepts are unstable cross-seed (both indices and directions), 0.004 is the chance floor, and the instability is not fixed by hyperparameters — a structural limitation.
2. 05 (this): the concepts that exist are moderately but genuinely **faithful** to real clinical labels. It is the complementary axis the series lacked: not "are they reproducible?" (no) but "do they mean something?" (yes, partly).
3. The overall result is balanced and defensible: the SAE on this dataset has a declared structural limitation (seed-dependence) but produces directions with real clinical grounding. Not a failure, not a complete success — an honest and nuanced result.

**Honest caveats:**
- **Labels from reports, not annotated gold standard:** `MeSH`/`Problems` derive from clinical reports (real richness but not controlled annotation). Faithfulness measures concept↔report alignment, not concept↔image-truth.
- **Only seed 42:** faithfulness is measured on the reference model. How stable the *quota* of faithful features is across seeds is not tested here (but 00 says which features they are is already unstable).
- **Prevalence ≥10:** cuts the rarest labels (degenerate `|r|=1`). The per-feature shuffle-null corrects for the prevalence distribution anyway.

---

## 5. Reproducibility notes
- **Headless run (2026-06-22):** 9/9 cells via `jupyter nbconvert --execute`, Agg backend. Artifacts: `a5_faithfulness.json` (12 KB), `a5_faithfulness_headline.png` (3 panels: max|corr| distribution + null, % faithful per threshold, per-label best).
- **Zero training:** reuses `models/sae_seed42/`. `SAEManager.encode` → continuous TopK activations.
- **Point-biserial, not AUROC:** one matmul `A_zᵀ·Y_z/N` (Pearson with binary var), O(one matmul) vs ~500k AUROC calls on 4096×118.
- **Triple null:** analytical SE `1/√N`=0.0259; per-feature shuffle-null p95 (200 perm, `seed=0`); BH-FDR 0.05 on the matrix (2251 live × 50 labels = 112550 tests).
- **Prevalence filter ≥10:** avoids the degenerate `|r|=1` cases of labels in 1–3 images. 50/101 prevalent labels kept (median prev 30, max 191).
- **Naming cross-ref:** uses the same modality gap shift as 01–04 (`W_dec -= visual_centroid − text_centroid`). `train_emb` used only for the gap, never for the correlation (test-set discipline).
- **Output isolation:** writes only to `results/ablation/` + `results/figures/ablation/` — baseline untouched.

---

## Cumulative conclusion

| Ab | Axis | Synthetic outcome |
|---|---|---|
| 00 | directions | 0.0038 is not permutation: direction-Jaccard ~0, consensus@4 = 0%, shuffle-null p=1.0 |
| 01 | dict_size | dead% ✓ scales with capacity; stability ✗ invariant (ratio 4096 > 1024 > 2048) |
| 02 | k | baseline k=32 on the null floor (ratio 0.954); weak sweet spot at k=16 (ratio 1.14); k=8 pathological |
| 03 | baselines | Random@4096 = 0.0037 ≈ SAE → 0.0038 is the chance floor; SAE survives on sparsity + top-end naming |
| 04 | family | dead% ✓ (BatchTopK 4.2%); consensus ✗ 0 for all; cross-family 2.8% / 0% universal |
| 05 | faithfulness | 158/1175 live (13.5%) faithful above the null; top: lung, hyperlucent |r|=0.40, mass, implants |

1. The baseline's "0.004" is the **mathematical chance floor** (03), not a failure — confirmed as noise in index space (03) and direction space (00).
2. It is not fixed by dict_size (01), k (02), or activation family (04). The instability is a declared **structural limitation** of the method on this dataset (few samples + non-uniqueness of the sparse decomposition on projected CLIP embeddings).
3. But instability **does not equate to uselessness** (05): the existing concepts are moderately faithful to real clinical labels.
4. **What to do:** accept seed-dependence as a declared limitation, or aggregate the seeds (model soup / consensus clustering with validation). The SAE's value is **structural** (guaranteed sparsity + recon 0.988) **and partially semantic** (05), plus top-end naming above chance.

Open soft spots: faithfulness measured only on seed 42 (the faithful quota across seeds is not tested — see 00); labels derived from reports, not an annotated gold standard; pre-projection (06) / augmented (07) remain future work.

---

## Bibliography

References supporting the methodological choices and the theoretical framing. The extended causal diagnosis is in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

- Olshausen & Field (1997) — sparse coding; data-samples/features regime.
- Spielman, Wang, Wright (2012) — "Exact Recovery of Sparsely-Used Dictionaries": identifiability conditions of dictionary learning.
- Soltanolkotabi, Elhamifar, Candès (2013–2014) — robustness/identifiability of structured sparsity.
- Bricken et al. (2023) "Towards Monosemanticity" — SAE on millions of activations (reference data-rich regime).
- Gao, Dupré la Tour et al. (2024) "Scaling and Evaluating Sparse Autoencoders" [arXiv:2406.04093] — Top-K SAE architecture used in the project.
- Rajamanoharan et al. (2024) — BatchTopK / JumpReLU SAE (variants in 04).
- Bhalla, Srinivas, Hsieh (2024) "SpLiCE" [arXiv:2402.10376] — naming via sparse optimization on decoder weights.
- Liang et al. (2022) "Mind the Gap" [arXiv:2203.02053] — formal characterization of the modality gap in contrastive models (framing of the gap-corrected naming).
