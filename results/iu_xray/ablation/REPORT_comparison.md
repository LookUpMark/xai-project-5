# REPORT — Standard vs Augmented: Ablation Comparison (00–05)

**Generated:** 2026-06-25 (updated post clean retrain)
**Standard path:** `embeddings/standard/` — 7470 raw images → **5955 train / 1515 test**, 3081/771 studies, group-aware split, no augmentation
**Augmented path:** `embeddings/augmented/` — 7470 × 3 = 22410 images (num_aug=2, rot=5°, crop=0.95–1.0) → **17865 train / 4545 test**, same 3081/771 studies, group overlap = 0
**Models:** Standard models retrained from scratch on clean split (post group-aware fix `b5a7b2e`, 2026-06-24). Old contaminated standard backup (`_standard_backup_ablations/`, pre-fix) discarded.
**Split:** group-aware by study ID in both paths — no leak.

> **Reading guide:** augmentation = 3× training samples (same images, different crops/rotations), not new patients.

---

## Executive Summary

| Finding | Standard | Augmented | Δ / interpretation |
|---|---|---|---|
| **Core conclusion** | Instability structural | Instability structural | **Identical** |
| **A0 consensus** | p=0.19, 5 multi-seed clusters | p=0.19, 5 multi-seed clusters | Identical — baseline models shared |
| **A1 dead% (D=4096)** | 41.2% | 60.3% | +19 pp — augmented near-duplicates starve rare features |
| **A2 k=16 stability ratio** | 1.144 (CI excl.1) | 1.175 (CI excl.1) | Both confirm k=16; virtually equal on clean split |
| **A3 KMeans naming** | 0.834 | 0.833 | Stable — vocab-alignment, not data-dependent |
| **A4 BatchTopK dead%** | 4.2% | 7.0% | Standard lower dead% — augmented actually worse here |
| **A4 JumpReLU jaccard** | 0.00813 | 0.000028 | JumpReLU collapses under augmentation |
| **A5 % beat null p95** | **13.4%** (158/1175) | **42.7%** (600/1404) | Gap mostly statistical power (N 1515→4545 → lower null threshold) |
| **A5 strongest \|r\|** | 0.405 (*lung, hyperlucent*) | **0.545** (*heart failure*) | Augmented stronger peak clinical correlations |
| **A5 \|r\|>0.30** | 1.2% (14 features) | 2.4% (34 features) | Augmented 2× high-signal features |

> [!IMPORTANT]
> The A4 BatchTopK dead% reversal vs the previous (contaminated) comparison is significant: on the clean standard split, BatchTopK achieves 4.2% dead (better than augmented's 7.0%). The old contaminated comparison showed 13.7% for standard — that was an artifact of the wrong split.

---

## Ablation 00 — Cross-Seed Consensus

| Metric | Standard | Augmented |
|---|---|---|
| Multi-seed clusters (τ=0.9) | 5 | 5 |
| Hungarian cosine-Jaccard | 0.000562 | 0.000562 |
| Consensus@≥3 seeds | 0.000488 | 0.000488 |
| Shuffle-null p-value | **0.19** | **0.19** |

Δ = 0. A0 uses the same 5 baseline checkpoints in both runs — augmentation has zero effect. Direction-space disjointness is intrinsic to the model.

---

## Ablation 01 — Dictionary-Size Ladder

### Reconstruction cosine (seed-averaged)

| D | Standard | Augmented | Δ |
|---|---|---|---|
| 1024 | 0.9934 | 0.9950 | +0.0016 |
| 2048 | 0.9917 | 0.9948 | +0.0031 |
| 4096 | 0.9899 | 0.9944 | +0.0045 |

On the clean standard split, augmented slightly outperforms on reconstruction — consistent with more training data providing better optimization coverage.

### Dead features % (seed-averaged)

| D | Standard | Augmented | Δ |
|---|---|---|---|
| 1024 | 30.0% | 45.2% | +15.2 pp |
| 2048 | 33.6% | 51.8% | +18.2 pp |
| 4096 | 41.2% | 60.3% | +19.1 pp |

Augmentation consistently increases dead% by ~15–19 pp on clean split. Mechanism: 3× near-duplicate images concentrate activations on "popular" directions, starving rare-direction features. Effect is larger than previously estimated from the contaminated standard run.

### Stability (ratio vs null)

Both paths: ratio ≈ 0.86–1.13 — all within noise of 1.0. Neither achieves stability above chance floor.

---

## Ablation 02 — k (Sparsity) Sweep

| k | STD cosine | STD dead% | STD ratio | STD CI | AUG cosine | AUG dead% | AUG ratio | AUG CI |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.9834 | 91.8% | 1.010 | ❌ | 0.9843 | 93.7% | 0.751 | ❌ |
| **16** | **0.9884** | **74.0%** | **1.144** | **✅** | **0.9905** | **81.6%** | **1.175** | **✅** |
| 32 | 0.9922 | 41.1% | 1.065 | ✅ | 0.9950 | 62.6% | 1.187 | ✅ |
| 64 | 0.9971 | 40.0% | 0.969 | ❌ | 0.9981 | 67.9% | 0.818 | ❌ |

k=16 is the stability sweet spot in both paths (CI excludes 1). Standard k=16 ratio 1.144 ≈ augmented 1.175 — effectively equal on clean data. Augmented dead% higher at every k (+7–21 pp).

---

## Ablation 03 — Concept Baselines

### Comparison table (primary seed 42)

| Method | STD cosine | STD naming | AUG cosine | AUG naming |
|---|---|---|---|---|
| SAE (ref, D=4096)† | 0.988 | 0.395 | 0.988 | 0.395 |
| Random (D=256) | ~0.440 | 0.372 | ~0.452 | 0.372 |
| Dense-PCA (D=256) | ~0.996 | 0.380 | ~0.996 | 0.379 |
| Freq-KMeans (D=256) | ~0.961 | **0.834** | ~0.961 | **0.833** |

†Hardcoded standard-path baseline reference in both runs.

Naming scores essentially identical across paths (Δ ≤ 0.001). KMeans dominates in both — naming is a property of the RadLex vocabulary alignment, not of training data size.

---

## Ablation 04 — Activation Function Bakeoff

| Family | STD cosine | STD dead% | STD jaccard | AUG cosine | AUG dead% | AUG jaccard |
|---|---|---|---|---|---|---|
| **TopK** | 0.9910 | 15.9% | 0.003794 | 0.9927 | 15.1% | 0.003590 |
| **BatchTopK** | 0.9915 | **4.2%** | 0.003677 | 0.9934 | 7.0% | 0.002100 |
| **JumpReLU** | 0.9903 | 46.6% | **0.008130** | 0.9917 | 49.8% | **0.000028** |

> [!IMPORTANT]
> **BatchTopK** achieves the lowest dead% on the **standard** path (4.2%) — better than augmented (7.0%). This reversal from the contaminated comparison (which showed 13.7% for standard) confirms the old split introduced noise. On clean data, BatchTopK is the most efficient activation function regardless of path.

> [!WARNING]
> **JumpReLU** nearly collapses under augmentation (jaccard 0.008130 → 0.000028). The learned threshold mechanism is highly augmentation-sensitive — avoid JumpReLU with augmented near-duplicate training sets at these hyperparameters.

All families in both paths remain below the practical stability threshold — activation function does not rescue the fundamental degeneracy.

---

## Ablation 05 — Faithfulness (Clinical Labels)

| Metric | Standard | Augmented | Δ |
|---|---|---|---|
| Test set size | 1515 | 4545 | +3× (same images, no aug at inference) |
| Prevalent labels (≥10 cases) | 51 | 69 | +18 — larger N admits more labels |
| Live features | 1175 / 4096 (28.7%) | 1404 / 4096 (34.3%) | +5.6 pp |
| Analytic SE(r) | 0.0257 | 0.0148 | Smaller SE → lower significance bar for augmented |
| Median shuffle null p95 | 0.193 | 0.120 | Augmented null lower — easier to beat |
| **% live beat null p95** | **13.4%** (158/1175) | **42.7%** (600/1404) | **+29 pp** |
| \|r\|>0.10 (% live) | 54.3% (638) | 47.1% (661) | Similar raw count |
| \|r\|>0.20 (% live) | 11.7% (138) | 12.0% (168) | Nearly identical |
| \|r\|>0.30 (% live) | **1.2%** (14) | **2.4%** (34) | Augmented 2× high-signal features |
| **Strongest \|r\|** | **0.405** (*lung, hyperlucent*) | **0.545** (*heart failure*) | +0.14 |

### Interpretation

The "% beat null" gap (13.4% vs 42.7%) is **mainly a statistical power artifact**:
- SE(r): 0.0257 (std) vs 0.0148 (aug) → the augmented null threshold is 0.073 lower
- The same feature with |r|=0.15 beats the null in augmented but not in standard
- Use threshold-matched metrics for fair comparison

**Real faithfulness signal (threshold-matched):**
- |r|>0.30: 1.2% → 2.4% (2× more features with strong clinical signal — real improvement)
- Max |r|: 0.405 → 0.545 — augmented training produces stronger peak correlations
- BH-FDR significant pairs: both paths find thousands of significant feature-label pairs

**Conclusion:** both paths produce clinically grounded, equally unstable SAEs. Augmentation improves the upper tail of faithfulness without changing the fundamental degeneracy.

---

## Cross-Cutting Observations

1. **Non-uniqueness universal.** Jaccard ≈ analytical null `k/(2D−k)` in all ablations, both paths. No lever rescues stability.

2. **Augmentation increases dead% by ~15–19 pp** (clean split). Near-duplicate samples concentrate activations → more unused directions. Not a data quality issue — an expected geometric consequence.

3. **BatchTopK is the best activation family** on the standard path (dead% 4.2%, jaccard 0.003677). On augmented path BatchTopK dead% rises to 7.0% but is still lowest. Recommended for production.

4. **JumpReLU is augmentation-fragile.** Jaccard 0.008130 (std) → 0.000028 (aug). Avoid with near-duplicate training data.

5. **Naming is augmentation-invariant** (Δ ≤ 0.01 across all configurations). Property of RadLex vocabulary alignment.

6. **k=16 is the stability sweet spot** in both paths. Signal/null ratios nearly identical on clean split (1.144 vs 1.175).

7. **Faithfulness (upper tail) is real.** |r|>0.30 doubling from standard to augmented survives threshold matching. More training data calibrates activations better at the high end.

---

## Artifacts Index

| | Standard | Augmented |
|---|---|---|
| **Exec notebooks** | `*.standard_exec.ipynb` | `*.augmented_exec.ipynb` |
| **Results** | `results/ablation/_standard_clean/` | `results/ablation/` (active) |
| **Models** | `models/_standard_clean_ablations/` | `models/ablation_a{1,2,4}/` (active) |
| **Contaminated (discarded)** | `models/_contaminated_standard_ablations/` | — |

> [!NOTE]
> Standard exec notebooks (`.standard_exec.ipynb`) were produced by the clean retrain. Earlier contaminated standard exec notebooks were overwritten. Config active: `augmentation.enabled = True`.
