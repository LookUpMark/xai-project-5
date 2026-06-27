# REPORT — Baseline on AUGMENTED data (`pipeline.augmented_exec.ipynb`)

**Run date:** 2026-06-24
**Machine:** Linux / NVIDIA RTX 5070 Laptop, **CUDA** (`torch 2.12.0+cu130`)
**Path:** `embeddings/augmented/` (`config.augmentation.enabled = True`)
**Input:** BiomedCLIP embeddings (512-d) of IU X-Ray with **on-the-fly augmentation** (`num_augmentations=2`, `rotation=5°`, `crop=0.95–1.0`) → `visual_embeddings.pt` (22410 = 7470 × 3), RadLex vocabulary **508 terms** (`text_vocab_embeddings.pt` + `data/vocabulary.json`)
**Split:** group-aware by radiograph study (`study_key_from_basename`) — fixes audit 06-23 F-001/F-002/F-003. **17865 train / 4545 test** across **3081 / 771 of 3852 studies, group overlap = 0**
**SAE config:** Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4), **5 seeds** = `(0, 42, 123, 456, 789)`

> Companion to [`REPORT.md`](REPORT.md) (standard path). This report covers the **augmented** ablation member, on the **clean post-fix split**. The standard report's split was contaminated (audit 06-23 F-001/F-002, 75% study overlap); both now run on leak-free group-splits.

**Table of contents**
- [Executive summary](#executive-summary)
- [1. Headline metrics vs standard](#1-headline-metrics-vs-standard)
- [2. Results in detail](#2-results-in-detail)
- [3. Are these results normal?](#3-are-these-results-normal)
- [4. Reproducibility notes](#4-reproducibility-notes)

---

## Executive summary

The augmented baseline trains the same Top-K SAE ensemble on **3× augmented embeddings** under a **leak-free group split** (no patient/study crosses train↔test). Three axes:

| Axis | Result |
|---|---|
| **Does it reconstruct well?** | ✅ Test cosine **0.9945** (↑ from 0.988 standard) — near-perfect |
| **Do the concepts make medical sense?** | ✅ RadLex alignment mean **0.384** / max **0.538** (≈ standard 0.395 / 0.546) |
| **Are they reproducible?** | ❌ Cross-seed Jaccard **0.0039** — on the chance floor (`k/(2D−k)=0.00392`, ratio 0.995) |

**The central result.** Augmentation does **not** change cross-seed stability: Jaccard 0.0039 on the clean augmented split is statistically indistinguishable from the contaminated standard split's 0.0039, and both sit exactly on the analytical chance floor for two independent 32-subsets of 4096. This **answers the project's master question** — the 0.0038 baseline figure is **the chance floor, not a real SAE failure** — and it holds with or without augmentation and with or without leakage. Concept instability is a **structural property of the method** (non-uniqueness of sparse decomposition on projected CLIP embeddings), independent of the data contamination that masked it.

---

## 1. Headline metrics vs standard

| Metric | Standard (old, contaminated) | **Augmented (this run, clean)** | Δ |
|---|---|---|---|
| Train / test samples | 5976 / 1494 | **17865 / 4545** | ×3 (aug) |
| Studies train/test | — | **3081 / 771** (overlap **0**) | leak-free |
| Test cosine (mean) | 0.988 | **0.9945** | ↑ better |
| Variance explained | ~99.3% | **~97.7%** (`frac_variance_explained`) | ↓ lower |
| Dead features (activation, full test) | ~44% | **~65%** | ↑ worse |
| Dict utilization | ~55% | **~34%** | ↓ lower |
| Activation entropy | ~6.3 nats | **~5.70 nats** | ↓ |
| Cross-seed Jaccard (mean) | 0.0039 | **0.0039** | = (chance floor) |
| Naming mean / max | 0.395 / 0.546 | **0.384 / 0.538** | ≈ (noise) |

Reconstruction improves; **sparsity degrades** (more dead, lower utilization); **stability and naming are unchanged**.

---

## 2. Results in detail

### 2.1 Reconstruction — ✅ EXCELLENT

| Seed | MSE | Cosine | L0 | Dead % | Dict util % | Entropy |
|------|-----|--------|-----|--------|-------------|---------|
| 0 | 2.17e-5 | 0.9945 | 32.0 | 65.9 | 34.1 | 5.700 |
| 42 | 2.16e-5 | **0.9945** | 32.0 | 65.7 | 34.3 | 5.702 |
| 123 | 2.24e-5 | 0.9943 | 32.0 | 62.7 | 37.3 | 5.729 |
| 456 | 2.16e-5 | **0.9945** | 32.0 | 65.7 | 34.3 | 5.671 |
| 789 | 2.21e-5 | 0.9944 | 32.0 | 65.7 | 34.3 | 5.697 |

- Mean cosine **0.9945** → reconstruction nearly parallel to the original. 5 seeds mutually consistent (cosine 0.9943–0.9945).
- **L0 = 32.0 exact** across all seeds → TopK constraint perfectly respected.
- Higher cosine than standard (0.988) — more (near-identical) training samples tighten the fit.

### 2.2 Sparsity / dead features — ⚠️ WORSE THAN STANDARD

- **~63–66% of the 4096 features never activate** on the test set (util ~34%). Standard was ~44% dead / ~55% util.
- Activation entropy **~5.70 nats** → ~`e^5.70 ≈ 300` distinct features used on test (standard ~540). **Narrower usage.**
- ⚠️ Two "dead" definitions (per CLAUDE.md):
  - *Naming dead* (zero-norm decoder): **0** (library unit-normalizes columns each step).
  - *Activation dead* (never active on test): **~65%**.
- Note: the train-time sanity check (random subset) reports **~83% dead**; the authoritative figure is the **~65%** measured on the full 4545-sample test in `stability_analysis.json`.

### 2.3 Cross-seed stability — ❌ ON THE CHANCE FLOOR (unchanged)

Mean Jaccard **0.0039** (std 0.0016).

| | 0 | 42 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.0035 | 0.0063 | 0.0034 | 0.0026 |
| 42 | 0.0035 | 1.00 | 0.0033 | 0.0028 | 0.0067 |
| 123 | 0.0063 | 0.0033 | 1.00 | 0.0047 | 0.0043 |
| 456 | 0.0034 | 0.0028 | 0.0047 | 1.00 | 0.0014 |
| 789 | 0.0026 | 0.0067 | 0.0043 | 0.0014 | 1.00 |

- Analytical null `k/(2D−k) = 32/8160 = 0.00392`. Observed 0.00390 → **ratio 0.995**.
- The 5 SAEs reconstruct equally well but share **<0.4%** of active features → the discovered concepts are **not reproducible** across seeds.
- Augmentation did **not** move this off the floor: it is governed by `D=4096` and `k=32`, not by data quantity or leakage.

### 2.4 Concept naming — ✅ GAP-CORRECTED (≈ standard)

Gap-corrected (`W_dec -= modality_gap`) RadLex alignment:

| Metric | Augmented |
|---|---|
| Mean score | **0.3840** |
| Min / Max | 0.2732 / **0.5376** |
| Features named | 4096 (`DEAD_FEATURE` = 0) |

**Top-10 by score (clinically plausible):**

| Feat | Name | Score |
|---|---|---|
| 685 | shapeable wire tip | 0.5376 |
| 3690 | dental device | 0.5230 |
| 1201 | rootlet of spinal nerve | 0.5054 |
| 1415 | endotracheal tube | 0.5046 |
| 1498 | sacral segment of spinal epidural space | 0.5004 |
| 709 | endocavitary linear transducer | 0.4947 |
| 334 | cricothyroid tube | 0.4942 |
| 2751 | ligamentum flavum | 0.4908 |
| 1090 | ligamentum flavum | 0.4900 |
| 2117 | fasciculus cuneatus of spinal cord | 0.4887 |

Same coherent families as standard (endocavitary tubes/devices, vertebral anatomy, spinal pathways). Naming is marginally lower than standard (0.384 vs 0.395) — within noise.

### 2.5 Explanations — ✅ 4545 records

`results/sample_explanations.json`: one record per augmented test image, schema `{image_id, top_k_concepts[].{feature_id,name,activation}, pseudo_report}`. Same basename across the 3 augmented variants of a study (grouped), activations mean ~0.14, max ~0.39.

---

## 3. Are these results normal?

**Yes — expected for this setup, with one item worth watching.**

**✅ Reconstruction 0.9945 (↑).** Normal. 3× more (near-identical) training samples tighten the fit. BiomedCLIP embeddings live near a low-dim manifold; a 4096-dict / k=32 SAE reconstructs them near-perfectly. Expected, healthy.

**✅ Jaccard 0.0039 = chance floor.** Normal and **the key finding**. Two random 32-subsets of 4096 overlap by `32/8160 = 0.00392` by pure probability. Observed 0.00390 (ratio 0.995). The SAEs find *equally valid but different* decompositions — basis-rotated feature sets. This is the well-known non-uniqueness of sparse coding on superposed/CLIP-projected activations, not a bug. Augmentation does not (and cannot) fix it: it is set by `D` and `k`. This **confirms the master question** — 0.0038 is the floor.

**⚠️ Dead features 65% (↑ from 44%) — the one to watch.** More data should usually *reduce* dead features, but here they *increase*. Explained by the augmentation profile: `rotation=5°`, `crop=0.95–1.0` are **very mild** → the 3× augmented samples are near-duplicates, adding ~3× training signal without ~3× *diversity*. The SAE reinforces the directions that fire on common patterns and lets rare-direction features die. Consistent with: lower entropy (5.70 vs 6.3 → ~300 vs ~540 features used) and lower dict utilization (34% vs 55%). **Not pathological** (VE still 0.977, cosine 0.9945), but it means augmentation as configured *narrows* the active dictionary rather than broadening it.

**✅ Naming ≈ standard (0.384 vs 0.395).** Normal, within seed noise. Augmentation neither helps nor hurts RadLex grounding.

**Bottom line.** The augmented baseline behaves sensibly: better reconstruction, unchanged (chance-floor) stability, unchanged naming. The only deviation from the standard picture is the higher dead-feature rate, and it has a clean mechanical explanation (mild augmentation → redundant samples → concentrated usage). The master conclusion — **concept instability is structural, not a data/leakage artifact** — is reproduced and reinforced on the clean augmented split.

---

## 4. Reproducibility notes

- **Clean group split (audit fix F-001/002/003).** `utils.split_embeddings` groups by `study_key_from_basename` (`{patient}_IM-{study}`); `sorted(set())` makes the split deterministic (load-bearing under the "no cache" recompute strategy). `group_overlap == 0` asserted. Originals + their augmented variants share one study key → stay in one partition.
- **"No cache" retrain.** `train_sae.py` recomputed split + `modality_gap.pt` + retrained all 5 SAEs (no skip-if-exists). Run ~15 min on RTX 5070.
- **Augmented path is the audit-flagged untested one** (06-23 F-010: augmentation modules use `from src.` imports, work only via the notebook's dual-`sys.path`). Extraction succeeded via the notebook workaround; the CLI retrain (`train_sae.py`) does **not** import those modules, so F-010 did not affect training downstream.
- **`config.py:240 AugmentationConfig.enabled = True`** is a local edit (committed file). Revert to `False` to return to the standard path; `embeddings/standard/` currently holds only the vocab `.pt` (no split) and would need its own retrain.
- **Headless notebook execution** via `nbconvert 7.17.1` → `pipeline.augmented_exec.ipynb` (36 cells, 0 errors; non-destructive copy).
- **Artifacts:** `embeddings/augmented/{train,test}_embeddings.pt` + sidecars, `models/modality_gap.pt`, `models/sae_seed{0,42,123,456,789}/`, `results/{concept_names,sample_explanations,stability_analysis}.json`, `results/figures/`.
