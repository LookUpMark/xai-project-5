# Implementation Plan v3.0 — Project 5 XAI 2025/26
## Unsupervised Concept Discovery for Medical VLMs

> **Version:** 3.0 (post-audit reframe)
> **Supersedes:** `docs/archive/IMPLEMENTATION-PLAN-v1-original.md`
> **Driven by:** `docs/design/PROJECT-STRATEGY.md` v2.0, `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md`, `docs/audits/ML-AUDIT-2026-06-25.md`
> **Status:** Active plan — governs all new implementation work.

---

## Architecture Overview

```
Methods:
  Baseline   → SAE TopK on 512-d projected embedding  [done — re-narrated as failure case]
  Main A     → SAE on 768-d pre-projection hidden state [to implement]
  Main B     → SPLiCE — sparse decomposition on RadLex  [to implement, start here]
  Extension  → Structured concept organisation           [to implement]
  Evaluation → MedGemma LLM judge                        [code done, pending M-007 fixes]
```

**Backbone:** `chuhac/BiomedCLIP-vit-bert-hf` — ViT-B/16, hidden 768-d, projected 512-d.
**Dataset:** IU X-Ray — 7,470 images; train/test split group-aware.
**Stack:** PyTorch + HuggingFace Transformers + dictionary-learning + LangGraph + MedGemma.

---

## Repository Structure

```
xai-project-5/
├── data/
│   └── iu_xray/
│       ├── images/images_normalized/   # PNG radiographs
│       └── reports/                    # indiana_reports.csv, indiana_projections.csv
├── embeddings/
│   ├── standard/                       # 512-d projected embeddings (baseline)
│   │   ├── visual_embeddings.pt        # (7470, 512) L2-normalised
│   │   ├── train_embeddings.pt / test_embeddings.pt
│   │   ├── visual_image_ids.json       # PNG basenames (row-aligned)
│   │   └── text_vocab_embeddings.pt   # (508, 512) RadLex text embeddings
│   └── standard_hidden/               # 768-d hidden state embeddings [Path A]
│       ├── visual_embeddings_768.pt
│       ├── train_embeddings_768.pt / test_embeddings_768.pt
│       └── visual_image_ids.json
├── models/
│   ├── modality_gap.pt                 # visual_centroid - text_centroid
│   ├── sae_seed{0,42,123,456,789}/     # trainer_0/ae.pt (baseline 512-d)
│   ├── ablation_{a0..a5}/             # ablation models (baseline)
│   └── sae_hidden_seed{0,42,...}/     # [Path A] 768-d SAE models
├── results/
│   ├── concept_names.json              # {feat_id: {name, score}} baseline
│   ├── sample_explanations.json        # per-sample explanations, baseline
│   ├── ablation/                       # ablation results (baseline)
│   ├── spliece/                        # [Path B] SPLiCE outputs
│   └── sae_hidden/                     # [Path A] SAE-768 outputs
├── src/
│   ├── config.py                       # all configs (SAEConfig, EmbeddingConfig, ...)
│   ├── utils.py                        # load_vlm, set_global_seed, load_tensor, ...
│   ├── embedding_extraction/
│   │   └── extract_embeddings.py       # 512-d (existing) + 768-d branch [Path A]
│   ├── vocabulary_building/
│   │   └── build_vocabulary.py
│   ├── autoencoder/
│   │   ├── sae_module.py               # SAEManager facade
│   │   ├── train_sae.py                # prepare_split + train_single
│   │   ├── concept_naming.py           # cosine naming + gap correction
│   │   ├── generate_explanations.py    # per-sample pseudo-reports
│   │   ├── stability_analysis.py       # cross-seed Jaccard
│   │   └── contracts.py / protocols.py / ...
│   ├── concept_discovery/              # NEW [Paths B + Extension]
│   │   ├── __init__.py
│   │   ├── spliece.py                  # [Path B] SPLiCE implementation
│   │   └── organize.py                 # [Extension] clustering + hierarchy
│   └── evaluate_llm_judge.py           # LangGraph + MedGemma judge
├── notebooks/
│   ├── vlm/
│   │   └── extract_embeddings.ipynb   # 512-d (existing); add 768-d mode
│   └── autoencoder/
│       ├── baseline/pipeline.ipynb
│       ├── ablation/0{0..5}_*.ipynb
│       ├── 07_spliece.ipynb            # [Path B]
│       ├── 06_concept_organization.ipynb # [Extension]
│       └── 08_sae_hidden.ipynb         # [Path A]
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── CLAUDE.md
├── AGENTS.md
├── HANDOFF.md
└── requirements.txt
```

---

## Priority 0 — Judge Fix (M-007) — Independent, parallel

**Files:** `src/evaluate_llm_judge.py`, `src/config.py`

Prerequisite for any faithfulness number. From `ML-AUDIT-2026-06-24.md`:

| Fix | Issue | Change |
|---|---|---|
| **F-001** | Verb/label mismatch in prompt | Map prompt verbs (SUPPORTS/CONTRADICTS/AMBIGUOUS) consistently to labels (Aligned/Unaligned/Uncertain) |
| **F-002** | Frozen on resume | On `--resume`, retry failed/missing pairs, not all pairs |
| **F-003** | Non-reproducible temperature | Set `temperature=0.0` + `seed` kwarg explicitly |
| **F-007** | No `JudgeConfig` | Add `JudgeConfig` frozen dataclass to `config.py`; wire into `evaluate_llm_judge.py` |

**Test:** existing `tests/unit/test_llm_judge.py` (currently uncollected due to langgraph).

---

## Priority 1 — Path B: SPLiCE (implement first)

**New file:** `src/concept_discovery/spliece.py`
**New notebook:** `notebooks/autoencoder/07_spliece.ipynb`

### Algorithm

```python
# src/concept_discovery/spliece.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit

@dataclass(frozen=True)
class SpliCEConfig:
    k: int = 32                    # number of concepts per image
    use_gap_correction: bool = True
    vocab_path: Path = Path("data/vocabulary.json")
    vocab_emb_path: Path = Path("embeddings/standard/text_vocab_embeddings.pt")
    gap_path: Path = Path("models/modality_gap.pt")
    output_dir: Path = Path("results/spliece")


def decompose_image(
    image_emb: torch.Tensor,      # (512,) L2-normalised
    vocab_emb: torch.Tensor,      # (508, 512) text embeddings
    gap: torch.Tensor | None,     # (512,) modality gap vector
    k: int = 32,
) -> torch.Tensor:
    """Return sparse coefficient vector (508,) via OMP."""
    emb = image_emb.clone()
    if gap is not None:
        emb = emb - gap            # modality-gap correction
    # OMP: solve min ||emb - vocab_emb.T @ c||  s.t. nnz(c) <= k, c >= 0
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    # Note: OMP from sklearn does not enforce non-negativity; use Lasso(positive=True)
    # as alternative. See implementation notes below.
    X = vocab_emb.numpy()          # (508, 512) — dictionary atoms as rows
    y = emb.numpy()                # (512,) — target
    omp.fit(X.T, y)                # X.T is (512, 508); solve y = X.T @ c
    coeffs = torch.from_numpy(omp.coef_).float()  # (508,)
    coeffs = coeffs.clamp(min=0)   # enforce non-negativity post-hoc
    return coeffs


def run(config: SpliCEConfig, test_embeddings: torch.Tensor,
        image_ids: list[str], vocab_terms: list[str]) -> list[dict]:
    """Decompose all test images; return per-image concept lists."""
    vocab_emb = torch.load(config.vocab_emb_path, weights_only=True)
    gap = None
    if config.use_gap_correction:
        gap = torch.load(config.gap_path, weights_only=True)

    results = []
    for i, (emb, img_id) in enumerate(zip(test_embeddings, image_ids)):
        coeffs = decompose_image(emb, vocab_emb, gap, k=config.k)
        top_k = coeffs.topk(config.k)
        concepts = [
            {"term": vocab_terms[idx], "coefficient": float(val)}
            for idx, val in zip(top_k.indices.tolist(), top_k.values.tolist())
        ]
        pseudo_report = "Findings suggest: " + ", ".join(
            c["term"] for c in concepts[:5]
        )
        results.append({
            "image_id": img_id,
            "top_k_concepts": concepts,
            "pseudo_report": pseudo_report,
        })
    return results
```

### Implementation notes

- **OMP vs Lasso:** `OrthogonalMatchingPursuit` gives exact L0 sparsity (exactly k terms)
  but does not enforce non-negativity natively. Use `clamp(min=0)` post-hoc or switch to
  `Lasso(positive=True, alpha=...)` for true non-negative L1. The SPLiCE paper uses L1
  non-negative. Test both; OMP is faster for a fixed k budget.
- **Modality gap:** use the existing `models/modality_gap.pt` computed in `train_sae.py`.
- **Vocabulary:** reuse `data/vocabulary.json` + `embeddings/standard/text_vocab_embeddings.pt`.
  Both already on disk.
- **Stability:** SPLiCE is deterministic — no seed, no Jaccard instability. Stability
  analysis is trivially a no-op (document as such).

### Output schema

```jsonc
// results/spliece/sample_explanations.json — same as SAE producer schema
{
  "image_id": "1000_IM-0003-1001.dcm.png",
  "top_k_concepts": [
    {"term": "cardiomegaly", "coefficient": 0.234},
    {"term": "pleural effusion", "coefficient": 0.187}
  ],
  "pseudo_report": "Findings suggest: cardiomegaly, pleural effusion, ..."
}
```

### Verification checklist

- [ ] Top concepts on sample images are clinically coherent (manual inspection: ≥10 images).
- [ ] No concept appears with coefficient = 0 in top-k (clamp is effective).
- [ ] Faithfulness (judge, post M-007): % Aligned ≥ baseline SAE.
- [ ] Coverage: what fraction of RadLex terms appear in at least one image?

---

## Priority 2 — Extension: Structured Concept Organisation

**New file:** `src/concept_discovery/organize.py`
**New notebook:** `notebooks/autoencoder/06_concept_organization.ipynb`

### Design

Applied on top of the winning method (prototype on B, adapt to A):

1. **Redundancy detection:** agglomerative clustering of concept term embeddings
   (from `text_vocab_embeddings.pt`) by cosine similarity. Use
   `sklearn.cluster.AgglomerativeClustering(metric='cosine', linkage='average')`.
2. **Hierarchy mapping:** assign each cluster a label by the most frequently activated
   term in that cluster. If RadLex provides a hierarchy, map clusters to top-level
   anatomical/category nodes.
3. **Per-sample structured explanation:** instead of flat top-k, report active clusters
   and the top term per cluster. Example output:
   ```
   CARDIAC: cardiomegaly (0.23), cardiac silhouette (0.11)
   PULMONARY: pleural effusion (0.19), atelectasis (0.08)
   ```
4. **Metric:** compare intra-cluster cosine mean (should be > inter-cluster mean).

```python
# src/concept_discovery/organize.py (sketch)
from sklearn.cluster import AgglomerativeClustering
import torch

def cluster_concepts(
    vocab_emb: torch.Tensor,   # (V, 512)
    n_clusters: int = 20,
) -> list[int]:
    """Return cluster label per vocabulary term."""
    X = vocab_emb.numpy()
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(X).tolist()
```

---

## Priority 3 — Path A: SAE on 768-d hidden state

**Files to modify:** `src/embedding_extraction/extract_embeddings.py`,
`src/config.py`, `src/autoencoder/concept_naming.py`
**New notebook:** `notebooks/autoencoder/08_sae_hidden.ipynb`

### Step 1 — New extraction mode

Add a `mode` parameter to `extract_embeddings.py`:

```python
# src/embedding_extraction/extract_embeddings.py — addition
def extract_visual_embeddings_hidden(
    model, processor, image_paths, config
) -> tuple[torch.Tensor, list[str]]:
    """Extract 768-d CLS token (pre-projection) from BiomedCLIP."""
    all_embeddings, all_ids = [], []
    for batch_paths in batches(image_paths, config.batch_size):
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(config.device)
        with torch.no_grad():
            out = model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=False,
            )
            # CLS token of last hidden state: (B, 197, 768) -> (B, 768)
            hidden = out.last_hidden_state[:, 0, :]
            hidden = hidden / hidden.norm(dim=-1, keepdim=True)  # L2-normalise
        all_embeddings.append(hidden.cpu())
        all_ids.extend([Path(p).name for p in batch_paths])
    return torch.cat(all_embeddings), all_ids
```

> **Verification note:** confirm `model.vision_model` attribute name on
> `chuhac/BiomedCLIP-vit-bert-hf`. Run `print(type(model.vision_model))` at load.
> Expected: `transformers.models.vit.modeling_vit.ViTModel`.

Persist to `embeddings/standard_hidden/visual_embeddings_768.pt`.

### Step 2 — Config

Add `EmbeddingMode = Literal["projected_512", "hidden_768"]` to `config.py`:

```python
@dataclass(frozen=True)
class EmbeddingConfig:
    mode: str = "projected_512"        # "projected_512" or "hidden_768"
    activation_dim: int = 512          # set to 768 for hidden mode

    @property
    def embeddings_dir(self) -> Path:
        base = Path("embeddings")
        return base / ("standard_hidden" if self.mode == "hidden_768" else "standard")
```

Update `SAEConfig.activation_dim` to read from `EmbeddingConfig` at instantiation.

### Step 3 — Training

Retrain 5 seeds with `activation_dim=768`, same training pipeline:

```bash
# After extraction:
python src/autoencoder/train_sae.py --mode hidden_768 --dict_size 4096 \
    --steps 8000 --lr 5e-5 --seeds 0 42 123 456 789
```

Config changes: `n_training_steps 50000 → 8000`; `lr` auto → `5e-5` explicit;
`dict_size` 4096 (≈ 5.3× of 768); `batch_size` 256 unchanged.
Persist models to `models/sae_hidden_seed{seed}/`.

### Step 4 — Naming bridge (768-d → 512-d)

Use BiomedCLIP's frozen linear projection to map decoder rows into the shared text space:

```python
# src/autoencoder/concept_naming.py — add hidden_naming branch
def name_concepts_hidden(
    W_dec_768: torch.Tensor,    # (dict_size, 768) SAE decoder
    text_emb: torch.Tensor,     # (V, 512) RadLex text embeddings
    model,                       # BiomedCLIP — provides W_proj
    gap: torch.Tensor | None,
    vocab_terms: list[str],
) -> dict:
    # Access frozen projection weight: (512, 768) — verify attribute at load
    W_proj = model.visual_projection.weight.detach()  # (512, 768)
    # Project decoder rows into 512-d shared space
    dec_512 = W_dec_768 @ W_proj.T                    # (dict_size, 512)
    dec_512 = dec_512 / dec_512.norm(dim=-1, keepdim=True)
    # Apply modality-gap correction
    if gap is not None:
        dec_512 = dec_512 - gap.unsqueeze(0)
    # Cosine similarity with RadLex (same as baseline naming)
    sim = dec_512 @ text_emb.T                        # (dict_size, V)
    best_idx = sim.argmax(dim=-1)
    best_score = sim.max(dim=-1).values
    return {
        i: {"name": vocab_terms[best_idx[i]], "score": float(best_score[i])}
        for i in range(len(W_dec_768))
    }
```

> **Attribute verification:** run at load:
> ```python
> print(hasattr(model, "visual_projection"))
> # If not present, try: model.vision_model.post_layernorm or model.projection
> ```

### Step 5 — Verification

- [ ] Cross-seed Jaccard for SAE-768 > baseline (0.0038). Target: meaningfully above analytical null.
- [ ] Dead-feature rate < baseline (40–60%).
- [ ] Naming mean (SAE-768) > random baseline (0.372) by a clear margin.
- [ ] Faithfulness (judge) ≥ baseline SAE.

---

## Priority 4 — Path C: Hyperparameter Hygiene (apply to Path A)

From `ML-AUDIT-2026-06-25.md` M-006:

| Parameter | Current (baseline) | Corrected (Path A) |
|---|---|---|
| `n_training_steps` | 50,000 | 5,000–10,000 |
| `lr` | auto ~4e-4 | 5e-5 (explicit) |
| `dict_size` | 4,096 | 4,096 (≈5.3× of 768) or 2,048 |
| `batch_size` | 256 | 256 |

**File:** `src/config.py` — update `SAEConfig` defaults or pass as CLI args.

---

## Sequencing Summary

```
Priority 0: Fix judge M-007         [independent, can be done by one person]
    ↓
Priority 1: Implement Path B SPLiCE [start here; guarantees positive result]
    ↓
Priority 2: Extension (organise)    [on top of B; low cost, immediate rubric value]
    ↓
Priority 3: Path A SAE 768-d        [centerpiece; higher cost; after B stable]
    ↓
Priority 4: Path C hygiene          [applied while doing Path A]
    ↓
Consolidation: recap, slides, repo
```

Paths A and B can be parallelised across team members (group of 3).

---

## File Ownership

| Person | Responsibility | Files |
|---|---|---|
| **A** | Path A (SAE 768-d): extraction + training + naming bridge | `extract_embeddings.py`, `config.py`, `train_sae.py`, `concept_naming.py`, `08_sae_hidden.ipynb` |
| **B** | Path B (SPLiCE) + Extension: decomposition + clustering | `src/concept_discovery/spliece.py`, `organize.py`, `07_spliece.ipynb`, `06_concept_organization.ipynb` |
| **C** | Judge M-007 fixes + statistics + figures + writing | `evaluate_llm_judge.py`, `config.py`, result notebooks, recap document, slides |

---

## Deliverable Checklist

### Baseline (already done — verify artifacts exist)
- [ ] `embeddings/standard/visual_embeddings.pt` (7470×512)
- [ ] `embeddings/standard/train_embeddings.pt` + `test_embeddings.pt`
- [ ] `models/modality_gap.pt`
- [ ] `results/concept_names.json` (mean score ≈ 0.3943)
- [ ] `results/sample_explanations.json` (1494 entries)
- [ ] `results/ablation/` clean (a0–a5, a1_naming)
- [ ] Baseline REPORT.md (English, MedConcept deviations declared)

### Path B (SPLiCE)
- [ ] `src/concept_discovery/spliece.py`
- [ ] `results/spliece/sample_explanations.json`
- [ ] `results/spliece/concept_coverage.json`
- [ ] Faithfulness numbers (post M-007 judge)
- [ ] `notebooks/autoencoder/07_spliece.ipynb` (executed, clean)

### Extension
- [ ] `src/concept_discovery/organize.py`
- [ ] `results/concept_clusters.json`
- [ ] `notebooks/autoencoder/06_concept_organization.ipynb`

### Path A (SAE 768-d)
- [ ] `embeddings/standard_hidden/*.pt` (768-d, all splits)
- [ ] `models/sae_hidden_seed{0,42,...}/` (5 trained models)
- [ ] `results/sae_hidden/concept_names.json`
- [ ] `results/sae_hidden/sample_explanations.json`
- [ ] Cross-seed Jaccard + dead-feature rate measured
- [ ] Naming mean > random baseline
- [ ] `notebooks/autoencoder/08_sae_hidden.ipynb` (executed)

### Evaluation
- [ ] `results/aligned_scores.csv` (Aligned/Unaligned/Uncertain per method)
- [ ] Judge run on: baseline, SPLiCE, SAE-768

### Final deliverables
- [ ] Recap document 2–3 pages (6-section structure)
- [ ] Slides (15 min presentation)
- [ ] GitHub repo public with updated README
- [ ] ZIP for Portale della Didattica (slides + doc + repo link)

---

## References

- `docs/design/PROJECT-STRATEGY.md` v2.0
- `docs/design/proposals/PIPELINE-REFRAME-MAIN-VS-BASELINE.md`
- `docs/audits/ML-AUDIT-2026-06-25.md` (M-001..M-008)
- `docs/audits/ML-AUDIT-2026-06-24.md` (F-001..F-007 judge bugs)
- `docs/design/proposals/CONCEPT-INSTABILITY-DIAGNOSIS.md`
- `docs/design/proposals/VOCAB-BUILDING-ALTERNATIVES.md`
- `docs/design/proposals/SAE-SMALL-DATASET-MITIGATION.md`
- `HANDOFF.md` (session handoff notes)
