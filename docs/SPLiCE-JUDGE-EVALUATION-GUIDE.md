# SPLiCE Path B — LLM Judge Evaluation Guide

> **Target**: Membro 3 (LLM Judge Evaluation Team)  
> **Branch**: `feat/spliece-path-b`  
> **Status**: Ready for evaluation  
> **Date**: 2026-06-27

---

## Executive Summary

SPLiCE (Path B) è un nuovo metodo di **sparse decomposition deterministica** sul vocabolario RadLex che risolve il problema di non-identificabilità (M-001) che affligge sia il baseline che Path A.

**Richiesta**: Valutare la faithfulness clinica dei concetti SPLiCE usando il LLM judge e confrontare con il baseline SAE.

---

## Background: Cos'è SPLiCE?

### Differenza vs SAE

| Aspect | SAE (Baseline/Path A) | SPLiCE (Path B) |
|--------|----------------------|-----------------|
| **Metodo** | Sparse AutoEncoder (training) | Orthogonal Matching Pursuit (no training) |
| **Spazio latente** | Appreso (2048 o 768-d features) | Vincolato (1030 RadLex terms) |
| **Training time** | ~1.7h baseline, ~20 min/seed Path A | **0s** (deterministico) |
| **Stabilità** | Instabile (M-001 weak universality) | **Stabile** (deterministico) |
| **Concepts/image** | 5.0 fixed | 18.3 avg (14-22 range) |
| **Vocab coverage** | Non measured | **98.0%** (1009/1030 terms) |

### Risultati Preliminari

- **Processing**: 1515 images in 2.3s (CPU)
- **Vocabulary coverage**: 98.0% del vocabolario RadLex
- **Top term**: "chronic obstructive pulmonary disease" (21.0% images)
- **Determinism**: 100% (nessuna instabilità cross-seed)

---

## Input per LLM Judge

### File Principale

**Path**: `results/spliece/sample_explanations.json`

**Struttura** (compatibile con SAE):
```json
{
  "image_id": "1000_IM-0003-1001.dcm.png",
  "top_k_concepts": [
    {"term": "sail sign of chest", "coefficient": 0.5037},
    {"term": "superior division bronchus", "coefficient": 0.3879},
    {"term": "upper extremity artery", "coefficient": 0.3557}
  ],
  "pseudo_report": "Findings suggest: sail sign of chest, superior division bronchus, ..."
}
```

**Note**:
- `image_id` sono i **filename reali IU X-Ray** (es. `"1000_IM-0003-1001.dcm.png"`)
- Mapping `image_id → report` funziona tramite `indiana_projections.csv` → `indiana_reports.csv`
- **1515 images total** (stesso test set del baseline)

### Dati di Supporto

**Files già presenti nel repo**:
- `data/iu_xray/reports/indiana_reports.csv` — Report radiologici con uid
- `data/iu_xray/reports/indiana_projections.csv` — Mapping filename → uid
- `data/vocabulary.json` — Vocabolario RadLex (1030 termini)

---

## Come Eseguire il LLM Judge su SPLiCE

### Metodo 1: Modifica script esistente

**File**: `src/evaluate_llm_judge.py`

**Modifica richiesta** (riga 39):
```python
# Modifica da:
EXPLANATIONS_PATH = paths.results_dir / "sample_explanations.json"

# A:
EXPLANATIONS_PATH = paths.results_dir / "spliece" / "sample_explanations.json"
```

**Esecuzione**:
```bash
.venv/bin/python src/evaluate_llm_judge.py
```

**Output**: `results/spliece/aligned_scores.csv`

### Metodo 2: Crea script wrapper (consigliato)

Crea `src/evaluate_spliece_judge.py`:
```python
"""Evaluate SPLiCE concepts with LLM judge."""

import sys
sys.path.insert(0, "src/")

# Monkey-patch the explanations path BEFORE importing evaluate_llm_judge
import config
from pathlib import Path

# Override to point to SPLiCE output
config.paths.results_dir = Path("results/spliece")

# Now import and run the judge
import evaluate_llm_judge
evaluate_llm_judge.main()
```

**Esecuzione**:
```bash
.venv/bin/python src/evaluate_spliece_judge.py
```

---

## Output Atteso

### File: `results/spliece/aligned_scores.csv`

**Struttura** (stesso del baseline):
```csv
image_id,concept,verdict,aligned_score
1000_IM-0003-1001.dcm.png,sail sign of chest,Aligned,1.0
1000_IM-0003-1001.dcm.png,superior division bronchus,Uncertain,0.0
1000_IM-0003-1001.dcm.png,upper extremity artery,Unaligned,0.0
...
```

**Colonne**:
- `image_id` — Filename IU X-Ray
- `concept` — Termine RadLex
- `verdict` — Aligned/Unaligned/Uncertain
- `aligned_score` — 1.0 se Aligned, 0.0 altrimenti

---

## Metriche da Calcolare

### Metrica Primaria: % Aligned

**Definizione**: Percentuale di concetti classificati come "Aligned" dal LLM judge.

**Formula**:
```
% Aligned = (Aligned count) / (Total evaluations) × 100
```

**Confronto richiesto**:
```
SPLiCE % Aligned vs Baseline % Aligned
```

### Metriche Secondarie

1. **% Uncertain** — Concetti ambigui (report non menziona il finding)
2. **% Unaligned** — Concetti contraddetti dal report
3. **Per-image % Aligned** — Distribution della faithfulness per immagine
4. **Concept-frequency correlation** — Concetti più frequenti sono più aligned?

---

## Risultati Baseline (Riferimento)

**Fonte**: `results/aligned_scores.csv` (baseline SAE 512-d)

**Metriche precedenti** (da HANDOFF.md):
- Naming cosine (random ~0.372): 0.420
- Cross-seed Jaccard: 0.0038 (chance floor)
- Instabilità: "weak universality" (non identifiabile)

⚠️ **Nota**: Il baseline % Aligned NON è stato ancora misurato con il LLM judge!

---

## Piano di Valutazione Proposto

### FASE 1: Setup (5 min)

1. **Checkout branch**:
   ```bash
   git checkout feat/spliece-path-b
   .venv/bin/python -m pytest tests/unit/test_spliece.py -v  # Verify install
   ```

2. **Verifica input**:
   ```bash
   cat results/spliece/sample_explanations.json | jq '.[0]'
   ```

3. **Verifica mapping**:
   ```bash
   grep "1000_IM-0003-1001" data/iu_xray/reports/indiana_projections.csv
   grep "^1000," data/iu_xray/reports/indiana_reports.csv
   ```

### FASE 2: Esecuzione LLM Judge (~30-60 min)

1. **Modifica `src/evaluate_llm_judge.py`** (riga 39):
   ```python
   EXPLANATIONS_PATH = paths.results_dir / "spliece" / "sample_explanations.json"
   ```

2. **Esegui valutazione**:
   ```bash
   .venv/bin/python src/evaluate_llm_judge.py
   ```

3. **Verifica output**:
   ```bash
   head -20 results/spliece/aligned_scores.csv
   wc -l results/spliece/aligned_scores.csv  # Should be ~1500 lines (18.3 × 1515)
   ```

### FASE 3: Analisi Risultati (10 min)

1. **Calcola % Aligned**:
   ```python
   import pandas as pd
   df = pd.read_csv('results/spliece/aligned_scores.csv')
   aligned_pct = (df['aligned_score'] == 1.0).mean() * 100
   print(f"SPLiCE % Aligned: {aligned_pct:.1f}%")
   ```

2. **Distribuzione verdicts**:
   ```python
   print(df['verdict'].value_counts(normalize=True) * 100)
   ```

3. **Top aligned concepts**:
   ```python
   aligned_concepts = df[df['aligned_score'] == 1.0]['concept'].value_counts()
   print(aligned_concepts.head(20))
   ```

### FASE 4: Confronto con Baseline

**Opzione A**: Se baseline % Aligned esiste:
```bash
# Confronto diretto
python -c "
import pandas as pd
spliece = pd.read_csv('results/spliece/aligned_scores.csv')
baseline = pd.read_csv('results/aligned_scores.csv')
spliece_pct = (spliece['aligned_score'] == 1.0).mean() * 100
baseline_pct = (baseline['aligned_score'] == 1.0).mean() * 100
print(f'SPLiCE: {spliece_pct:.1f}% vs Baseline: {baseline_pct:.1f}%')
print(f'Delta: {spliece_pct - baseline_pct:+.1f}%')
"
```

**Opzione B**: Se baseline % Aligned NON esiste:
- Prima calcola baseline % Aligned (stesso procedura su `results/baseline/sample_explanations.json`)
- Poi confronta

---

## Risultati Attesi e Interpretazione

### Scenario 1: SPLiCE > Baseline ✅

**Verdict**: **SPLiCE vince** — Riesce a risolvere M-001 mantenendo/superando la faithfulness del baseline.

**Implicazioni**:
- SPLiCE è la soluzione deterministica superior
- Paper → "SPLiCE: Deterministic Concept Discovery in Medical VLMs"
- Non-identificabilità era il problema, non la tecnica

### Scenario 2: SPLiCE ≈ Baseline ≈

**Verdict**: **SPLiCE equivalente** — Risolve M-131 senza sacrificare faithfulness.

**Implicazioni**:
- SPLiCE è preferibile per determinismo e velocità
- Validazione della strategia "deterministico > stocastico"

### Scenario 3: SPLiCE < Baseline ❌

**Verdict**: **Investigare why** — Potrebbero essere problemi:
- Modality gap correction peggiora decomposizione?
- OMP vs Lasso algorithm choice?
- k=32 troppo basso/alto?

**Azioni**:
- Analizzare concetti Unaligned/Uncertain
- Testare con `--no-gap-correction`
- Aumentare/diminuire `--k 64` o `--k 16`

---

## Troubleshooting

### Problema: "Image ID not found in projections"

**Symptom**: Messaggio "skipped X images due to missing report"

**Fix**: Verifica che `data/test_image_ids.json` contenga i filename corretti:
```bash
head -5 data/test_image_ids.json  # Should be: ["1000_IM-0003-1001.dcm.png", ...]
```

### Problema: "GPU not available"

**Symptom**: CUDA/MPS error

**Fix**: Il judge funziona anche su CPU (più lento). Verifica che `unsloth/medgemma-4b-it` sia accessibile:
```bash
.venv/bin/python -c "from transformers import pipeline; print('OK')"
```

### Problema: "Sample too small"

**Symptom**: Meno di 1515 immagini valutate

**Fix**: Verifica che `sample_explanations.json` abbia 1515 entries:
```bash
cat results/spliece/sample_explanations.json | jq '. | length'
```

---

## Deliverables Richiesti

Al termine della valutazione, fornire:

1. **File principale**: `results/spliece/aligned_scores.csv`
2. **Metriche chiave**:
   - % Aligned SPLiCE
   - % Aligned Baseline (se non esiste, calcolalo prima)
   - Delta (SPLiCE - Baseline)
3. **Analisi qualitativa** (opzionale ma raccomandata):
   - Top 10 concetti più aligned
   - Top 10 concetti meno aligned
   - Esempi di Aligned/Uncertain/Unaligned
4. **Conclusioni**: SPLiCE vince/pareggio/perde vs baseline?

---

## Riferimenti Tecnici

### Implementazione SPLiCE

- **Code**: `src/concept_discovery/spliece.py`
- **Script**: `scripts/run_spliece.py`
- **Config**: `src/config.py` (SpliCEConfig)
- **Tests**: `tests/unit/test_spliece.py`, `tests/integration/test_spliece_pipeline.py`

### Documentazione

- **Implementation Plan**: `docs/plans/2026-06-27-spliece-path-b.md`
- **Verification Audit**: `docs/audits/ML-AUDIT-2026-06-27.md`
- **Release Notes**: `docs/releases/CHANGELOG-v0.5.0-2026-06-27.md`

### Risultati

- **Main output**: `results/spliece/sample_explanations.json`
- **Coverage analysis**: `results/spliece/REPORT_coverage.md`
- **Run report**: `results/spliece/REPORT_run.md`

---

## Contatti

Per domande tecniche su SPLiCE:
- **Branch**: `feat/spliece-path-b`
- **Commit SPLiCE**: `0a92d40` (fix per image IDs corretti)
- **Implementation**: Vedi `docs/plans/2026-06-27-spliece-path-b.md`

Per domande sul LLM judge:
- **Script**: `src/evaluate_llm_judge.py`
- **Config**: `src/config.py` (JudgeConfig)

---

## Checklist Membro 3

Prima di iniziare:
- [ ] Branch `feat/spliece-path-b` checkoutato
- [ ] Tests passano (`pytest tests/unit/test_spliece.py`)
- [ ] `results/spliece/sample_explanations.json` esiste (1515 images)
- [ ] Mapping image_id → report verificato (test con uid 1000)

Durante valutazione:
- [ ] LLM judge eseguito su SPLiCE
- [ ] `aligned_scores.csv` generato
- [ ] % Aligned calcolato
- [ ] Confronto con baseline eseguito (se necessario)

Dopo valutazione:
- [ ] Risultati comunicati al team
- [ ] File `aligned_scores.csv` salvato in `results/spliece/`
- [ ] Metriche chiave documentate
- [ ] Conclusioni fornite (SPLiCE vince/pareggio/perde)

---

**Good luck! 🎯**

SPLiCE è un approccio innovativo che potrebbe risolvere il problema fondamentale di non-identificabilità. La tua valutazione determinerà se questa soluzione deterministica supera gli SAE tradizionali.
