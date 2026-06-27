# LLM Judge Evaluation — Complete Guide (All Methods)

> **Target**: Member 3 (LLM Judge Evaluation Team)  
> **Branch**: `dev` — tutti e 3 i metodi (Baseline + Path A + SPLiCE) sono qui, nessun checkout necessario  
> **Status**: Ready for evaluation — All 3 methods prepared  
> **Date**: 2026-06-27

---

## Executive Summary

**Tutti e 3 i metodi sono pronti per la valutazione LLM judge**:

| Method | Input File | Concepts/Image | Status |
|--------|-----------|----------------|--------|
| **Baseline SAE (512-d)** | `results/baseline/sample_explanations.json` | 5.0 fixed | ✅ Ready |
| **Path A SAE (768-d)** | `results/sae_hidden/sample_explanations.json` | 5.0 fixed | ✅ Ready |
| **SPLiCE Path B** | `results/spliece/sample_explanations.json` | 18.3 avg (14-22) | ✅ Ready |

**Obiettivo**: Valutare la faithfulness clinica (% Aligned) di tutti e 3 i metodi e determinare il vincitore.

---

## Background: I 3 Metodi a Confronto

### Baseline SAE (512-d)
- **Metodo**: Sparse AutoEncoder su projected embeddings BiomedCLIP (512-d)
- **Training**: ~1.7h su MPS (5 seeds)
- **Stabilità**: Weak universality (Jaccard 0.0038 = chance floor)
- **Problema**: Non-identificabilità (M-001) — 0% features con forte identifiabilità

### Path A SAE (768-d)  
- **Metodo**: Sparse AutoEncoder su hidden state pre-projection (768-d)
- **Training**: ~20 min/seed (5 seeds, ~1.7h total)
- **Stabilità**: Weak universality (stesso problema del baseline)
- **Problema**: Non-identificabilità (M-001) — 0% features con forte identifiabilità
- **Naming**: 0.471 cosine vs RadLex (leggermente meglio del baseline 0.420)

### SPLiCE Path B ⭐
- **Metodo**: Orthogonal Matching Pursuit (deterministico, no training)
- **Training**: 0s (2.3s total per 1515 images)
- **Stabilità**: 100% deterministico (nessuna instabilità cross-seed)
- **Vantaggio**: Risolve M-131 per costruzione (spazio vincolato a RadLex)
- **Coverage**: 98.0% vocabolario RadLex (1009/1030 termini usati)

---

## Input per LLM Judge

### File 1: Baseline SAE (512-d)

**Path**: `results/baseline/sample_explanations.json`

**Struttura**:
```json
{
  "image_id": "1000_IM-0003-1001.dcm.png",
  "top_k_concepts": [
    {"feature_id": 1853, "name": "cricothyroid tube", "activation": 0.2231},
    {"feature_id": 431, "name": "stapedius nerve", "activation": 0.1386},
    {"feature_id": 1634, "name": "strap muscle of neck", "activation": 0.1134}
  ],
  "pseudo_report": "The model identifies the following visual concepts..."
}
```

**Statistiche**:
- 1515 images
- 5.0 concepts/image (fixed)
- Attivazioni basse (0.1-0.2 range)

### File 2: Path A SAE (768-d)

**Path**: `results/sae_hidden/sample_explanations.json`

**Struttura**: (stessa del baseline)
```json
{
  "image_id": "1000_IM-0003-1001.dcm.png",
  "top_k_concepts": [
    {"feature_id": 1410, "name": "deep fascia of leg", "activation": 9.1007},
    {"feature_id": 1937, "name": "third part of subclavian artery", "activation": 6.9703}
  ],
  "pseudo_report": "..."
}
```

**Statistiche**:
- 1515 images  
- 5.0 concepts/image (fixed)
- ⚠️ **Attivazioni molto più alte** (6-9 range vs 0.1-0.2 baseline)

### File 3: SPLiCE Path B

**Path**: `results/spliece/sample_explanations.json`

**Struttura**: (diversa, usa `term` e `coefficient`)
```json
{
  "image_id": "1000_IM-0003-1001.dcm.png",
  "top_k_concepts": [
    {"term": "sail sign of chest", "coefficient": 0.5037},
    {"term": "superior division bronchus", "coefficient": 0.3879}
  ],
  "pseudo_report": "Findings suggest: sail sign of chest, superior division bronchus..."
}
```

**Statistiche**:
- 1515 images
- **18.3 concepts/image** (14-22 range, molto più granulare)
- Coefficienti medi (0.17 range)

---

## Piano di Valutazione

### FASE 1: Setup (10 min)

1. **Verifica che tutti gli input esistano** (tutti su `dev`, nessun checkout):
   ```bash
   ls -lh results/baseline/sample_explanations.json    # Baseline (512-d)
   ls -lh results/sae_hidden/sample_explanations.json  # Path A (768-d)
   ls -lh results/spliece/sample_explanations.json     # SPLiCE (Path B)
   ```

2. **Verifica image_id mapping** (tutti i metodi):
   ```bash
   # Dovrebbe restituire: "1000_IM-0003-1001.dcm.png"
   cat results/baseline/sample_explanations.json | jq -r '.[0].image_id'
   cat results/sae_hidden/sample_explanations.json | jq -r '.[0].image_id'
   cat results/spliece/sample_explanations.json | jq -r '.[0].image_id'
   ```

### FASE 2: Esegui LLM Judge (3 methods × ~30-60 min = 1.5-3h total)

#### Method 1: Baseline SAE (512-d)

**Modifica `src/evaluate_llm_judge.py`** (riga 39) — già il default baseline:
```python
EXPLANATIONS_PATH = paths.baseline_results_dir / "sample_explanations.json"
```

**Esegui**:
```bash
.venv/bin/python src/evaluate_llm_judge.py
```

**Output**: `results/aligned_scores_baseline.csv` (rinomina per chiarezza)

#### Method 2: Path A SAE (768-d)

**Modifica `src/evaluate_llm_judge.py`** (riga 39):
```python
EXPLANATIONS_PATH = paths.results_dir / "sae_hidden" / "sample_explanations.json"
```

**Esegui**:
```bash
.venv/bin/python src/evaluate_llm_judge.py
```

**Output**: `results/sae_hidden/aligned_scores.csv` oppure rinomina in `results/aligned_scores_path_a.csv`

#### Method 3: SPLiCE Path B

**Modifica `src/evaluate_llm_judge.py`** (riga 39):
```python
EXPLANATIONS_PATH = paths.results_dir / "spliece" / "sample_explanations.json"
```

**Esegui**:
```bash
.venv/bin/python src/evaluate_llm_judge.py
```

**Output**: `results/spliece/aligned_scores.csv`

### FASE 3: Analisi Comparativa (20 min)

1. **Calcola % Aligned per metodo**:
```python
import pandas as pd

def pct_aligned(path):
    """% Aligned from the judge CSV. Column is `verdict` (string), NOT `aligned_score`.

    The CSV columns are: image_id, feature_id, concept, activation, verdict, raw_response.
    `verdict` is one of {Aligned, Unaligned, Uncertain}. Infrastructure-error rows are
    written with verdict=Uncertain and a `raw_response` starting with "ERROR:" — exclude
    them so the denominator matches the judge's own `judge_scores.json["aligned_rate"]`
    (aligned_count / total_valid, F-006). Uncertain *model* verdicts stay in the denominator.
    """
    df = pd.read_csv(path)
    valid = df[~df["raw_response"].astype(str).str.startswith("ERROR:")]
    return (valid["verdict"] == "Aligned").mean() * 100

baseline_pct = pct_aligned('results/aligned_scores_baseline.csv')
path_a_pct   = pct_aligned('results/aligned_scores_path_a.csv')
spliece_pct  = pct_aligned('results/spliece/aligned_scores.csv')
null_pct     = pct_aligned('results/null/aligned_scores.csv')  # F-005: chance floor

print(f"Baseline SAE (512-d):  {baseline_pct:.1f}% Aligned")
print(f"Path A SAE (768-d):    {path_a_pct:.1f}% Aligned")
print(f"SPLiCE Path B:         {spliece_pct:.1f}% Aligned")
print(f"Random-k NULL:         {null_pct:.1f}% Aligned  (chance floor)")
print(f"\nWinner by LIFT over null (the only fair cross-method comparison):")
for name, pct in [('Baseline', baseline_pct), ('Path A', path_a_pct), ('SPLiCE', spliece_pct)]:
    print(f"  {name}: lift = {pct / null_pct:.2f}x")
```

> **⚠ F-005 — Confronta il LIFT, non il % Aligned grezzo.** I metodi emettono un numero diverso di concetti per immagine (Baseline/Path A ~5, SPLiCE ~13). Più concetti = più "biglietti della lotteria" su termini comuni (consolidation, effusion, tube). Confronta `pct_metodo / pct_null` contro un null **count-matched**: `results/null_k5/` (k=5, per Baseline/Path A) e `results/null/` (k≈13, per SPLiCE) — `python scripts/generate_null_explanations.py --k 5 --output results/null_k5`. Solo il lift è confrontabile tra metodi.

> **ℹ F-018 — Coverage del judge.** ~7/1515 immagini (0.46%) non hanno report mappabile in `indiana_projections.csv`/`indiana_reports.csv` (4 PNG assenti dalle projections per provenienza upstream IU X-Ray, 3 con report vuoto). Il judge le scarta e le registra in `results/judge_coverage.json` — dropout simmetrico tra metodi, non influenza il confronto.

2. **Analisi distribuzioni**:
```python
# Verdict distribution per method
for name, df in [('Baseline', baseline), ('Path A', path_a), ('SPLiCE', spliece)]:
    print(f"\n{name}:")
    print(df['verdict'].value_counts(normalize=True) * 100)
```

3. **Top aligned concepts**:
```python
# Top 10 concetti più aligned per method
for name, df in [('Baseline', baseline), ('Path A', path_a), ('SPLiCE', spliece)]:
    aligned = df[df['verdict'] == 'Aligned']
    print(f"\n{name} Top 10 Aligned:")
    print(aligned['concept'].value_counts().head(10))
```

---

## Risultati Attesi e Interpretazione

### Scenario 1: SPLiCE > Baseline & Path A ✅

**Verdict**: **SPLiCE vince** — Risolve M-001 E supera la faithfulness SAE.

**Implicazioni**:
- Non-identificabilità era il problema fondamentale, non la tecnica
- Metodo deterministico superior per design
- Paper → "SPLiCE: Deterministic Concept Discovery in Medical VLMs"

### Scenario 2: Path A > Baseline, SPLiCE ≈ Path A

**Verdict**: **Path A preferibile** — Meglio del baseline, SPLiCE equivalente.

**Implicazioni**:
- Hidden state 768-d migliore di projected 512-d
- SPLiCE valida alternativa deterministica a Path A
- Trade-off: Path A (training) vs SPLiCE (determinismo)

### Scenario 3: Tutti ≈ Equivalenti

**Verdict**: **Tie** — Nessuna differenza significativa in faithfulness.

**Implicazioni**:
- SPLiCE vince per determinismo + velocità (0 training vs 1.7h)
- Confronta costi/benefici: stabilità vs setup cost

### Scenario 4: SPLiCE < Entrambi

**Verdict**: **Investigare why** — Potenziali problemi:
- Modality gap correction peggiora?
- k=32 troppo alto/basso?
- OMP vs Lasso algorithm choice?

---

## Troubleshooting Specífico

### Problema: Path A activations molto alte

**Symptom**: Valori 6-9+ vs 0.1-0.2 baseline

**Spiegazione**: Path A usa hidden state pre-projection (768-d) con scale diverse. Questo è **normale e non un errore**.

**Impatto sul judge**: Il LLM judge guarda solo il **nome del concetto**, non l'activation. Quindi non è un problema.

### Problema: Diverso numero di evaluations

**Symptom**: 
- Baseline: 1515 × 5 = 7,575 evaluations
- Path A: 1515 × 5 = 7,575 evaluations  
- SPLiCE: 1515 × 18.3 = ~27,700 evaluations

**Spiegazione**: SPLiCE ha più concetti per image.

**Impatto sul confronto**: Confronta **% Aligned** (non count assoluto) per normalizzare.

### Problema: Branch switching tra evaluations

**Symptom**: Files non trovati

**Fix**: Tutti i risultati dei 3 metodi sono su `dev` in `results/{baseline,sae_hidden,spliece}/` — nessun branch switch necessario. Se un file manca, rigeneralo con lo script corrispondente (`scripts/run_baseline.py`, `scripts/run_path_a.py`, `scripts/run_spliece.py`).

---

## Deliverables Richiesti

Al termine della valutazione, fornire:

1. **File CSV**:
   - `results/aligned_scores_baseline.csv`
   - `results/aligned_scores_path_a.csv` (o nella dir Path A)
   - `results/spliece/aligned_scores.csv`

2. **Metriche chiave**:
   ```
   Baseline SAE (512-d):  XX.X% Aligned
   Path A SAE (768-d):     XX.X% Aligned  
   SPLiCE Path B:          XX.X% Aligned
   
   Winner: [Baseline/Path A/SPLiCE/Tie]
   ```

3. **Analisi aggiuntiva** (raccomandata):
   - Distribuzione verdicts per method
   - Top 10 aligned concepts per method
   - Esempi di Aligned/Uncertain/Unaligned per method
   - Correlation: concepts più frequenti = più aligned?

4. **Conclusioni**:
   - Quale method vince e perché
   - Se SPLiCE vince: conferma che M-131 era il problema
   - Se Path A vince: hidden state migliore di projected
   - Se SPLiCE perde: analizzare why (modality gap? OMP vs Lasso?)

---

## Riferimenti Tecnici

### Code Locations

| Method | Input Script | Config | Reports |
|--------|------------|--------|---------|
| Baseline | `scripts/run_baseline.py` | `src/config.py` (SAEConfig) | `results/REPORT_*.md` |
| Path A | `scripts/run_path_a.py` | `src/config.py` (SAEHiddenConfig) | `results/sae_hidden/REPORT_*.md` |
| SPLiCE | `scripts/run_spliece.py` | `src/config.py` (SpliCEConfig) | `results/spliece/REPORT_*.md` |

### LLM Judge

- **Script**: `src/evaluate_llm_judge.py`
- **Config**: `src/config.py` (JudgeConfig)
- **Model**: `unsloth/medgemma-4b-it` (HuggingFace)

---

## Tempo Totale Stimato

- **Setup**: 10 min (verifica files, branches)
- **LLM Judge execution**: 1.5-3h (3 methods × 30-60 min)
- **Analysis**: 20 min
- **Total**: **2-3.5 ore**

---

## Checklist Member 3

### Prima di iniziare:
- [ ] Tutti e 3 i branches verificati
- [ ] Tutti e 3 i `sample_explanations.json` esistono
- [ ] Image_id mapping verificato (stesso UID per tutti)
- [ ] LLM judge script funziona (`pytest tests/test_llm_judge.py`)

### Durante valutazione:
- [ ] Baseline evaluation completata
- [ ] Path A evaluation completata
- [ ] SPLiCE evaluation completata
- [ ] % Aligned calcolato per tutti e 3
- [ ] Confronto eseguito

### Dopo valutazione:
- [ ] 3 file CSV salvati
- [ ] Metriche chiave documentate
- [ ] Vincitore identificato
- [ ] Report conclusivo fornito al team

---

## Domande Frequenti

### Q: Perché Path A ha activations così alte?

**A**: Path A usa hidden state pre-projection (768-d) che ha scale diverse dalle projected embeddings (512-d). Il LLM judge guarda solo il **nome del concetto**, non l'activation, quindi non è un problema.

### Q: Perché SPLiCE ha più concetti per image?

**A**: SPLiCE usa k=32 ma filtra gli zeri, risultando in 18.3 concetti/image in media vs 5.0 fissi degli SAE. Questo dà una rappresentazione più granulare.

### Q: Come confronto se il numero di evaluations è diverso?

**A**: Usa **% Aligned** (non count assoluto) che normalizza per il numero totale di evaluations.

### Q: Che succede se SPLiCE perde?

**A**: Analizzare il why:
- Modality gap correction peggiora la decomposizione?
- k=32 non ottimale?
- OMP vs Lasso algorithm choice?
- Investigare e documentare i findings.

---

## Good Luck! 🎯

Hai 3 metodi da valutare: 2 stochastici (Baseline, Path A) e 1 deterministico (SPLiCE). 

La tua valutazione determinerà:
1. Se SPLiCE risolve M-001 senza sacrificare faithfulness
2. Se Path A (768-d) è superiore al baseline (512-d)
3. Qual è il metodo migliore per concept discovery nei medical VLM

**Risultati attesi**: 3 percentuali (% Aligned) + 1 vincitore chiaro.

Let's find out which method wins! 🏆
