# generate_explanations.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/generate_explanations.py`,
lo script che genera spiegazioni strutturate (pseudo-report) per ogni immagine del
TEST set a partire dai concetti SAE attivati, producendo output per la valutazione
con LLM Judge.

---

## 1. Docstring e metadata

```python
"""
generate_explanations.py -- Generate SAE-based explanations

For each image, extract the top-k activated SAE concepts and generate
a structured explanation (pseudo-report) for the LLM Judge.

Uses HELD-OUT test embeddings for evaluation.

Prerequisites:
    - models/sae_seed{PRIMARY_SEED}/ae.pt
    - embeddings/test_embeddings.pt
    - results/concept_names.json (output of concept_naming.py)

Run:
    python src/autoencoder/generate_explanations.py
"""
```

**Perche:**

Due dettagli critici: (1) "Uses HELD-OUT test embeddings" -- le spiegazioni sono
generate su dati mai visti durante il training. (2) Prerequisiti chiari per fail-fast.

---

## 2. Importazioni e costanti

```python
import json, logging, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
from autoencoder.tracking import init_tracking, log_artifact, finish_tracking

SEED = config.training.primary_seed
CONCEPT_NAMES_PATH = config.paths.results_dir / "concept_names.json"
OUTPUT_PATH = config.paths.results_dir / "sample_explanations.json"
```

**Perche:**

- **`config.training.primary_seed`**: coerenza con `concept_naming.py`. I nomi
  e le attivazioni devono provenire dallo stesso modello (seed 42 di default),
  altrimenti le spiegazioni sarebbero incoerenti.
- Input: `concept_names.json` (output di concept_naming).
- Output: `sample_explanations.json` (consumato da 05_evaluate_llm_judge.py).

---

## 3. Funzione `generate_explanation()`

```python
def generate_explanation(
    top_concepts: list[tuple[int, float]],
    concept_names: dict[str, dict],
) -> dict:
```

### 3.1 Costruzione dei findings

```python
    findings = []
    for feat_id, activation in top_concepts:
        feat_key = str(feat_id)
        if feat_key in concept_names:
            name = concept_names[feat_key]["name"]
            similarity = concept_names[feat_key]["score"]
        else:
            name = f"unknown_feature_{feat_id}"
            similarity = 0.0

        findings.append({
            "concept": name,
            "feature_id": feat_id,
            "activation": round(activation, 4),
            "naming_confidence": round(similarity, 4),
        })
```

**Perche:**

Per ogni concetto attivato: nome medico, feature_id, intensita' attivazione,
e naming_confidence (quanto il nome e' affidabile). Il fallback `unknown_feature_`
e' difesa difensiva: non dovrebbe mai attivarsi, ma protegge da inconsistenze.
`str(feat_id)` perche' le chiavi JSON sono sempre stringhe.

### 3.2 Guard contro findings vuoti

```python
    if not findings:
        return {
            "findings": [],
            "pseudo_report": "No active concepts detected.",
            "n_active_concepts": 0,
        }
```

**Perche:**

Se il SAE non attiva nessuna feature (embedding tutto zeri, o bug), restituisce
un placeholder sicuro invece di crashare con indice out-of-bounds.

### 3.3 Generazione del pseudo-report

```python
    concept_list = ", ".join(f["concept"] for f in findings[:5])
    pseudo_report = (
        f"The model identifies the following visual concepts in this "
        f"radiograph: {concept_list}. "
        f"The dominant concept is '{findings[0]['concept']}' "
        f"(activation={findings[0]['activation']:.3f})."
    )

    return {
        "findings": findings,
        "pseudo_report": pseudo_report,
        "n_active_concepts": len(findings),
    }
```

**Perche:**

- **Solo top-5 nel testo**: includere tutti renderebbe il report illeggibile per
  un LLM. I primi 5 sono sufficienti per un giudizio qualitativo.
- **`findings` contiene tutti**: dati strutturati completi per analisi quantitative.
- **"Dominant concept"**: aiuta l'LLM Judge a focalizzarsi sull'aspetto piu'
  rilevante.
- Restituisce sia dati strutturati sia testo leggibile.

---

## 4. Funzione `run()`

### 4.1 Validazione prerequisiti

```python
def run() -> Path:
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"
    embeddings_path = config.paths.test_embeddings_path

    for path, desc in [
        (model_dir, "SAE model"),
        (embeddings_path, "Test embeddings"),
        (CONCEPT_NAMES_PATH, "Concept names"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")
```

**Perche:**

Pattern standard. Critico: `embeddings_path = config.paths.test_embeddings_path`
usa il test set hold-out, non le embedding complete. Le spiegazioni valutano
capacita' di generalizzazione, non memorizzazione.

### 4.2 Caricamento e limit opzionale

```python
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    with open(CONCEPT_NAMES_PATH) as f:
        concept_names = json.load(f)

    if config.explanation.explanation_max_samples:
        embeddings = embeddings[: config.explanation.explanation_max_samples]

    logger.info(f"Generating explanations for {embeddings.shape[0]} test samples...")
```

**Perche:**

- `weights_only=True`: sicurezza pickle.
- `explanation_max_samples`: parametro opzionale per limitare i campioni in
  sviluppo. `None` = processa tutto il test set.

### 4.3 Encoding e generazione

```python
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    all_top_concepts = mgr.get_top_concepts(
        embeddings, n=config.explanation.explanation_top_n
    )

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["sample_idx"] = idx
        explanations.append(explanation)
```

**Perche:**

`get_top_concepts(embeddings, n=5)`: per ogni immagine del test set, i top-5
concetti. `sample_idx` per tracciabilita' (corrispondenza con le immagini).

### 4.4 Salvataggio e esempio

```python
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")

    if explanations:
        logger.info(f"\nExample (sample 0):\n  {explanations[0]['pseudo_report']}")
```

**Perche:**

Log dell'esempio per verifica visiva rapida. `if explanations:` guard prima
di accedere a `explanations[0]`.

### 4.5 Tracking Weights & Biases

```python
    if config.wandb_cfg.enabled:
        init_tracking("generate_explanations", {
            "project": config.wandb_cfg.project,
            "seed": SEED,
            "n_samples": len(explanations),
        })
        log_artifact(OUTPUT_PATH, "sample_explanations", "results")
        finish_tracking()
```

**Perche:**

Run wandb dedicata. Seed, numero campioni e JSON come artefatto. No-op se
disabilitato.

---

## 5. Funzione `main()`

```python
def main():
    run()

if __name__ == "__main__":
    main()
```

**Perche:**

Pattern standard: `run()` testabile, `main()` wrapper, `__name__` guard.

---

## Diagramma del flusso

```text
[Input 1: embeddings/test_embeddings.pt (N_test, 512)]   <-- HELD-OUT
            |
[Input 2: models/sae_seed{primary_seed}/ae.pt]
            |
    [get_top_concepts(n=5)]
            |
[Input 3: results/concept_names.json]
            |
    +--[generate_explanation()]--+
    |   feat_id -> nome medico   |
    |   Guard: empty findings    |
    |   Costruisci pseudo-report |
    +----------------------------+
            |
    [Aggiungi sample_idx]
            |
[Output: results/sample_explanations.json]
```

---

## Formato dell'output

```json
[
  {
    "findings": [
      {"concept": "cardiomegaly", "feature_id": 127,
       "activation": 2.3412, "naming_confidence": 0.7234},
      {"concept": "pleural effusion", "feature_id": 892, ...}
    ],
    "pseudo_report": "The model identifies the following visual concepts...",
    "n_active_concepts": 5,
    "sample_idx": 0
  }
]
```

---

## Perche' si usa il test set

1. **No data leakage**: spiegazioni su dati nuovi, non contaminati dal training.
2. **Valutazione realistica**: simula lo scenario d'uso reale.
3. **Coerenza metodologica**: train_sae.py fa sanity check sul test set; usare
   lo stesso set qui mantiene coerenza.

---

## Dipendenze dalla configurazione

| Variabile | Section | Default | Usata per |
|-----------|---------|---------|-----------|
| `config.training.primary_seed` | TrainingConfig | 42 | Modello di riferimento |
| `config.hardware.device` | HardwareConfig | auto | Device SAE |
| `config.paths.test_embeddings_path` | PathsConfig | `embeddings/test_embeddings.pt` | Input held-out |
| `config.explanation.explanation_top_n` | ExplanationConfig | 5 | Top concetti |
| `config.explanation.explanation_max_samples` | ExplanationConfig | None | Limit sviluppo |
| `config.wandb_cfg.enabled` | WandbConfig | False | Tracking |

---

## Ruolo nella pipeline

```text
01 (embedding) -> train_sae -> concept_naming -> generate_explanations (su test) -> 05 (LLM Judge)
```
