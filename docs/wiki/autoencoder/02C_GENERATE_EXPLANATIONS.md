# 02c_generate_explanations.py - Documentazione completa

Questo documento descrive ogni sezione di `src/02c_generate_explanations.py`,
lo script che genera spiegazioni strutturate (pseudo-report) per ogni immagine
del dataset a partire dai concetti SAE attivati.

---

## 1. Docstring e metadata

```python
"""
02c_generate_explanations.py - Generate SAE-based explanations

For each image, extract the top-k activated SAE concepts and generate
a structured explanation (pseudo-report) for the LLM Judge.

Prerequisites:
    - models/sae_seed{SEED}/ae.pt
    - embeddings/visual_embeddings.pt
    - results/concept_names.json (output of 02b)

Run:
    python src/02c_generate_explanations.py
"""
```

**Perche:**

Questo script e' il ponte tra l'analisi SAE e la valutazione con LLM Judge.
Prende i concetti grezzi (feature_id, activation_value) e li trasforma in
spiegazioni leggibili da un LLM, che poi giudichera' se la spiegazione e'
clinicamente coerente.

---

## 2. Costanti

```python
SEED = config.training.seeds[1]  # Use seed 42 as primary
CONCEPT_NAMES_PATH = config.paths.results_dir / "concept_names.json"
OUTPUT_PATH = config.paths.results_dir / "sample_explanations.json"
```

**Perche:**

- Stesso seed di 02b (seed=42) per coerenza: i concept names sono stati calcolati
  su questo modello, le spiegazioni devono usare lo stesso.
- Input: `concept_names.json` prodotto da 02b
- Output: `sample_explanations.json` consumato da 05_evaluate_llm_judge.py

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

Per ogni concetto attivato nell'immagine, costruisce un "finding" strutturato:

- `concept`: nome medico assegnato in 02b (es. "pneumothorax")
- `feature_id`: ID numerico della feature SAE (0-4095)
- `activation`: intensita' dell'attivazione (piu' alta = piu' presente nell'immagine)
- `naming_confidence`: cosine similarity del naming - quanto siamo sicuri che
  il nome sia corretto

Il caso `unknown_feature_{feat_id}` e' un fallback difensivo: non dovrebbe
mai attivarsi se 02b e' stato eseguito correttamente (produce nomi per tutte
le 4096 feature), ma protegge da inconsistenze.

`str(feat_id)` e' necessario perche' le chiavi JSON sono sempre stringhe,
anche se rappresentano numeri.

### 3.2 Generazione del pseudo-report

```python
    concept_list = ", ".join(f["concept"] for f in findings[:5])
    pseudo_report = (
        f"The model identifies the following visual concepts in this radiograph: {concept_list}. "
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

Il pseudo-report e' una frase in linguaggio naturale che riassume i concetti
attivati. Questo formato e' pensato per essere consumato dall'LLM Judge (05)
che valuta:
1. Se i concetti identificati hanno senso clinico per una radiografia
2. Se il concetto dominante e' plausibile
3. Se la spiegazione e' coerente nel suo insieme

Restituisce sia i dati strutturati (`findings`) sia il testo leggibile
(`pseudo_report`), permettendo sia analisi automatiche sia valutazione qualitativa.

---

## 4. Funzione `main()`

### 4.1 Validazione e caricamento

```python
    embeddings = torch.load(config.paths.visual_embeddings_path, map_location="cpu", weights_only=True)
    with open(CONCEPT_NAMES_PATH) as f:
        concept_names = json.load(f)

    if config.explanation.explanation_max_samples:
        embeddings = embeddings[: config.explanation.explanation_max_samples]
```

**Perche:**

- Carica le embedding visive (7400, 512) e i nomi dei concetti
- `explanation_max_samples`: parametro opzionale per limitare il numero di campioni
  processati. Utile in sviluppo per testare velocemente su un sottoinsieme.
  Se `None` (default), processa tutto il dataset.

### 4.2 Encoding e generazione

```python
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    all_top_concepts = mgr.get_top_concepts(embeddings, n=config.explanation.explanation_top_n)

    explanations = []
    for idx, top_concepts in enumerate(all_top_concepts):
        explanation = generate_explanation(top_concepts, concept_names)
        explanation["sample_idx"] = idx
        explanations.append(explanation)
```

**Perche:**

1. `get_top_concepts(embeddings, n=5)`: per ogni immagine, trova i top-5 concetti
   piu' attivati. Restituisce lista di liste di tuple (feat_id, activation).

2. Per ogni campione, genera la spiegazione strutturata e aggiunge `sample_idx`
   per tracciabilita' (sapere a quale immagine corrisponde).

3. Il risultato e' una lista di 7400 spiegazioni, una per immagine.

### 4.3 Salvataggio e esempio

```python
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    logger.info(f"Explanations generated: {len(explanations)}")
    logger.info(f"Saved to: {OUTPUT_PATH}")

    if explanations:
        logger.info(f"\nExample (sample 0):\n  {explanations[0]['pseudo_report']}")
```

**Perche:**

Salva il JSON completo e logga un esempio per verifica visiva rapida.
L'output sara' tipo:
```
Example (sample 0):
  The model identifies the following visual concepts in this radiograph:
  cardiomegaly, pleural effusion, lung opacity, rib, diaphragm.
  The dominant concept is 'cardiomegaly' (activation=2.341).
```

---

## Diagramma del flusso

```
[Input 1: embeddings/visual_embeddings.pt (7400, 512)]
            |
[Input 2: models/sae_seed42/ae.pt]
            |
    [SAEManager.get_top_concepts(n=5)]
            |
    Per ogni immagine: [(feat_id, activation), ...] x 5
            |
[Input 3: results/concept_names.json]
            |
    [generate_explanation(): feat_id -> nome medico]
            |
    [Costruzione pseudo-report in linguaggio naturale]
            |
[Output: results/sample_explanations.json]
```

---

## Formato dell'output

```json
[
  {
    "findings": [
      {"concept": "cardiomegaly", "feature_id": 127, "activation": 2.3412, "naming_confidence": 0.7234},
      {"concept": "pleural effusion", "feature_id": 892, "activation": 1.8901, "naming_confidence": 0.6541},
      ...
    ],
    "pseudo_report": "The model identifies the following visual concepts...",
    "n_active_concepts": 5,
    "sample_idx": 0
  },
  ...
]
```

---

## Ruolo nella pipeline

Questo script e' il penultimo prima della valutazione finale:

```
01 (embedding) -> 02a (train) -> 02b (naming) -> 02c (explanations) -> 05 (LLM Judge)
```

Le spiegazioni generate qui sono l'input per l'LLM Judge che le valuta
qualitativamente, completando il ciclo di XAI unsupervised concept discovery.
