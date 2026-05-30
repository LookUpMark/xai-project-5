# concept_naming.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/concept_naming.py`, lo script
che assegna nomi medici alle 4096 feature apprese dal SAE tramite cosine similarity
con un vocabolario medico (RadLex).

---

## 1. Docstring e metadata

```python
"""
concept_naming.py - Assign names to SAE concepts

Assign medical names to the 4096 SAE features using cosine similarity
between decoder weights and vocabulary embeddings.

Prerequisites:
    - models/sae_seed{SEED}/ae.pt
    - embeddings/text_vocab_embeddings.pt
    - data/vocabulary.json

Run:
    python src/autoencoder/concept_naming.py
"""
```

**Perche:**

Lo script necessita di tre input:
1. Un SAE addestrato (da train_sae) - per estrarre i pesi del decoder
2. Le embedding del vocabolario medico - vettori 512-dim dei termini RadLex
3. Le label del vocabolario - i nomi human-readable dei termini

L'idea: ogni colonna del decoder rappresenta una "direzione" nello spazio 512-dim.
Se quella direzione e' simile all'embedding di "pneumothorax", allora quella
feature cattura il concetto di pneumotorace.

---

## 2. Importazioni e configurazione

```python
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
```

**Perche:**

- `json`: per caricare il vocabolario (lista di stringhe) e salvare i risultati
- Stessa struttura di train_sae: path injection, config centralizzata, SAEManager facade

---

## 3. Costanti

```python
SEED = config.training.seeds[1]  # Use seed 42 as the primary/reference model
OUTPUT_PATH = config.paths.results_dir / "concept_names.json"
```

**Perche:**

- Usa seed 42 (indice 1 nella tupla) come modello di riferimento. E' il seed
  piu' usato in letteratura e sara' lo stesso usato per le spiegazioni (generate_explanations).
- Il risultato va in `results/concept_names.json`, un JSON con mapping
  feature_id -> nome medico.

---

## 4. Validazione prerequisiti

```python
def main():
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"

    for path, desc in [
        (model_dir, "SAE model"),
        (config.paths.vocab_embeddings_path, "Vocab embeddings"),
        (config.paths.vocab_labels_path, "Vocabulary labels"),
    ]:
        if not path.exists():
            logger.error(f"{desc} not found: {path}")
            sys.exit(1)
```

**Perche:**

Pattern riutilizzato in tutti gli script: prima di fare qualsiasi operazione
costosa, verifica che tutti i file prerequisiti esistano. Il loop con tuple
(path, descrizione) evita ripetizione di codice.

---

## 5. Caricamento vocabolario

```python
    with open(config.paths.vocab_labels_path) as f:
        vocab_labels = json.load(f)
    logger.info(f"Vocabulary: {len(vocab_labels)} terms")

    vocab_embeddings = torch.load(config.paths.vocab_embeddings_path, map_location="cpu", weights_only=True)
    logger.info(f"Vocab embeddings shape: {vocab_embeddings.shape}")
```

**Perche:**

Due file complementari:
- `vocabulary.json`: lista di stringhe tipo `["pneumothorax", "cardiomegaly", "pleural effusion", ...]`
- `text_vocab_embeddings.pt`: tensor (V, 512) dove ogni riga e' l'embedding BiomedCLIP
  del termine corrispondente nella lista

La corrispondenza e' posizionale: `vocab_labels[i]` corrisponde a `vocab_embeddings[i]`.
Entrambi sono prodotti dallo script 00_build_vocabulary.py che processa i termini RadLex
attraverso il text encoder di BiomedCLIP.

---

## 6. Caricamento SAE e naming

```python
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    logger.info(f"Computing concept names (top_n={config.explanation.concept_top_n})...")
    concept_names = mgr.name_concepts(vocab_embeddings, vocab_labels, top_n=config.explanation.concept_top_n)
```

**Perche:**

- Carica il SAE seed=42
- `name_concepts()` internamente:
  1. Estrae la matrice decoder W_dec (4096, 512)
  2. Normalizza W_dec e vocab_embeddings a norma unitaria
  3. Calcola la cosine similarity tra ogni feature e ogni termine
  4. Per ogni feature, seleziona i top_n=3 termini piu' simili

Il risultato e' un dizionario:
```json
{
  "0": {"name": "pneumothorax", "score": 0.72, "candidates": [...]},
  "1": {"name": "cardiomegaly", "score": 0.68, "candidates": [...]},
  ...
  "4095": {"name": "rib fracture", "score": 0.45, "candidates": [...]}
}
```

---

## 7. Salvataggio risultati

```python
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(concept_names, f, indent=2, ensure_ascii=False)
```

**Perche:**

- `mkdir(parents=True, exist_ok=True)`: crea la directory `results/` se non esiste.
- `ensure_ascii=False`: mantiene caratteri Unicode nei nomi medici (es. accenti
  in terminologia non-inglese) senza escape `\uXXXX`.
- `indent=2`: formattazione leggibile per ispezione manuale.

---

## 8. Statistiche riassuntive

```python
    scores = [v["score"] for v in concept_names.values()]
    logger.info(f"Concept naming complete:")
    logger.info(f"  Total features: {len(concept_names)}")
    logger.info(f"  Mean score: {sum(scores)/len(scores):.4f}")
    logger.info(f"  Min/Max: {min(scores):.4f} / {max(scores):.4f}")
    logger.info(f"  Saved to: {OUTPUT_PATH}")
```

**Perche:**

Feedback immediato sulla qualita' del naming:
- **Mean score**: cosine similarity media tra feature e nome assegnato.
  Valori attesi: 0.3-0.6 per un buon naming. Se < 0.2 i concetti non hanno
  corrispondenza chiara nel vocabolario.
- **Min/Max**: identifica feature con naming eccellente (>0.7) o problematico (<0.2).

---

## 9. Top-10 log

```python
    sorted_concepts = sorted(concept_names.items(), key=lambda x: x[1]["score"], reverse=True)
    logger.info(f"\nTop-10 concepts:")
    for feat_id, info in sorted_concepts[:10]:
        logger.info(f"  Feature {feat_id:>4s}: {info['name']:30s} ({info['score']:.4f})")
```

**Perche:**

Mostra le 10 feature con naming piu' sicuro (score piu' alto). Questo da'
un'idea rapida della qualita' senza dover aprire il JSON. Se i nomi hanno senso
clinico e gli score sono >0.5, il naming e' riuscito.

---

## Diagramma del flusso

```
[Input 1: models/sae_seed42/ae.pt]
            |
    [Estrai W_dec (4096, 512)]
            |
[Input 2: text_vocab_embeddings.pt (V, 512)]
            |
    [Normalizza entrambi]
            |
    [Cosine similarity: W_dec_norm @ V_norm.T = (4096, V)]
            |
    [Per ogni feature: top-3 termini piu' simili]
            |
[Output: results/concept_names.json]
```

---

## Interpretazione dei risultati

| Score range | Significato |
|-------------|-------------|
| > 0.7 | Feature fortemente allineata con un termine medico preciso |
| 0.4 - 0.7 | Buon allineamento, nome probabilmente corretto |
| 0.2 - 0.4 | Allineamento debole, il nome e' una approssimazione |
| < 0.2 | Feature non ben catturata dal vocabolario (potrebbe rappresentare un concetto visivo non medico) |

Features con score basso non sono necessariamente "cattive" - potrebbero catturare
concetti visivi legittimi (orientamento, contrasto, struttura) che non hanno
corrispondenti nel vocabolario medico testuale.
