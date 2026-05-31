# concept_naming.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/concept_naming.py`, lo script
che assegna nomi medici alle feature apprese dal SAE tramite cosine similarity con un
vocabolario medico (RadLex), produce statistiche riassuntive e genera una visualizzazione
della distribuzione degli score.

---

## 1. Docstring e metadata

```python
"""
concept_naming.py -- Assign names to SAE concepts

Assign medical names to the SAE features using cosine similarity
between decoder weights and vocabulary embeddings.

Prerequisites:
    - models/sae_seed{PRIMARY_SEED}/ae.pt
    - embeddings/text_vocab_embeddings.pt
    - data/vocabulary.json

Run:
    python src/autoencoder/concept_naming.py
"""
```

**Perche:**

Lo script necessita di tre input: un SAE addestrato (pesi decoder), embedding
del vocabolario medico (vettori 512-dim RadLex), e le label human-readable.
L'idea: ogni riga del decoder e' una "direzione" nello spazio 512-dim. Se quella
direzione e' simile all'embedding di "pneumothorax", la feature cattura il
concetto di pneumotorace. `{PRIMARY_SEED}` indica che il seed e' configurabile.

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
from autoencoder.visualization import plot_concept_score_distribution

# Use primary_seed from config (not fragile seeds[1] index)
SEED = config.training.primary_seed
OUTPUT_PATH = config.paths.results_dir / "concept_names.json"
```

**Perche:**

- `tracking`: integrazione wandb per tracciare risultati del naming.
- `visualization.plot_concept_score_distribution`: genera istogramma degli score.
- **`config.training.primary_seed` vs `seeds[1]`**: la versione precedente usava
  `seeds[1]` (fragile: riordinando la tupla si cambia il modello silenziosamente).
  Ora usa `primary_seed`, campo esplicito validato da `__post_init__` che verifica
  sia nella tupla dei seed. Impossibile usare un seed non addestrato.

---

## 3. Funzione `run()`

La funzione `run()` contiene tutta la logica e restituisce il path dell'output.
Separata da `main()` per testabilita' e composizione in pipeline.

### 3.1 Validazione prerequisiti

```python
def run() -> Path:
    model_dir = config.paths.models_dir / f"sae_seed{SEED}"
    for path, desc in [
        (model_dir, "SAE model"),
        (config.paths.vocab_embeddings_path, "Vocab embeddings"),
        (config.paths.vocab_labels_path, "Vocabulary labels"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")
```

**Perche:**

Pattern standard della pipeline: verifica tutti i prerequisiti prima di operazioni
costose. `raise FileNotFoundError` (non `sys.exit(1)`) permette al chiamante di
gestire l'errore nel contesto appropriato (test, pipeline orchestrata).

### 3.2 Caricamento vocabolario

```python
    with open(config.paths.vocab_labels_path) as f:
        vocab_labels = json.load(f)
    vocab_embeddings = torch.load(
        config.paths.vocab_embeddings_path, map_location="cpu", weights_only=True
    )
```

**Perche:**

Due file complementari con corrispondenza posizionale: `vocab_labels[i]` corrisponde
a `vocab_embeddings[i]`. Entrambi prodotti da `00_build_vocabulary.py`.

### 3.3 Naming dei concetti

```python
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dir)

    concept_names = mgr.name_concepts(
        vocab_embeddings, vocab_labels, top_n=config.explanation.concept_top_n
    )
```

**Perche:**

`name_concepts()` internamente: (1) estrae W_dec (dict_size, 512), (2) normalizza
W_dec e vocab a norma unitaria, (3) calcola cosine similarity, (4) per ogni feature
seleziona i top_n=3 termini piu' simili. Risultato:

```json
{
  "0": {"name": "pneumothorax", "score": 0.72, "candidates": [...]},
  ...
}
```

### 3.4 Statistiche riassuntive

```python
    scores = [v["score"] for v in concept_names.values()]
    mean_score = sum(scores) / len(scores)
    logger.info(f"Total features: {len(concept_names)}")
    logger.info(f"Mean score: {mean_score:.4f}")
    logger.info(f"Min/Max: {min(scores):.4f} / {max(scores):.4f}")
```

**Perche:**

Feedback immediato sulla qualita'. Mean score atteso: 0.3-0.6 per buon naming.
Min/Max identifica feature eccellenti (>0.7) o problematiche (<0.2). Total features
deve essere uguale a `dict_size` (4096).

### 3.5 Top-10 per score

```python
    sorted_concepts = sorted(concept_names.items(), key=lambda x: x[1]["score"], reverse=True)
    for feat_id, info in sorted_concepts[:10]:
        logger.info(f"  Feature {feat_id:>4}: {info['name']:30s} ({info['score']:.4f})")
```

**Perche:**

Verifica rapida senza aprire il JSON. Se i top-10 nomi hanno senso clinico e
score >0.5, il naming e' riuscito. Formattazione tabellare con allineamento.

### 3.6 Visualizzazione

```python
    fig_path = config.paths.figures_dir / "concept_score_distribution.png"
    plot_concept_score_distribution(scores, fig_path)
```

**Perche:**

Genera un istogramma con KDE della distribuzione degli score. Rivela:
- Distribuzione unimodale (buono) vs bimodale ("interpretabili" vs "non interpretabili").
- Molti score bassi: vocabolario inadeguato per il dominio visivo.
- Linea rossa verticale per la media. Salvato in `results/figures/`.

### 3.7 Tracking Weights & Biases

```python
    if config.wandb_cfg.enabled:
        init_tracking("concept_naming", {
            "project": config.wandb_cfg.project,
            "seed": SEED,
            "total_features": len(concept_names),
            "mean_score": mean_score,
        })
        log_artifact(OUTPUT_PATH, "concept_names", "results")
        finish_tracking()
```

**Perche:**

Run wandb dedicata per il naming. Passa seed, numero feature e score medio come
config. Salva JSON come artefatto. Protetto da `if wandb_cfg.enabled`.

---

## 4. Funzione `main()`

```python
def main():
    run()

if __name__ == "__main__":
    main()
```

**Perche:**

Pattern standard: `run()` testabile e con valore di ritorno, `main()` wrapper
per esecuzione diretta. `__name__` guard per import senza side effects.

---

## Diagramma del flusso

```
[Input 1: models/sae_seed{primary_seed}/ae.pt]
            |
    [Estrai W_dec (dict_size, 512)]
            |
[Input 2: text_vocab_embeddings.pt (V, 512)]
            |
    [Normalizza + cosine similarity]
    [Per ogni feature: top-n termini piu' simili]
            |
    +--[Salva results/concept_names.json]--+
    |   +--[Statistiche: mean, min, max]    |
    |   +--[Top-10 per score]              |
    |   +--[Istogramma score distribution] |
    |   +--[Tracking wandb]                |
    +----------------------------------------+
```

---

## Formato dell'output

```json
{
  "0": {
    "name": "pneumothorax",
    "score": 0.7234,
    "candidates": [
      {"label": "pneumothorax", "score": 0.7234},
      {"label": "pleural line", "score": 0.6512},
      {"label": "air space", "score": 0.6021}
    ]
  }
}
```

`candidates` (top_n=3) utile per debugging: se il nome #1 e' errato, il #2
potrebbe essere piu' appropriato.

---

## Interpretazione dei risultati

| Score range | Significato |
|-------------|-------------|
| > 0.7 | Feature fortemente allineata con termine medico preciso |
| 0.4 - 0.7 | Buon allineamento, nome probabilmente corretto |
| 0.2 - 0.4 | Allineamento debole, approssimazione |
| < 0.2 | Non catturata dal vocabolario (concetto visivo non medico?) |

---

## Dipendenze dalla configurazione

| Variabile | Section | Default | Usata per |
|-----------|---------|---------|-----------|
| `config.training.primary_seed` | TrainingConfig | 42 | Modello di riferimento |
| `config.hardware.device` | HardwareConfig | auto | Device SAE |
| `config.paths.vocab_embeddings_path` | PathsConfig | `embeddings/text_vocab_embeddings.pt` | Embedding vocab |
| `config.paths.vocab_labels_path` | PathsConfig | `data/vocabulary.json` | Label vocab |
| `config.paths.results_dir` | PathsConfig | `results/` | Output JSON |
| `config.paths.figures_dir` | PathsConfig | `results/figures/` | Output grafico |
| `config.explanation.concept_top_n` | ExplanationConfig | 3 | Candidati per feature |
| `config.wandb_cfg.enabled` | WandbConfig | False | Tracking |

---

## Relazione con gli altri script

```
train_sae --> concept_naming (usa sae_seed{primary_seed})
                +---> generate_explanations (usa concept_names.json)
```

I concept names sono il ponte tra feature IDs numeriche e spiegazioni leggibili.
Senza questo step, le spiegazioni sarebbero solo numeri.
