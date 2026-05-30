# 02a_train_sae.py - Documentazione completa

Questo documento descrive ogni sezione di `src/02a_train_sae.py`, lo script
che addestra multipli SAE (uno per seed) per l'analisi di stabilita'.

---

## 1. Docstring e metadata

```python
"""
02a_train_sae.py - Train Sparse Autoencoders (Top-K)

Train SAEs on BiomedCLIP embeddings with multiple seeds for stability analysis.

Prerequisites:
    - embeddings/visual_embeddings.pt (output of 01_extract_embeddings.py)

Run:
    python src/02a_train_sae.py
"""
```

**Perche:**

Il docstring documenta:
- Lo scopo dello script (training multi-seed)
- I prerequisiti (quali file devono esistere prima di eseguirlo)
- Il comando per lanciarlo

Questo script e' il terzo della pipeline (02a) e presuppone che le embedding
siano gia' state estratte con 01_extract_embeddings.py.

---

## 2. Importazioni e setup del path

```python
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from sae_module import SAEManager
```

**Perche:**

- `sys.path.insert(0, str(Path(__file__).parent))`: aggiunge la directory `src/`
  al Python path, cosi' `import config` e `from sae_module import SAEManager`
  funzionano indipendentemente da dove viene lanciato lo script (root del progetto
  o dalla directory src/ stessa).
- `config`: configurazione centralizzata in dataclass (paths, hyperparams, ecc.)
- `SAEManager`: facade class che gestisce l'intero ciclo di vita del SAE.

---

## 3. Configurazione del logging

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
```

**Perche:**

- `level=logging.INFO`: mostra messaggi informativi e superiori (WARNING, ERROR).
- Il formato include timestamp (ore:minuti:secondi), livello e messaggio.
- `__name__` identifica il logger con il nome del modulo, utile per filtrare
  i messaggi quando piu' moduli loggano contemporaneamente.

---

## 4. Funzione `train_single(seed)`

### 4.1 Creazione del manager e training

```python
def train_single(seed: int):
    """Train a single SAE with the given seed."""
    logger.info(f"Training SAE with seed={seed}")

    mgr = SAEManager({"device": config.hardware.device})
    model_dir = mgr.train(
        embeddings_path=config.paths.visual_embeddings_path,
        seed=seed,
        save_dir=config.paths.models_dir,
        steps=config.sae.steps,
        batch_size=config.sae.batch_size,
    )
```

**Perche:**

- Crea un `SAEManager` passandogli solo il device dalla configurazione.
  Gli altri parametri (activation_dim, dict_size, k, lr, ecc.) vengono dai
  DEFAULT_CONFIG del SAEManager. Se si volessero sovrascrivere, basterebbe
  aggiungerli al dizionario.
- `mgr.train()` fa tutto: carica embedding, crea DataLoader, configura trainer,
  lancia il training, salva il modello, e lo carica in memoria.
- Il `seed` cambia ad ogni iterazione del loop in `main()`, producendo SAE
  con inizializzazione diversa per misurare la stabilita'.
- Il modello viene salvato in `models/sae_seed{N}/ae.pt`.

### 4.2 Sanity check post-training

```python
    embeddings = torch.load(config.paths.visual_embeddings_path, map_location="cpu", weights_only=True)
    sample = embeddings[:256]

    mse = mgr.compute_reconstruction_mse(sample)
    sparsity = mgr.compute_sparsity_metrics(sample)

    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  L0 mean: {sparsity['l0_mean']:.1f} (expected ~{config.sae.k})")
    logger.info(f"  Dead features: {sparsity['dead_features_pct']:.1f}%")
    logger.info(f"  Saved to: {model_dir}")
```

**Perche:**

Dopo il training, verifica immediatamente che il modello funzioni correttamente
su un sottoinsieme di 256 campioni:

1. **MSE (reconstruction loss)**: deve essere basso (ordine 1e-3 o inferiore).
   Se e' alto, il SAE non ha imparato una buona ricostruzione.

2. **L0 mean**: conta media delle attivazioni non-zero per campione.
   Con k=32, ci aspettiamo L0 ~ 32.0. Se diverge significativamente,
   il Top-K enforcement ha un problema.

3. **Dead features %**: percentuale di feature nel dizionario che non si attivano
   mai. Idealmente < 20%. Se e' alta, il dizionario e' troppo grande per i dati.

Questo check non blocca l'esecuzione ma da' un feedback immediato sulla qualita'
del training senza dover lanciare script di analisi separati.

---

## 5. Funzione `main()`

```python
def main():
    if not config.paths.visual_embeddings_path.exists():
        logger.error(f"Embeddings not found: {config.paths.visual_embeddings_path}")
        logger.error("Run first: python src/01_extract_embeddings.py")
        sys.exit(1)

    logger.info(f"Training {len(config.training.seeds)} SAEs: seeds={config.training.seeds}")

    model_dirs = []
    for seed in config.training.seeds:
        model_dir = train_single(seed)
        model_dirs.append(model_dir)

    logger.info(f"All {len(model_dirs)} SAEs trained successfully.")
```

**Perche:**

### Validazione prerequisiti
Prima di iniziare un training potenzialmente lungo, verifica che il file degli
embedding esista. Se manca, stampa un messaggio chiaro con il comando da
eseguire e termina con exit code 1 (errore).

### Loop multi-seed
Itera sui 5 seed definiti in `config.training.seeds = (0, 42, 123, 456, 789)`.
Per ogni seed:
1. Allena un SAE completo (50k step)
2. Esegue il sanity check
3. Salva il path del modello

Al termine si hanno 5 modelli in:
```
models/sae_seed0/ae.pt
models/sae_seed42/ae.pt
models/sae_seed123/ae.pt
models/sae_seed456/ae.pt
models/sae_seed789/ae.pt
```

Questi verranno confrontati in `02d_stability_analysis.py` con Jaccard similarity
per verificare che i concetti appresi siano robusti all'inizializzazione random.

### Raccolta model_dirs
La lista `model_dirs` potrebbe essere usata per operazioni successive nello
stesso script (es. un'analisi di stabilita' inline), ma attualmente serve solo
per il log finale.

---

## 6. Entry point

```python
if __name__ == "__main__":
    main()
```

**Perche:**

Guard standard Python che permette di:
- Eseguire lo script direttamente: `python src/02a_train_sae.py`
- Importare le funzioni senza eseguire il training: `from 02a_train_sae import train_single`

---

## Diagramma del flusso

```
[Input: embeddings/visual_embeddings.pt (7400, 512)]
            |
    for seed in (0, 42, 123, 456, 789):
            |
    [SAEManager.train()] --> models/sae_seed{N}/ae.pt
            |
    [Sanity check: MSE, L0, dead features %]
            |
[Output: 5 modelli addestrati pronti per concept naming e stability analysis]
```

---

## Dipendenze dalla configurazione

| Variabile | Valore default | Usata per |
|-----------|---------------|-----------|
| `config.hardware.device` | `"cuda"` | Device per il training |
| `config.paths.visual_embeddings_path` | `embeddings/visual_embeddings.pt` | Input del training |
| `config.paths.models_dir` | `models/` | Dove salvare i SAE |
| `config.sae.steps` | 50000 | Numero di step di training |
| `config.sae.batch_size` | 256 | Dimensione del batch |
| `config.sae.k` | 32 | Atteso nel sanity check |
| `config.training.seeds` | (0, 42, 123, 456, 789) | Semi per multi-seed |

---

## Tempo di esecuzione stimato

Con 7400 embedding su GPU (RTX 3090): ~5-10 minuti per seed, ~30-50 minuti totali.
Su CPU: significativamente piu' lento (~1-2 ore per seed).
Su Apple Silicon (MPS): ~15-30 minuti per seed.
