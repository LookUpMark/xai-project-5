# train_sae.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/train_sae.py`, lo script
che prepara lo split train/test e addestra multipli SAE (uno per seed) per l'analisi
di stabilita'.

---

## 1. Docstring e metadata

```python
"""
train_sae.py -- Train Sparse Autoencoders (Top-K)

Train SAEs on BiomedCLIP embeddings with multiple seeds for stability analysis.
Creates train/test split if not already on disk, trains on train split only,
evaluates sanity checks on held-out test set.

Prerequisites:
    - embeddings/visual_embeddings.pt (output of 01_extract_embeddings.py)

Run:
    python src/autoencoder/train_sae.py
"""
```

**Perche:**

Il docstring documenta lo scopo (training multi-seed con split train/test),
il prerequisito (embedding estratte) e il comando di lancio. Questo script e'
il primo della pipeline SAE e implementa due fasi: split delle embedding e
training con sanity check su test set held-out.

---

## 2. Importazioni e setup

```python
import logging, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
from autoencoder.tracking import init_tracking, log_metrics, log_artifact, finish_tracking
```

**Perche:**

- `sys.path.insert(0, ...)`: aggiunge `src/` al Python path per importare
  `config` e `autoencoder.*` indipendentemente dalla directory di lancio.
- `numpy`: usato per `train_test_split` e generazione di sottoinsiemi randomici.
- `SAEManager`: facade class per il ciclo di vita del SAE.
- `_set_global_seed`: propagazione del seed a torch/numpy/random.
- `tracking.*`: wrapper wandb con degrado elegante (no-op se disabilitato).

---

## 3. Funzione `prepare_split()`

```python
def prepare_split() -> None:
    train_path = config.paths.train_embeddings_path
    test_path = config.paths.test_embeddings_path

    if train_path.exists() and test_path.exists():
        logger.info("Train/test splits already exist -- skipping.")
        return

    from sklearn.model_selection import train_test_split

    source = config.paths.visual_embeddings_path
    if not source.exists():
        raise FileNotFoundError(f"Embeddings not found: {source}.")

    embeddings = torch.load(source, map_location="cpu", weights_only=True)
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=config.training.train_split_ratio,
        random_state=config.training.split_seed,
    )

    torch.save(embeddings[train_idx], train_path)
    torch.save(embeddings[test_idx], test_path)
```

**Perche:**

### Skip se gia' esiste

Se `train_embeddings.pt` e `test_embeddings.pt` esistono gia', non ricrea lo
split. Lo split e' deterministico (`split_seed=42`), quindi riproducibile.

### Import lazy di sklearn

L'import e' dentro la funzione perche' sklearn e' pesante. Se lo split esiste,
non viene mai importato. Rende sklearn una dipendenza de facto opzionale.

### train_test_split con indici

```python
indices = np.arange(len(embeddings))
train_idx, test_idx = train_test_split(indices, train_size=0.8, random_state=42)
```

Si passano indici (non le embedding direttamente) per garantire deterministicita'
e per poter tracciare quali campioni vanno in train vs test.

### Config

- `train_split_ratio=0.8`: 80/20 e' lo standard per dataset di dimensione media.
- `split_seed=42`: garantisce riproducibilita' dello split.

---

## 4. Funzione `train_single(seed)`

### 4.1 Config injection completa

```python
def train_single(seed: int) -> Path:
    mgr = SAEManager({
        "device": config.hardware.device,
        "activation_dim": config.sae.activation_dim,
        "dict_size": config.sae.dict_size,
        "k": config.sae.k,
        "lr": config.sae.lr,
        "warmup_steps": config.sae.warmup_steps,
        "log_steps": config.sae.log_steps,
        "decay_start_frac": config.sae.decay_start_frac,
    })
```

**Perche:**

Tutti i campi di `SAEConfig` vengono iniettati esplicitamente, non si affidano ai
DEFAULT_CONFIG del SAEManager. Questo perche':

1. **Singola fonte di verita**: `config.py` e' l'unico posto per gli iperparametri.
2. **Tracciabilita'**: ogni run ha parametri tracciati in wandb e nel manifest.
3. **Ablazione**: basta cambiare config.py senza toccare il codice.

I parametri: `activation_dim=512`, `dict_size=4096`, `k=32` (Top-K), `lr=None`
(auto-scale dalla libreria: ~4e-4 per dict_size=4096), `warmup_steps=1000`,
`log_steps=1000`, `decay_start_frac=0.8` (inizio decay LR all'80% degli step).

### 4.2 Training su train set

```python
    model_dir = mgr.train(
        embeddings_path=config.paths.train_embeddings_path,
        seed=seed, save_dir=config.paths.models_dir,
        steps=config.sae.steps, batch_size=config.sae.batch_size,
    )
```

**Perche:**

Usa `train_embeddings_path` (non `visual_embeddings.pt`): il training avviene
solo sull'80% dei dati. Il 20% e' tenuto da parte per il sanity check.

### 4.3 Sanity check su TEST set

```python
    test_emb = torch.load(config.paths.test_embeddings_path, map_location="cpu", weights_only=True)
    n_check = min(config.training.sanity_check_samples, len(test_emb))
    rng = np.random.default_rng(seed)
    check_idx = rng.choice(len(test_emb), size=n_check, replace=False)
    sample = test_emb[check_idx]

    mse = mgr.compute_reconstruction_mse(sample)
    cosine = mgr.compute_cosine_reconstruction(sample)
    sparsity = mgr.compute_sparsity_metrics(sample)
```

**Perche:**

Il sanity check valuta su dati **mai visti durante il training** (held-out test set):

1. **Overfitting detection**: MSE su test molto piu' alto di train = overfitting.
2. **Seed propagation**: `np.random.default_rng(seed)` seleziona un sottoinsieme
   casuale ma deterministico. Non uno slice posizionale (che potrebbe introdurre
   bias se le embedding sono ordinate per classe).
3. **Metriche complementari**:

| Metrica | Misura | Valore atteso |
|---------|--------|---------------|
| Test MSE | Errore ricostruzione | 1e-3 o inferiore |
| Test Cosine | Allineamento direzionale | > 0.95 |
| L0 mean | Feature attive/campione | ~k (es. 32) |
| Dead features % | Feature mai attivate | < 20% |
| Dict utilization % | Feature usate almeno una volta | > 80% |

### 4.4 Tracking Weights & Biases

```python
    if config.wandb_cfg.enabled:
        log_metrics({
            f"train/seed{seed}/test_mse": mse,
            f"train/seed{seed}/test_cosine": cosine,
            f"train/seed{seed}/dead_pct": sparsity["dead_features_pct"],
            f"train/seed{seed}/dict_util": sparsity["dict_utilization_pct"],
        })
        log_artifact(model_dir / "training_manifest.json", f"sae_seed{seed}_manifest", "manifest")
```

**Perche:**

Metriche namespaced per seed (`train/seed42/test_mse`) per confronto diretto
in dashboard wandb. Il manifest viene salvato come artefatto per riproducibilita'.
Protetto da `if wandb_cfg.enabled`: zero overhead se disabilitato.

---

## 5. Funzione `main()`

```python
def main():
    prepare_split()

    if not config.paths.train_embeddings_path.exists():
        logger.error("Train embeddings not found.")
        sys.exit(1)

    logger.info(f"PyTorch: {torch.__version__}, Device: {config.hardware.device}")
    logger.info(f"Training {len(config.training.seeds)} SAEs: seeds={config.training.seeds}")

    if config.wandb_cfg.enabled:
        init_tracking("train_sae", {
            "project": config.wandb_cfg.project, "entity": config.wandb_cfg.entity,
            "seeds": list(config.training.seeds), "k": config.sae.k,
            "dict_size": config.sae.dict_size, "steps": config.sae.steps,
        })

    model_dirs = []
    for seed in config.training.seeds:
        model_dir = train_single(seed)
        model_dirs.append(model_dir)

    logger.info(f"All {len(model_dirs)} SAEs trained successfully.")
    finish_tracking()
```

**Perche:**

### Validazione post-split

Dopo `prepare_split()`, verifica che il file di train esista. Se `prepare_split()`
ha fallito (es. sklearn non installato e file mancanti), questo check intercetta
il problema.

### Log ambiente

Versione PyTorch, device, seed e iperparametri per confronto tra run.

### Loop multi-seed

Itera su `config.training.seeds = (0, 42, 123, 456, 789)`. Per ogni seed:
allena, sanity check su test, salva. Al termine: 5 modelli in
`models/sae_seed{N}/ae.pt`.

### Tracking

Una singola run wandb per l'intero training multi-seed. `finish_tracking()`
chiude la run garantendo sincronizzazione.

---

## Diagramma del flusso

```text
[Input: embeddings/visual_embeddings.pt (7400, 512)]
            |
    +--[prepare_split()]-------+
    |   sklearn 80/20         |
    |   split_seed=42          |
    +-------------------------+
            |
    train (5920)              test (1480)
            |                       |
    for seed in (0,42,123,456,789):
            |
    +--[SAEManager.train()]--+     models/sae_seed{N}/ae.pt
    |   Full config injection |
    +------------------------+
            |
    +--[Sanity Check]--------+
    |   TEST set held-out     |
    |   Random subset (seed)  |
    |   MSE, Cosine, L0, etc |
    +------------------------+
```

---

## Dipendenze dalla configurazione

| Variabile | Section | Default | Usata per |
|-----------|---------|---------|-----------|
| `config.hardware.device` | HardwareConfig | auto | Device |
| `config.paths.visual_embeddings_path` | PathsConfig | `embeddings/visual_embeddings.pt` | Sorgente split |
| `config.paths.train_embeddings_path` | PathsConfig | `embeddings/train_embeddings.pt` | Training |
| `config.paths.test_embeddings_path` | PathsConfig | `embeddings/test_embeddings.pt` | Sanity check |
| `config.sae.*` | SAEConfig | k=32, dict=4096, ... | Tutti gli iperparametri |
| `config.training.seeds` | TrainingConfig | (0,42,123,456,789) | Semi multi-seed |
| `config.training.split_seed` | TrainingConfig | 42 | Seed per lo split |
| `config.training.train_split_ratio` | TrainingConfig | 0.8 | Rapporto train/test |
| `config.training.sanity_check_samples` | TrainingConfig | 256 | Campioni per check |
| `config.wandb_cfg.*` | WandbConfig | disabled | Tracking |

---

## Relazione con gli altri script

```text
train_sae (split + train 5 SAEs + sanity check)
    +---> concept_naming (usa sae_seed42)
    +---> generate_explanations (usa sae_seed42 + test embeddings)
    +---> stability_analysis (confronta tutti 5 i seed su test embeddings)
```
