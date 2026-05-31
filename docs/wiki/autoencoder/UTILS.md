# utils.py - Documentazione completa

Questo documento descrive `src/utils.py`, il modulo delle utility condivise
dall'intera pipeline (sia estrazione VLM che autoencoder).

---

## 1. Panoramica

```python
"""Shared utilities for the SAE concept-discovery pipeline."""

from __future__ import annotations

import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from config import VLMConfig
```

Il modulo raccoglie funzioni generiche usate da piu' moduli della pipeline,
eliminando duplicazione di codice. Ogni funzione e' domain-agnostic: nessuna
dipendenza da strutture specifiche del SAE o del naming.

---

## 2. load_vlm(config)

```python
def load_vlm(config: VLMConfig):
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(config.processor_name, trust_remote_code=True)
    model.eval().to(config.device)
    return model, processor
```

**Perche:**

- Carica il backbone BiomedCLIP e il processor per l'estrazione embedding.
- `trust_remote_code=True`: necessario per modelli con codice custom su HuggingFace.
- `model.eval()`: disabilita dropout e batch-norm in training mode.
- `.to(config.device)`: usa il device specificato in `VLMConfig` (auto-detected).
- Usato da `src/extract_embeddings.py` (Member 1 pipeline).

---

## 3. set_global_seed(seed)

```python
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Perche:**

Garantisce riproducibilita' completa impostando i seed di tutti i generatori
random (Python stdlib, NumPy, PyTorch CPU e CUDA). `cudnn.deterministic = True`
forza algoritmi deterministici (piu' lenti ma riproducibili).
`cudnn.benchmark = False` disabilita l'auto-tuning di cuDNN che puo' variare
tra run.

Chiamato da `SAEManager.train()` prima di ogni training per seed.

---

## 4. load_tensor(path, device)

```python
def load_tensor(path: str | Path, device: str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)
```

**Perche:**

- `weights_only=True`: protezione di sicurezza contro pickle injection
  (deserializzazione di codice arbitrario). Senza questo flag, un file `.pt`
  malevolo potrebbe eseguire codice al caricamento.
- `map_location=device`: carica direttamente sul device target senza passare
  per la GPU (evita OOM su modelli pesanti).
- Usato in 7 punti della pipeline: train_sae (2x), concept_naming, 
  generate_explanations, stability_analysis, sae_module (2x).

---

## 5. ensure_dir(path)

```python
def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
```

**Perche:**

Crea le directory parent prima di salvare un file. `parents=True` crea anche
le directory intermedie. `exist_ok=True` non solleva errore se esistono gia'.

Usato da `visualization.py` prima di salvare ogni figura.

---

## 6. setup_logging(name)

```python
def setup_logging(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
```

**Perche:**

Configura il logging con formato standard per tutti gli script della pipeline:
`HH:MM:SS | LEVEL | messaggio`. Restituisce un logger nominato per il modulo
chiamante. `basicConfig` e' idempotente (la seconda chiamata non ha effetto).

Usato da: train_sae, concept_naming, generate_explanations, stability_analysis.

---

## 7. dataclass_to_dict(obj)

```python
def dataclass_to_dict(obj) -> dict:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
```

**Perche:**

Converte qualsiasi dataclass (frozen o mutable) in un dict piatto. Utile per:
- Serializzazione JSON di configurazioni
- Passaggio di config a librerie che accettano dict (es. `dictionary_learning`)
- Logging di iperparametri su wandb

A differenza di `dataclasses.asdict()`, non effettua deep-copy ricorsiva dei
campi (piu' efficiente per dataclass semplici senza campi annidati).

Usato da `SAEManager._extract_sae_config()`.
