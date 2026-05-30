# config.py - Documentazione completa

Questo documento descrive ogni sezione di `src/config.py`, il modulo di
configurazione centralizzata per tutta la pipeline SAE.

---

## 1. Architettura della configurazione

```python
"""
config.py - Central configuration for all pipeline scripts.

Uses dataclasses to group related settings. Each dataclass represents
a logical component of the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
```

**Perche:**

La configurazione e' organizzata in dataclass tematiche anziche' in variabili
globali sparse. Ogni dataclass raggruppa parametri dello stesso dominio:
paths, backbone, SAE, training, ecc.

Vantaggi:
- Autocompletamento nell'IDE (`config.sae.` mostra solo parametri SAE)
- Documentazione integrata (ogni classe ha docstring)
- Validazione dei tipi tramite annotazioni
- `frozen=True` (dove usato) impedisce modifiche accidentali a runtime
- Facile da passare come argomento senza import circolari

---

## 2. PathsConfig

```python
@dataclass
class PathsConfig:
    """Project directory layout and derived file paths."""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    visual_embeddings_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.embeddings_dir = self.project_root / "embeddings"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.vocab_embeddings_path = self.embeddings_dir / "text_vocab_embeddings.pt"
        self.vocab_labels_path = self.data_dir / "vocabulary.json"
```

**Perche:**

### project_root
`Path(__file__).parent.parent` calcola la root del progetto relativamente alla
posizione di config.py stesso (`src/config.py` -> `src/` -> root del progetto).
Questo rende i path indipendenti dalla directory di lavoro corrente.

### field(init=False)
Questi campi non possono essere passati al costruttore. Sono sempre derivati
da `project_root`. `field(init=False)` li esclude da `__init__` ma li include
in `__repr__` e `__eq__`.

### __post_init__
Chiamato automaticamente alla fine di `__init__` generato dalla dataclass.
Calcola tutti i path derivati. I path dei file composti (es. `visual_embeddings_path`)
riutilizzano i campi directory (`self.embeddings_dir`) per evitare ripetizione.

### Non frozen
A differenza delle altre dataclass, `PathsConfig` non e' frozen perche'
`__post_init__` deve assegnare valori ai campi `init=False`. Con `frozen=True`
servirebbe il workaround `object.__setattr__`.

### Struttura delle directory

```
project_root/
  data/             -> vocabulary.json, radlex.csv
  embeddings/       -> visual_embeddings.pt, text_vocab_embeddings.pt
  models/           -> sae_seed{N}/ae.pt
  results/          -> concept_names.json, sample_explanations.json, stability_analysis.json
```

---

## 3. BackboneConfig

```python
@dataclass(frozen=True)
class BackboneConfig:
    """BiomedCLIP backbone model settings."""
    model_id: str = "chuhac/BiomedCLIP-vit-bert-hf"
    embedding_dim: int = 512
```

**Perche:**

- `model_id`: identificativo HuggingFace del modello BiomedCLIP. Usato nello script
  01 per caricare il modello e estrarre le embedding.
- `embedding_dim`: dimensione dell'output di BiomedCLIP (512-dim). Deve corrispondere
  all'`activation_dim` del SAE. Se si cambia backbone, bisogna aggiornare entrambi.
- `frozen=True`: questi valori non devono mai cambiare a runtime - cambiarli
  renderebbe le embedding incompatibili con il SAE.

---

## 4. SAEConfig

```python
@dataclass(frozen=True)
class SAEConfig:
    """Sparse Autoencoder (Top-K) hyperparameters."""
    activation_dim: int = 512
    dict_size: int = 4096
    k: int = 32
    lr: float = 5e-5
    steps: int = 50_000
    warmup_steps: int = 1000
    batch_size: int = 256
```

**Perche:**

### activation_dim = 512
Dimensione degli embedding in input/output del SAE. Deve corrispondere a
`backbone.embedding_dim`. E' la dimensione dello spazio in cui il SAE opera.

### dict_size = 4096
Numero di "concetti" (feature) nel dizionario sparse. Rapporto di overcompleteness
= 4096/512 = 8x. Un dizionario overcomplete permette al SAE di catturare
piu' concetti distinti rispetto alla dimensionalita' dello spazio.

### k = 32
Top-K sparsity: per ogni input, solo le 32 feature con attivazione piu' alta
vengono mantenute, le altre sono azzerate. Questo forza una rappresentazione
sparse dove ogni immagine e' descritta da esattamente 32 concetti.

### lr = 5e-5
Learning rate basso, tipico per training non supervisionato dove la convergenza
deve essere stabile. Valori troppo alti causano instabilita', troppo bassi
rallentano la convergenza.

### steps = 50_000
Il training e' step-based (non epoch-based). Con batch_size=256 e 7400 campioni,
50k step corrispondono a ~1730 epoche. Valore di default - potrebbe essere
ridotto per dataset piccoli (vedi docs/suggestions/).

### warmup_steps = 1000
Riscaldamento lineare del learning rate: LR cresce da 0 a `lr` nei primi 1000 step.
Stabilizza l'inizio del training quando i pesi sono random e i gradienti grandi.

### batch_size = 256
Numero di campioni per step di training. Con 7400 campioni totali, ogni epoca
completa richiede 28.9 step. 256 e' un buon compromesso tra stabilita' del
gradiente e velocita'.

---

## 5. TrainingConfig

```python
@dataclass(frozen=True)
class TrainingConfig:
    """Multi-seed training and stability analysis settings."""
    seeds: tuple[int, ...] = (0, 42, 123, 456, 789)
    stability_max_samples: Optional[int] = None
```

**Perche:**

### seeds = (0, 42, 123, 456, 789)
Cinque semi per l'analisi di stabilita'. Ogni seed produce un SAE con
inizializzazione diversa. Confrontando i risultati si misura la robustezza.
- 5 seed e' sufficiente per stimare media e varianza della Jaccard similarity
- Tuple (immutabile) per coerenza con `frozen=True`

### stability_max_samples = None
Limita opzionalmente il numero di campioni usati nell'analisi di stabilita'.
`None` = usa tutto il dataset. Utile in sviluppo per velocizzare i test.

---

## 6. ExplanationConfig

```python
@dataclass(frozen=True)
class ExplanationConfig:
    """Concept naming and explanation generation settings."""
    concept_top_n: int = 3
    explanation_top_n: int = 5
    explanation_max_samples: Optional[int] = None
```

**Perche:**

### concept_top_n = 3
Quanti candidati di nome restituire per ogni feature nel concept naming (02b).
Top-3 permette di vedere alternative quando il primo nome non e' convincente.

### explanation_top_n = 5
Quanti concetti includere nella spiegazione di ogni immagine (02c).
5 concetti bilanciano completezza e leggibilita' nel pseudo-report.

### explanation_max_samples = None
Come `stability_max_samples`: limita le immagini processate in 02c.
`None` = processa tutto il dataset.

---

## 7. HardwareConfig

```python
@dataclass(frozen=True)
class HardwareConfig:
    """Device and compute settings."""
    device: str = "cuda"
```

**Perche:**

Centralizza la scelta del device. Opzioni:
- `"cuda"`: GPU NVIDIA (default, richiede CUDA toolkit)
- `"mps"`: GPU Apple Silicon (PyTorch 1.12+)
- `"cpu"`: fallback universale, piu' lento

Un unico punto da cambiare per passare da GPU a CPU durante sviluppo/debug.

---

## 8. Istanziazione

```python
# Instantiate configs
paths = PathsConfig()
backbone = BackboneConfig()
sae = SAEConfig()
training = TrainingConfig()
explanation = ExplanationConfig()
hardware = HardwareConfig()

# Hardware
DEVICE = hardware.device
```

**Perche:**

Le istanze sono create a livello di modulo cosi' gli script possono fare:
```python
import config
config.sae.dict_size  # 4096
config.paths.models_dir  # Path(".../models")
```

Le istanze vengono create una sola volta all'import del modulo (singleton pattern
implicito di Python). Ogni script che fa `import config` condivide le stesse
istanze.

`DEVICE` e' un alias di compatibilita' per codice legacy che usava `config.DEVICE`.

---

## Come modificare la configurazione

### Per cambiare un default
Modificare il valore nel campo della dataclass:
```python
dict_size: int = 2048  # era 4096
```

### Per aggiungere un parametro
Aggiungere un campo nella dataclass appropriata:
```python
@dataclass(frozen=True)
class SAEConfig:
    ...
    decay: float = 0.99  # nuovo parametro
```

### Per override temporaneo in uno script
Non modificare config.py. Passare il valore come argomento:
```python
mgr = SAEManager({"device": "cpu", "dict_size": 2048})
```

---

## Relazione tra le configurazioni

```
BackboneConfig.embedding_dim = 512
        |
        v
SAEConfig.activation_dim = 512  (devono coincidere)
        |
        v
SAEConfig.dict_size = 4096  (concetti totali)
        |
        v
SAEConfig.k = 32  (concetti per immagine)
        |
        v
ExplanationConfig.explanation_top_n = 5  (deve essere <= k)
ExplanationConfig.concept_top_n = 3  (candidati per feature)
```

Se `embedding_dim` cambia (es. altro backbone), `activation_dim` deve cambiare
di conseguenza e tutti i modelli vanno ri-addestrati.
