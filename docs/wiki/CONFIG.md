# config.py - Documentazione completa

Questo documento descrive ogni sezione di `src/config.py`, il modulo di
configurazione centralizzata per tutta la pipeline SAE.

---

## 1. Architettura della configurazione

```python
"""
config.py - Central configuration for all pipeline scripts.

Uses dataclasses to group related settings. Each dataclass represents
a logical component of the pipeline. Frozen dataclasses provide
immutability guarantees; validation happens in __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
```

**Perche:**

La configurazione e' organizzata in dataclass tematiche anziche' in variabili
globali sparse. Ogni dataclass raggruppa parametri dello stesso dominio:
paths, backbone, SAE, training, explanation, wandb, hardware.

Vantaggi:

- Autocompletamento nell'IDE (`config.sae.` mostra solo parametri SAE)
- Documentazione integrata (ogni classe ha docstring)
- Validazione dei tipi tramite annotazioni e controlli in `__post_init__`
- `frozen=True` (dove usato) impedisce modifiche accidentali a runtime
- Facile da passare come argomento senza import circolari

Novita' rispetto alla versione precedente:

- `import torch` e' stato aggiunto per il rilevamento automatico della GPU
  in `HardwareConfig`.
- `from __future__ import annotations` permette di usare `str | Path` come type hint
  anche su Python < 3.10 (valutazione lazy delle annotazioni).
- La validazione dei parametri non e' piu' solo dichiarativa (annotazioni di tipo)
  ma esecutiva tramite `__post_init__` nelle dataclass che lo supportano.

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
    figures_dir: Path = field(init=False)
    visual_embeddings_path: Path = field(init=False)
    train_embeddings_path: Path = field(init=False)
    test_embeddings_path: Path = field(init=False)
    vocab_embeddings_path: Path = field(init=False)
    vocab_labels_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.embeddings_dir = self.project_root / "embeddings"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.results_dir / "figures"
        self.visual_embeddings_path = self.embeddings_dir / "visual_embeddings.pt"
        self.train_embeddings_path = self.embeddings_dir / "train_embeddings.pt"
        self.test_embeddings_path = self.embeddings_dir / "test_embeddings.pt"
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

### **post_init**

Chiamato automaticamente alla fine di `__init__` generato dalla dataclass.
Calcola tutti i path derivati. I path dei file composti (es. `visual_embeddings_path`)
riutilizzano i campi directory (`self.embeddings_dir`) per evitare ripetizione.

### Non frozen

A differenza delle altre dataclass, `PathsConfig` non e' frozen perche'
`__post_init__` deve assegnare valori ai campi `init=False`. Con `frozen=True`
servirebbe il workaround `object.__setattr__`.

### Novita': figures_dir

`figures_dir = self.results_dir / "figures"` e' una nuova directory dedicata
all'output dei grafici e delle figure (es. loss curves, activation histograms).
Prima le figure venivano salvate direttamente in `results_dir`, creando confusione
tra file JSON di risultati e immagini PNG/SVG.

### Novita': train_embeddings_path e test_embeddings_path

I path separati per le embedding di train e test riflettono il nuovo flusso della
pipeline in cui le embedding vengono divise in due set fin dallo script di split
(vedi `TrainingConfig.train_split_ratio`). Questo e' fondamentale per la
valutazione out-of-sample del SAE, che prima non era possibile.

### Struttura delle directory

```text
project_root/
  data/                 -> vocabulary.json, radlex.csv
  embeddings/           -> visual_embeddings.pt, train_embeddings.pt,
                          test_embeddings.pt, text_vocab_embeddings.pt
  models/               -> sae_seed{N}/ae.pt
  results/
    figures/            -> loss_curves.png, activation_histograms.png, ...
    concept_names.json
    sample_explanations.json
    stability_analysis.json
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
  di estrazione embedding per caricare il modello.
- `embedding_dim`: dimensione dell'output di BiomedCLIP (512-dim). Deve corrispondere
  all'`activation_dim` del SAE. Se si cambia backbone, bisogna aggiornare entrambi.
- `frozen=True`: questi valori non devono mai cambiare a runtime -- cambiarli
  renderebbe le embedding incompatibili con il SAE.

---

## 3b. VLMConfig

```python
@dataclass
class VLMConfig:
    """VLM embedding extraction settings (Member 1 pipeline)."""

    model_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    processor_name: str = "chuhac/BiomedCLIP-vit-bert-hf"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir: str = "data/iu_xray/images/images_normalized"
    reports_dir: str = "data/iu_xray/reports"
    output_dir: str = "embeddings"
    visual_output_filename: str = "visual_embeddings.pt"
    text_output_filename: str = "text_embeddings.pt"
```

**Perche:**

- Config per lo script di estrazione embedding (`src/extract_embeddings.py`).
- `model_name` / `processor_name`: identificativi HuggingFace separati (per BiomedCLIP
  sono identici, ma la separazione supporta modelli con processor diverso).
- `device`: auto-detected — usa CUDA se disponibile, altrimenti CPU.
- `batch_size` / `num_workers`: parametri DataLoader per l'estrazione.
- Properties `visual_output_path` / `text_output_path`: path derivati per output.
- Non frozen: permette override da test.
- Usata da `utils.load_vlm()` e `tests/test_load_vlm.py`.

---

## 4. SAEConfig

```python
@dataclass(frozen=True)
class SAEConfig:
    """Sparse Autoencoder (Top-K) hyperparameters.

    Ablation presets for small datasets (N ~ 7400):
        Conservative:  k=16, dict_size=2048, lr=None, steps=30_000
        Default:       k=32, dict_size=4096, lr=None, steps=50_000
        Aggressive:    k=64, dict_size=4096, lr=None, steps=80_000

    lr=None triggers the library's auto-scaling: 2e-4 / sqrt(dict_size / 16384).
    For dict_size=4096 this gives ~4e-4. For small datasets, consider overriding
    to a lower value (e.g. 5e-5) to avoid overfitting.
    """

    activation_dim: int = 512
    dict_size: int = 4096
    k: int = 32
    lr: Optional[float] = None  # None = auto-scale from library
    steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    log_steps: int = 1_000
    decay_start_frac: float = 0.8  # fraction of steps to start LR decay

    def __post_init__(self):
        if self.dict_size <= self.activation_dim:
            raise ValueError(
                f"dict_size ({self.dict_size}) must exceed "
                f"activation_dim ({self.activation_dim})"
            )
        if self.k >= self.dict_size:
            raise ValueError(
                f"k ({self.k}) must be less than dict_size ({self.dict_size})"
            )
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.warmup_steps >= self.steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < steps ({self.steps})"
            )
        if not (0.0 < self.decay_start_frac <= 1.0):
            raise ValueError(
                f"decay_start_frac must be in (0, 1], got {self.decay_start_frac}"
            )
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

### lr: Optional[float] = None (novita' importante)

Il learning rate e' ora `Optional[float]` con default `None`. Questo attiva
l'auto-scaling della libreria `dictionary_learning`:

```text
lr_auto = 2e-4 / sqrt(dict_size / 16384)
```

Per `dict_size=4096`: `lr_auto = 2e-4 / sqrt(0.25) = 2e-4 / 0.5 = 4e-4`.
Per `dict_size=2048`: `lr_auto = 2e-4 / sqrt(0.125) = 2e-4 / 0.354 ~ 5.66e-4`.
Per `dict_size=16384`: `lr_auto = 2e-4 / sqrt(1.0) = 2e-4`.

**Perche' None e non un valore fisso?** La formula adatta automaticamente il
learning rate alla dimensione del dizionario. Senza questo, ogni cambio di
`dict_size` richiederebbe anche un ri-tuning manuale del learning rate.
L'alternativa (lr=5e-5 fisso) era fragile: ottima per `dict_size=4096` ma
potenzialmente troppo bassa per dizionari piu' grandi o troppo alta per
quelli piu' piccoli.

Per dataset piccoli dove il rischio di overfitting e' reale, si puo'
sovrascrivere con un valore fisso piu' basso (es. `lr=5e-5`) direttamente
nel costruttore o come argomento.

### steps = 50_000

Il training e' step-based (non epoch-based). Con batch_size=256 e ~5900 campioni
di train (80% di ~7400), 50k step corrispondono a ~2160 epoche. Valore di
default -- potrebbe essere ridotto per dataset piccoli (vedi ablation presets).

### warmup_steps = 1_000

Riscaldamento lineare del learning rate: LR cresce da 0 a `lr` nei primi 1000 step.
Stabilizza l'inizio del training quando i pesi sono random e i gradienti grandi.

### batch_size = 256

Numero di campioni per step di training. Con ~5900 campioni di train, ogni epoca
completa richiede ~23 step. 256 e' un buon compromesso tra stabilita' del
gradiente e velocita'.

### log_steps = 1_000 (novita')

Frequenza di logging delle metriche di training (loss, sparsity, ecc.).
Ogni 1000 step vengono stampati/loggati i valori correnti.
Valore ragionevole: con 50k step totali si ottengono 50 punti di log,
sufficienti per una loss curve senza inondare l'output.

### decay_start_frac = 0.8 (novita')

Frazione dei passi totali a cui inizia il decay lineare del learning rate.
Con `steps=50_000`, il decay inizia al passo 40.000 e il LR scende a zero
al passo 50.000.

**Perche' 0.8 e non 0.5 o 1.0?**

- 0.5 sarebbe troppo presto: il modello perderebbe capacita' di apprendimento
  nella meta' finale del training, quando le feature meno frequenti stanno ancora
  venendo imparate.
- 1.0 (nessun decay) lascerebbe il LR costante fino alla fine, causando
  oscillazioni della loss negli ultimi step.
- 0.8 permette un plateau di apprendimento solido (40k step) seguito da un
  raffinamento graduale (10k step).

### **post_init** (novita')

La validazione e' stata introdotta per catturare errori di configurazione
immediatamente all'import, anziche' scoprirli a runtime durante il training
(magari dopo ore di calcolo). I controlli sono:

1. **`dict_size > activation_dim`**: un dizionario sotto-completo non ha senso
   per un SAE -- lo scopo e' avere piu' feature che dimensioni di input.
2. **`k < dict_size`**: non si possono selezionare piu' feature di quante ne
   esistano nel dizionario.
3. **`lr > 0` (se fornito)**: un learning rate negativo o nullo e' chiaramente
   un errore. Non si controlla `lr is not None` perche' `None` e' un valore
   legittimo (auto-scaling).
4. **`warmup_steps < steps`**: il warmup non puo' durare piu' del training
   stesso.
5. **`0 < decay_start_frac <= 1`**: la frazione deve essere nell'intervallo
   valido. Non si accetta 0 (decay immediato) ma si accetta 1 (decay solo
   all'ultimo step).

Tutti i controlli usano `raise ValueError` con messaggi descrittivi che
includono il valore errato e il vincolo violato.

### Preset di ablation (novita' nella docstring)

La docstring include tre preset per esperimenti di ablation, pensati per
confrontare sistematicamente l'effetto dei parametri sul dataset piccolo:

| Preset | k | dict_size | lr | steps | Rapporto overcompleteness |
|---|---|---|---|---|---|
| Conservative | 16 | 2048 | None | 30_000 | 4x |
| Default | 32 | 4096 | None | 50_000 | 8x |
| Aggressive | 64 | 4096 | None | 80_000 | 8x |

**Perche' tre preset?** Permettono di studiare tre assi:

- **Conservative vs Default**: effetto della sparsita' (k) e della dimensione
  del dizionario insieme.
- **Default vs Aggressive**: effetto della sparsita' a parita' di dizionario
  (piu' concetti per immagine = spiegazioni piu' ricche ma meno interpretabili).
- **Conservative vs Aggressive**: effetto combinato di tutti i parametri.

Tutti usano `lr=None` (auto-scaling) per eliminare il learning rate come
variabile confondente.

---

## 5. TrainingConfig

```python
@dataclass(frozen=True)
class TrainingConfig:
    """Multi-seed training and stability analysis settings."""

    seeds: tuple[int, ...] = (0, 42, 123, 456, 789)
    primary_seed: int = 42  # reference model for naming/explanations
    sanity_check_samples: int = 256
    train_split_ratio: float = 0.8  # 80/20 train/test split
    split_seed: int = 42  # deterministic split
    stability_max_samples: Optional[int] = None
    correlation_threshold: float = 0.7

    def __post_init__(self):
        if self.primary_seed not in self.seeds:
            raise ValueError(
                f"primary_seed ({self.primary_seed}) must be in seeds {self.seeds}"
            )
        if not (0.0 < self.train_split_ratio < 1.0):
            raise ValueError(
                f"train_split_ratio must be in (0, 1), got {self.train_split_ratio}"
            )
```

**Perche:**

### seeds = (0, 42, 123, 456, 789)

Cinque semi per l'analisi di stabilita'. Ogni seed produce un SAE con
inizializzazione diversa. Confrontando i risultati si misura la robustezza.

- 5 seed e' sufficiente per stimare media e varianza della Jaccard similarity.
- Tuple (immutabile) per coerenza con `frozen=True`.

### primary_seed = 42 (novita')

Seme di riferimento usato per il modello "principale" su cui vengono eseguiti
il concept naming e la generazione delle spiegazioni.

**Perche' non usare semplicemente `seeds[1]`?** Nella versione precedente,
il codice assumeva che `seeds[1]` fosse il seed primario: un accoppiamento
fragile tra indice e significato. Se qualcuno avesse riordinato la tuple
(`(42, 0, 123, 456, 789)`), il seed primario sarebbe cambiato silenziosamente.
Con `primary_seed` esplicito, l'intento e' dichiarativo e la validazione in
`__post_init__` garantisce che il seed scelto sia effettivamente nella lista.

### sanity_check_samples = 256 (novita')

Numero di campioni usati nei sanity check pre-training (verifica che le
embedding siano caricate correttamente, che le dimensioni siano coerenti, ecc.).
256 e' un sottoinsieme rappresentativo ma veloce da processare.

**Perche' non usare tutto il dataset?** I sanity check vengono eseguiti prima
di ogni training multi-seed. Con 7400 campioni, il check richiederebbe tempo
inutile. 256 campioni sono sufficienti per rilevare problemi di forma,
valori NaN, o dimensioni errate.

### train_split_ratio = 0.8 (novita')

Frazione del dataset usata per il training (80%). Il restante 20% e' usato
come test set per la valutazione out-of-sample del SAE.

**Perche' 0.8?** E' lo standard de facto per dataset piccoli-medi. Con ~7400
campioni, si ottengono ~5900 di train e ~1500 di test. Il test set e'
abbastanza grande per stime affidabili della reconstruction loss e della
sparsita', ma abbastanza piccolo da non "sprecare" dati preziosi per il
training (i dataset biomedici sono tipicamente piccoli).

### split_seed = 42 (novita')

Seme per la riproducibilita' dello split train/test. Garantisce che lo stesso
campione finisca sempre nello stesso set, indipendentemente dall'ordine
dei dati su disco.

### stability_max_samples = None

Limita opzionalmente il numero di campioni usati nell'analisi di stabilita'.
`None` = usa tutto il dataset. Utile in sviluppo per velocizzare i test.

### correlation_threshold = 0.7 (novita')

Soglia di correlazione usata nell'analisi di stabilita' per determinare se
due feature (da seed diversi) rappresentano lo stesso concetto. Due feature
con correlazione cosine >= 0.7 sono considerate "la stessa feature".

**Perche' 0.7?** Valore empirico bilanciato:

- Troppo basso (0.5): unisce feature che catturano concetti diversi ma
  con direzioni simili nel spazio latente.
- Troppo alto (0.9): non riesce a unire feature che rappresentano lo
  stesso concetto ma con piccole variazioni dovute all'inizializzazione.

### **post_init** (novita')

La validazione garantisce che:

1. **`primary_seed in seeds`**: se il seed primario non e' nella lista,
   il training multi-seed non addestrerebbe un modello con quel seed,
   causando un errore a runtime quando si cerca di caricare il modello
   di riferimento per il concept naming.

2. **`0 < train_split_ratio < 1`**: impedisce split degeneri (0% o 100%
   di training), che renderebbero il training o la valutazione impossibili.

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

Quanti candidati di nome restituire per ogni feature nel concept naming (concept_naming).
Top-3 permette di vedere alternative quando il primo nome non e' convincente.

### explanation_top_n = 5

Quanti concetti includere nella spiegazione di ogni immagine (generate_explanations).
5 concetti bilanciano completezza e leggibilita' nel pseudo-report.

### explanation_max_samples = None

Come `stability_max_samples`: limita le immagini processate in generate_explanations.
`None` = processa tutto il dataset.

Questa dataclass non e' stata modificata rispetto alla versione precedente.

---

## 7. WandbConfig (nuova)

```python
@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases experiment tracking."""

    enabled: bool = False
    project: str = "sae-concept-discovery"
    entity: Optional[str] = None
```

**Perche:**

### Perche' una dataclass dedicata?

Weights & Biases (wandb) e' uno strumento di experiment tracking che logga
metriche di training (loss, learning rate, sparsita') su un dashboard web.
Avere una config dedicata permette di:

- Disabilitare wandb in sviluppo (`enabled=False`) senza dover cercare
  e commentare chiamate sparse nel codice.
- Cambiare progetto/entity per esperimenti diversi senza modificare il
  codice di training.
- Centralizzare la configurazione del logging in un unico punto.

### enabled = False

Wandb e' disabilitato di default. Questo evita:

- Richieste di login a chiunque importi il modulo senza account wandb.
- Upload non voluti di metriche durante esperimenti esplorativi.
- Dipendenza implicita da wandb in ambienti dove non e' installato.

Per abilitare: `WandbConfig(enabled=True)` oppure modificare il default.

### project = "sae-concept-discovery"

Nome del progetto su wandb. Tutti i run (diversi seed, diversi preset)
vengono raggruppati sotto questo progetto.

### entity: Optional[str] = None

L'entity (utente o team) su wandb. `None` usa l'entity di default
dell'utente loggato. Puo' essere sovrascritto per loggare sotto un team
condiviso (es. `"my-research-lab"`).

---

## 8. HardwareConfig

```python
@dataclass(frozen=True)
class HardwareConfig:
    """Device and compute settings."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

**Perche:**

### Rilevamento automatico (novita')

Nella versione precedente, `device` era hardcoded a `"cuda"`. Questo causava
un errore immediato su macchine senza GPU NVIDIA (es. MacBook, server CPU-only).
Ora il rilevamento e' automatico:

```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

**Perche' `torch.cuda.is_available()` e non un detect universale?**

- MPS (Apple Silicon) non e' supportato dalla libreria `dictionary_learning`
  usata per il training SAE. Anche se torch lo supporta, il codice del trainer
  presume l'uso di CUDA o CPU.
- Il fallback a `"cpu"` e' sempre sicuro ma piu' lento. Per dataset di ~7400
  campioni con embedding 512-dim, il training su CPU e' comunque fattibile
  (minuti, non ore).

Per forzare un device specifico (es. `"mps"` per test su Apple Silicon),
istanziare esplicitamente: `HardwareConfig(device="mps")`.

### frozen=True

Una volta determinato il device, non dovrebbe cambiare durante l'esecuzione.
Modificare il device a runtime causerebbe copie di tensori tra device
impreviste e possibili errori.

---

## 9. Istanziazione

```python
# Instantiate configs
paths = PathsConfig()
backbone = BackboneConfig()
sae = SAEConfig()
training = TrainingConfig()
explanation = ExplanationConfig()
wandb_cfg = WandbConfig()
hardware = HardwareConfig()

# Backward compatibility alias
DEVICE = hardware.device
```

**Perche:**

Le istanze sono create a livello di modulo cosi' gli script possono fare:

```python
import config
config.sae.dict_size  # 4096
config.paths.models_dir  # Path(".../models")
config.wandb_cfg.enabled  # False
```

Le istanze vengono create una sola volta all'import del modulo (singleton pattern
implicito di Python). Ogni script che fa `import config` condivide le stesse
istanze.

### Nota sull'ordine di istanziazione

L'ordine non e' casuale: `PathsConfig()` viene prima perche' non ha dipendenze,
ma le altre dataclass con `__post_init__` vengono istanziate subito dopo.
Se una di queste validazioni fallisse (es. `primary_seed` non in `seeds`),
l'errore viene sollevato all'import, prima che qualsiasi script possa
usare una configurazione inconsistente.

### wandb_cfg (novita')

Il nome usa l'abbreviazione `wandb_cfg` invece di `wandb` per evitare
conflitti di namespace con il package `wandb` stesso importato negli script
di training. Un modulo che fa `import wandb` e `import config` non avrebbe
ambiguita'.

### DEVICE

Alias di compatibilita' per codice legacy che usava `config.DEVICE` invece
di `config.hardware.device`.

---

## Come modificare la configurazione

### Per cambiare un default

Modificare il valore nel campo della dataclass:

```python
dict_size: int = 2048  # era 4096
```

Attenzione: se la dataclass ha `__post_init__`, verificare che il nuovo
valore soddisfi tutti i vincoli di validazione.

### Per aggiungere un parametro

Aggiungere un campo nella dataclass appropriata:

```python
@dataclass(frozen=True)
class SAEConfig:
    ...
    decay: float = 0.99  # nuovo parametro
```

Se il parametro richiede validazione, aggiungere il controllo in `__post_init__`.

### Per usare un preset di ablation

Sovrascrivere i campi del preset nel costruttore:

```python
# Preset Conservative per dataset piccolo
sae = SAEConfig(k=16, dict_size=2048, steps=30_000)
```

Oppure creare un'istanza separata nel punto di uso:

```python
from config import SAEConfig
sae_conservative = SAEConfig(k=16, dict_size=2048, steps=30_000)
```

### Per abilitare wandb

Non modificare config.py direttamente. Creare l'istanza con override:

```python
wandb_cfg = WandbConfig(enabled=True, entity="my-team")
```

### Per forzare un device

```python
hardware = HardwareConfig(device="cpu")
```

### Per override temporaneo in uno script

Non modificare config.py. Passare il valore come argomento alla funzione
o al manager che lo usa:

```python
mgr = SAEManager({"device": "cpu", "dict_size": 2048})
```

### Per cambiare lo split train/test

```python
training = TrainingConfig(train_split_ratio=0.9, split_seed=123)
```

---

## Relazione tra le configurazioni

```text
BackboneConfig.embedding_dim = 512
        |
        v
SAEConfig.activation_dim = 512  (devono coincidere)
        |
        +---> SAEConfig.__post_init__: dict_size > activation_dim
        |           |
        |           v
        |     SAEConfig.dict_size = 4096  (concetti totali)
        |           |
        |           v
        |     SAEConfig.__post_init__: k < dict_size
        |           |
        |           v
        |     SAEConfig.k = 32  (concetti per immagine)
        |           |
        |           +---> SAEConfig.lr: None -> auto-scaling
        |           |          formula: 2e-4 / sqrt(dict_size / 16384)
        |           |
        |           +---> SAEConfig.decay_start_frac = 0.8
        |           |          decay inizia a: steps * 0.8 = 40_000
        |           |
        +---> SAEConfig.warmup_steps = 1_000
                    deve essere < steps (validato in __post_init__)
        |
        v
ExplanationConfig.explanation_top_n = 5  (deve essere <= k)
ExplanationConfig.concept_top_n = 3  (candidati per feature)
        |
        v
TrainingConfig.primary_seed = 42
        |
        +---> TrainingConfig.__post_init__: primary_seed in seeds
                    |
                    v
              TrainingConfig.seeds = (0, 42, 123, 456, 789)
                    |
                    +---> TrainingConfig.train_split_ratio = 0.8
                    |          produce train_embeddings.pt e test_embeddings_path
                    |
                    +---> TrainingConfig.correlation_threshold = 0.7
                               usato in stability analysis per feature matching
        |
        v
PathsConfig.train_embeddings_path  <- TrainingConfig.train_split_ratio
PathsConfig.test_embeddings_path   <- TrainingConfig.train_split_ratio
PathsConfig.figures_dir            <- nuovo output per grafici
        |
        v
HardwareConfig.device              <- rilevato automaticamente (cuda/cpu)
        |
        v
WandbConfig.enabled = False       <- toggle per experiment tracking
WandbConfig.project/entity        <- dove loggare le metriche
```

### Vincoli chiave

1. `BackboneConfig.embedding_dim == SAEConfig.activation_dim`: se il backbone
   cambia, il SAE deve essere ri-addestrato con `activation_dim` aggiornato.
2. `SAEConfig.dict_size > SAEConfig.activation_dim`: validato in `__post_init__`.
3. `SAEConfig.k < SAEConfig.dict_size`: validato in `__post_init__`.
4. `TrainingConfig.primary_seed in TrainingConfig.seeds`: validato in `__post_init__`.
5. `TrainingConfig.train_split_ratio` in `(0, 1)`: validato in `__post_init__`.
6. `SAEConfig.lr=None` adatta automaticamente il learning rate a `dict_size`:
   se si cambia `dict_size`, il learning rate si adatta di conseguenza.

Se `embedding_dim` cambia (es. altro backbone), `activation_dim` deve cambiare
di conseguenza e tutti i modelli vanno ri-addestrati. Le embedding di train/test
andranno rigenerate con il nuovo split (o ri-estratte se il backbone cambia).
