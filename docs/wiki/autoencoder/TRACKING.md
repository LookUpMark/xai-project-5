# tracking.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/tracking.py`,
il modulo che integra Weights & Biases per il tracciamento degli esperimenti,
degradando in modo elegante quando W&B non e' installato o disabilitato.

---

## 1. Docstring e importazioni

```python
"""
tracking.py — Experiment tracking integration (Weights & Biases).

Thin wrapper around wandb that degrades gracefully when wandb
is not installed or not enabled. All functions are no-ops when
tracking is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_tracking_enabled: bool = False
```

**Perche:**

Questo file esiste per risolvere un problema pratico: il tracciamento degli
esperimenti con W&B e' incredibilmente utile per monitorare training,
confrontare run, e condividere risultati — ma imporre `import wandb` come
dipendenza obbligatoria blocca chi non ha W&B installato o non vuole usarlo.

L'approccio adottato e' un **thin wrapper** con **graceful degradation**:
- Se W&B e' disponibile e abilitato: le funzioni loggano normalmente.
- Se W&B non e' installato: le funzioni sono no-op silenziosi.
- Se W&B lancia un errore: l'errore viene catturato e loggato, la pipeline
  continua senza tracking.

Le importazioni standard:
- `logging`: per messaggi informativi e di warning.
- `Path`: per il path degli artefatti da loggare.
- `typing.Any, Optional`: per type hint flessibili.

Notare che `wandb` **non e' importato a livello di modulo** — l'import
e' differito dentro ogni funzione, dentro un blocco `try`. Questo garantisce
che il modulo non crasha all'import se wandb non e' installato.

---

## 2. Variabile di stato `_tracking_enabled`

```python
_tracking_enabled: bool = False
```

**Perche:**

Flag globale (a livello di modulo) che indica se il tracking e' attivo.
Inizia come `False` (tracking disabilitato di default) e viene settato a
`True` da `init_tracking()` se l'inizializzazione W&B ha successo.

E' un nome con prefisso `_` (convenzione Python per "privato") perche':
- Non e' parte dell'API pubblica del modulo.
- I consumer dovrebbero usare le funzioni pubbliche, non controllare
  il flag direttamente.
- Le funzioni pubbliche (`log_metrics`, `log_artifact`, `finish_tracking`)
  controllano questo flag internamente.

Il prefisso `_` e' significativo: informa il type checker (e gli sviluppatori)
che questo e' stato interno del modulo.

**Perche' globale (non in una classe)?**

Il tracking e' per natura un singleton — c'e' al piu' una run W&B attiva
alla volta per processo. Un flag globale e' piu' semplice e piu' appropriato
di una classe con stato, dato che non serve istanziare nulla.

---

## 3. Funzione `init_tracking()`

```python
def init_tracking(stage_name: str, config: dict[str, Any]) -> None:
    """Initialize wandb run for a pipeline stage."""
    global _tracking_enabled
    try:
        import wandb

        wandb.init(
            project=config.get("project", "sae-concept-discovery"),
            entity=config.get("entity"),
            name=stage_name,
            config=config,
        )
        _tracking_enabled = True
        logger.info(f"wandb tracking enabled: {stage_name}")
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
    except Exception as e:
        logger.warning(f"wandb init failed: {e}. Tracking disabled.")
```

**Perche:**

E' il punto di ingresso per abilitare il tracking. Deve essere chiamato
all'inizio di ogni stadio della pipeline che vuole tracciare.

### Parametri

- `stage_name`: il nome dello stadio (es. `"train_sae"`, `"concept_naming"`).
  Diventa il nome della run W&B, permettendo di identificare quale stadio
  sta loggando.
- `config`: dizionario di configurazione. Viene passato interamente a
  `wandb.init(config=...)` perche' W&B lo salva come metadata della run,
  permettendo confronti tra run con parametri diversi.

### Perche' due except separati

1. `except ImportError`: cattura il caso in cui wandb non e' installato.
   Messaggio specifico con istruzione di installazione. E' il caso piu'
   comune in sviluppo locale.

2. `except Exception`: cattura qualsiasi altro errore (autenticazione fallita,
   connessione di rete assente, W&B server down, etc.). Logga l'errore
   specifico per debugging ma non blocca la pipeline.

### Perche' `global _tracking_enabled`

Le funzioni usano `global` perche' `_tracking_enabled` e' definito a livello
di modulo. In Python, se una funzione vuole riassegnare una variabile di
modulo, deve dichiararla `global` — altrimenti crea una variabile locale
con lo stesso nome (shadowing), e il flag globale resterebbe `False`.

### Perche' valori default per `project` e `entity`

- `project="sae-concept-discovery"`: nome default del progetto W&B.
  Raggruppa tutte le run di questo progetto nella stessa dashboard.
- `entity=None`: lascia che W&B usi l'entity default dell'utente (solamente
  il username personale). Non e' forzato perche' in alcuni ambienti
  (enterprise W&B) l'entity e' diversa dal username.

### Posizione nella pipeline

```
stadio.begin()
    |
    v
tracking.init_tracking(stage_name, config)  <-- abilita il tracking
    |
    v
... esegue il lavoro ...
    |
    v
tracking.log_metrics(metrics)  <-- logga se abilitato
    |
    v
tracking.finish_tracking()  <-- chiude la run
```

---

## 4. Funzione `log_metrics()`

```python
def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to wandb (no-op if tracking disabled)."""
    if not _tracking_enabled:
        return
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception:
        pass
```

**Perche:**

Registra metriche numeriche nella run W&B corrente. E' il cuore del
tracking — ogni stadio chiama questa funzione per loggare le proprie
metriche (MSE, L0, dead%, Jaccard, ecc.).

### Early return pattern

```python
if not _tracking_enabled:
    return
```

Il check piu' rapido possibile: se tracking disabilitato, torna
immediatamente senza overhead. Evita l'import di wandb (che potrebbe
essere costoso) e qualsiasi allocazione.

### Parametri

- `metrics`: dict di nome -> valore. W&B crea automaticamente un grafico
  per ogni metrica. Esempio: `{"mse": 0.0012, "l0_mean": 32.0}`.
- `step`: passo opzionale per asse x del grafico. Se `None`, W&B
  incrementa automaticamente. Utile per metriche di training dove
  lo step e' significativo.

### Perche' `except Exception: pass`

A differenza di `init_tracking()`, qui gli errori sono silent (non loggati).
Motivazione:
- `log_metrics()` e' chiamato molto frequentemente (decine di volte per stadio).
- Un warning per ogni chiamata fallita inonderebbe i log.
- Se `init_tracking()` ha avuto successo ma `log_metrics()` fallisce,
  probabilmente e' un errore temporaneo di rete W&B — non vale la pena
  interrompere il training per questo.
- La pipeline continua a funzionare senza tracking, che e' il comportamento
  di graceful degradation desiderato.

---

## 5. Funzione `log_artifact()`

```python
def log_artifact(path: Path, name: str, artifact_type: str) -> None:
    """Log a file artifact to wandb (no-op if tracking disabled)."""
    if not _tracking_enabled:
        return
    try:
        import wandb

        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    except Exception:
        pass
```

**Perche:**

Carica un file come artefatto nella run W&B. Gli artefatti sono file
salvati nella W&B Artifact Store — persistono anche dopo la fine della
run e possono essere scaricati, versionati e confrontati.

### Parametri

- `path`: percorso del file da caricare (es. `models/sae_seed42/ae.pt`,
  `results/stability_analysis.json`).
- `name`: nome dell'artefatto (es. `"sae-model-seed42"`, `"stability-report"`).
  W&B usa questo nome per il versioning.
- `artifact_type`: categoria dell'artefatto (es. `"model"`, `"dataset"`,
  `"report"`). W&B raggruppa gli artefatti per tipo nella UI.

### Esempio d'uso

```python
tracking.log_artifact(
    path=Path("models/sae_seed42/ae.pt"),
    name="sae-model-seed42",
    artifact_type="model",
)
```

Questo carica il modello nella W&B Artifact Store, permettendo di:
- Scaricarlo da qualsiasi macchina con `wandb.use_artifact("name:latest")`.
- Confrontare modelli tra run diverse.
- Tenere uno storico versionato dei modelli prodotti.

### Perche' lo stesso pattern di `log_metrics()`

Stesso early return + except silenzioso. Stessa motivazione: graceful
degradation. Se l'upload fallisce (file troppo grande, rete assente),
la pipeline continua — il file e' comunque sul disco locale.

---

## 6. Funzione `finish_tracking()`

```python
def finish_tracking() -> None:
    """Finish the current wandb run."""
    global _tracking_enabled
    if _tracking_enabled:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
        _tracking_enabled = False
```

**Perche:**

Chiude la run W&B corrente. Deve essere chiamato alla fine di ogni stadio
per:
- Flushare eventuali metriche pendenti (W&B potrebbe bufferizzare).
- Finalizzare la run nella dashboard W&B.
- Rilasciare risorse (connessione di rete, thread interni).

### Perche' `_tracking_enabled = False` fuori dal try

```python
if _tracking_enabled:
    try:
        wandb.finish()
    except Exception:
        pass
    _tracking_enabled = False  # fuori dal try
```

Anche se `wandb.finish()` fallisce, il flag viene comunque resettato a
`False`. Motivazione: se `finish()` fallisce, non ha senso tentare
nuovamente — resettare il flag permette di ricominciare da capo nella
prossima chiamata a `init_tracking()`.

### Perche' il check `if _tracking_enabled`

Evita una chiamata inutile a `wandb.finish()` se il tracking non era mai
stato abilitato. Questo e' particolarmente importante perche' `wandb.finish()`
su una run non inizializzata potrebbe lanciare un errore o un warning.

---

## Diagramma del flusso

```
[Inizio stadio]
       |
       v
init_tracking("train_sae", config)
       |
       +-- wandb installato? --NO--> logger.warning(), _tracking_enabled=False
       |                                    |
       YES                                  |
       |                                    |
       v                                    |
wandb.init(project, name, config)           |
_tracking_enabled = True                   |
       |                                    |
       +<-----------------------------------+
       |
       v
[Training / Processing]
       |
       v
log_metrics({"mse": 0.0012, "l0": 32.0}, step=1000)
       |
       +-- _tracking_enabled? --NO--> return (no-op)
       |                               |
       YES                             |
       |                               |
       v                               |
wandb.log(metrics, step)               |
       |                               |
       +<------------------------------+
       |
       v
log_artifact(model_path, "sae-model", "model")
       |  (stesso pattern)
       v
[Fine stadio]
       |
       v
finish_tracking()
       |
       v
wandb.finish()
_tracking_enabled = False
```

---

## Relazione con gli altri file

```
tracking.py  (questo file)
    |
    +---> chiamato da: train_sae.py (logga metriche training)
    +---> chiamato da: concept_naming.py (logga score naming)
    +---> chiamato da: stability_analysis.py (logga Jaccard, per-seed)
    +---> chiamato da: generate_explanations.py (logga metriche explanation)
    |
    +---> collabora con: protocols.py (TrackedStage restituisce dict
    |     compatibili con log_metrics)
    +---> collabora con: contracts.py (le metriche nei contratti sono
    |     originate qui)
```

tracking.py e' un modulo trasversale (cross-cutting concern): non e' uno
stadio della pipeline, ma un servizio usato da tutti gli stadi. La sua
esistenza come modulo separato (invece di codice inline in ogni stadio)
garantisce un unico punto di configurazione e un unico punto di failure
per il tracking.
