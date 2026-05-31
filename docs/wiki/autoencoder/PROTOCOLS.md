# protocols.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/protocols.py`,
il modulo che definisce le interfacce (Protocol) per gli stadi della
pipeline SAE, abilitando composizione type-safe e testing con mock.

---

## 1. Docstring e importazioni

```python
"""
protocols.py — Interface definitions for SAE pipeline stages.

Each pipeline stage should implement the relevant protocol,
enabling type-safe composition, testing with mocks, and
independent development of each stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
```

**Perche:**

Questo file esiste per risolvere un problema di disaccoppiamento nella
pipeline SAE. Senza interfacce formali, ogni stadio e' legato alle
implementazioni concrete degli altri stadi. Con i Protocol, ogni stadio
dipende solo dall'interfaccia (contratto comportamentale), non dalla
classe specifica.

I benefici concreti:

1. **Testing con mock**: si puo' testare `generate_explanations` usando un
   mock di `concept_naming` che implementa `PipelineStage` senza dover
   eseguire il naming reale.
2. **Sviluppo indipendente**: due sviluppatori possono lavorare su stadi
   diversi, ciascuno implementando il Protocol concordato.
3. **Composizione flessibile**: si puo' sostituire un'implementazione con
   un'altra (es. concept naming con vocabolario diverso) senza modificare
   il codice downstream.
4. **Type checking statico**: mypy/pyright verificano che ogni stadio
   rispetti l'interfaccia a compile-time.

Le importazioni chiave:
- `Protocol` (PEP 544): definisce interfacce strutturali — una classe
  implementa un Protocol se ha i metodi giusti, senza ereditarieta' esplicita.
- `runtime_checkable`: abilita `isinstance(obj, MyProtocol)` per verificare
  a runtime che un oggetto rispetti il Protocol.
- `Path`: il tipo restituito da `run()` — il path dell'artefatto prodotto.
- `Optional`: per il parametro opzionale `run_id` di `TrackedStage`.

---

## 2. PipelineStage

```python
@runtime_checkable
class PipelineStage(Protocol):
    """Base protocol for any pipeline stage."""

    @property
    def name(self) -> str:
        """Human-readable stage name."""
        ...

    def run(self) -> Path:
        """Execute the stage, return path to primary output artifact."""
        ...
```

**Perche:**

### Il problema risolto
Nella pipeline SAE, ogni stadio (train, naming, explanation, stability)
condivide lo stesso pattern: ha un nome, produce un artefatto su disco,
e restituisce il path. Senza `PipelineStage`, questo pattern e' implicito
— ogni script e' un file standalone senza struttura comune.

### Perche' un Protocol (non una classe astratta)
Un Protocol Python e' un'interfaccia strutturale (structural typing):
una classe la implementa automaticamente se ha gli attributi/metodi giusti,
senza bisogno di `class MyStage(PipelineStage)`.

Questo e' cruciale perche':
- Gli script esistenti (`train_sae.py`, `concept_naming.py`, ecc.) possono
  implementare il Protocol **senza modificare il loro codice** — basta che
  abbiano una property `name` e un metodo `run() -> Path`.
- Non introduce un gerarchia di classi rigida che forzerebbe refactoring
  del codice legacy.

### Perche' `@runtime_checkable`
Normalmente i Protocol funzionano solo con il type checker statico (mypy).
`@runtime_checkable` aggiunge la possibilita' di fare `isinstance(stage, PipelineStage)`
a runtime, utile per:
- Validazione degli stadi in un orchestratore di pipeline.
- Errori precoci se uno stadio non rispetta l'interfaccia.

### `name` come property (non metodo)
E' una `@property` perche' il nome di uno stadio e' un attributo immutabile,
non un'azione. Consente di scrivere `stage.name` invece di `stage.name()`.

### `run() -> Path`
Restituisce un `Path` al artefatto principale prodotto. Potrebbe essere:
- Un file JSON (es. `results/stability_analysis.json`)
- Un file pickle/tensor (es. `models/sae_seed42/ae.pt`)
- Una directory di output

Il valore di ritorno e' sempre un singolo `Path` per semplicita' — se uno
stadio produce piu' file, il `Path` punta al principale e gli altri sono
nella stessa directory.

### Esempio di implementazione implicita

```python
# Questa classe implementa PipelineStage automaticamente:
class TrainSAE:
    @property
    def name(self) -> str:
        return "train_sae"

    def run(self) -> Path:
        # ... addestra il SAE ...
        return Path("models/sae_seed42")
```

---

## 3. TrackedStage

```python
@runtime_checkable
class TrackedStage(PipelineStage, Protocol):
    """Pipeline stage with experiment tracking support."""

    def run(self, run_id: Optional[str] = None) -> tuple[Path, dict]:
        """Execute stage, return (output_path, metrics_dict)."""
        ...
```

**Perche:**

### Estensione di PipelineStage
`TrackedStage` eredita da `PipelineStage` (tramite Protocol, non
classica ereditarieta'). Questo significa che un `TrackedStage` deve avere:
- La property `name` (da `PipelineStage`)
- Il metodo `run()` ( Ridefinito con firma diversa)

### Il problema risolto
Alcuni stadi della pipeline producono metriche che devono essere tracciate
in Weights & Biases (o sistemi simili). `PipelineStage.run()` restituisce
solo un Path — non c'e' modo di estrarre le metriche senza leggere il
file di output.

`TrackedStage.run()` restituisce una tupla `(Path, dict)`:
- `Path`: stesso artefatto di output di `PipelineStage`.
- `dict`: metriche strutturate pronte per il logging (MSE, L0, dead%, ecc.).

### Perche' `run_id` e' Optional
`run_id` permette di associare piu' stadi alla stessa run di tracking.
Se `None`, ogni stadio crea la propria run. Se valorizzato (es. `"exp_42"`),
tutti gli stadi con lo stesso `run_id loggano nella stessa run W&B.

E' `Optional` perche' non tutti gli stadi tracciati hanno bisogno di
raggruppamento — in development, ogni stadio ha la sua run indipendente.

### Override di `run()` con firma diversa
Attenzione: `TrackedStage.run()` ha una firma diversa da `PipelineStage.run()`
— prende `run_id` e restituisce `tuple[Path, dict]` invece di `Path`. Questo
e' intenzionale: i Protocol Python supportano l'overloading tramite
sottotipizzazione strutturale. Un type checker tratta `TrackedStage` come
un sottotipo specializzato di `PipelineStage`.

### Esempio di implementazione

```python
# Questa classe implementa TrackedStage automaticamente:
class ConceptNaming:
    @property
    def name(self) -> str:
        return "concept_naming"

    def run(self, run_id=None) -> tuple[Path, dict]:
        concepts = self._compute_names()
        path = self._save(concepts)
        metrics = {"mean_score": concepts.mean_score}
        return path, metrics
```

### Connessione con tracking.py

Il dict restituito da `TrackedStage.run()` e' progettato per essere passato
direttamente a `tracking.log_metrics()`:

```python
path, metrics = tracked_stage.run(run_id="exp_42")
tracking.log_metrics(metrics)  # perfetta compatibilita' dei tipi
```

---

## Diagramma delle interfacce

```
PipelineStage (Protocol)
    |
    | @property name -> str
    | def run() -> Path
    |
    +---> TrackedStage (Protocol)
              |
              | def run(run_id=None) -> tuple[Path, dict]
              |
              +---> implementato da: train_sae, concept_naming,
              |     generate_explanations, stability_analysis
              |
              +---> consumato da: orchestratore pipeline, tracking.py
```

### Relazione con i contratti (contracts.py)

```
protocols.py  <-- definisce COME si comportano gli stadi
contracts.py  <-- definisce COSA producono e consumano (dati)

TrackedStage.run() -> tuple[Path, dict]
                          |          |
                          v          v
                    file su disco  SeedMetrics / ConceptMap / etc.
                                    (contratti.py)
```

---

## Perche' non una classe base astratta?

Alternativa considerata: usare `ABC` (Abstract Base Class) con `@abstractmethod`.

Motivi per cui Protocol e' superiore in questo contesto:

| Aspect | ABC | Protocol |
|--------|-----|----------|
| Ereditarieta' | Richiede `class X(ABC)` | Strutturale, nessun ereditarieta' |
| Implementazione multipla | `class X(ABC1, ABC2)` possibile ma fragile | Composizione automatica |
| Codice legacy | Deve ereditare esplicitamente | Implementa automaticamente |
| Type checking | Funziona | Funziona |
| `isinstance()` | Sempre | Con `@runtime_checkable` |

Il progetto ha script esistenti scritti come funzioni standalone (`main()`).
Convertirli in classi che ereditano da ABC richiederebbe refactoring
massiccio. Con i Protocol, basta aggiungere `name` e `run()` — zero
modifiche strutturali.

---

## Relazione con gli altri file

```
protocols.py  (questo file)
    |
    +---> train_sae.py       (implementa implicitamente PipelineStage)
    +---> concept_naming.py   (implementa implicitamente TrackedStage)
    +---> generate_explanations.py  (implementa implicitamente TrackedStage)
    +---> stability_analysis.py     (implementa implicitamente TrackedStage)
    +---> contracts.py       (i dict restituiti contengono istanze dei contratti)
    +---> tracking.py        (consuma i dict da TrackedStage.run())
    +---> visualization.py    (consuma Path e dati dai contratti)
```
