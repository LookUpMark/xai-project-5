# contracts.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/contracts.py`,
il modulo che definisce i dataclass immutable che strutturano il flusso
di dati tra gli stadi della pipeline SAE.

---

## 1. Docstring e importazioni

```python
"""
contracts.py — Typed data contracts for the SAE pipeline.

These dataclasses define the shape of data flowing between pipeline stages.
They wrap tensors and raw types in structured containers for type safety
and validation at stage boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
```

**Perche:**

Questo file esiste per risolvere un problema architetturale fondamentale:
senza contratti espliciti, gli stadi della pipeline si passano dizionari
non tipizzati, tensor nudi, o tuple senza nome. Questo porta a:

- **Bug silenziosi**: uno stadio passa un dict con chiave `"mse"` ma il
  successivo legge `"loss"` — nessun errore a compile-time.
- **Refactoring impossibile**: cambiare la firma di uno stadio rompe
  silenziosamente tutti gli stadi downstream.
- **Documentazione frammentata**: la forma dei dati e' disseminata nei
  docstring e nei commenti, mai centralizzata.

I dataclass `frozen=True` sono immutabili — una volta creati non possono
essere modificati. Questo garantisce che ogni stadio produce un risultato
coerente che non verra' alterato involontariamente dagli stadi successivi.

`from __future__ import annotations` abilita la valutazione lazy delle
annotazioni di tipo, permettendo riferimenti circolari e tipi forward.

---

## 2. CandidateName

```python
@dataclass(frozen=True)
class CandidateName:
    """A candidate concept name with its similarity score."""

    label: str
    score: float
```

**Perche:**

Rappresenta un singolo candidato per il nome di un concetto SAE. Quando il
`concept_naming` calcola la cosine similarity tra una feature del decoder
e tutte le embedding del vocabolario medico, ogni termine simile diventa
un `CandidateName`.

- `label`: il termine del vocabolario (es. `"pneumonia"`, `"cardiomegaly"`).
- `score`: cosine similarity tra il vettore del concetto e l'embedding del
  termine. Valore in [0, 1] dove 1 = perfetta somiglianza direzionale.

E' `frozen=True` perche' un candidato nome non dovrebbe mai cambiare dopo
la creazione — e' un dato osservazionale, non uno stato mutabile.

**Posizione nella pipeline:**

```text
sae_module.name_concepts()
    --> calcola similarita' per ogni feature
        --> crea CandidateName per ogni termine candidato
            --> raggruppato in ConceptName.candidates
```

---

## 3. ConceptName

```python
@dataclass(frozen=True)
class ConceptName:
    """Named SAE feature with candidates from vocabulary."""

    feature_id: int
    name: str
    score: float
    candidates: list[CandidateName]
```

**Perche:**

E' il risultato unitario del naming per una singola feature SAE. Raggruppa:

- `feature_id`: l'indice della feature nel dizionario (0-4095).
- `name`: il termine migliore (primo candidato).
- `score`: il cosine similarity del termine migliore.
- `candidates`: la lista completa dei candidati (tipicamente top-3 o top-5),
  ognuno come `CandidateName`.

La struttura gerarchica `ConceptName -> list[CandidateName]` permette sia
l'uso rapido (prendi `name` e `score` per il migliore) sia l'analisi
approfondita (ispeziona tutti i candidati per valutare la confidenza del
naming).

**Esempio concreto:**

```python
ConceptName(
    feature_id=142,
    name="pneumonia",
    score=0.87,
    candidates=[
        CandidateName(label="pneumonia", score=0.87),
        CandidateName(label="lung opacity", score=0.82),
        CandidateName(label="consolidation", score=0.71),
    ],
)
```

**Posizione nella pipeline:**

```text
concept_naming --> ConceptName (per ogni feature attiva)
    --> aggregato in ConceptMap.concepts
```

---

## 4. ConceptMap

```python
@dataclass(frozen=True)
class ConceptMap:
    """Output of concept_naming stage — maps feature IDs to names."""

    concepts: dict[int, ConceptName]
    source_model_seed: int
    total_features: int
    mean_score: float
```

**Perche:**

E' l'output completo dello stadio `concept_naming`. Non e' solo un dizionario
di nomi — include metadata essenziali per la riproducibilita' e il monitoring:

- `concepts`: mappa feature_id -> ConceptName. Tipicamente contiene solo le
  feature attive (esclude le dead features che non hanno nome significativo).
- `source_model_seed`: quale seed SAE e' stato usato per il naming. Fondamentale
  per tracciabilita' — lo stesso dataset con seed diversi produce ConceptMap
  diverse.
- `total_features`: dimensione del dizionario (4096). Permette di calcolare
  la copertura: `len(concepts) / total_features` = percentuale di feature nominate.
- `mean_score`: media dei cosine similarity di tutti i concetti. Metrica di
  qualita' globale del naming — se bassa, i concetti non sono ben allineati
  con il vocabolario medico.

**Posizione nella pipeline:**

```text
concept_naming --> ConceptMap
    --> consumato da generate_explanations (per associare nomi alle attivazioni)
    --> consumato da visualization.plot_concept_score_distribution (per i grafici)
```

---

## 5. Finding

```python
@dataclass(frozen=True)
class Finding:
    """A single concept activation in an explanation."""

    concept: str
    feature_id: int
    activation: float
    naming_confidence: float
```

**Perche:**

Rappresenta un singolo "ritrovamento" nella spiegazione di un'immagine.
Quando lo stadio `generate_explanations` analizza un campione, identifica
quali concetti sono attivi e li impacchetta come `Finding`:

- `concept`: il nome leggibile del concetto (es. `"pneumonia"`), derivato
  dal `ConceptName` associato alla feature.
- `feature_id`: l'indice raw nel dizionario SAE, per tracciabilita' verso
  il modello.
- `activation`: l'intensita' dell'attivazione per questa feature in questo
  campione specifico. Valore float, tipicamente > 0 per le feature selezionate
  dal Top-K.
- `naming_confidence`: il cosine similarity con cui il concetto e' stato
  nominato. Permette di distinguere concetti affidabili (score alto) da
  concetti incerti (score basso).

Ogni `Finding` e' immutabile perche' rappresenta un'osservazione fissa
del modello su un campione specifico.

**Posizione nella pipeline:**

```text
generate_explanations
    --> per ogni campione, crea una lista di Finding
    --> raggruppati in Explanation.findings
```

---

## 6. Explanation

```python
@dataclass(frozen=True)
class Explanation:
    """Output of generate_explanations stage — per-sample explanation."""

    sample_idx: int
    findings: list[Finding]
    pseudo_report: str
    n_active_concepts: int
```

**Perche:**

E' l'output dello stadio `generate_explanations` per un singolo campione.
Contiene tutto il necessario per interpretare una predizione:

- `sample_idx`: indice del campione nel dataset originale. Permette di
  risalire all'immagine specifica.
- `findings`: lista dei concetti attivi per questo campione, ordinati per
  attivazione decrescente. Ogni `Finding` ha nome, feature, attivazione
  e confidenza del naming.
- `pseudo_report`: testo generato che riassume i concetti attivi in un
  formato simile a un report radiologico. E' il deliverable principale
  per l'utente finale.
- `n_active_concepts`: numero di concetti attivi. Diagnostico — se e' sempre
  uguale a k (32), la sparsita' e' perfetta; se varia, ci potrebbero essere
  feature con attivazioni sotto soglia.

La struttura `Explanation` separa i dati strutturati (`findings`, `sample_idx`)
dal testo interpretativo (`pseudo_report`), permettendo sia l'uso programmatico
sia la presentazione umana.

**Posizione nella pipeline:**

```text
generate_explanations --> list[Explanation] (una per campione)
    --> serializzate in JSON per il report finale
```

---

## 7. SeedMetrics

```python
@dataclass(frozen=True)
class SeedMetrics:
    """Per-seed training metrics."""

    seed: int
    mse: float
    l0_mean: float
    l0_std: float
    dead_features_pct: float
    dict_utilization_pct: float
    activation_entropy: float
    feature_frequency_mean: float
    feature_frequency_std: float
```

**Perche:**

Cattura tutte le metriche di qualita' per un singolo seed SAE. Usato
principalmente nello stadio `stability_analysis` per confrontare i seed
tra loro e identificare outlier:

- `seed`: il seed di addestramento (es. 42). Identificativo.
- `mse`: errore di ricostruzione. Piu' basso = migliore ricostruzione.
  Dovrebbe essere simile tra seed se il training converge stabilmente.
- `l0_mean` / `l0_std`: media e deviazione standard del numero di feature
  attive per campione. Con k=32, ci aspettiamo `l0_mean` circa 32.
- `dead_features_pct`: percentuale di feature mai attivate. Se alta,
  il dizionario e' sovradimensionato.
- `dict_utilization_pct`: complementare di dead_features_pct. Percentuale
  di feature almeno una volta attive.
- `activation_entropy`: entropia della distribuzione delle attivazioni.
  Misura quanto "uniformemente" il dizionario viene usato.
- `feature_frequency_mean` / `feature_frequency_std`: statistica sulla
  frequenza di attivazione delle feature (media e variazione).

L'insieme di queste metriche permette di distinguere:

- Seed sani (MSE basso, L0 ~ k, dead% moderato, entropia alta)
- Seed problematici (MSE alto, dead% alto, entropia bassa)

**Posizione nella pipeline:**

```text
stability_analysis
    --> per ogni seed: calcola SeedMetrics
    --> aggregati in StabilityResult.per_seed_metrics
```

---

## 8. ClusteringResult

```python
@dataclass(frozen=True)
class ClusteringResult:
    """Concept clustering analysis result."""

    n_active_features: int
    n_dead_features: int
    high_correlation_pairs: int
    correlation_threshold: float
    mean_co_occurrence: float
```

**Perche:**

Cattura il risultato dell'analisi di clustering sui concetti. Risponde alla
domanda: "ci sono concetti ridondanti nel dizionario?"

- `n_active_features`: numero di feature attivate almeno una volta.
- `n_dead_features`: feature mai attivate (n_dead = dict_size - n_active).
- `high_correlation_pairs`: numero di coppie di feature con cosine similarity
  > threshold nel loro pattern di co-occorrenza. Coppie correlate suggeriscono
  ridondanza.
- `correlation_threshold`: la soglia usata (default 0.7).
- `mean_co_occurrence`: media della matrice di co-occorrenza normalizzata.
  Valore basso indica feature ben differenziate.

Un `high_correlation_pairs` alto suggerisce che il `dict_size` potrebbe essere
ridotto senza perdere informazione — supporto empirico per il suggerimento
di ridurre da 4096.

**Posizione nella pipeline:**

```text
stability_analysis.compute_concept_clustering()
    --> ClusteringResult
    --> aggregato in StabilityResult.clustering
```

---

## 9. StabilityResult

```python
@dataclass(frozen=True)
class StabilityResult:
    """Output of stability_analysis stage — cross-seed comparison."""

    mean_jaccard: float
    std_jaccard: float
    jaccard_matrix: list[list[float]]
    per_seed_metrics: dict[int, SeedMetrics]
    clustering: ClusteringResult
    config_snapshot: dict
```

**Perche:**

E' l'output completo dello stadio `stability_analysis`. E' il contratto
piu' ricco perche' aggrega tre analisi distinte (Jaccard, per-seed metrics,
clustering) in un unico contenitore strutturato:

- `mean_jaccard` / `std_jaccard`: riassunto numerico della stabilita'.
  Mean Jaccard > 0.6 indica concetti robusti; < 0.2 indica artefatti.
- `jaccard_matrix`: matrice simmetrica (n_seeds x n_seeds) con i valori
  Jaccard per ogni coppia di seed. Permette ispezione visiva (heatmap).
- `per_seed_metrics`: mappa seed -> SeedMetrics. Per confronto dettagliato
  e identificazione di seed outlier.
- `clustering`: il ClusteringResult con l'analisi di ridondanza.
- `config_snapshot`: copia della configurazione usata (seeds, n_samples).
  Fondamentale per riproducibilita'.

`config_snapshot` come `dict` (non dataclass) e' intenzionale: la config
e' dinamica e varia tra run, non ha una struttura fissa. Un dict permette
flessibilita' senza rompere il contratto.

**Posizione nella pipeline:**

```text
stability_analysis --> StabilityResult
    --> serializzato in results/stability_analysis.json
    --> consumato da visualization per le heatmap Jaccard
    --> consumato da visualization per i grafici per-seed
```

---

## Diagramma del flusso dei contratti

```python
                    contracts.py (tutti i dataclass)
                               |
        +------+--------+------+--------+
        |      |        |      |        |
        v      v        v      v        v
   Candidate  Concept   Concept   Finding Explanation
      Name     Name       Map
        |      |        |
        +------+--+  +--+--+
                  |  |     |
                  v  v     v
              SeedMetrics  ClusteringResult
                  |           |
                  +-----+-----+
                        |
                        v
                  StabilityResult
```

I contratti si organizzano in tre "gruppi" che corrispondono agli stadi:

1. **Naming**: `CandidateName` -> `ConceptName` -> `ConceptMap`
2. **Explanation**: `Finding` -> `Explanation`
3. **Stability**: `SeedMetrics` + `ClusteringResult` -> `StabilityResult`

---

## Relazione con gli altri file

```text
sae_module.py  --> produce i dati grezzi (tensor, similarita')
concept_naming.py  --> usa CandidateName, ConceptName, ConceptMap
generate_explanations.py  --> usa Finding, Explanation
stability_analysis.py  --> usa SeedMetrics, ClusteringResult, StabilityResult
protocols.py  --> definisce le interfacce che restituiscono questi contratti
visualization.py  --> consuma questi contratti per generare grafici
tracking.py  --> logga metriche estratte da questi contratti
```

I contratti sono il "lingua franca" della pipeline: ogni stadioproduce e
consuma tipi definiti qui. Senza di essi, la pipeline sarebbe un groviglio
di dict e tensor non tipizzati.
