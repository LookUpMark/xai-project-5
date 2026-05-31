# visualization.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/visualization.py`,
il modulo che genera e salva grafici per la diagnostica del training, l'analisi
dei concetti e la valutazione della stabilita' dell'SAE.

---

## 1. Docstring e importazioni

```python
"""
visualization.py — Standard SAE visualizations using seaborn/matplotlib.

Generates and saves figures for training diagnostics, concept analysis,
and stability evaluation. Figures are saved to results/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
```

**Perche:**

Questo file esiste per risolvere un problema di osservabilita': i risultati
della pipeline SAE sono numeri (JSON, tensor) che da soli non permettono
di capire rapidamente la qualita' del modello. I grafici trasformano questi
numeri in informazioni visive immediate:

- Un Jaccard heatmap con valori alti ovunque = concetti stabili.
- Una distribuzione di score centrata su 0.8 = naming affidabile.
- Un bar chart con dead features al 30% = dizionario inefficiente.

### Perche' seaborn + matplotlib (non Plotly o altro)
- **matplotlib + seaborn**: standard de facto in scientific Python, zero
  dipendenze extra (già usati da pandas, sklearn), output come file statici
  (PNG) perfetti per paper e report.
- **Plotly/Bokeh**: richiedono browser, sono interattivi ma pesanti, non
  ideali per report statici o CI/CD.
- La scelta riflette il principio: usare l'outil piu' semplice che risolve
  il problema.

### Perche' import lazy (dentro le funzioni)
`matplotlib.pyplot` e `seaborn` non sono importati a livello di modulo.
L'import e' dentro ogni funzione. Motivo:
- Evita l'inizializzazione del backend grafico quando il modulo viene
  importato ma le funzioni non sono chiamate.
- Evita crash se matplotlib non e' installato — il modulo si importa
  comunque, e l'errore compare solo quando si tenta di generare un grafico.
- In contesti headless (server CI, Docker senza display), previene errori
  di backend TkAgg/Qt5Agg.

`numpy` e' importato a livello di modulo perche' usato per calcoli
semplici (`np.mean()`) — e' una dipendenza gia' presente.

---

## 2. Funzione helper `_ensure_dir()`

```python
def _ensure_dir(path: Path) -> None:
    """Create parent directories if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
```

**Perche:**

Funzione helper usata da tutte le funzioni di plotting. Garantisce che
la directory di output esista prima di salvare il file. Senza di essa,
`fig.savefig()` crasherebbe con `FileNotFoundError` se `results/figures/`
non esiste.

- `path.parent`: ottiene la directory (es. `results/figures/` da
  `results/figures/jaccard_heatmap.png`).
- `parents=True`: crea tutte le directory intermedie se necessario.
- `exist_ok=True`: non lancia errore se la directory esiste gia'.

E' un helper privato (prefisso `_`) perche' non e' parte dell'API pubblica
del modulo — e' dettaglio implementativo condiviso.

---

## 3. Funzione `plot_jaccard_heatmap()`

```python
def plot_jaccard_heatmap(
    jaccard_matrix: np.ndarray,
    seeds: list[int],
    save_path: Path,
) -> Path:
    """Plot pairwise Jaccard similarity heatmap across seeds."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = [str(s) for s in seeds]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Cross-Seed Jaccard Similarity")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Jaccard heatmap to {save_path}")
    return save_path
```

**Perche:**

Visualizza la stabilita' dei concetti tra seed diversi come una matrice
di calore. E' il grafico piu' importante per valutare la robustezza del SAE.

### Parametri

- `jaccard_matrix`: matrice numpy (n_seeds x n_seeds) con valori in [0, 1].
  La diagonale e' 1.0 (auto-similarita'), gli off-diagonal sono i valori
  Jaccard medi tra coppie di seed.
- `seeds`: lista degli seed usati (es. `[0, 42, 123, 456, 789]`).
  Usata come etichette sugli assi.
- `save_path`: dove salvare il PNG.

### Scelte di design

**Colormap `YlOrRd`** (giallo-arancio-rosso):
- Giallo = Jaccard basso (concetti instabili tra questa coppia).
- Rosso = Jaccard alto (concetti stabili tra questa coppia).
- Questa colormap e' percettivamente uniforme e distinguibile anche in
  stampa in bianco e nero.

**`vmin=0, vmax=1`**: forza l'intervallo [0, 1] indipendentemente dai
dati. Fondamentale perche' il Jaccard e' semanticamente limitato a [0, 1].
Senza questo, seaborn adatterebbe la scala ai dati, rendendo un Jaccard
di 0.5 "rosso scuro" quando in realta' e' mediocre.

**`annot=True, fmt=".3f"`**: mostra il valore numerico in ogni cella
con 3 decimali. Permette letture precise oltre al colore.

**`bbox_inches="tight"`**: rimuove lo spazio bianco extra attorno al grafico.
Produce file PNG piu' compatti e professionali.

**`plt.close(fig)`**: rilascia la memoria del grafico. Senza questo, se
si generano molti grafici in sequenza, matplotlib accumula oggetti Figure
in memoria causando memory leak.

**Restituisce `save_path`**: permette chaining — `path = plot_jaccard_heatmap(...)`
e poi usare `path` per altri scopi (es. logging dell'artefatto).

### Dati di input

La matrice proviene da `StabilityResult.jaccard_matrix` (contratto in
`contracts.py`), prodotto da `stability_analysis.py`.

---

## 4. Funzione `plot_concept_score_distribution()`

```python
def plot_concept_score_distribution(
    scores: list[float],
    save_path: Path,
) -> Path:
    """Plot histogram of concept naming cosine similarity scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True, ax=ax)
    ax.set_title("Concept Naming Score Distribution")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    mean_score = float(np.mean(scores))
    ax.axvline(mean_score, color="red", linestyle="--", label=f"Mean={mean_score:.3f}")
    ax.legend()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved concept score distribution to {save_path}")
    return save_path
```

**Perche:**

Visualizza la distribuzione dei cosine similarity tra i concetti SAE
e i termini del vocabolario medico. Risponde alla domanda: "quanto bene
il vocabolario riesce a spiegare i concetti appresi?"

### Parametri

- `scores`: lista dei cosine similarity (uno per feature nominata).
  Tipicamente 3000-4000 valori (le feature attive).
- `save_path`: dove salvare il PNG.

### Interpretazione del grafico

| Forma della distribuzione | Interpretazione |
|---------------------------|----------------|
| Picco su 0.8-0.9, stretta | Naming eccellente, vocabolario ben allineato |
| Picco su 0.5-0.7, larga | Naming accettabile, ma concetti ambigui |
| Picco su 0.2-0.4 | Vocabolario inadeguato per questi concetti |
| Bimodale | Alcuni concetti spiegabili, altri no |

### Scelte di design

**`bins=50`**: 50 bin per l'istogramma. Numero sufficiente per vedere
la struttura senza eccessivo rumore. Con 3500 feature, ogni bin contiene
in media ~70 valori.

**`kde=True`**: aggiunge una Kernel Density Estimate (curva smooth)
sovrapposta all'istogramma. Aiuta a visualizzare la forma della
distribuzione senza dipendere dal binning.

**Linea rossa per la media** (`axvline`):
- Linea verticale tratteggiata in rosso con etichetta "Mean=X.XXX".
- Permette di valutare rapidamente se la media e' in una zona accettabile.
- Il rosso e' un colore ad alto contrasto che si distingue dall'istogramma
  (tipicamente in blu/grigio di seaborn).

**`figsize=(10, 6)`**: formato orizzontale largo, adatto per istogrammi
dove l'asse x (similarity) ha una gamma continua e l'asse y (count)
ha valori alti.

### Dati di input

I scores provengono dall'output di `concept_naming.py`, contenuto in
`ConceptMap` come `concepts[feature_id].score` per ogni feature.

---

## 5. Funzione `plot_per_seed_metrics()`

```python
def plot_per_seed_metrics(
    metrics: dict[int, dict],
    save_path: Path,
) -> Path:
    """Plot grouped bar chart comparing metrics across seeds."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for seed, m in metrics.items():
        rows.append({
            "seed": str(seed),
            "MSE": m.get("mse", 0),
            "Dead %": m.get("dead_features_pct", 0),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df, x="seed", y="MSE", ax=axes[0],
                hue="seed", palette="Blues_d", legend=False)
    axes[0].set_title("Reconstruction MSE per Seed")
    sns.barplot(data=df, x="seed", y="Dead %", ax=axes[1],
                hue="seed", palette="Reds_d", legend=False)
    axes[1].set_title("Dead Features % per Seed")
    fig.suptitle("Per-Seed Metrics Comparison")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved per-seed metrics to {save_path}")
    return save_path
```

**Perche:**

Confronta visivamente le metriche principali tra i seed. Permette di
identificare outlier: un seed con MSE molto piu' alto degli altri indica
un problema di convergenza; un seed con dead features molto piu' alto
indica un dizionario inefficiente.

### Parametri

- `metrics`: dict seed -> dict di metriche. Ogni seed ha almeno `mse` e
  `dead_features_pct`. E' compatibile con `StabilityResult.per_seed_metrics`.
- `save_path`: dove salvare il PNG.

### Scelte di design

**Costruzione del DataFrame**:
I dati grezzi sono un dict di dict. Li converto in un DataFrame pandas
perche' seaborn richiede dati "tidy" (long format) per i barplot — ogni
riga e' un'osservazione (un seed), ogni colonna e' una variabile.

**`m.get("mse", 0)`**: usa `.get()` con default 0 per robustezza —
se un seed non ha una metrica particolare, non crasha ma mostra 0.

**Due subplot affiancati** (`1, 2`):
- Sinistra: MSE (errore di ricostruzione) — palette "Blues_d" (blu scuro
  per valori alti = cattivo).
- Destra: Dead Features % — palette "Reds_d" (rosso scuro per valori alti
  = cattivo).

Le palette hanno un significato semantico: valori alti in entrambe le
metriche sono negativi, quindi colori intensi = attenzione.

**`hue="seed"`**: richiesto da seaborn >= 0.13 per evitare il warning
"Setting with a copy". Ogni barra ha il proprio colore nella palette,
e `legend=False` rimuove la legenda ridondante (il label e' gia' sull'asse x).

**`fig.tight_layout()`**: evita sovrapposizione tra titolo, assi e subplot.

**`figsize=(14, 5)`**: formato largo orizzontale per due subplot affiancati.

### Dati di input

Le metriche provengono da `StabilityResult.per_seed_metrics` (contratto
in `contracts.py`), prodotto da `stability_analysis.py`. Il campo `seed`
di ogni `SeedMetrics` diventa la chiave del dict.

---

## 6. Funzione `plot_sparsity_summary()`

```python
def plot_sparsity_summary(
    dead_pct: float,
    utilization: float,
    entropy: float,
    save_path: Path,
) -> Path:
    """Plot sparsity summary metrics as annotated bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {
        "Dict Util %": utilization,
        "Dead %": dead_pct,
    }
    colors = ["#2ecc71" if v > 50 else "#e74c3c" for v in metrics.values()]
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax,
                palette=colors, hue=list(metrics.keys()), legend=False)
    ax.set_title(f"Sparsity Metrics (Entropy={entropy:.2f})")
    ax.set_ylabel("Percentage")
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sparsity summary to {save_path}")
    return save_path
```

**Perche:**

Riassume la salute del dizionario SAE in due metriche visive. E' il grafico
piu' semplice ma uno dei piu' informativi: a colpo d'occhio si capisce se
il dizionario e' ben utilizzato o sprecato.

### Parametri

- `dead_pct`: percentuale di feature mai attivate (0-100).
- `utilization`: percentuale di feature attive = 100 - dead_pct.
- `entropy`: entropia della distribuzione delle attivazioni. Inclusa nel
  titolo ma non come barra perche' e' su una scala diversa (non percentuale).
- `save_path`: dove salvare il PNG.

### Scelte di design

**Colori semantici condizionali**:
```python
colors = ["#2ecc71" if v > 50 else "#e74c3c" for v in metrics.values()]
```

- `#2ecc71` (verde): se la metrica e' > 50%, colore positivo.
  Per "Dict Util %": verde se piu' della meta' del dizionario e' usato.
  Per "Dead %": verde se meno del 50% delle feature sono morte — cioe'
  per "Dead %", verde e' strano semanticamente, ma il threshold e'
  invertito: un "Dead %" basso (< 50) e' positivo, e un "Dead %" alto
  (> 50) e' negativo.

In realta', per "Dead %", il valore e' "buono" quando e' basso (< 20%),
ma il threshold > 50 e' usato come limite di attenzione generico.
Un miglior design userebbe logiche diverse per le due metriche, ma la
semplicita' prevale in questo caso.

**Entropy nel titolo**: inclusa come annotazione testuale nel titolo perche'
e' una metrica su scala diversa (0-~6.5 bit, non percentuale) e non
puo' essere confrontata direttamente con le barre nel grafico.

**Due sole metriche**: il grafico intenzionalmente mostra solo Dict Utilization
e Dead Features %. Metriche piu' specifiche (L0, Hoyer, feature frequency)
sono nei grafici per-seed. Qui lo scopo e' un indicatore rapido.

### Dati di input

I valori provengono dall'output di `stability_analysis.py`, in
`StabilityResult.clustering` (n_active/n_dead per le percentuali) e
dalle `SeedMetrics` (entropy).

---

## 6. Funzione `plot_loss_curve()`

```python
def plot_loss_curve(
    steps: list[int],
    train_losses: list[float],
    test_losses: list[float],
    save_path: Path,
    title: str | None = None,
) -> Path:
    """Plot training and test loss curves over training steps."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_losses, "b-o", label="Train MSE", markersize=4)
    ax.plot(steps, test_losses, "r-s", label="Test MSE", markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE (Reconstruction Loss)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Training & Test Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved loss curve to {save_path}")
    return save_path
```

**Perche:**

Visualizza l'andamento della loss durante il training, permettendo di
diagnosticare convergenza, overfitting (divergenza train/test) e problemi
di learning rate.

### Parametri

- `steps`: lista degli step di training (asse x).
- `train_losses`: MSE di ricostruzione valutata sul train set ad ogni step.
- `test_losses`: MSE di ricostruzione valutata sul test set held-out ad ogni step.
- `save_path`: dove salvare il PNG.
- `title`: titolo custom opzionale. Se `None`, usa "Training & Test Loss Curve".

### Scelte di design

**Due curve separate**: mostrare train e test loss insieme permette di individuare
overfitting (train scende ma test sale) o underfitting (entrambe alte e piatte).

**Marker diversi** (`"b-o"` blu cerchi, `"r-s"` rosso quadrati): distinguibili
anche in stampa in bianco e nero (forma diversa), accessibili per daltonici
(marker shape diverso oltre al colore).

**Grid con alpha=0.3**: griglia leggera per leggere i valori senza dominare
il grafico.

### Dati di input

I dati provengono tipicamente dal notebook `pipeline_smoke_test.ipynb`, dove
il SAE viene addestrato con `save_steps` e la loss viene valutata a intervalli
regolari. Non e' chiamata dagli script CLI automatici, ma dal notebook per
analisi interattiva e per logging su W&B.

---

## Diagramma del flusso

```
[Dati dalla pipeline]
        |
        +---> StabilityResult.jaccard_matrix + seeds
        |           |
        |           v
        |     plot_jaccard_heatmap()
        |           |
        |           v
        |     results/figures/jaccard_heatmap.png
        |
        +---> ConceptMap scores
        |           |
        |           v
        |     plot_concept_score_distribution()
        |           |
        |           v
        |     results/figures/concept_score_distribution.png
        |
        +---> StabilityResult.per_seed_metrics
        |           |
        |           v
        |     plot_per_seed_metrics()
        |           |
        |           v
        |     results/figures/per_seed_metrics.png
        |
        +---> ClusteringResult + entropy
        |           |
        |           v
        |     plot_sparsity_summary()
        |           |
        |           v
        |     results/figures/sparsity_summary.png
        |
        +---> Training step losses (from notebook)
                    |
                    v
              plot_loss_curve()
                    |
                    v
              results/figures/loss_curve.png
```

---

## Relazione con gli altri file

```
visualization.py  (questo file)
    |
    +---> consuma dati da: stability_analysis.py (jaccard_matrix,
    |     per_seed_metrics, clustering)
    +---> consuma dati da: concept_naming.py (concept scores)
    +---> consuma dati da: notebook (loss curves at training steps)
    +---> consuma contratti da: contracts.py (StabilityResult,
    |     ConceptMap, ClusteringResult, SeedMetrics)
    +---> salva output su: results/figures/ (directory di output)
    +---> collabora con: tracking.py (i path restituiti possono essere
          loggati come artefatti W&B)
```

visualization.py e' l'ultimo stadio della catena di valore: prende i
dati strutturati prodotti dagli altri moduli e li trasforma in
rappresentazioni visive per l'analisi umana. Non produce dati per la
pipeline, ma produce insight per il ricercatore.
