# stability_analysis.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/stability_analysis.py`,
lo script che valuta la robustezza dei concetti SAE confrontando le attivazioni
di 5 modelli addestrati con seed diversi.

---

## 1. Docstring e metadata

```python
"""
stability_analysis.py - Multi-seed stability analysis and clustering

Evaluate robustness of SAE concepts by comparing activations across 5 SAEs
trained with different seeds. Computes Jaccard similarity and clustering metrics.

Prerequisites:
    - models/sae_seed{0,42,123,456,789}/ae.pt (all 5 seeds)
    - embeddings/visual_embeddings.pt

Run:
    python src/autoencoder/stability_analysis.py
"""
```

**Perche:**

La domanda fondamentale a cui questo script risponde: "I concetti che il SAE
impara sono reali o sono artefatti dell'inizializzazione random?"

Se 5 SAE addestrati con seed diversi attivano le stesse feature per le stesse
immagini, allora quei concetti sono robusti e probabilmente catturano struttura
reale nei dati.

---

## 2. Costanti e importazioni

```python
OUTPUT_PATH = config.paths.results_dir / "stability_analysis.json"
```

**Perche:**

Il risultato e' un singolo JSON con tutte le metriche di stabilita', salvato
in `results/stability_analysis.json`.

---

## 3. Funzione `compute_feature_frequency()`

```python
def compute_feature_frequency(mgr: SAEManager, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute activation frequency of each feature across the dataset."""
    with torch.no_grad():
        sparse = mgr.encode(embeddings)
    return (sparse != 0).float().mean(dim=0)
```

**Perche:**

Per ogni feature (0-4095), calcola la frazione di campioni in cui si attiva.

- `sparse != 0` crea una maschera booleana (B, 4096): True dove la feature e' attiva
- `.float().mean(dim=0)` media lungo i campioni -> vettore (4096,) con frequenze

Esempio: se feature 127 ha frequenza 0.15, significa che si attiva nel 15% delle
immagini. Feature con frequenza 0 sono "dead features" (mai usate).

Questa metrica serve per:
- Identificare dead features
- Capire la distribuzione di utilizzo del dizionario
- Individuare feature troppo generiche (frequenza alta) o troppo specifiche (frequenza ~0)

---

## 4. Funzione `compute_concept_clustering()`

```python
def compute_concept_clustering(model_dirs: list[Path], embeddings: torch.Tensor) -> dict:
    """Compute correlation between concept activation patterns."""
    mgr = SAEManager({"device": config.hardware.device})
    mgr.load(model_dirs[0])

    with torch.no_grad():
        sparse = mgr.encode(embeddings)
```

**Perche:**

Analizza se ci sono concetti ridondanti nel dizionario: feature diverse che
si attivano sempre insieme sugli stessi campioni.

### 4.1 Filtraggio dead features

```python
    active_mask = (sparse != 0).float().sum(dim=0) > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    n_active = active_indices.shape[0]

    logger.info(f"  Active features: {n_active}/{sparse.shape[1]}")
```

**Perche:**

Prima di calcolare correlazioni, rimuove le feature mai attivate (dead features).
Includerle introdurrebbe correlazioni spurie (due feature entrambe sempre zero
hanno correlazione perfetta ma non e' informativa).

- `sum(dim=0) > 0`: una feature e' "attiva" se si attiva almeno una volta sul dataset
- `.nonzero()`: ottiene gli indici delle feature attive

### 4.2 Matrice di co-occorrenza normalizzata

```python
    sparse_active = sparse[:, active_indices]
    binary = (sparse_active != 0).float()

    norms = binary.norm(dim=0, keepdim=True) + 1e-8
    binary_norm = binary / norms
    co_occurrence = (binary_norm.T @ binary_norm).cpu()
```

**Perche:**

Calcola la cosine similarity tra i pattern di attivazione delle feature:

1. `binary`: converte in 0/1 (ignora l'intensita', interessa solo se attiva o no)
2. `binary_norm`: normalizza ogni colonna (feature) a norma unitaria
3. `binary_norm.T @ binary_norm`: il prodotto scalare tra colonne normalizzate
   e' il cosine similarity tra i pattern di attivazione

Il risultato e' una matrice (n_active, n_active) dove [i,j] misura quanto
spesso le feature i e j co-occorrono.

- 1.0: si attivano sempre sugli stessi campioni (ridondanti)
- 0.0: non si attivano mai insieme (complementari)

### 4.3 Conteggio coppie correlate

```python
    threshold = 0.7
    high_corr_pairs = (co_occurrence > threshold).sum().item() - n_active
    high_corr_pairs //= 2

    return {
        "n_active_features": n_active,
        "n_dead_features": sparse.shape[1] - n_active,
        "high_correlation_pairs": high_corr_pairs,
        "correlation_threshold": threshold,
        "mean_co_occurrence": co_occurrence.mean().item(),
    }
```

**Perche:**

- `threshold = 0.7`: coppie con correlazione > 0.7 sono considerate potenzialmente
  ridondanti (catturano informazione simile)
- `- n_active`: sottrae la diagonale (ogni feature ha correlazione 1.0 con se stessa)
- `// 2`: la matrice e' simmetrica, ogni coppia e' contata due volte

Un numero alto di coppie correlate suggerisce che il `dict_size` potrebbe essere
ridotto senza perdere informazione (supporta il suggerimento di ridurre da 4096).

---

## 5. Funzione `main()`

### 5.1 Setup e validazione

```python
    model_dirs = [config.paths.models_dir / f"sae_seed{s}" for s in config.training.seeds]
    missing = [d for d in model_dirs if not (d / "ae.pt").exists()]
    if missing:
        logger.error(f"Missing models: {[str(m) for m in missing]}")
        logger.error("Run first: python src/autoencoder/train_sae.py")
        sys.exit(1)
```

**Perche:**

Verifica che tutti e 5 i modelli esistano. A differenza degli altri script che
necessitano di un solo modello, qui servono tutti perche' lo scopo e' il confronto
cross-seed.

### 5.2 Caricamento embeddings con limit opzionale

```python
    embeddings = torch.load(config.paths.visual_embeddings_path, map_location="cpu", weights_only=True)
    if config.training.stability_max_samples:
        embeddings = embeddings[: config.training.stability_max_samples]
    logger.info(f"Embeddings: {embeddings.shape}")
```

**Perche:**

`stability_max_samples` permette di limitare i campioni usati per l'analisi.
L'analisi di stabilita' e' O(n_seeds^2 * n_samples) quindi con 7400 campioni
e 5 seed puo' richiedere tempo significativo. In sviluppo si puo' testare con
un sottoinsieme.

### 5.3 Cross-seed Jaccard stability

```python
    logger.info("Computing Jaccard stability across seeds...")
    stability = SAEManager.compute_stability(
        model_dirs, embeddings, config={"device": config.hardware.device}
    )
    logger.info(f"  Mean Jaccard: {stability['mean_jaccard']:.4f}")
    logger.info(f"  Std Jaccard:  {stability['std_jaccard']:.4f}")
```

**Perche:**

Delega al metodo statico `SAEManager.compute_stability()` (documentato nella
wiki di sae_module). Il risultato chiave e' `mean_jaccard`:

| Mean Jaccard | Interpretazione |
|-------------|-----------------|
| > 0.6 | Concetti molto stabili, alta robustezza |
| 0.4 - 0.6 | Stabilita' ragionevole |
| 0.2 - 0.4 | Stabilita' debole, i concetti sono parzialmente artefatti |
| < 0.2 | I concetti dipendono fortemente dal seed - scarsa affidabilita' |

### 5.4 Per-seed metrics

```python
    per_seed_metrics = {}
    for seed, model_dir in zip(config.training.seeds, model_dirs):
        mgr = SAEManager({"device": config.hardware.device})
        mgr.load(model_dir)

        mse = mgr.compute_reconstruction_mse(embeddings)
        sparsity = mgr.compute_sparsity_metrics(embeddings)
        freq = compute_feature_frequency(mgr, embeddings)

        per_seed_metrics[seed] = {
            "mse": mse,
            **sparsity,
            "feature_frequency_mean": freq.mean().item(),
            "feature_frequency_std": freq.std().item(),
        }
```

**Perche:**

Per ogni seed calcola metriche individuali:
- **MSE**: dovrebbe essere simile tra i seed (stessa capacita' ricostruttiva)
- **Sparsity (L0, Hoyer, dead %)**: dovrebbe essere simile
- **Feature frequency**: distribuzione dell'uso del dizionario

Se un seed ha metriche molto diverse dagli altri, potrebbe indicare un problema
di convergenza o un minimo locale particolarmente diverso.

### 5.5 Clustering analysis

```python
    clustering = compute_concept_clustering(model_dirs, embeddings)
    logger.info(f"  High-correlation pairs (>{clustering['correlation_threshold']}): "
                f"{clustering['high_correlation_pairs']}")
```

**Perche:**

Aggiunge informazione complementare alla Jaccard: non solo "i seed sono
d'accordo?" ma anche "ci sono concetti ridondanti nel dizionario?"

### 5.6 Salvataggio risultati

```python
    results = {
        "stability": {
            "mean_jaccard": stability["mean_jaccard"],
            "std_jaccard": stability["std_jaccard"],
            "jaccard_matrix": stability["jaccard_matrix"].tolist(),
        },
        "per_seed_metrics": per_seed_metrics,
        "clustering": clustering,
        "config": {"seeds": list(config.training.seeds), "n_samples": embeddings.shape[0]},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
```

**Perche:**

- `.tolist()`: converte tensor PyTorch in liste Python (JSON-serializzabile)
- Include la config usata per riproducibilita' (quanti campioni, quali seed)
- Il JSON e' consultabile da notebook di analisi o da report automatici

---

## Diagramma del flusso

```
[Input: 5 modelli + embeddings]
            |
    +---[1. Jaccard Stability]---+
    |       |                    |
    |   Per ogni coppia di seed: |
    |   J(A,B) = |A&B| / |A|B|  |
    |       |                    |
    |   mean_jaccard, std_jaccard|
    +----------------------------+
            |
    +---[2. Per-seed Metrics]----+
    |   MSE, L0, Hoyer, Dead%,  |
    |   Feature frequency        |
    +----------------------------+
            |
    +---[3. Clustering]----------+
    |   Co-occurrence matrix     |
    |   High-correlation pairs   |
    +----------------------------+
            |
[Output: results/stability_analysis.json]
```

---

## Output di esempio

```json
{
  "stability": {
    "mean_jaccard": 0.4523,
    "std_jaccard": 0.0312,
    "jaccard_matrix": [[1.0, 0.45, 0.46, ...], ...]
  },
  "per_seed_metrics": {
    "0":   {"mse": 0.0012, "l0_mean": 32.0, "dead_features_pct": 15.2, ...},
    "42":  {"mse": 0.0011, "l0_mean": 32.0, "dead_features_pct": 14.8, ...},
    ...
  },
  "clustering": {
    "n_active_features": 3480,
    "n_dead_features": 616,
    "high_correlation_pairs": 234,
    "correlation_threshold": 0.7,
    "mean_co_occurrence": 0.023
  },
  "config": {"seeds": [0, 42, 123, 456, 789], "n_samples": 7400}
}
```

---

## Relazione con gli altri script

```
train_sae (train 5 SAEs) --> stability_analysis (compare them)
                             |
                       [Jaccard matrix]
                       [Per-seed quality]
                       [Redundancy check]
```

I risultati di stability_analysis informano decisioni su:
- Se il `dict_size` e' appropriato (troppi dead features? troppi pair correlati?)
- Se il training e' stabile (Jaccard alto?)
- Se servono piu' seed o piu' dati
