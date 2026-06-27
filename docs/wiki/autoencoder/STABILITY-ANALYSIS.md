# stability_analysis.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/stability_analysis.py`,
lo script che valuta la robustezza dei concetti SAE confrontando le attivazioni
di 5 modelli addestrati con seed diversi sul TEST set held-out, con visualizzazione
della matrice Jaccard e delle metriche per-seed.

---

## 1. Docstring e metadata

```python
"""
stability_analysis.py -- Multi-seed stability analysis and clustering

Evaluate robustness of SAE concepts by comparing activations across
multiple SAEs trained with different seeds. Uses HELD-OUT test embeddings.
Computes Jaccard similarity, per-seed metrics, and concept clustering.

Prerequisites:
    - models/sae_seed{0,42,123,456,789}/ae.pt (all 5 seeds)
    - embeddings/test_embeddings.pt

Run:
    python src/autoencoder/stability_analysis.py
"""
```

**Perche:**

La domanda fondamentale: "I concetti SAE sono reali o artefatti dell'inizializzazione
random?" Se 5 SAE con seed diversi attivano le stesse feature, i concetti sono robusti.
"HELD-OUT test embeddings": stabilita' su dati mai visti, non da overfitting.
Servono tutti e 5 i modelli per il confronto cross-seed.

---

## 2. Importazioni e costanti

```python
import json, logging, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from autoencoder.sae_module import SAEManager
from autoencoder.tracking import init_tracking, log_artifact, finish_tracking
from autoencoder.visualization import plot_jaccard_heatmap, plot_per_seed_metrics

OUTPUT_PATH = config.paths.results_dir / "stability_analysis.json"
```

**Perche:**

- `tracking`: integrazione wandb per metriche di stabilita'.
- `plot_jaccard_heatmap`: heatmap Jaccard (seaborn, colormap "YlOrRd").
- `plot_per_seed_metrics`: bar chart comparativo (MSE, dead %) per seed.

---

## 3. Funzione `compute_feature_frequency()`

```python
def compute_feature_frequency(mgr: SAEManager, embeddings: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        sparse = mgr.encode(embeddings)
    return (sparse != 0).float().mean(dim=0)
```

**Perche:**

Per ogni feature (0-4095), la frazione di campioni in cui si attiva. Feature con
frequenza 0 = "dead features". Serve per identificare feature inutilizzate e
capire la distribuzione del dizionario.

---

## 4. Funzione `compute_concept_clustering()`

```python
def compute_concept_clustering(
    model_dirs: list[Path], embeddings: torch.Tensor, device: str
) -> dict:
```

Analizza ridondanza nel dizionario: feature diverse che si attivano sempre insieme.
Usa il primo modello (seed[0]) perche' la co-occorrenza e' proprieta' del
dizionario, non dell'inizializzazione.

```python
    # Filtra dead features (correlazione spuria se entrambe zero)
    active_mask = (sparse != 0).float().sum(dim=0) > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    # Cosine similarity tra pattern binari (ignora intensita')
    binary = (sparse[:, active_indices] != 0).float()
    norms = binary.norm(dim=0, keepdim=True) + 1e-8
    co_occurrence = (binary / norms).T @ (binary / norms)
```

**Perche:**

- Dead features (sempre zero) hanno cosine similarity perfetta ma non informativa.
- `binary` ignora l'intensita': interessa solo attiva/non-attiva.
- `1e-8` previene divisione per zero. Risultato: matrice (n_active, n_active).

```python
    threshold = config.training.correlation_threshold
    high_corr_pairs = (co_occurrence > threshold).sum().item() - n_active
    high_corr_pairs //= 2
```

**Perche:**

`threshold` da config (default: 0.7), non hardcoded. `- n_active`: sottrae la
diagonale. `// 2`: matrice simmetrica, coppie contate due volte. Numero alto di
coppie correlate suggerisce dict_size riducibile.

---

## 5. Funzione `run()`

### 5.1 Validazione prerequisiti

```python
def run() -> Path:
    model_dirs = [config.paths.models_dir / f"sae_seed{s}" for s in config.training.seeds]
    missing = [d for d in model_dirs if not (d / "ae.pt").exists()
               and not (d / "trainer_0" / "ae.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing models: {[str(m) for m in missing]}.")
```

**Perche:**

Servono tutti e 5 i modelli. Il check cerca `ae.pt` e `trainer_0/ae.pt` per
compatibilita' con diverse versioni della libreria.

### 5.2 Caricamento TEST embeddings

```python
    embeddings_path = config.paths.test_embeddings_path
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    if config.training.stability_max_samples:
        embeddings = embeddings[: config.training.stability_max_samples]
```

**Perche:**

Test set held-out: stabilita' generalizzabile, non stabilita' da overfitting.
`stability_max_samples` limita in sviluppo (analisi e' O(n_seeds^2 * n_samples)).

### 5.3 Cross-seed Jaccard

```python
    stability = SAEManager.compute_stability(
        model_dirs, embeddings, config={"device": config.hardware.device}
    )
```

| Mean Jaccard | Significato |
|-------------|------------|
| > 0.6 | Concetti molto stabili |
| 0.4 - 0.6 | Stabilita' ragionevole |
| 0.2 - 0.4 | Stabilita' debole |
| < 0.2 | Dipendono fortemente dal seed |

### 5.4 Per-seed metrics (caricamento uno alla volta)

```python
    per_seed_metrics = {}
    for seed, model_dir in zip(config.training.seeds, model_dirs):
        mgr = SAEManager({"device": config.hardware.device})
        mgr.load(model_dir)

        mse = mgr.compute_reconstruction_mse(embeddings)
        cosine = mgr.compute_cosine_reconstruction(embeddings)
        sparsity = mgr.compute_sparsity_metrics(embeddings)
        freq = compute_feature_frequency(mgr, embeddings)

        per_seed_metrics[seed] = {
            "mse": mse, "cosine_sim": cosine, **sparsity,
            "feature_frequency_mean": freq.mean().item(),
            "feature_frequency_std": freq.std().item(),
        }
```

**Perche:**

**Caricamento uno alla volta**: ogni SAE viene caricato e valutato, poi rimpiazzato
nel prossimo ciclo (~5x risparmio memoria GPU). Metriche: MSE e Cosine
(ricostruzione), sparsity (L0, Hoyer, dead %), feature frequency (mean, std).
Un seed con metriche anomale indica convergenza problematica.

### 5.5 Clustering e visualizzazione

```python
    clustering = compute_concept_clustering(model_dirs, embeddings, config.hardware.device)

    jaccard_np = stability["jaccard_matrix"].numpy()
    plot_jaccard_heatmap(jaccard_np, list(config.training.seeds),
        config.paths.figures_dir / "jaccard_heatmap.png")
    plot_per_seed_metrics(per_seed_metrics,
        config.paths.figures_dir / "per_seed_metrics.png")
```

**Perche:**

Complementare alla Jaccard: "ci sono concetti ridondanti?" Due grafici:
(1) Heatmap Jaccard 5x5 -- coppie simili/dissimili a colpo d'occhio.
(2) Bar chart MSE e dead % per seed -- uniformita' tra modelli.

### 5.6 Salvataggio e tracking

```python
    results = {
        "stability": {
            "mean_jaccard": stability["mean_jaccard"],
            "std_jaccard": stability["std_jaccard"],
            "jaccard_matrix": stability["jaccard_matrix"].tolist(),
        },
        "per_seed_metrics": per_seed_metrics,
        "clustering": clustering,
        "config": {"seeds": list(config.training.seeds),
                   "n_samples": embeddings.shape[0], "dataset": "test"},
    }
```

**Perche:**

`.tolist()` converte tensor in liste JSON-serializzabili. `"dataset": "test"`
documenta l'uso del test set. Config inclusa per riproducibilita'.

```python
    if config.wandb_cfg.enabled:
        init_tracking("stability_analysis", {
            "project": config.wandb_cfg.project,
            "mean_jaccard": stability["mean_jaccard"],
            "std_jaccard": stability["std_jaccard"],
        })
        log_artifact(OUTPUT_PATH, "stability_analysis", "results")
        finish_tracking()
```

No-op se wandb disabilitato. Jaccard mean/std sono metriche chiave per
confrontare esperimenti.

---

## 6. Funzione `main()`

```python
def main():
    run()

if __name__ == "__main__":
    main()
```

Pattern standard: `run()` testabile con valore di ritorno, `main()` wrapper.

---

## Diagramma del flusso

```text
[Input: 5 modelli + embeddings/test_embeddings.pt]
            |
    +--[1. Jaccard Stability]--+
    |   J(A,B)=|A&B|/|A|B|   |
    |   mean_jaccard, std      |
    +--------------------------+
            |
    +--[2. Per-seed Metrics]--+
    |   UNO ALLA VOLTA (mem)  |
    |   MSE, Cosine, L0, ...  |
    +--------------------------+
            |
    +--[3. Clustering]-------+
    |   Co-occurrence, thresh  |
    +--------------------------+
            |
    +--[4. Visualization]-----+
    |   Heatmap + bar chart    |
    +--------------------------+
            |
[Output: results/stability_analysis.json]
```

---

## Dipendenze dalla configurazione

| Variabile | Section | Default | Usata per |
|-----------|---------|---------|-----------|
| `config.training.seeds` | TrainingConfig | (0,42,123,456,789) | Quali modelli |
| `config.paths.test_embeddings_path` | PathsConfig | `embeddings/test_embeddings.pt` | Input held-out |
| `config.paths.models_dir` | PathsConfig | `models/` | Directory modelli |
| `config.paths.results_dir` | PathsConfig | `results/` | Output JSON |
| `config.paths.figures_dir` | PathsConfig | `results/figures/` | Output grafici |
| `config.training.stability_max_samples` | TrainingConfig | None | Limit sviluppo |
| `config.training.correlation_threshold` | TrainingConfig | 0.7 | Soglia ridondanza |
| `config.wandb_cfg.enabled` | WandbConfig | False | Tracking |

---

## Relazione con gli altri script

```text
train_sae (5 SAEs) --> stability_analysis (confronta su test set)
                          |
                    [Jaccard matrix]
                    [Per-seed quality]
                    [Redundancy check]
                    [Heatmap + bar chart]
```

I risultati informano decisioni su: dict_size appropriato, stabilita' del training,
necessita' di piu' seed/dati, correlation_threshold ottimale.
