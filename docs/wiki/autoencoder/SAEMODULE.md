# SAEModule - Documentazione completa

Questo documento descrive ogni sezione di `src/sae_module.py`, il modulo facade
che espone un'interfaccia unificata per il ciclo di vita di un Sparse Autoencoder
Top-K: training, caricamento, inferenza, naming dei concetti e metriche.

---

## 1. Docstring e importazioni

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE

logger = logging.getLogger(__name__)
```

**Perche:**

- `from __future__ import annotations` permette di usare `str | Path` come type hint
  anche su Python < 3.10 (valutazione lazy delle annotazioni).
- `torch.nn.functional` (aliasato come `F`) fornisce operazioni stateless come
  `mse_loss` e `normalize` senza dover istanziare moduli.
- `DataLoader` e `TensorDataset` servono per creare un iteratore efficiente sui batch.
- `dictionary_learning` e' la libreria esterna (saprmarks/dictionary_learning) che
  fornisce l'architettura `AutoEncoderTopK` e il training loop `trainSAE`.
- Il logger standard permette di tracciare il progresso senza print() sparse.

---

## 2. DEFAULT_CONFIG

```python
DEFAULT_CONFIG = {
    "activation_dim": 512,
    "dict_size": 4096,
    "k": 32,
    "lr": 5e-5,
    "steps": 50_000,
    "warmup_steps": 1000,
    "batch_size": 256,
    "lm_name": "BiomedCLIP",
    "layer": 0,
    "device": "cuda",
}
```

**Perche:**

Centralizza gli iperparametri default del SAE in un unico dizionario. Ogni istanza
di `SAEManager` puo' sovrascrivere singoli valori passando un dict parziale al
costruttore. I valori chiave:

- `activation_dim=512`: dimensione degli embedding BiomedCLIP (input/output del SAE).
- `dict_size=4096`: dimensione del dizionario sparse (numero di "concetti" appresi).
  Rapporto di overcompleteness = 4096/512 = 8x.
- `k=32`: Top-K sparsity - solo 32 neuroni attivi su 4096 per ogni input.
  Questo forza il SAE ad apprendere rappresentazioni altamente sparse.
- `lr=5e-5`: learning rate basso, adatto alla natura non-supervisionata del task.
- `steps=50_000`: numero di step di training (non epoche).
- `warmup_steps=1000`: riscaldamento lineare del learning rate per stabilizzare
  le prime iterazioni.
- `lm_name` e `layer`: metadati richiesti da `trainSAE` per il logging interno.

---

## 3. Classe SAEManager - Costruttore

```python
class SAEManager:
    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._ae: Optional[AutoEncoderTopK] = None
        self._model_dir: Optional[Path] = None
```

**Perche:**

- `{**DEFAULT_CONFIG, **(config or {})}` fa un merge: i default vengono sovrascritti
  solo se l'utente passa valori espliciti. Pattern standard per configurazione
  a cascata.
- `self._ae` contiene il modello caricato; e' `None` fino a quando non si chiama
  `load()` o `train()`. Il prefisso `_` segnala che e' interno.
- `self._model_dir` tiene traccia di da dove il modello e' stato caricato.

---

## 4. Property `is_loaded`

```python
@property
def is_loaded(self) -> bool:
    return self._ae is not None
```

**Perche:**

Guard semplice per verificare se il SAE e' pronto prima di usare encode/decode.
Usata internamente da `_check_loaded()`.

---

## 5. Metodo `train()`

```python
def train(
    self,
    embeddings_path: str | Path,
    seed: int = 42,
    save_dir: str | Path = "models",
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Path:
```

### 5.1 Caricamento embeddings

```python
embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
if embeddings.dim() != 2 or embeddings.shape[1] != self.config["activation_dim"]:
    raise ValueError(
        f"Expected shape (N, {self.config['activation_dim']}), got {embeddings.shape}"
    )
```

**Perche:**

- `map_location="cpu"`: carica sempre su CPU indipendentemente da dove il tensor
  e' stato salvato (evita crash se salvato su GPU e caricato su macchina senza GPU).
- `weights_only=True`: sicurezza - impedisce l'esecuzione di codice arbitrario
  contenuto nel file .pt (pickle exploits).
- La validazione della shape con `ValueError` previene errori criptici piu'
  avanti nel training (es. dimensioni incompatibili nelle matmul).

### 5.2 DataLoader con generatore infinito

```python
loader = DataLoader(
    TensorDataset(embeddings),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=(device != "cpu"),
)

def batch_generator():
    while True:
        for (batch,) in loader:
            yield batch.to(device)
```

**Perche:**

- `TensorDataset(embeddings)` wrappa il tensor in un dataset indicizzabile.
- `shuffle=True`: randomizza l'ordine dei campioni ad ogni epoca.
- `drop_last=True`: scarta l'ultimo batch incompleto per mantenere dimensioni costanti.
- `pin_memory=True` (quando su GPU): pre-alloca la memoria in page-locked RAM,
  accelerando il trasferimento CPU -> GPU.
- Il generatore infinito `while True` e' necessario perche' `trainSAE` ragiona
  in step, non in epoche. Consuma batch uno alla volta fino a `steps` totali,
  ciclando automaticamente sul dataset.
- `(batch,)` con la virgola: `TensorDataset` restituisce tuple, anche con un
  solo tensor. La destructuring estrae il tensor dalla tupla singola.

### 5.3 Configurazione del trainer

```python
trainer_config = {
    "trainer": TopKTrainer,
    "activation_dim": self.config["activation_dim"],
    "dict_size": self.config["dict_size"],
    "k": self.config["k"],
    "steps": steps,
    "layer": self.config["layer"],
    "lm_name": self.config["lm_name"],
    "lr": self.config["lr"],
    "warmup_steps": self.config["warmup_steps"],
    "seed": seed,
    "device": device,
}
```

**Perche:**

`trainSAE` accetta una lista di trainer configs (supporta training parallelo di
piu' SAE). Noi ne passiamo uno solo. Il `TopKTrainer` implementa la variante
Top-K dell'autoencoder sparse dove la sparsita' viene imposta selezionando solo
le k attivazioni piu' alte, anziche' usare un termine di penalita' L1 nel loss.

### 5.4 Chiamata a trainSAE

```python
trainSAE(
    data=batch_generator(),
    trainer_configs=[trainer_config],
    steps=steps,
    save_dir=str(model_dir),
    device=device,
    autocast_dtype=torch.bfloat16,
    verbose=True,
)
```

**Perche:**

- `data=batch_generator()`: passa il generatore (lazy) non una lista concreta.
- `autocast_dtype=torch.bfloat16`: mixed precision training - le forward/backward
  pass usano bfloat16 per accelerare su GPU moderne, mentre i parametri master
  restano in float32.
- `save_dir`: la libreria salva il modello come `ae.pt` dentro questa directory.
- Dopo il training, il metodo chiama `self.load(model_dir)` per rendere il SAE
  immediatamente utilizzabile senza doverlo ricaricare manualmente.

---

## 6. Metodo `load()`

```python
def load(self, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    ae_path = model_dir / "ae.pt"

    if not ae_path.exists():
        raise FileNotFoundError(f"Model not found: {ae_path}")

    self._ae = AutoEncoderTopK.from_pretrained(
        str(ae_path),
        k=self.config["k"],
        device=self.config["device"],
    )
    self._ae.eval()
    self._model_dir = model_dir
```

**Perche:**

- `from_pretrained` ricostruisce l'architettura e carica i pesi. Necessita di `k`
  perche' il file salvato non contiene questo iperparametro.
- `.eval()`: mette il modello in modalita' inferenza (disattiva dropout, batchnorm
  in modalita' running stats). Anche se questo SAE non ha dropout, e' best practice.
- `FileNotFoundError` con path esplicito aiuta il debugging quando il path e' sbagliato.

---

## 7. Metodo `encode()`

```python
def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
    self._check_loaded()
    with torch.no_grad():
        return self._ae.encode(embeddings.to(self._device))
```

**Perche:**

- `torch.no_grad()`: disabilita il calcolo dei gradienti, riducendo consumo di
  memoria e accelerando l'inferenza. Non serve il gradiente durante l'encoding.
- `.to(self._device)`: sposta l'input sullo stesso device del modello (CPU/GPU).
- Il risultato e' un tensor (B, 4096) con esattamente k=32 valori non-zero per riga.
  I valori non-zero rappresentano le attivazioni dei concetti piu' rilevanti.

---

## 8. Metodo `encode_topk()`

```python
def encode_topk(self, embeddings):
    self._check_loaded()
    with torch.no_grad():
        encoded, values, indices, _ = self._ae.encode(
            embeddings.to(self._device), return_topk=True
        )
    return encoded, values, indices
```

**Perche:**

Variante di `encode()` che restituisce anche i valori e gli indici dei k neuroni
attivati. Utile per:
- Analisi di stabilita' (confronto degli indici tra seed diversi)
- Naming dei concetti (sapere quali feature sono attive)
- `_` ignora il quarto valore di ritorno (dettaglio implementativo della libreria)

---

## 9. Metodo `decode()`

```python
def decode(self, sparse: torch.Tensor) -> torch.Tensor:
    self._check_loaded()
    with torch.no_grad():
        return self._ae.decode(sparse.to(self._device))
```

**Perche:**

La decodifica moltiplica la rappresentazione sparse (4096-dim) per la matrice decoder
per riproiettarla nello spazio embedding (512-dim). E' l'operazione inversa di
encode: `x_hat = W_dec @ sparse + bias`.

---

## 10. Metodo `reconstruct()`

```python
def reconstruct(self, embeddings: torch.Tensor) -> torch.Tensor:
    self._check_loaded()
    with torch.no_grad():
        return self._ae(embeddings.to(self._device))
```

**Perche:**

Shortcut per encode + decode in un singolo forward pass. `self._ae(x)` chiama
il metodo `forward()` del modello che internamente fa encode -> decode.
Usato per calcolare l'errore di ricostruzione (MSE).

---

## 11. Metodo `get_decoder_weights()`

```python
def get_decoder_weights(self) -> torch.Tensor:
    self._check_loaded()
    # decoder.weight is (activation_dim, dict_size), transpose to (dict_size, activation_dim)
    return self._ae.decoder.weight.data.T.clone()
```

**Perche:**

La matrice decoder W_dec ha forma (512, 4096) in PyTorch (convenzione: out_features
x in_features per nn.Linear). La trasponiamo a (4096, 512) cosi' ogni riga
rappresenta un "concetto" - un vettore 512-dim nello spazio embedding.

- `.data`: accede ai dati raw senza autograd.
- `.T`: traspone.
- `.clone()`: crea una copia indipendente per evitare che modifiche esterne
  corrompano i pesi del modello.

Ogni riga di questa matrice e' la "direzione" del concetto nello spazio
embedding. Confrontandola (cosine similarity) con le embedding del vocabolario
medico, possiamo assegnare un nome a ciascun concetto.

---

## 12. Metodo `get_top_concepts()`

```python
def get_top_concepts(self, embeddings, n=5):
    self._check_loaded()
    with torch.no_grad():
        sparse = self._ae.encode(embeddings.to(self._device))

    results = []
    for row in sparse:
        topk = row.topk(n)
        concepts = [
            (idx.item(), val.item())
            for idx, val in zip(topk.indices, topk.values)
        ]
        results.append(concepts)
    return results
```

**Perche:**

Per ogni campione, identifica i top-n concetti con attivazione piu' alta.
Restituisce una lista di tuple (feature_id, activation_value) ordinate per
attivazione decrescente. Usato nella pipeline di spiegazione per selezionare
i concetti dominanti di un'immagine.

- `row.topk(n)`: operazione PyTorch che trova i top-n valori e i rispettivi indici.
- `.item()`: converte un tensor scalare in un float/int Python.

---

## 13. Metodo `name_concepts()`

```python
def name_concepts(self, vocab_embeddings, vocab_labels, top_n=3):
    self._check_loaded()
    W_dec = self.get_decoder_weights()  # (dict_size, 512)

    W_norm = F.normalize(W_dec, dim=1)
    V_norm = F.normalize(vocab_embeddings.to(self._device), dim=1)

    similarities = W_norm @ V_norm.T  # (dict_size, V)

    concept_names = {}
    for feat_id in range(self.config["dict_size"]):
        topk = similarities[feat_id].topk(top_n)
        candidates = [
            {"label": vocab_labels[idx.item()], "score": val.item()}
            for val, idx in zip(topk.values, topk.indices)
        ]
        concept_names[feat_id] = {
            "name": candidates[0]["label"],
            "score": candidates[0]["score"],
            "candidates": candidates,
        }
    return concept_names
```

**Perche:**

Assegna nomi medici ai 4096 concetti appresi dal SAE:

1. **Ottiene W_dec** (4096, 512): ogni riga e' un vettore-concetto.
2. **Normalizza** entrambe le matrici (concetti e vocabolario) a norma L2 unitaria.
   Dopo la normalizzazione, il prodotto scalare equivale alla cosine similarity.
3. **Matmul** W_norm @ V_norm.T produce una matrice (4096, V) dove ogni cella
   [i,j] e' la similarita' tra il concetto i e il termine j del vocabolario.
4. **Per ogni concetto**, seleziona i top_n termini piu' simili come candidati
   per il naming.

Il risultato e' un dizionario dove ogni feature ha:
- `name`: il termine piu' simile (candidato #1)
- `score`: il cosine similarity con quel termine
- `candidates`: lista dei top_n candidati con i rispettivi score

---

## 14. Metodo `compute_reconstruction_mse()`

```python
def compute_reconstruction_mse(self, embeddings: torch.Tensor) -> float:
    self._check_loaded()
    with torch.no_grad():
        x = embeddings.to(self._device)
        x_hat = self._ae(x)
        return F.mse_loss(x_hat, x).item()
```

**Perche:**

Metrica fondamentale per valutare la qualita' del SAE: quanto bene ricostruisce
gli input dopo la compressione sparse. Un MSE basso indica che le 32 feature
attive catturano la maggior parte dell'informazione nell'embedding originale.

`F.mse_loss` calcola la media di (x - x_hat)^2 su tutte le dimensioni e campioni.

---

## 15. Metodo `compute_sparsity_metrics()`

```python
def compute_sparsity_metrics(self, embeddings: torch.Tensor) -> dict:
    self._check_loaded()
    with torch.no_grad():
        sparse = self._ae.encode(embeddings.to(self._device))

    # L0: count of non-zero activations per sample (expected ~k)
    l0 = (sparse != 0).float().sum(dim=1)

    # Hoyer sparsity
    n = sparse.shape[1]
    l1 = sparse.abs().sum(dim=1)
    l2 = sparse.norm(dim=1)
    hoyer = (n**0.5 - l1 / (l2 + 1e-8)) / (n**0.5 - 1)

    # Dead features
    active_per_feature = (sparse != 0).float().sum(dim=0)
    dead_pct = (active_per_feature == 0).float().mean().item() * 100

    return {
        "l0_mean": l0.mean().item(),
        "l0_std": l0.std().item(),
        "hoyer_mean": hoyer.mean().item(),
        "dead_features_pct": dead_pct,
    }
```

**Perche:**

Tre metriche complementari sulla sparsita':

### L0 (pseudo-norma)
Conta il numero di attivazioni non-zero per campione. Con k=32, ci aspettiamo
L0 ~ 32. Se diverge, c'e' un problema nel Top-K enforcement.

### Hoyer sparsity
Formula: `(sqrt(n) - L1/L2) / (sqrt(n) - 1)`

- Se il vettore e' perfettamente sparse (un solo valore non-zero): Hoyer = 1.0
- Se il vettore e' uniforme (tutti i valori uguali): Hoyer = 0.0
- `1e-8` previene la divisione per zero quando L2 = 0.

La Hoyer sparsity cattura non solo QUANTI elementi sono attivi, ma anche quanto
e' concentrata l'energia nelle poche attivazioni.

### Dead features
Features che non si attivano MAI su nessun campione del batch. Una percentuale
alta di dead features indica capacita' sprecata del dizionario.
`sum(dim=0)` conta le attivazioni per feature (non per campione).

---

## 16. Metodo statico `compute_stability()`

```python
@staticmethod
def compute_stability(model_dirs, embeddings, config=None, n=32):
    managers = []
    for d in model_dirs:
        mgr = SAEManager(config)
        mgr.load(d)
        managers.append(mgr)

    active_sets = []
    for mgr in managers:
        _, _, indices = mgr.encode_topk(embeddings)
        sets_per_sample = [set(row.tolist()) for row in indices.cpu()]
        active_sets.append(sets_per_sample)
```

**Perche:**

### Obiettivo
Misurare la robustezza dei concetti: se alleniamo il SAE con seed diversi,
i concetti appresi sono gli stessi? Usiamo la Jaccard similarity per confrontare
gli insiemi di feature attivate.

### Fase 1: Caricamento modelli
Carica un `SAEManager` per ogni seed. `@staticmethod` perche' il metodo opera
su multipli modelli, non su `self`.

### Fase 2: Estrazione insiemi attivi
Per ogni modello, `encode_topk` restituisce gli indici delle k feature attive
per ogni campione. Li convertiamo in set Python per il calcolo Jaccard.
`.cpu()` e' necessario per convertire da tensor GPU a lista Python.

### Fase 3: Matrice Jaccard

```python
    n_seeds = len(model_dirs)
    n_samples = embeddings.shape[0]
    jaccard_matrix = torch.zeros(n_seeds, n_seeds)

    for i in range(n_seeds):
        for j in range(i, n_seeds):
            if i == j:
                jaccard_matrix[i, j] = 1.0
                continue
            jaccards = []
            for s in range(n_samples):
                a, b = active_sets[i][s], active_sets[j][s]
                if len(a | b) > 0:
                    jaccards.append(len(a & b) / len(a | b))
                else:
                    jaccards.append(0.0)
            mean_j = sum(jaccards) / len(jaccards)
            jaccard_matrix[i, j] = mean_j
            jaccard_matrix[j, i] = mean_j
```

**Perche:**

Jaccard Index J(A, B) = |A intersect B| / |A union B|:
- J = 1.0: i due modelli attivano esattamente le stesse feature per quel campione.
- J = 0.0: nessuna feature in comune.

La matrice e' simmetrica e ha 1.0 sulla diagonale. Calcoliamo solo il triangolo
superiore e specchiamo.

### Fase 4: Statistiche riassuntive

```python
    mask = torch.triu(torch.ones(n_seeds, n_seeds), diagonal=1).bool()
    upper_vals = jaccard_matrix[mask]

    return {
        "jaccard_matrix": jaccard_matrix,
        "mean_jaccard": upper_vals.mean().item(),
        "std_jaccard": upper_vals.std().item(),
    }
```

**Perche:**

`torch.triu(..., diagonal=1)` crea una maschera per il triangolo superiore
(esclusa la diagonale). Estrae solo i confronti unici (evitando duplicati e
self-similarity). La media e la deviazione standard danno un riassunto
della stabilita' complessiva.

---

## 17. Metodi privati

```python
def _check_loaded(self):
    if not self.is_loaded:
        raise RuntimeError("SAE not loaded. Call .load(model_dir) or .train() first.")

@property
def _device(self) -> str:
    return self.config["device"]
```

**Perche:**

- `_check_loaded()`: guard chiamata all'inizio di ogni metodo che richiede il modello.
  Lancia `RuntimeError` con un messaggio che spiega come risolvere.
- `_device`: property per accedere al device senza scrivere `self.config["device"]`
  ogni volta. Convenienza interna.

---

## Diagramma del flusso dati

```
Input: embeddings (B, 512)
         |
    [encode: x @ W_enc + b_enc]
         |
    [top-k: mantieni solo k=32 valori piu' alti, azzera il resto]
         |
    Sparse: (B, 4096) con 32 non-zero per riga
         |
    [decode: sparse @ W_dec + b_dec]
         |
Output: x_hat (B, 512) - ricostruzione approssimata
```

Il loss durante il training e' MSE(x, x_hat). La sparsita' e' imposta
architetturalmente dal Top-K (non da un termine L1 nel loss).
