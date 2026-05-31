# sae_module.py - Documentazione completa

Questo documento descrive ogni sezione di `src/autoencoder/sae_module.py`, il modulo facade
che espone un'interfaccia unificata per il ciclo di vita di un Sparse Autoencoder
Top-K: training, caricamento, inferenza, naming dei concetti e metriche.

---

## 1. Docstring e importazioni

```python
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
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
- `dataclasses`: modulo standard per convertire una dataclass frozen `SAEConfig`
  in un dizionario piatto tramite `_extract_sae_config()`.
- `hashlib`: calcola l'hash SHA-256 dei primi 100 embedding nel manifest di
  training per verificare che i dati non siano cambiati.
- `random` + `numpy.random` + `torch.manual_seed`: la propagazione completa del
  seed richiede di impostare tutti e tre i generatori pseudo-casuali (vedi
  `_set_global_seed()`).
- `torch.nn.functional` (aliasato come `F`) fornisce operazioni stateless come
  `mse_loss`, `cosine_similarity` e `normalize` senza dover istanziare moduli.
- `DataLoader` e `TensorDataset` servono per creare un iteratore efficiente
  e deterministico sui batch durante il training.
- `dictionary_learning` e' la libreria esterna (saprmarks/dictionary_learning)
  che fornisce l'architettura `AutoEncoderTopK` e il training loop `trainSAE`.
- Il logger standard permette di tracciare il progresso senza print() sparse.

---

## 2. Dizionario `_DEFAULTS`

```python
# Default config values -- kept in sync with config.py's SAEConfig.
# Change config.py, not here. These are only used when SAEManager
# is constructed without a config (e.g. in tests).
_DEFAULTS = {
    "activation_dim": 512,
    "dict_size": 4096,
    "k": 32,
    "lr": None,
    "steps": 50_000,
    "warmup_steps": 1_000,
    "batch_size": 256,
    "log_steps": 1_000,
    "decay_start_frac": 0.8,
    "lm_name": "BiomedCLIP",
    "layer": 0,
    "device": "cpu",
}
```

**Perche:**

Centralizza gli iperparametri default del SAE in un dizionario privato (il prefisso
`_` segnala che non deve essere usato esternamente). Rispetto alla versione precedente
(`DEFAULT_CONFIG`), questo dizionario e' esplicitamente marcato come **fonte di verita'
secondaria**: il valore canonico vive nella dataclass `SAEConfig` in `config.py`.

I campi `log_steps` e `decay_start_frac` sono nuovi rispetto alla vecchia
`DEFAULT_CONFIG`:
- `log_steps=1_000`: ogni quanti step la libreria stampa le metriche di training
  durante l'addestramento.
- `decay_start_frac=0.8`: frazione dei passi totali a cui inizia il decay
  del learning rate (es. con 50k step, il decay inizia al passaggio 40k).
- `lr=None`: significa che il learning rate viene auto-scalato dalla libreria
  (`2e-4 / sqrt(dict_size / 16384)`). Per dict_size=4096 restituisce circa 4e-4.

I valori fondamentali rimangono:
- `activation_dim=512`: dimensione degli embedding BiomedCLIP (input/output del SAE).
- `dict_size=4096`: dimensione del dizionario sparse (numero di "concetti" appresi).
  Rapporto di overcompleteness = 4096/512 = 8x.
- `k=32`: Top-K sparsity -- solo 32 neuroni attivi su 4096 per ogni input.
  Questo forza il SAE ad apprendere rappresentazioni altamente sparse.
- `device="cpu"`: il default e' CPU per sicurezza (i test non richiedono GPU);
  gli script della pipeline sovrascrivono con `"cuda"` quando necessario.

---

## 3. Funzione `_extract_sae_config()`

```python
def _extract_sae_config(cfg) -> dict:
    """Convert a frozen SAEConfig dataclass to a plain dict."""
    result = {}
    for f in dataclasses.fields(cfg):
        result[f.name] = getattr(cfg, f.name)
    return result
```

**Perche:**

`SAEConfig` e' una dataclass `frozen=True` (immutabile). `SAEManager` invece
lavora internamente con un dizionario. Questa funzione di conversione serve
quando l'utente passa un oggetto `SAEConfig` al costruttore di `SAEManager`
anziche' un dict.

Senza questa funzione, ogni call site dovrebbe gestire manualmente la conversione.
Qui invece e' centralizzata: `dataclasses.fields()` riflette su tutti i campi
della dataclass e `getattr()` ne estrae il valore, producendo un dict piatto
con le stesse chiavi di `_DEFAULTS`.

Il vantaggio di usare `dataclasses.fields()` rispetto all'accesso diretto ai
campi (es. `cfg.activation_dim`) e' che la funzione funziona per qualsiasi
dataclass senza dover conoscere i nomi dei campi a compile-time.

---

## 4. Classe SAEManager - Costruttore

```python
class SAEManager:
    def __init__(self, config: Optional[dict] = None):
        if config is not None and not isinstance(config, dict):
            config = _extract_sae_config(config)
        self.config = {**_DEFAULTS, **(config or {})}
        self._ae: Optional[AutoEncoderTopK] = None
        self._model_dir: Optional[Path] = None
```

**Perche:**

- La prima riga e' nuova rispetto alla versione precedente: se l'utente passa
  un oggetto `SAEConfig` (dataclass frozen) anziche' un dict, lo converte
  automaticamente tramite `_extract_sae_config()`. Questo permette di usare
  sia `SAEManager({"device": "cuda"})` che `SAEManager(config.sae)`.
- `{**_DEFAULTS, **(config or {})}` fa un merge: i default vengono sovrascritti
  solo se l'utente passa valori espliciti. Pattern standard per configurazione
  a cascata.
- `self._ae` contiene il modello caricato; e' `None` fino a quando non si chiama
  `load()` o `train()`. Il prefisso `_` segnala che e' interno.
- `self._model_dir` tiene traccia di da dove il modello e' stato caricato.

---

## 5. Property `is_loaded`

```python
@property
def is_loaded(self) -> bool:
    return self._ae is not None
```

**Perche:**

Guard semplice per verificare se il SAE e' pronto prima di usare encode/decode.
Usata internamente da `_check_loaded()`.

---

## 6. Funzione `_set_global_seed()`

```python
def _set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Perche:**

Per garantire la **riproducibilita' totale** del training, non basta impostare
il seed di PyTorch. Ci sono almeno 5 sorgenti di non-determinismo che devono
essere controllate:

1. **`random.seed(seed)`**: il modulo `random` standard di Python e' usato
   internamente da DataLoader per lo shuffle. Senza questo seed, l'ordine
   dei campioni varia tra run diverse.
2. **`np.random.seed(seed)`**: NumPy e' usato dalla libreria `dictionary_learning`
   per alcune inizializzazioni interne.
3. **`torch.manual_seed(seed)`**: il generatore pseudo-casuale di PyTorch
   (CPU). Controlla l'inizializzazione dei pesi del modello e altre operazioni
   random di PyTorch.
4. **`torch.cuda.manual_seed_all(seed)`**: imposta il seed per **tutte** le GPU.
   Il suffisso `_all` e' necessario perche' `manual_seed()` agisce solo sulla
   GPU corrente; in sistemi multi-GPU le altre rimarrebbero non deterministiche.
5. **`cudnn.deterministic = True`** e **`cudnn.benchmark = False`**: cuDNN usa
   algoritmi non deterministici di default per massimizzare la performance.
   `deterministic=True` forza algoritmi deterministici (piu' lenti ma riproducibili).
   `benchmark=False` disabilita la selezione automatica dell'algoritmo piu' veloce
   (che dipende dall'hardware e varierebbe tra macchine).

La versione precedente del modulo non impostava `cudnn.deterministic` e
`cudnn.benchmark`, rendendo il training non pienamente riproducibile su GPU.

---

## 7. Metodo `train()`

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

### 7.1 Propagazione del seed e caricamento embeddings

```python
    _set_global_seed(seed)

    embeddings = torch.load(
        embeddings_path, map_location="cpu", weights_only=True
    )
    if embeddings.dim() != 2 or embeddings.shape[1] != self.config["activation_dim"]:
        raise ValueError(
            f"Expected shape (N, {self.config['activation_dim']}), "
            f"got {embeddings.shape}"
        )
```

**Perche:**

- `_set_global_seed(seed)` e' la **prima cosa** che viene chiamata, prima ancora
  di caricare i dati. Questo garantisce che ogni operazione random successiva
  (incluso lo shuffle del DataLoader) sia deterministica.
- `map_location="cpu"`: carica sempre su CPU indipendentemente da dove il tensor
  e' stato salvato (evita crash se salvato su GPU e caricato su macchina senza GPU).
- `weights_only=True`: sicurezza -- impedisce l'esecuzione di codice arbitrario
  contenuto nel file .pt (pickle exploits).
- La validazione della shape con `ValueError` previene errori criptici piu'
  avanti nel training (es. dimensioni incompatibili nelle matmul).

### 7.2 DataLoader con generatore deterministico

```python
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(embeddings),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device != "cpu"),
        generator=generator,
    )

    def batch_generator():
        while True:
            for (batch,) in loader:
                yield batch.to(device)
```

**Perche:**

Cambiamento chiave rispetto alla versione precedente: il `DataLoader` riceve
ora un `generator` esplicito. Senza di esso, `DataLoader(shuffle=True)` crea
un generatore interno con seed random, rendendo lo shuffle non deterministico
anche dopo `_set_global_seed()`.

- `torch.Generator().manual_seed(seed)`: crea un generatore di numeri casuali
  PyTorch con un seed specifico. Passarlo al DataLoader garantisce che
  l'ordine di shuffle sia identico in ogni run con lo stesso seed.
- `TensorDataset(embeddings)` wrappa il tensor in un dataset indicizzabile.
- `shuffle=True`: randomizza l'ordine dei campioni ad ogni epoca.
- `drop_last=True`: scarta l'ultimo batch incompleto per mantenere dimensioni
  costanti. Senza di questo, l'ultimo batch potrebbe avere dimensione diversa
  causando batch statistics irregolari.
- `pin_memory=True` (quando su GPU): pre-alloca la memoria in page-locked RAM,
  accelerando il trasferimento CPU -> GPU tramite DMA diretto.
- Il generatore infinito `while True` e' necessario perche' `trainSAE` ragiona
  in step, non in epoche. Consuma batch uno alla volta fino a `steps` totali,
  ciclando automaticamente sul dataset.
- `(batch,)` con la virgola: `TensorDataset` restituisce tuple, anche con un
  solo tensor. La destructuring estrae il tensor dalla tupla singola.

### 7.3 Calcolo di `decay_start`

```python
    decay_start = int(steps * self.config["decay_start_frac"])
```

**Perche:**

Converte la frazione `decay_start_frac` (default 0.8) in un numero assoluto
di step. Per steps=50_000 e frac=0.8, il decay inizia al passaggio 40_000.
Questo valore viene passato nel `trainer_config` alla libreria `dictionary_learning`.

Il decay del learning rate e' importante per la convergenza finale: nelle ultime
fasi del training, un LR piu' basso permette di "raffinare" i pesi senza
oscillare intorno al minimo.

### 7.4 Configurazione del trainer

```python
    trainer_config = {
        "trainer": TopKTrainer,
        "activation_dim": self.config["activation_dim"],
        "dict_size": self.config["dict_size"],
        "k": self.config["k"],
        "steps": steps,
        "layer": self.config["layer"],
        "lm_name": self.config["lm_name"],
        "lr": lr,
        "warmup_steps": self.config["warmup_steps"],
        "decay_start": decay_start,
        "seed": seed,
        "device": device,
    }
```

**Perche:**

`trainSAE` accetta una lista di trainer configs (supporta training parallelo di
piu' SAE). Noi ne passiamo uno solo. Il `TopKTrainer` implementa la variante
Top-K dell'autoencoder sparse dove la sparsita' viene imposta selezionando solo
le k attivazioni piu' alte, anziche' usare un termine di penalita' L1 nel loss.

Novita' rispetto alla versione precedente:
- `decay_start` viene calcolato e passato esplicitamente (prima non era presente).
- `seed` viene incluso nel trainer_config (prima non veniva passato alla
  libreria).

### 7.5 Chiamata a trainSAE

```python
    trainSAE(
        data=batch_generator(),
        trainer_configs=[trainer_config],
        steps=steps,
        save_dir=str(model_dir),
        log_steps=self.config.get("log_steps", 1000),
        device=device,
        autocast_dtype=torch.bfloat16,
        normalize_activations=True,
        verbose=True,
    )
```

**Perche:**

- `data=batch_generator()`: passa il generatore (lazy) non una lista concreta.
- `log_steps`: controlla la frequenza di logging durante il training. Prima non
  veniva passato e veniva usato il default della libreria. Con `log_steps=1000`
  e 50k step totali, otteniamo 50 log intermedi.
- `autocast_dtype=torch.bfloat16`: mixed precision training -- le forward/backward
  pass usano bfloat16 per accelerare su GPU moderne, mentre i parametri master
  restano in float32.
- `normalize_activations=True`: **nuovo** rispetto alla versione precedente.
  Normalizza le attivazioni in input prima della forward pass. Questo e'
  cruciale quando le embedding hanno scale diverse tra dataset, ed e'
  raccomandato dalla documentazione della libreria per migliorare la stabilita'
  del training.
- `save_dir`: la libreria salva il modello come `ae.pt` dentro questa directory.
- Dopo il training, il metodo chiama `self.load(model_dir)` per rendere il SAE
  immediatamente utilizzabile senza doverlo ricaricare manualmente.

### 7.6 Salvataggio del manifest

```python
    self._save_manifest(model_dir, seed, steps, batch_size, embeddings_path, embeddings)
```

**Perche:**

Dopo il training, salva un file `training_manifest.json` nella directory del
modello. Questo file contiene tutte le informazioni necessarie per riprodurre
esattamente lo stesso training (vedi sezione dedicata a `_save_manifest()`).

---

## 8. Metodo `load()`

```python
def load(self, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)

    ae_path = model_dir / "ae.pt"
    if not ae_path.exists():
        trainer_path = model_dir / "trainer_0" / "ae.pt"
        if trainer_path.exists():
            ae_path = trainer_path
        else:
            raise FileNotFoundError(
                f"Model not found at {model_dir / 'ae.pt'} or {trainer_path}"
            )
```

**Perche:**

Cambiamento importante rispetto alla versione precedente che usava
`AutoEncoderTopK.from_pretrained()`. Ora il caricamento e' manuale per
avere controllo completo su ogni fase.

### 8.1 Fallback `trainer_0/`

La libreria `dictionary_learning` potrebbe salvare il modello dentro una
sottodirectory `trainer_0/` (convenzione interna della libreria). Il codice
prima cerca `ae.pt` direttamente, poi in `trainer_0/ae.pt`. Questo rende
il caricamento robusto indipendentemente dalla versione della libreria usata
per il training.

Senza questo fallback, i modelli addestrati con versioni recenti della libreria
avrebbero un path diverso da quelli addestrati con versioni vecchie.

### 8.2 Caricamento sicuro

```python
    state_dict = torch.load(
        ae_path, map_location=self.config["device"], weights_only=True
    )
```

**Perche:**

- `weights_only=True` sovrascrive il default della libreria (che usa
  `weights_only=False`). Questo e' una scelta di sicurezza deliberata: impedisce
  il caricamento di oggetti arbitrari tramite pickle, che potrebbe essere un
  vettore di attacco se il file .pt e' stato manomesso.
- `map_location=self.config["device"]`: mappa i tensori direttamente sul device
  target, evitando un transfer CPU -> device successivo.

### 8.3 Estrazione delle dimensioni e validazione del k

```python
    dict_size, activation_dim = state_dict["encoder.weight"].shape
    k = self.config["k"]

    if "k" in state_dict and k != state_dict["k"].item():
        raise ValueError(
            f"Config k={k} != saved k={state_dict['k'].item()}"
        )
```

**Perche:**

Le dimensioni del modello vengono lette direttamente dal `state_dict` (non
dalla config). Questo e' piu' robusto: se il file contiene un modello con
dimensioni diverse da quelle attese, lo rileva immediatamente.

Inoltre, se il `state_dict` contiene un campo `"k"` esplicito (le versioni
recenti della libreria lo salvano), verifica che il k nella config corrisponda.
Un mismatch tra k usato per l'addestramento e k usato per l'inferenza
produrrebbe risultati completamente errati senza dare errori evidenti.

### 8.4 Costruzione e caricamento del modello

```python
    self._ae = AutoEncoderTopK(activation_dim, dict_size, k)
    self._ae.load_state_dict(state_dict)
    self._ae = self._ae.float()  # Ensure float32 for consistent inference
    self._ae.eval()
    self._ae.to(self.config["device"])
```

**Perche:**

- `AutoEncoderTopK(activation_dim, dict_size, k)`: costruisce l'architettura
  con le dimensioni lette dal `state_dict`. Non usa le dimensioni dalla config
  per la costruzione, garantendo che architettura e pesi siano compatibili.
- `load_state_dict(state_dict)`: carica i pesi pre-addestrati.
- `.float()` (nuovo): forza il modello in **float32**. Il training usa bfloat16,
  ma per l'inferenza vogliamo la massima precisione. Senza questo, i pesi
  rimarrebbero in bfloat16 se salvati in quel formato, causando piccole
  differenze numeriche nelle attivazioni.
- `.eval()`: mette il modello in modalita' inferenza (disattiva dropout,
  batchnorm in modalita' running stats). Anche se questo SAE non ha dropout,
  e' best practice.
- `.to(device)`: sposta il modello sul device target.

### 8.5 Validazione config vs modello

```python
    if activation_dim != self.config["activation_dim"]:
        raise ValueError(
            f"Config activation_dim={self.config['activation_dim']} != "
            f"model activation_dim={activation_dim}"
        )
    if dict_size != self.config["dict_size"]:
        raise ValueError(
            f"Config dict_size={self.config['dict_size']} != "
            f"model dict_size={dict_size}"
        )
```

**Perche:**

Dopo il caricamento, verifica che le dimensioni del modello corrispondano
alla config. Questo previene errori silenziosi: se l'utente carica un modello
4096-dim con una config che dice 512-dim, i tensori avrebbero shape
incompatibili e l'errore apparirebbe solo molto piu' avanti nella pipeline
(come un crash criptico in una matmul).

Le ValueError forniscono il valore atteso e il valore trovato, facilitando
il debugging.

---

## 9. Metodo `encode()`

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
  `self._device` e' una property (vedi sezione 17) che restituisce
  `self.config["device"]`.
- Il risultato e' un tensor (B, 4096) con esattamente k=32 valori non-zero per
  riga. I valori non-zero rappresentano le attivazioni dei concetti piu'
  rilevanti.

L'API e' invariata rispetto alla versione precedente; l'unica differenza
interna e' l'uso della property `_device` per centralizzare l'accesso al
device.

---

## 10. Metodo `encode_topk()`

```python
def encode_topk(
    self, embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
- `_` ignora il quarto valore di ritorno (dettaglio implementativo della
  libreria -- pre-topk ReLU activations).

I tre tensori restituiti sono:
- `encoded` (B, 4096): la rappresentazione sparse completa (stessa di `encode()`)
- `values` (B, k): i valori delle k attivazioni massime
- `indices` (B, k): gli indici delle feature corrispondenti

---

## 11. Metodo `decode()`

```python
def decode(self, sparse: torch.Tensor) -> torch.Tensor:
    self._check_loaded()
    with torch.no_grad():
        return self._ae.decode(sparse.to(self._device))
```

**Perche:**

La decodifica moltiplica la rappresentazione sparse (4096-dim) per la matrice
decoder per riproiettarla nello spazio embedding (512-dim). E' l'operazione
inversa di encode: `x_hat = W_dec @ sparse + bias`.

API invariata rispetto alla versione precedente.

---

## 12. Metodo `reconstruct()`

```python
def reconstruct(self, embeddings: torch.Tensor) -> torch.Tensor:
    self._check_loaded()
    with torch.no_grad():
        return self._ae(embeddings.to(self._device))
```

**Perche:**

Shortcut per encode + decode in un singolo forward pass. `self._ae(x)` chiama
il metodo `forward()` del modello che internamente fa encode -> decode.
Usato per calcolare l'errore di ricostruzione (MSE e cosine similarity).

---

## 13. Metodo `get_decoder_weights()`

```python
def get_decoder_weights(self) -> torch.Tensor:
    self._check_loaded()
    return self._ae.decoder.weight.data.T.clone()
```

**Perche:**

La matrice decoder W_dec ha forma (512, 4096) in PyTorch (convenzione: out_features
x in_features per nn.Linear). La trasponiamo a (4096, 512) cosi' ogni riga
rappresenta un "concetto" -- un vettore 512-dim nello spazio embedding.

- `.data`: accede ai dati raw senza autograd.
- `.T`: traspone.
- `.clone()`: crea una copia indipendente per evitare che modifiche esterne
  corrompano i pesi del modello.

Ogni riga di questa matrice e' la "direzione" del concetto nello spazio
embedding. **Nota bene**: le righe **non** sono normalizzate. Per calcolare la
cosine similarity (come fa `name_concepts()`), e' necessario normalizzarle
prima con `F.normalize()`.

---

## 14. Metodo `get_top_concepts()`

```python
def get_top_concepts(
    self, embeddings: torch.Tensor, n: int = 5
) -> list[list[tuple[int, float]]]:
    self._check_loaded()
    with torch.no_grad():
        sparse = self._ae.encode(embeddings.to(self._device))

    # Vectorized topk (much faster than row-by-row Python loop)
    topk = sparse.topk(n, dim=1)  # values [B, n], indices [B, n]
    results = []
    for i in range(sparse.shape[0]):
        concepts = [
            (idx.item(), val.item())
            for idx, val in zip(topk.indices[i], topk.values[i])
        ]
        results.append(concepts)
    return results
```

**Perche:**

Per ogni campione, identifica i top-n concetti con attivazione piu' alta.
Restituisce una lista di tuple (feature_id, activation_value) ordinate per
attivazione decrescente. Usato nella pipeline di spiegazione per selezionare
i concetti dominanti di un'immagine.

**Cambiamento rispetto alla versione precedente**: la versione vecchia faceva
il `topk` riga per riga in un loop Python:

```python
# Vecchia versione (lenta)
for row in sparse:
    topk = row.topk(n)
    ...
```

La nuova versione esegue `sparse.topk(n, dim=1)` **sulla matrice intera**,
sfruttando la vettorizzazione di PyTorch. Il risultato e' identico, ma
l'operazione e' significativamente piu' veloce perche':
- Un'unica chiamata C++/CUDA invece di B chiamate
- Migliore sfruttamento della parallelizzazione SIMD/GPU
- Meno overhead del bridge Python <-> C++

Il loop Python rimane solo per la conversione dei tensori in liste di tuple
per l'output, che non puo' essere vettorizzata (il formato di ritorno richiede
oggetti Python).

---

## 15. Metodo `name_concepts()`

```python
def name_concepts(
    self,
    vocab_embeddings: torch.Tensor,
    vocab_labels: list[str],
    top_n: int = 3,
) -> dict[int, dict]:
```

### 15.1 Validazione degli input

```python
    self._check_loaded()

    if vocab_embeddings.dim() != 2:
        raise ValueError(
            f"vocab_embeddings must be 2D, got {vocab_embeddings.dim()}D"
        )
    if vocab_embeddings.shape[1] != self.config["activation_dim"]:
        raise ValueError(
            f"vocab_embeddings dim-1 ({vocab_embeddings.shape[1]}) != "
            f"activation_dim ({self.config['activation_dim']})"
        )
    if len(vocab_labels) != vocab_embeddings.shape[0]:
        raise ValueError(
            f"vocab_labels length ({len(vocab_labels)}) != "
            f"vocab_embeddings rows ({vocab_embeddings.shape[0]})"
        )
```

**Perche:**

**Nuovo rispetto alla versione precedente**: la vecchia versione non validava
gli input, il che poteva portare a errori criptici piu' avanti (es. una
matmul con shape incompatibile che crashava con un messaggio poco chiaro).

Le tre validazioni verificano che:
1. `vocab_embeddings` sia 2D (B, dim) -- non un tensor 1D o 3D.
2. La seconda dimensione sia `activation_dim` (512) -- altrimenti la
   cosine similarity con i concetti non ha senso.
3. Il numero di labels corrisponda al numero di embedding -- altrimenti
   l'indicizzazione `vocab_labels[idx]` potrebbe causare un IndexError o
   assegnare label sbagliate ai concetti.

Queste validazioni falliscono presto (fail-fast), con messaggi di errore
che spiegano esattamente cosa e' sbagliato.

### 15.2 Calcolo della cosine similarity

```python
    W_dec = self.get_decoder_weights()  # (dict_size, 512)

    # Normalize -> dot product equals cosine similarity
    W_norm = F.normalize(W_dec, dim=1)
    V_norm = F.normalize(vocab_embeddings.to(self._device), dim=1)

    similarities = W_norm @ V_norm.T  # (dict_size, V)
```

**Perche:**

Assegna nomi medici ai 4096 concetti appresi dal SAE:

1. **Ottiene W_dec** (4096, 512): ogni riga e' un vettore-concetto.
2. **Normalizza** entrambe le matrici (concetti e vocabolario) a norma L2
   unitaria. Dopo la normalizzazione, il prodotto scalare equivale alla
   cosine similarity. Questo evita di chiamare `F.cosine_similarity`
   (che richiederebbe un broadcasting costoso).
3. **Matmul** `W_norm @ V_norm.T` produce una matrice (4096, V) dove ogni
   cella [i,j] e' la similarita' tra il concetto i e il termine j del
   vocabolario.

### 15.3 Selezione dei candidati

```python
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

Per ogni concetto, seleziona i top_n termini piu' simili come candidati per
il naming. Il risultato e' un dizionario dove ogni feature ha:
- `name`: il termine piu' simile (candidato #1)
- `score`: il cosine similarity con quel termine
- `candidates`: lista dei top_n candidati con i rispettivi score

La logica e' invariata rispetto alla versione precedente; l'unica aggiunta
e' la validazione degli input.

---

## 16. Metodo `compute_reconstruction_mse()`

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

`F.mse_loss` calcola la media di (x - x_hat)^2 su tutte le dimensioni e
campioni. Invariato rispetto alla versione precedente.

---

## 17. Metodo `compute_cosine_reconstruction()` (NUOVO)

```python
def compute_cosine_reconstruction(self, embeddings: torch.Tensor) -> float:
    self._check_loaded()
    with torch.no_grad():
        x = embeddings.to(self._device)
        x_hat = self._ae(x)
        return F.cosine_similarity(x_hat, x, dim=-1).mean().item()
```

**Perche:**

Questo metodo e' **completamente nuovo** rispetto alla versione precedente.
Misura la qualita' della ricostruzione usando la cosine similarity anziche'
il MSE.

La cosine similarity cattura una metrica diversa rispetto all'MSE:
- **MSE** e' sensibile alla scala: un errore grande su una dimensione
  con valore alto pesa molto.
- **Cosine similarity** e' invariante alla scala: misura solo la "direzione"
  dell'errore, non la sua grandezza.

Per gli embedding (che spesso sono normalizzati o quasi-normalizzati), la
cosine similarity e' spesso piu' informativa del MSE. Valori vicini a 1.0
indicano una ricostruzione eccellente. Valori inferiori a 0.9 suggeriscono
che il SAE sta perdendo informazione direzionale importante.

- `dim=-1`: calcola la cosine similarity lungo l'ultima dimensione (le 512
  feature dell'embedding), per ogni campione.
- `.mean()`: media sulle cosine similarity di tutti i campioni.

---

## 18. Metodo `compute_sparsity_metrics()`

```python
def compute_sparsity_metrics(self, embeddings: torch.Tensor) -> dict:
    self._check_loaded()
    with torch.no_grad():
        sparse = self._ae.encode(embeddings.to(self._device))

    l0 = (sparse != 0).float().sum(dim=1)

    active_per_feature = (sparse != 0).float().sum(dim=0)
    n_total = sparse.shape[1]
    n_dead = (active_per_feature == 0).sum().item()
    dead_pct = n_dead / n_total * 100
    utilization_pct = (n_total - n_dead) / n_total * 100

    freq = active_per_feature / (active_per_feature.sum() + 1e-8)
    freq = freq[freq > 0]
    entropy = -(freq * freq.log()).sum().item()

    return {
        "l0_mean": l0.mean().item(),
        "l0_std": l0.std().item(),
        "dead_features_pct": dead_pct,
        "activation_entropy": entropy,
        "dict_utilization_pct": utilization_pct,
    }
```

**Perche:**

Riscrittura significativa rispetto alla versione precedente. La vecchia versione
restituiva `l0_mean`, `l0_std`, `hoyer_mean` e `dead_features_pct`. La nuova
versione sostituisce `hoyer_mean` con due metriche piu' informative.

### L0 (pseudo-norma)
Conta il numero di attivazioni non-zero per campione. Con k=32, ci aspettiamo
L0 ~ 32.0 (con deviazione standard ~0, dato che il Top-K e' deterministico).
Viene mantenuto come **sanity check**: se L0 diverge significativamente da k,
c'e' un problema nel Top-K enforcement.

### Dead features e dict utilization (rimaneggiati)
La logica e' la stessa, ma ora il risultato include sia `dead_features_pct`
(Feature mai attivate) sia `dict_utilization_pct` (il complemento):
- `dead_features_pct`: percentuale di feature nel dizionario che non si attivano
  mai. Una percentuale alta indica capacita' sprecata.
- `dict_utilization_pct`: percentuale di feature che si attivano almeno una
  volta. E' il complemento: `100 - dead_features_pct`.

`utilization_pct` e' piu' intuitivo da leggere: "il 85% del dizionario viene
usato" e' piu' chiaro di "il 15% delle feature sono morte".

### Activation entropy (NUOVA)
```python
    freq = active_per_feature / (active_per_feature.sum() + 1e-8)
    freq = freq[freq > 0]
    entropy = -(freq * freq.log()).sum().item()
```

Calcola l'**entropia di Shannon** della distribuzione di frequenza di
attivazione delle feature.

**Perche' questa metrica e' importante**: un SAE "sano" dovrebbe usare tutte le
feature in modo relativamente uniforme. Se poche feature catturano la maggior
parte delle attivazioni, il dizionario non e' sfruttato efficacemente.

Interpretazione dell'entropia:
- **Entropia alta** (vicina a log(dict_size)): le feature sono usate in modo
  uniforme -- buona diversita' concettuale.
- **Entropia bassa**: poche feature dominano -- possibile "collasso" del
  dizionario dove la maggior parte delle feature sono poco usate.

Il filtro `freq[freq > 0]` esclude le dead features dal calcolo dell'entropia
(perche' `0 * log(0)` e' indefinito). L'`1e-8` nel denominatore previene
la divisione per zero.

### Perche' la Hoyer sparsity e' stata rimossa
La vecchia metrica Hoyer sparsity, formula `(sqrt(n) - L1/L2) / (sqrt(n) - 1)`,
e' **tautologica** per un Top-K SAE: il Top-K garantisce una sparsita'
strutturale identica per ogni campione (esattamente k elementi non-zero),
quindi la Hoyer sparsity e' essenzialmente una costante. Non porta informazione
aggiuntiva rispetto al semplice conteggio L0.

L'entropia di attivazione e la dict utilization sono metriche molto piu'
informative perche' misurano **come** il dizionario viene usato, non solo
**quanto** e' sparse la rappresentazione.

---

## 19. Metodo statico `compute_stability()`

```python
@staticmethod
def compute_stability(
    model_dirs: list[str | Path],
    embeddings: torch.Tensor,
    config: Optional[dict] = None,
    n: Optional[int] = None,
) -> dict:
```

**Perche:**

Misurare la robustezza dei concetti: se alleniamo il SAE con seed diversi,
i concetti appresi sono gli stessi? Usiamo la Jaccard similarity per
confrontare gli insiemi di feature attivate.

`@staticmethod` perche' il metodo opera su multipli modelli, non su `self`.

### 19.1 Supporto per SAEConfig e parametro `n`

```python
    if config is not None and not isinstance(config, dict):
        config = _extract_sae_config(config)
    effective_config = {**_DEFAULTS, **(config or {})}
    if n is None:
        n = effective_config["k"]
```

**Perche:**

Cambiamenti rispetto alla versione precedente:

1. **Supporto SAEConfig**: come il costruttore, se viene passata una dataclass
   `SAEConfig` invece di un dict, viene convertita tramite `_extract_sae_config()`.
   Questo permette di chiamare `SAEManager.compute_stability(dirs, emb, config.sae)`.

2. **Parametro `n` (nuovo)**: prima il confronto Jaccard usava sempre k feature.
   Ora e' possibile confrontare solo le top-n feature (con n < k). Questo e'
   utile perche' le feature con attivazione piu' alta sono le piu' affidabili;
   confrontarle separatamente dalle feature a bassa attivazione puo' rivelare
   se il SAE ha imparato concetti robusti nella "testa" della distribuzione.

3. **`n` default dalla config**: se non specificato, usa `k` (comportamento
   compatibile con la versione precedente).

### 19.2 Caricamento sequenziale con release della GPU

```python
    active_sets: list[list[set[int]]] = []
    for d in model_dirs:
        mgr = SAEManager(effective_config)
        mgr.load(d)

        sample_sets: list[set[int]] = []
        chunk_size = 512
        for start in range(0, embeddings.shape[0], chunk_size):
            chunk = embeddings[start : start + chunk_size]
            _, _, indices = mgr.encode_topk(chunk)
            if n < indices.shape[1]:
                indices = indices[:, :n]
            for row in indices.cpu():
                sample_sets.append(set(row.tolist()))
        active_sets.append(sample_sets)

        # Free GPU memory between models
        del mgr._ae
        mgr._ae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Perche:**

Cambiamento critico rispetto alla versione precedente. La vecchia versione
caricava **tutti i modelli simultaneamente** in una lista `managers`, poi
iterava per l'encoding. Questo causava **GPU OOM** (Out of Memory) quando
si confrontavano 5 o piu' SAE contemporaneamente, perche' ogni modello
occupa ~50-100 MB di VRAM.

La nuova versione:
1. **Carica un modello alla volta**: il loop `for d in model_dirs` carica,
   usa, e poi rilascia ogni modello prima di caricare il successivo.
2. **`del mgr._ae; mgr._ae = None`**: rimuove il riferimento al modello,
   permettendo al garbage collector di liberare la memoria.
3. **`torch.cuda.empty_cache()`**: forza PyTorch a rilasciare la VRAM
   non utilizzata verso il driver CUDA. Senza questa chiamata, la VRAM
   liberata dal garbage collector rimarrebbe "riservata" da PyTorch.
4. **Chunking (nuovo)**: `chunk_size=512` processa le embedding in chunk
   di 512 campioni alla volta. Questo previene OOM anche con dataset grandi
   dove l'encoding di tutto il batch alla volta richiederebbe troppo VRAM.

L'operazione `indices[:, :n]` tronca le feature ai top-n se n < k.
Per esempio, se k=32 e n=10, confronta solo le 10 feature con attivazione
massima. La Jaccard su meno feature e' tipicamente piu' alta (le feature
piu' forti tendono ad essere piu' stabili).

### 19.3 Matrice Jaccard

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
                union = len(a | b)
                if union > 0:
                    jaccards.append(len(a & b) / union)
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
superiore e specchiamo per efficienza (O(n^2/2) invece di O(n^2)).

Cambiamento rispetto alla versione precedente: il controllo `union > 0` e'
respi esplicito anziche' `len(a | b) > 0`. E' semanticamente equivalente ma
piu' leggibile perche' `union` e' gia' stato calcolato.

### 19.4 Statistiche riassuntive

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
della stabilita' complessiva. Invariato rispetto alla versione precedente.

---

## 20. Metodo `_save_manifest()`

```python
def _save_manifest(
    self,
    model_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    embeddings_path: str | Path,
    embeddings: torch.Tensor,
) -> None:
```

**Perche:**

Questo metodo e' **completamente nuovo** rispetto alla versione precedente.
Salva un file `training_manifest.json` nella directory del modello, contenente
tutte le informazioni necessarie per riprodurre esattamente lo stesso training.

### 20.1 Calcolo del learning rate effettivo

```python
    lr_used = self.config.get("lr")
    if lr_used is None:
        scale = self.config["dict_size"] / (2**14)
        lr_used = 2e-4 / scale**0.5
```

**Perche:**

Quando `lr=None` (auto-scaling), il learning rate effettivo viene calcolato
dalla libreria `dictionary_learning` usando la formula
`2e-4 / sqrt(dict_size / 16384)`. Questo codice replica la stessa formula
per registrare il valore effettivo nel manifest, cosi' da sapere esattamente
quale LR e' stato usato senza dover ispezionare i log di training.

Per `dict_size=4096`:
- `scale = 4096 / 16384 = 0.25`
- `lr = 2e-4 / sqrt(0.25) = 2e-4 / 0.5 = 4e-4`

### 20.2 Contenuto del manifest

```python
    manifest = {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr_auto_scaled": lr_used,
        "activation_dim": self.config["activation_dim"],
        "dict_size": self.config["dict_size"],
        "k": self.config["k"],
        "warmup_steps": self.config["warmup_steps"],
        "decay_start_frac": self.config["decay_start_frac"],
        "log_steps": self.config.get("log_steps", 1000),
        "autocast_dtype": "bfloat16",
        "normalize_activations": True,
        "device": self.config["device"],
        "embeddings_path": str(embeddings_path),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_hash": hashlib.sha256(
            embeddings[: min(100, len(embeddings))].cpu().numpy().tobytes()
        ).hexdigest()[:16],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
```

**Perche:**

Il manifest contiene diverse categorie di informazioni:

**Iperparametri di training**: seed, steps, batch_size, lr (effettivo), k,
warmup_steps, decay_start_frac, log_steps. Permettono di riprodurre
esattamente lo stesso addestramento.

**Meta-architettura**: activation_dim, dict_size, autocast_dtype,
normalize_activations, device. Documentano l'ambiente di training.

**Provenienza dei dati**: embeddings_path e embeddings_shape. Permettono
di verificare che lo stesso dataset venga usato nella riproduzione.

**Hash dei dati (nuovo e importante)**:
```python
"embeddings_hash": hashlib.sha256(
    embeddings[: min(100, len(embeddings))].cpu().numpy().tobytes()
).hexdigest()[:16]
```
Calcola l'hash SHA-256 dei primi 100 embedding (troncato ai primi 16 caratteri
esadecimali per leggibilita'). Questo serve a verificare che i dati di
training non siano cambiati:
- Se l'hash corrisponde, i primi 100 embedding sono identici.
- Se l'hash differisce, i dati sono cambiati e il modello non e'
  riproducibile con lo stesso seed.

Si usano solo i primi 100 perche' calcolare l'hash dell'intero dataset
(7400 x 512 x 4 bytes = ~14 MB) sarebbe troppo lento per ogni training.

**Ambiente**: torch_version, cuda_available (e gpu_name se disponibile).
Documentano l'ambiente hardware/software per diagnosticare differenze
di risultato tra macchine diverse.

### 20.3 Salvataggio

```python
    if torch.cuda.is_available():
        manifest["gpu_name"] = torch.cuda.get_device_name(0)

    with open(model_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
```

**Perche:**

- Il nome della GPU viene aggiunto solo se CUDA e' disponibile (per evitare
  chiavi vuote su sistemi CPU-only).
- `indent=2`: formattazione leggibile per ispezione manuale.

---

## 21. Metodi privati

```python
def _check_loaded(self):
    if not self.is_loaded:
        raise RuntimeError(
            "SAE not loaded. Call .load(model_dir) or .train() first."
        )

@property
def _device(self) -> str:
    return self.config["device"]
```

**Perche:**

- `_check_loaded()`: guard chiamata all'inizio di ogni metodo che richiede
  il modello. Lancia `RuntimeError` con un messaggio che spiega come risolvere.
- `_device`: property per accedere al device senza scrivere
  `self.config["device"]` ogni volta. Convenienza interna. Usato da
  `encode()`, `decode()`, `reconstruct()`, `compute_reconstruction_mse()`,
  `compute_cosine_reconstruction()` e `compute_sparsity_metrics()`.

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

---

## Diagramma del flusso del ciclo di vita

```
                        SAEManager
                            |
         +------------------+------------------+
         |                                     |
    [train()]                            [load()]
    - _set_global_seed()                - Fallback trainer_0/
    - DataLoader + generator             - weights_only=True
    - trainSAE(..., normalize=True)     - float32 forced
    - _save_manifest()                  - Config validation
    - self.load()                       - .eval()
         |                                     |
         +------------------+------------------+
                            |
                    [Modello pronto]
                            |
         +------------------+------------------+
         |          |           |              |
    encode()  decode()   get_top_concepts()  name_concepts()
    encode_topk()  reconstruct()
                            |
                    [Metriche]
                            |
    compute_reconstruction_mse()  compute_cosine_reconstruction()
    compute_sparsity_metrics()     compute_stability()
```

---

## Riepilogo delle differenze rispetto alla versione precedente

| Aspetto | Vecchia versione | Nuova versione |
|---------|-----------------|----------------|
| Config defaults | `DEFAULT_CONFIG` (public) | `_DEFAULTS` (private, synced con config.py) |
| Supporto SAEConfig | No | Si, tramite `_extract_sae_config()` |
| Propagazione seed | Solo `torch.manual_seed` | Completa: random, numpy, torch, cuda, cudnn |
| DataLoader generator | No (non deterministico) | Si, `torch.Generator().manual_seed(seed)` |
| `log_steps` | Non passato a trainSAE | Passato esplicitamente |
| `decay_start` | Non passato | Calcolato da `decay_start_frac` |
| `normalize_activations` | Non passato a trainSAE | `True` |
| Manifest di training | Non presente | `training_manifest.json` |
| Load fallback | No, crash su FileNotFoundError | Prova `trainer_0/` subdirectory |
| Load sicuro | `from_pretrained()` (weights_only=False) | `torch.load(..., weights_only=True)` |
| Float forzato | No | `.float()` per float32 |
| Validazione config vs model | No | Si, activation_dim e dict_size |
| `get_top_concepts()` | Loop Python riga per riga | Vectorized `topk(dim=1)` |
| `name_concepts()` validazione | Nessuna | Shape e lunghezza labels |
| `compute_sparsity_metrics()` | L0, Hoyer, dead% | L0, dead%, utilization%, entropy |
| `compute_cosine_reconstruction()` | Non presente | Nuovo metodo |
| `compute_stability()` OOM | Tutti i modelli in RAM | Uno alla volta + `empty_cache()` |
| `compute_stability()` chunking | Nessuno | `chunk_size=512` |
| `compute_stability()` param n | Fisso a k | Parametro opzionale (default k) |
| `compute_stability()` SAEConfig | Non supportato | Supportato via `_extract_sae_config()` |
