# Strategia di Implementazione: Data Augmentation e SAE Tuning

Questo documento descrive il piano di implementazione per mitigare il problema delle *dead features* del nostro Sparse Autoencoder (SAE) addestrato sul dataset IU X-Ray. La strategia prevede l'abbassamento della dimensionalità del dizionario e l'introduzione di tecniche di Data Augmentation sicure per le radiografie toraciche, il tutto mantenendo la rigida modularità e pulizia del codice del progetto.

## 1. Modifiche alla Configurazione (`src/config.py`)

La configurazione centralizzata verrà espansa per governare i nuovi parametri.

- **`SAEConfig`**: 
  - Verrà modificato il valore di default di `dict_size` da `4096` a `2048` (espansione 4x).
- **Nuova classe `AugmentationConfig`**:
  - `enabled`: `bool` (default: `True`) per accendere/spegnere la pipeline di augmentation.
  - `num_augmentations`: `int` (default: `3`) numero di versioni alterate da generare per ogni campione originale.
  - `rotation_degrees`: `int` (default: `5`) range per la rotazione randomica (da -5° a +5°).
  - `crop_scale`: `tuple` (default: `(0.95, 1.0)`) fattore di scala per un ritaglio randomico leggero.

## 2. Architettura dell'Augmentation e Dataset Wrapper

La logica di trasformazione e la gestione del dataset verranno separate per garantire la massima estensibilità futura.

### Logica Visiva (`src/augmentation/`)
- **`src/augmentation/__init__.py`**: Esporrà le funzioni principali del modulo verso l'esterno.
- **`src/augmentation/transforms.py`**:
  - Conterrà la logica per la costruzione delle pipeline di trasformazione `torchvision`.
  - Funzione `get_safe_cxr_transforms(config: AugmentationConfig)`: Restituisce una pipeline configurata per applicare rotazioni e crop leggeri, escludendo operazioni distorsive come flip e color jitter.

### Dataset Wrapper Modulare (`xai_datasets/augmentation.py`)
- Creeremo il file `augmentation.py` direttamente nella cartella `xai_datasets`, dove risiedono le altre definizioni dei dataset.
- **Classe `AugmentedImageDataset`**:
  - Funzionerà come un **wrapper generico** per qualsiasi dataset PyTorch (implementerà il pattern Decorator/Wrapper).
  - Prenderà in input un'istanza di un dataset di base (es. `IUXrayImageDataset`) e le funzioni di trasformazione.
  - **Funzionamento dinamico**: Quando il DataLoader richiederà un indice, il wrapper calcolerà a quale immagine originale corrisponde, se deve restituire la versione non alterata o una sua augmentazione (applicando in tempo reale le trasformazioni dal modulo `src/augmentation`).
  - **Suffissi ID**: Il wrapper modificherà dinamicamente gli ID restituiti (aggiungendo suffissi come `_orig`, `_aug1`, ecc.) per propagare i nuovi identificativi nei file di sidecar.
  - **Estensibilità**: Se in futuro si vorrà usare un nuovo dataset, basterà passare la nuova istanza al posto di `IUXrayImageDataset` e la logica di augmentation continuerà a funzionare senza modifiche.

## 3. Modifica all'Estrazione degli Embedding (`src/extract_embeddings.py`)

L'integrazione di questi nuovi componenti avverrà nel punto di ingresso per i modelli visivi. Non salveremo le immagini augmentate su disco per non sprecare spazio di storage, ma le genereremo *on-the-fly*.

- **Intervento in `extract_embeddings.py`**:
  - Si istanzierà il dataset di base (`IUXrayImageDataset`).
  - Se l'augmentation è abilitata in configurazione, il dataset di base verrà avvolto con `AugmentedImageDataset`.
  - Il DataLoader ciclerà sul wrapper, producendo `1 + N` sample per ogni immagine originale.
  - L'output salvato nel file `visual_embeddings.pt` passerà automaticamente da ~7.4k a ~29.8k righe (assumendo N=3).
  - L'estrazione dei `text_embeddings` rimarrà totalmente inalterata.

## 4. Impatto Trasparente a Valle (`src/autoencoder/train_sae.py`)

Grazie all'architettura modulare, la pipeline a valle non richiederà refactoring strutturali per gestire i nuovi dati:
- Lo script `prepare_split()` in `train_sae.py` leggerà in automatico il nuovo `visual_embeddings.pt` e la sua lista di ID estesa.
- La partizione train/test (80/20) spartirà un training set di quasi 24k campioni, assicurando che le varianti augmentate arricchiscano lo spazio senza alterare il formato tensoriale atteso dal trainer SAE.
- Il calcolo del `modality_gap.pt` si baserà sul nuovo mega-centroide, più stabile.
