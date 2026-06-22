# Diagnosi dell'instabilita' cross-seed dei concetti SAE

Documento di analisi retrospettiva: *perche'* il run baseline ha prodotto
concetti non robusti (Jaccard cross-seed 0.0038, ~44% dead, naming debole) e
quali scelte progettuali lo hanno determinato. Non e' un elenco di fix, ma la
lettura causale che giustifica il finding come **risultato atteso dalla teoria**,
non come bug. Da leggersi insieme al
`notebooks/autoencoder/baseline/REPORT.md` e al programma di ablation in
`notebooks/autoencoder/ablation/`.

## I numeri del run baseline (seed 0,42,123,456,789)

- Config SAE: Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4),
  `batch_size=256`, embedding BiomedCLIP **512-d** su IU X-Ray (5976 train / 1494 test).
- Ricostruzione: cosine **0.988**, varianza spiegata **99.3%**, L0 = 32.0 esatto.
- Dead features (activation-based): **~44%**.
- Stabilita' cross-seed: mean Jaccard **0.0038** (matrice 5x5 off-diagonal ~0.003-0.010).
- Concept naming: score mean **0.117**, max **0.291** (vocab RadLex 310 termini).
- Ablation a0 (consensus sulle direzioni del decoder, index-agnostic):
  consensus@4 = **0.0**, Hungarian direction-Jaccard = **0.0**, shuffle-null p = **1.0**
  (solo 3 cluster multi-membro a tau=0.80 su 20480 righe).

## Reframe: il SAE *funziona*. Il problema e' la non-univocita' della fattorizzazione

I numeri di ricostruzione (99.3% di varianza spiegata con soli 32 feature attivi)
dicono che il SAE impara a decomporre lo spazio in modo eccellente. La domanda
giusta quindi non e' "perche' il SAE non impara" ma **"perche' la fattorizzazione
sparse non e' unica/riproducibile tra seed?"**. Questa e' una domanda di teoria,
non di ingegneria.

I risultati "non buoni" sono in realta' **due fenomeni distinti** con cause diverse:

| Fenomeno | Numeri | Causa |
|---|---|---|
| Instabilita' cross-seed (Jaccard + direzioni) | 0.0038, a0 = 0.0 | Non-identificabilita' della sparse factorization in regime data-starved |
| Dead ~44% + naming debole (max 0.29) | baseline | Dizionario oversized + vocab mal allineato |

Vanno discussi separatamente: il primo e' strutturale, il secondo e' di capacita'/allineamento.

## Causa 1 (principale): regime data-starved su fattorizzazione non-identificabile

Due vincoli strutturali si combinano.

### 1a. Dataset minuscolo

7470 immagini totali (5976 train) e' *tiny* per gli standard SAE; la letteratura
di riferimento lavora su milioni-miliardi di attivazioni. Con 5976 punti, il
landscape di ottimizzazione contiene **una molteplicita' di ottimi equivalenti in
ricostruzione ma con dictionary completamente diversi**. Il prior di sparsita'
(Top-K) non e' abbastanza forte, con cosi' pochi dati, da selezionarne uno in
modo univoco.

### 1b. Lavoriamo su embedding CLIP *proiettati*, non su hidden states grezzi

L'estrazione usa `model.get_image_features()` che restituisce lo spazio di
*proiezione* testo-immagine (512-d): ottimizzato per essere liscio, allineato,
regolarizzato. E' **lo spazio sbagliato** per un SAE. La struttura sparse
interpretabile si trova normalmente nei hidden states pre-projection
(MLP/attention dell'encoder). Su uno spazio gia' compresso e smooth c'e' poca
struttura sparsa "reale" da scoprire -> piu' gradi di liberta' per divergere.

### 1c. Il colpo di teatro: e' quasi predetto dalla teoria

La fattorizzazione `X ~= D * A` con dizionario sparse e' nota per essere
**non-identificabile** senza dati sufficienti (permutazioni di colonne e
rotazioni nel null space sono tutte soluzioni valide in ricostruzione).
L'instabilita' cross-seed (Jaccard~0 tra 5 seed indipendenti) e' **esattamente
il sintomo** di non-identificabilita'. Non e' un fallimento ingegneristico ma la
conferma empirica di un limite teorico in questo regime di scala.

L'ablation a0 e' stata decisive qui: il Jaccard sugli indici sovrastima la
non-riproducibilita' perche' ignora le permutazioni (due SAE equivalenti possono
avere indici riordinati). Poteva essere la via di scapo. Ma a0, clusterizzando
le *direzioni* del decoder (index-agnostic), ha mostrato che **nemmeno le
direzioni coincidono** -> l'instabilita' e' reale a entrambi i livelli, non un
artefatto del protocollo. Corroborato dal null analitico (a2: baseline 0.0038 ~
random floor `k/(2D)` = 0.00398, signal-to-null ratio ~0.95) ed empirico
(a6: Random@4096 Jaccard 0.00372 ~ baseline).

## Causa 2: dead features e naming debole (capacita' + vocab)

Fenomeno piu' "ingegneristico" e separabile dall'instabilita'.

**Dizionario oversized.** `dict_size=4096` su embedding 512-d x 7470 sample ->
expansion ratio 8x su uno spazio gia' piccolo -> enorme ridondanza ->
~1800 feature non hanno nulla da rappresentare -> muoiono (44%). Gia' discusso
in `SAE_TRAINING_SMALL_DATASET.md`.

**Vocab RadLex 310 + naming greedy per coseno.** Due debolezze combinate:
310 termini clinici sono pochi per coprire la geometria di 4096 feature
(mismatch ~13x); RadLex e' un'ontologia formale clinica, non allineata per
costruzione allo spazio CLIP; il naming *greedy* sceglie "il migliore tra match
poveri" -> coseni bassi anche quando la feature cattura qualcosa di reale
(alternativa: SPLiCE, ottimizzazione sparsa sui pesi del decoder). Il naming
debole e' anche in parte *conseguenza* dell'instabilita': feature non stabili
faticano ad ancorarsi solidamente a un vocabolario.

## Causa 3: scelte di training amplificatrici

- **lr ~4e-4** su 7470 sample e' alto; `config` raccomanda 5e-5 per dataset
  piccoli. LR alto -> converge in optima locali *diversi* per seed -> piu'
  divergenza cross-seed.
- **50000 step = ~2100 epoche**. Troppe: ogni seed si "inchioda" alla sua
  soluzione specifica (overfit al training set), cristallizzando la divergenza
  invece di regolarizzarla. Fewer steps + early stopping su validation
  reconstruction sarebbero stati piu' sani.

## Onesta' bilanciata: le scelte *giuste*

Non tutto e' da buttare. Alcune decisioni sono state corrette:

- **TopK invece di ReLU+L1**: scelta migliore per la stabilita' (nessun
  shrinkage lambda da tunare). Non e' la causa dell'instabilita'.
- **5 seed**: averne 5 invece di 1 e' cio' che ha *permesso* di scoprire
  l'instabilita'. Con un seed solo si sarebbe stati falsamente rassicurati.
  Diagnostico, non un errore.
- **Valutazione null-calibrata** (a0/a2/a6): il confronto contro pavimenti
  analitici+empirici e' cio' che rende il finding *credibile* invece di
  aneddotico.
- **MPS ~ CUDA**: riproducibilita' cross-device confermata, quindi l'instabilita'
  non e' rumore hardware.

## Sintesi in una frase

> Si sta applicando una metodologia SAE progettata per il regime data-rich su
> attivazioni grezze a un regime data-starved su embedding CLIP proiettati e
> regolarizzati; l'instabilita' cross-seed osservata e' il sintomo atteso della
> non-identificabilita' della sparse factorization in queste condizioni,
> confermato (non artifatto) dall'analisi null-calibrata a livello di indice e
> di direzione.

## "Se dovessi rifarlo" -> ablation derivate

Questi sono i lever che attaccano le cause alla radice e **non sono coperti** dai
5 notebook gia' scritti (a0 consensus, a1 dict_size, a2 k_sweep, a3/a6 baselines,
a4 activation bakeoff). Ordinati per rapporto impatto/costo.

### Priorita' alta (attaccano la causa 1, la radice)

1. **Spazio di attivazione pre-projection.** Riestrarre gli embedding dai hidden
   states di `model.vision_model` *prima* della `visual_projection` (es.
   `last_hidden_state` pooled) invece di `get_image_features()`. Attacca
   direttamente la causa 1b (spazio sbagliato) e 1a (spazio piu' ricco di
   struttura sparse). **Costo alto**: nuova estrazione embedding (BiomedCLIP +
   GPU) + nuovo run SAE. E' l'ablation metodologicamente piu' interessante perche'
   testa se l'instabilita' e' del *setting* (embedding proiettati) o del *metodo*
   (SAE in se').
2. **Augmentation pre-embedding** (flip orizzontale, crop 90-95%, rotazione
   +/-5 gradi; evitare color jitter/cutout). 3-5x -> 22k-37k embedding.
   Attacca la causa 1a (dataset tiny). Gia' suggerito in
   `SAE_TRAINING_SMALL_DATASET.md`. **Costo alto**: nuova estrazione.
3. **Shared init / model soup cross-seed.** Inizializzare i 5 seed dallo stesso
   init (o pesarli in un model soup) per forzare feature comuni. Testa
   l'ipotesi "l'instabilita' nasce dall'init randomico". **Costo basso**:
   retrain con init condiviso, riusa la meccanica gia' scritta.

### Priorita' media (diagnostici, basso costo)

4. **Step-sweep / early stopping.** Salvare checkpoint a 5k/10k/20k/50k step e
   misurare il Jaccard cross-seed a ciascuno. Testa se l'instabilita' emerge
   subito (causa strutturale) o cresce col training (overfit). **Costo basso**:
   salva checkpoint intermedi nel training gia' pianificato.

### Priorita' bassa (post-hoc, no retrain)

5. **SPLiCE naming** invece di greedy cosine: ottimizzazione sparsa dei pesi del
   decoder sul vocabolario. Non cambia il SAE, valuta se il naming debole
   (max 0.29) e' dovuto al *metodo* di naming o al vocabolario. **Costo basso**.
6. **Vocab piu' ampio/curato**: piu' termini o descrizioni cliniche generate da
   LLM, o naming tipo SPLiCE. **Costo basso** (post-hoc).

### Gia' considerate e droppate (non riproporre)

- **auxk / dead-resampling full ablation**: null-by-construction al budget di
   12k step (`TopKTrainer.dead_feature_threshold` hardcoded a 10M token;
   12k x 256 = 3M token). Risorto come probe interno ad a1 con threshold
   abbassato via `RevivalTopKTrainer`.
- **Faithfulness vs 14 patologie NIH**: IU X-Ray non ha le etichette NIH
   ChestX-ray14 (solo projection Frontal/Lateral) -> irrealizzabile.

## Riferimenti

- Olshausen & Field (1997) - sparse coding, regime dati-campioni/feature.
- Spielman, Wang, Wright (2012) - "Exact Recovery of Sparsely-Used Dictionaries":
  condizioni di identificabilita' del dictionary learning.
- Soltanolkotabi, Elhamifar, Candes (2013-2014) - robustness/identifiability
  dello structured sparsity.
- Bricken et al. (2023) "Towards Monosemanticity" - SAE su milioni di
  attivazioni (regime data-rich di riferimento).
- Bhalla, Srinivas, Hsieh (2023/2024) "Compositional Explanations" (SPLiCE) -
  naming via ottimizzazione sparsa sui pesi del decoder.
- Rajamanoharan et al. (2024) - BatchTopK / JumpReLU SAE (varianti in a4).
