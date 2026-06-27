# Analisi delle Performance di Concept Naming

> **Stato (aggiornato 2026-06-22).** La tabella "Sintesi del Problema" qui sotto riporta i numeri **PRE-fix** del modality gap (max 0.322, mean 0.133). Con la correzione del gap (*Soluzione 1*, implementata e adottata nel run baseline) il naming sale a **mean 0.395 / max 0.547** (~3.4× medio) — vedi `notebooks/autoencoder/baseline/REPORT.md` §2.5. La letteratura considera 0.3–0.4 *normali e significativi* per SAE-su-CLIP (§"Cosa funziona davvero e perché" più sotto). Cross-check indipendente dall'**Ablation 05** (`notebooks/autoencoder/ablation/REPORT.md`): il naming RadLex (off-distribution) è talvolta rumoroso — una feature fedele a "implanted medical device" porta il nome RadLex "anterior segment of upper lobe". Il comportamento reale della feature è più informativo del suo nome.

## Sintesi del Problema

Il concept naming assegna un nome dal vocabolario (508 termini medici) a ciascuno dei 4096 nodi del SAE calcolando la **cosine similarity** tra le colonne del decoder (direzioni del SAE) e gli embeddings dei termini del vocabolario. I risultati sono pessimi:

| Metrica | Valore |
|---------|--------|
| Max score | 0.3220 |
| Mean score | 0.1331 |
| Min score | -0.0370 |
| Features con score > 0.2 | 11.4% |
| Features con score > 0.3 | 0.2% |

---

## Background: BiomedCLIP e lo Spazio Condiviso

### Come funziona BiomedCLIP

BiomedCLIP è un modello **contrastivo** (basato sull'architettura CLIP) addestrato su coppie immagine-testo biomediche. L'idea centrale è proiettare immagini e testi nello **stesso** spazio vettoriale a 512 dimensioni, in modo che coppie semanticamente correlate (ad esempio una radiografia che mostra cardiomegalia e il testo "cardiomegaly") finiscano vicine nello spazio, mentre coppie non correlate finiscano lontane.

In concreto:
- L'**image encoder** (ViT) trasforma un'immagine in un vettore a 512 dimensioni
- Il **text encoder** (BERT) trasforma un testo in un vettore a 512 dimensioni
- Il training contrastivo forza i vettori di coppie correlate ad avere alta cosine similarity

### Il Modality Gap: stesso spazio, regioni diverse

Tuttavia, esiste un fenomeno geometrico ben documentato in letteratura chiamato **modality gap** [1]: anche se immagini e testi condividono lo stesso spazio 512-d, non si mescolano al suo interno in modo uniforme. Le immagini occupano una regione, i testi un'altra, con uno scostamento (gap) sistematico tra i due cluster.

Questo fenomeno è stato formalmente caratterizzato da Liang et al. [1]:

> Gli autori dimostrano che nei modelli contrastivi (come CLIP) gli embeddings delle diverse modalità si collocano in "coni" separati dello spazio condiviso. Questo gap è causato da due fattori geometrici:
> 1. Il "cone effect" all'inizializzazione: le reti neurali tendono intrinsecamente a mappare gli input in un sottospazio molto stretto (un cono).
> 2. L'ottimizzazione contrastiva: la loss spinge le coppie accoppiate vicine, ma allo stesso tempo allontana tutti i campioni non accoppiati, spingendo intere modalità ad allontanarsi tra loro (dipende matematicamente dal parametro di "temperature").

### 2. Sulle soluzioni proposte dal paper "Mind the Gap" [1]

Le soluzioni proposte dalla letteratura (incluso il paper originale di Liang et al. [1]) per eliminare il modality gap si dividono in due categorie:

**A. Soluzioni in fase di addestramento (Training-time)**
La letteratura propone metodi come:

- Modificare il parametro di "temperature" nella funzione di loss contrastiva.
- Modality Swapping: scambiare immagini e testi durante il training per forzare l'encoder a fonderli.
- Gradient Reverse Layers (GRL): usare un classificatore di modalità per penalizzare la rete se i due spazi si separano.

**Possiamo implementarle? NO.** Tutte queste soluzioni richiedono di addestrare CLIP da zero. Noi stiamo usando BiomedCLIP come modello pre-addestrato (off-the-shelf). Non possiamo alterare i pesi interni dei suoi encoder.

**B. Soluzioni post-hoc (Geometriche)**
Il contributo più importante del paper [1] non sono tanto i fix di training, ma l'analisi matematica del problema. Loro dimostrano che il gap si comporta come una traslazione geometrica costante (un vettore di spostamento) tra i due "coni" (le regioni di spazio).
Questo risultato teorico è fondamentale perché giustifica matematicamente le soluzioni post-hoc (riportate di seguito). Sapendo che il gap è una semplice traslazione, possiamo prendere lo spazio del testo e "traslarlo" (shift) verso lo spazio visivo sottraendo il vettore differenza, oppure allineando tutto sul punto centrale visivo (`b_dec`).

Le soluzioni proposte sfruttano l'intuizione geometrica del paper [1] per allineare gli spazi a valle del modello, dato che non possiamo cambiare il modello a monte.

In termini pratici per il nostro progetto:

```
            Spazio condiviso 512-d di BiomedCLIP

   ┌────────────────────────────────────────────┐
   │                                            │
   │    ••••••                                  │
   │   • IMMAGINI •                             │
   │    • (CXR)  •      modality gap            │
   │     ••••••         ←───0.945───→  ○○○○○    │
   │                                  ○ TESTI ○ │
   │                                  ○(vocab)○ │
   │                                   ○○○○○    │
   │                                            │
   └────────────────────────────────────────────┘
```

CLIP funziona comunque bene per i task di **retrieval** (es. data un'immagine, trovare il testo più simile) perché il **ranking relativo** è preservato. Ma nel nostro concept naming non facciamo retrieval: calcoliamo una **cosine similarity diretta** e assoluta. Il modality gap abbatte sistematicamente questi valori.

---

## La Causa Principale: il Modality Gap nel Concept Naming

### Perché non addestriamo il SAE sul testo?
Un dubbio legittimo è: se il vocabolario è testuale, perché non addestrare il SAE direttamente sugli embeddings del testo per evitare il gap?
La risposta sta nell'obiettivo del progetto XAI. Noi vogliamo **spiegare come il modello "vede" le radiografie**. Vogliamo decomporre i feature estratti dalle immagini per capire quali patologie o strutture anatomiche (concept) si accendono in una specifica radiografia. Se addestrassimo il SAE sui testi (i referti), il SAE imparerebbe a decomporre la semantica linguistica, non l'anatomia visiva. Inoltre, durante l'inferenza, quando daremmo al "SAE testuale" l'embedding di un'immagine da spiegare, le performance crollerebbero a causa del modality gap, producendo attivazioni senza senso. Dobbiamo addestrare il SAE sulla modalità che vogliamo spiegare (le immagini).

### Evidenze numeriche dal nostro progetto

I dati sperimentali misurati sui nostri embeddings confermano il modality gap:

| Confronto | Cosine similarity media |
|-----------|------------------------|
| Immagine vs immagine (intra-modale) | **0.79** — strettamente raggruppate |
| Testo vs testo (intra-modale) | **0.65** — raggruppati |
| Immagine vs testo (cross-modale) | **0.27** — gap evidente |
| Centroide visivo vs centroide testuale | **0.38** |
| Distanza L2 tra centroidi | **0.945** |

Le immagini hanno alta similarità tra loro (0.79), i testi tra loro (0.65), ma cross-modalmente il valore crolla a 0.27.

### Come il SAE amplifica il problema

Il nostro SAE (Top-K Sparse Autoencoder [2]) ricostruisce un embedding visivo `x` come:

```
x̂ = W_dec · z + b_dec
```

dove:
- **`x`** ∈ ℝ⁵¹² è l'embedding visivo in input
- **`x̂`** ∈ ℝ⁵¹² è la ricostruzione fatta dal SAE
- **`z`** ∈ ℝ⁴⁰⁹⁶ è la rappresentazione **sparsa** intermedia
- **`W_dec`** ∈ ℝ⁵¹²ˣ⁴⁰⁹⁶ è la matrice del **decoder**: ogni colonna `w_i` rappresenta la "direzione" del concetto `i`
- **`b_dec`** ∈ ℝ⁵¹² è il **bias del decoder**: rappresenta il "punto base" o "centro" visivo attorno al quale il SAE opera.

La formula si riscrive come:
```
x̂ = b_dec + Σᵢ (zᵢ · wᵢ)
```
La ricostruzione parte dal "centro" visivo (`b_dec`) e aggiunge deviazioni date dalle direzioni del decoder (`wᵢ`). Le colonne di `W_dec` rappresentano quindi **variazioni relative** rispetto al centro. Infatti, il bias `b_dec` del nostro SAE ha `Cosine sim = 0.9962` con il centroide degli embeddings visivi.

#### Il problema nell'assegnazione
L'attuale `name_concepts` fa:
```python
similarities = normalize(W_dec) @ normalize(vocab_embeddings).T
```
Stiamo confrontando le direzioni di variazione (relative a `b_dec` nel cluster visivo) con le posizioni assolute del testo (nel cluster testuale). Stiamo confrontando deviazioni da un centro con coordinate spaziali assolute distanti.

### Prova diretta: la correzione del gap funziona

Sottraendo la differenza vettoriale (il gap) tra i due centroidi dai decoder weights:

| | Senza correzione | Con correzione gap |
|---|---|---|
| Mean max score | **0.1331** | **0.3946** |
| Max score | **0.3220** | **0.5470** |

Un miglioramento del **~3x**.

---

## Analisi dei Tre Componenti della Pipeline

### 1. Vocabolario (`build_vocabulary.py`) → Parzialmente Problematico

Il vocabolario è ben costruito, ma ha rumore:
- Termini troppo specialistici o in tedesco ("Hemidiaphragma", "Dura mater spinalis").
- Termini generici ("mass", "nodule") con score bassi contro gli anchor centroids.
*Impatto: Limita i matching ideali, ma non causa score globalmente a 0.13.*

### 2. Addestramento SAE (`sae_module.py`) → Funziona Bene

Il SAE ricostruisce lo spazio visivo in modo eccellente (Cosine sim: 0.997, MSE: 0.000044).
*L'addestramento non è il problema.*

### 3. Assegnamento Concetti (`name_concepts`) → CAUSA PRINCIPALE

Come ampiamente discusso, la cosine similarity diretta senza correzione del gap è l'errore geometrico di fondo.

---

## Revisione Critica dell'Analisi

Prima di proporre soluzioni, è necessario identificare alcuni errori e imprecisioni nell'analisi precedente.

### Errore 1: "Le colonne del decoder catturano direzioni relative a `b_dec`"

Questa affermazione, ripetuta più volte nel documento, è **imprecisa e fuorviante**. Rivediamo cosa fa realmente il SAE.

Dal codice sorgente della libreria `dictionary_learning` (`AutoEncoderTopK` [2]), il forward pass è:

```python
# ENCODE: sottrae b_dec dall'input prima di codificare
z = ReLU(W_enc · (x - b_dec) + b_enc)      # sparse code

# DECODE: aggiunge b_dec alla ricostruzione
x̂ = W_dec · z + b_dec
```

Riscrivendo:
```
x̂ = b_dec + Σᵢ (zᵢ · wᵢ)
```

Qui `b_dec` viene inizializzato alla **mediana geometrica** (non media aritmetica) degli embeddings di training, e poi viene **ottimizzato** durante il training come parametro apprendibile. Il fatto che `b_dec` sia vicino al centroide visivo (cosine sim 0.9962) è un risultato empirico, non un vincolo architetturale.

Il punto cruciale è: **le colonne del decoder `wᵢ` non sono "variazioni relative a `b_dec`"**. Sono i vettori base del dizionario che il SAE ha imparato. Il SAE ricostruisce `x` come combinazione lineare sparsa di queste basi, con `b_dec` che agisce come offset costante (bias). Le colonne `wᵢ` sono direzioni nello spazio 512-d, non "deviazioni dal centroide". Lo sono solo nel senso triviale che `x - b_dec ≈ Σᵢ(zᵢ · wᵢ)`, ma geometricamente `wᵢ` è una direzione a sé stante, normalizzata a norma unitaria.

Questa distinzione è importante perché cambia il ragionamento dietro la Soluzione 3.

### Errore 2: La Soluzione 3 non è geometricamente corretta come descritto

La Soluzione 3 propone di centrare i text embeddings sottraendo `b_dec`:
```python
text_centered = vocab_embeddings - b_dec
```

Il ragionamento era: "se `wᵢ` è una direzione relativa a `b_dec`, allora anche `tⱼ - b_dec` diventa una direzione relativa allo stesso punto, e la cosine similarity diventa significativa".

Ma questo ragionamento ha un problema. Quello che il SAE davvero fa è: `(x - b_dec) ≈ W_dec · z`, ovvero la matrice decoder spiega la differenza `x - b_dec`. Ciò significa che `W_dec` opera nello spazio di `(x - b_dec)`, non nello spazio originale di `x`.

Quando calcoliamo `tⱼ - b_dec`, stiamo proiettando i text embeddings in questo spazio "centrato sul visivo". Ma `tⱼ` parte dalla regione testuale (lontana da `b_dec`), quindi `tⱼ - b_dec` è un vettore molto lungo che punta dalla regione visiva alla regione testuale — non è una "piccola deviazione" nello stesso spazio in cui operano i `wᵢ`.

In pratica, `tⱼ - b_dec` e `xᵢ - b_dec` hanno geometrie diverse:
- `xᵢ - b_dec` sono piccoli (norma media ~0.45), isotropi, e coperti bene dai `wᵢ`
- `tⱼ - b_dec` sono grandi (norma media ~1.1), anisotropi, e puntano attraverso il modality gap

Dopo normalizzazione L2, parte di questa differenza viene assorbita, ma la **direzione** resta contaminata dal vettore di gap modale.

### Cosa funziona davvero e perché

Il risultato sperimentale resta valido: la Soluzione 1 (sottrazione del gap dei centroidi, giustificata dalla teoria di Liang et al. [1]) produce un miglioramento 3x. Questo non è un caso: **il modality gap è reale** e la correzione funziona. Ma la spiegazione corretta non è legata all'architettura del SAE. È semplicemente:

1. I decoder weights `wᵢ` sono direzioni unitarie nello spazio 512-d
2. I vocab embeddings `tⱼ` sono anche direzioni (quasi) unitarie nello stesso spazio
3. La cosine similarity tra `wᵢ` e `tⱼ` è bassa perché i due insiemi di vettori occupano regioni angolari diverse dello spazio (modality gap [1])
4. Sottrarre il vettore di gap dai `wᵢ` (o aggiungerlo ai `tⱼ`) trasla una delle due nuvole di punti, avvicinando le regioni angolari

Inoltre, l'analisi della letteratura conferma che per SAE applicate a CLIP, cosine similarity di 0.3-0.4 tra decoder directions e text embeddings **sono valori normali e significativi**, non pessimi [3][4][5][6]. Il metodo standard nella letteratura è proprio calcolare la cosine similarity tra le colonne del decoder e i text embeddings del vocabolario [3][4]. Il modality gap dimezza questi valori, ma anche dopo la correzione non ci si deve aspettare score vicini a 1.0: in spazi 512-d con centinaia di concetti, valori di 0.4-0.5 sono già eccellenti.

---

## Soluzioni Proposte (Revisionate)

### Soluzione 1: Correzione del Modality Gap Post-Hoc

Ispirata all'analisi di Liang et al. [1] (che dimostra che il gap è approssimativamente una traslazione costante), sottraiamo il gap sistematico. Questa è la soluzione più semplice e con evidenza sperimentale diretta.

```python
visual_centroid = train_embeddings.mean(dim=0)
text_centroid = vocab_embeddings.mean(dim=0)
gap = visual_centroid - text_centroid

W_dec_corrected = W_dec - gap.unsqueeze(0)
W_norm = F.normalize(W_dec_corrected, dim=1)
V_norm = F.normalize(vocab_embeddings, dim=1)
similarities = W_norm @ V_norm.T
```

**Risultato sperimentale**: Mean max score 0.13 → 0.39 (+3x), Max score 0.32 → 0.55.

**Pro**: Semplice, efficace, giustificata teoricamente.
**Contro**: Richiede di avere accesso ai `train_embeddings` durante il naming (per calcolare il centroide visivo). Questo è un dato esterno al SAE.

### Soluzione 2: Naming basato su Activation Matching (più robusto) [4]

Un approccio completamente diverso che **evita il confronto diretto nello spazio degli embeddings** e quindi bypassa del tutto il modality gap:

1. Per ogni feature `i` del SAE, calcolare il pattern di attivazione `aᵢ` su tutti i campioni del training set (un vettore binario o continuo di dimensione N)
2. Per ogni termine `j` del vocabolario, calcolare la similarity CLIP image-text `sⱼ` tra il testo `j` e tutte le N immagini (un vettore di dimensione N)
3. Fare la correlazione (Pearson o Spearman) tra `aᵢ` e `sⱼ`

In pratica: una feature si chiama "cardiomegaly" se si attiva sulle stesse immagini per cui BiomedCLIP dice "questa immagine è simile al testo cardiomegaly".

**Pro**: Non dipende dalla geometria dello spazio degli embeddings. Usa la semantica effettiva di CLIP.
**Contro**: Computazionalmente costoso (bisogna calcolare la similarity CLIP per ogni termine × ogni immagine). Richiede di caricare BiomedCLIP per l'inferenza.

### Soluzione 3: Centratura via `b_dec` (Variante della Soluzione 1)

Usare `b_dec` al posto del centroide calcolato esternamente. Poiché `b_dec ≈ centroide visivo` (cosine sim 0.9962), in pratica produce risultati molto simili alla Soluzione 1, ma con il vantaggio che **`b_dec` è già contenuto nel modello SAE salvato**, senza bisogno di ricalcolare il centroide dai training embeddings.

```python
b_dec = self._ae.b_dec.data              # già nel modello SAE
text_shifted = vocab_embeddings - b_dec   # sposta i testi verso l'origine dello spazio SAE
W_dec = self.get_decoder_weights()
W_norm = F.normalize(W_dec, dim=1)
T_norm = F.normalize(text_shifted, dim=1)
similarities = W_norm @ T_norm.T
```

**Attenzione**: questa soluzione NON è identica alla Soluzione 1 dal punto di vista matematico. Nella Soluzione 1 si trasla `W_dec`, qui si trasla il vocabolario. L'effetto geometrico è diverso dopo la normalizzazione L2. I risultati vanno testati sperimentalmente.

**Pro**: Autocontenuta (non serve ricalcolare nulla dai training data). Elegante.
**Contro**: L'approssimazione `b_dec ≈ centroide` potrebbe essere imperfetta. La traslazione del vocabolario vs la traslazione del decoder produce risultati diversi.

### Soluzione 4: Riduzione del Vocabolario

In parallelo a qualsiasi soluzione scelta, ripulire il vocabolario:
- Rimuovere termini non-inglesi (tedesco)
- Rimuovere termini anatomici troppo specifici (segmenti spinali, etc.)
- Concentrarsi sui ~100-200 termini più rilevanti per chest X-ray
- Aggiungere termini comuni nei report IU X-Ray (es. "clear lungs", "normal heart size")

### Raccomandazione

**Testare sperimentalmente tutte e tre le soluzioni (1, 2, 3) prima di sceglierne una.** La Soluzione 1 ha già evidenza sperimentale di miglioramento 3x. La Soluzione 3 ha il vantaggio pratico dell'autocontenimento. La Soluzione 2 è concettualmente la più solida ma anche la più costosa. 

Tutte le soluzioni vanno validate non solo sugli score numerici, ma anche qualitativamente: i nomi assegnati devono avere senso clinico quando confrontati con le immagini che attivano maggiormente ciascuna feature.

---

## Riferimenti

[1] Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. Y. (2022). **Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.** *Advances in Neural Information Processing Systems (NeurIPS), 35.* [arXiv:2203.02053](https://arxiv.org/abs/2203.02053) — Caratterizza formalmente il modality gap nei modelli contrastivi come CLIP, dimostrando che le diverse modalità occupano regioni separate ("coni") dello spazio condiviso. Dimostra che il gap è approssimativamente una traslazione costante, causata dal "cone effect" all'inizializzazione e dal parametro di temperatura della loss contrastiva.

[2] Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). **Scaling and Evaluating Sparse Autoencoders.** *arXiv preprint.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093) — Introduce l'architettura Top-K SAE utilizzata nel nostro progetto. Propone l'uso della funzione di attivazione TopK al posto della penalità L1 per controllare direttamente la sparsità, e introduce tecniche per mitigare le dead features su larga scala.

[3] Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F. P., & Lakkaraju, H. (2024). **Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE).** *Advances in Neural Information Processing Systems (NeurIPS), 37.* [arXiv:2402.10376](https://arxiv.org/abs/2402.10376) — Propone un metodo task-agnostic per interpretare gli embeddings CLIP decomponendoli in combinazioni lineari sparse di concetti umani. Utilizza cosine similarity tra concept vectors e text embeddings come metrica di naming.

[4] Rao, S., Mahajan, S., Böhle, M., & Schiele, B. (2024). **Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery.** *European Conference on Computer Vision (ECCV).* [arXiv:2407.14499](https://arxiv.org/abs/2407.14499) — Framework "Discover-then-Name" (DN-CBM) che utilizza SAE su CLIP per scoprire automaticamente concetti, poi li nomina tramite cosine similarity con text embeddings. Dimostra che i concetti scoperti tramite SAE sono semanticamente coerenti e utilizzabili per task di classificazione.

[5] Zaigrajew, V., Baniecki, H., & Biecek, P. (2025). **Interpreting CLIP with Hierarchical Sparse Autoencoders.** *International Conference on Machine Learning (ICML), 42.* [arXiv:2502.20578](https://arxiv.org/abs/2502.20578) — Introduce l'architettura Matryoshka Sparse Autoencoder (MSAE) per interpretare CLIP a multiple granularità. Utilizza cosine similarity tra colonne del decoder e text embeddings per il naming dei concetti, e valuta la qualità dei concetti estratti in termini di interpretabilità e bias analysis.

[6] Pach, M., Karthik, S., Bouniot, Q., Belongie, S., & Akata, Z. (2025). **Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models.** *Advances in Neural Information Processing Systems (NeurIPS).* [arXiv:2504.02821](https://arxiv.org/abs/2504.02821) — Dimostra che SAE addestrati su VLMs (incluso CLIP) producono features monosemantiche interpretabili. Introduce un benchmark umano per valutare la monosemanticity e mostra che la sparsità e la larghezza dei latenti sono i fattori più influenti sulla qualità delle features.

