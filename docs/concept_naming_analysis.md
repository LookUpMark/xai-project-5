# Analisi delle Performance di Concept Naming

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

Tuttavia, esiste un fenomeno geometrico ben documentato in letteratura chiamato **modality gap**: anche se immagini e testi condividono lo stesso spazio 512-d, non si mescolano al suo interno in modo uniforme. Le immagini occupano una regione, i testi un'altra, con uno scostamento (gap) sistematico tra i due cluster.

Questo fenomeno è stato formalmente caratterizzato nel paper:

> **Liang et al., "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning" (NeurIPS 2022)**
>
> Gli autori dimostrano che nei modelli contrastivi (come CLIP) gli embeddings delle diverse modalità si collocano in "coni" separati dello spazio condiviso. Questo gap è causato da due fattori geometrici:
> 1. Il "cone effect" all'inizializzazione: le reti neurali tendono intrinsecamente a mappare gli input in un sottospazio molto stretto (un cono).
> 2. L'ottimizzazione contrastiva: la loss spinge le coppie accoppiate vicine, ma allo stesso tempo allontana tutti i campioni non accoppiati, spingendo intere modalità ad allontanarsi tra loro (dipende matematicamente dal parametro di "temperature").

### 2. Sulle soluzioni proposte dal paper "Mind the Gap"

Le soluzioni proposte dalla letteratura (incluso il paper originale di Liang et al., NeurIPS 2022) per eliminare il modality gap si dividono in due categorie:

**A. Soluzioni in fase di addestramento (Training-time)**
La letteratura propone metodi come:

- Modificare il parametro di "temperature" nella funzione di loss contrastiva.
- Modality Swapping: scambiare immagini e testi durante il training per forzare l'encoder a fonderli.
- Gradient Reverse Layers (GRL): usare un classificatore di modalità per penalizzare la rete se i due spazi si separano.

**Possiamo implementarle? NO.** Tutte queste soluzioni richiedono di addestrare CLIP da zero. Noi stiamo usando BiomedCLIP come modello pre-addestrato (off-the-shelf). Non possiamo alterare i pesi interni dei suoi encoder.

**B. Soluzioni post-hoc (Geometriche)**
Il contributo più importante del paper "Mind the Gap" non sono tanto i fix di training, ma l'analisi matematica del problema. Loro dimostrano che il gap si comporta come una traslazione geometrica costante (un vettore di spostamento) tra i due "coni" (le regioni di spazio).
Questo risultato teorico è fondamentale perché giustifica matematicamente la soluzione 3 (riportata di seguito). Sapendo che il gap è una semplice traslazione, possiamo prendere lo spazio del testo e "traslarlo" (shift) verso lo spazio visivo sottraendo il vettore differenza, oppure allineando tutto sul punto centrale visivo (`b_dec`).

Ecco perché la "Soluzione 3" (l'encoding incrociato usando `b_dec`) è la via maestra: sfrutta l'intuizione geometrica del paper per allineare gli spazi a valle del modello, dato che non possiamo cambiare il modello a monte.

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

Il nostro SAE (Top-K Sparse Autoencoder) ricostruisce un embedding visivo `x` come:

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

## Soluzioni Proposte

### Soluzione 1: Correzione del Modality Gap Post-Hoc

Ispirata a "Mind the Gap", sottraiamo il gap sistematico per traslare i due spazi:

```python
visual_centroid = train_embeddings.mean(dim=0)
text_centroid = vocab_embeddings.mean(dim=0)
gap = visual_centroid - text_centroid

W_dec_corrected = W_dec - gap.unsqueeze(0)
W_norm = F.normalize(W_dec_corrected, dim=1)
V_norm = F.normalize(vocab_embeddings, dim=1)
similarities = W_norm @ V_norm.T
```

### Soluzione 2: Naming basato su Activation Matching

Un approccio *activation-based*: passare ogni termine testuale nel modello per ottenere similarity score su tutte le immagini del dataset. Fare poi il matching confrontando per ogni immagine il pattern di "similarity testuale" con il pattern di "attivazione della feature del SAE".

### Soluzione 3: Encoding Incrociato (Best Solution)

Soluzione più solida: usare il bias `b_dec` del SAE per correggere il gap. Le colonne di `W_dec` catturano direzioni *relative* a `b_dec`. Se vogliamo confrontarle con il testo, centriamo il testo su `b_dec`:

```python
b_dec = self._ae.b_dec.data              # centro visivo del SAE
text_centered = vocab_embeddings - b_dec  # centra i testi rendendoli direzioni relative
W_dec = self.get_decoder_weights()
W_norm = F.normalize(W_dec, dim=1)
T_norm = F.normalize(text_centered, dim=1)
similarities = W_norm @ T_norm.T
```
Questo rende il confronto geometricamente corretto senza calcoli complessi esterni.

### Soluzione 4: Riduzione del Vocabolario

Ripulire il vocabolario (filtrare lingue diverse, specificità estrema).
