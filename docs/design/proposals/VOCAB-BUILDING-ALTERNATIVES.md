# Costruzione del vocabolario: alternative e ablation derivate

> **Stato (aggiornato 2026-06-22).** Il vocab RadLex e' stato rebuildato a **508
> termini** (issue #7 risolta). Il naming debole e' stato **ridefinito** in
> `CONCEPT-NAMING-ANALYSIS.md`: con la correzione del modality gap (Soluzione 1)
> il naming sale a mean 0.395 / max 0.547 (~3×), e la letteratura considera
> 0.3-0.4 *normale e significativo* per SAE-su-CLIP — non un fallimento.
> **L'ablation #2 (faithfulness concept↔etichetta MeSH/Problems) e' STATO
> CONSEGNATO come `notebooks/autoencoder/ablation/05_faithfulness.ipynb` (a5)**:
> ~10% delle feature live (226/2251) sono fedeli a vere etichette cliniche oltre
> un null per-feature (le piu' forti: impianti medici |r|=0.46, versamento
> pleurico, enfisema). Le alternative qui sotto restano valide come **future
> work** (naming comparison RadLex-vs-gold #1, SPLiCE #3, vocab ibrido/mining).

Il concept naming del run baseline e' debole (score mean 0.117, **max 0.291**)
perche' il vocabolario RadLex (508 termini, ontologia radiologica generale) e'
*off-distribution* rispetto al dominio CXR di IU X-Ray. Questo doc sistematizza
le alternative di costruzione e gli **ablation aggiuntivi** che ne derivano
(naming comparison, faithfulness concept<->etichetta, SPLiCE). Da leggersi con
`CONCEPT-INSTABILITY-DIAGNOSIS.md` (causa 2: naming) e
`ADDITIONAL-ABLATION-STUDIES.md`.

## Contesto

Il naming confronta direzioni del decoder SAE (spazio *immagine*) con embedding
*testuali* dei termini del vocabolario. La qualita' del naming dipende da due
fattori, di cui solo il primo e' affrontabile col vocabolario:

1. **Allineamento del vocabolario alla distribuzione delle immagini** (questo doc).
2. **Stabilita' dei concetti** - feature non stabili (Jaccard 0.0038) non si
   ancorano a un nome stabile (causa 1 nella diagnosi). Il vocabolario alza il
   *soffitto* dell'allineamento ma non fissa l'instabilita': va fatto insieme a
   dict_size/steps minori (a1/a4 in corso).

## Il gold standard e' gia' sul disco

`indiana_reports.csv` (3851 referti) ha due colonne strutturate per referto:
- `MeSH` - termini MeSH (Medical Subject Headings, ontologia NLM/NIH), es.
  "Cardiomegaly/borderline;Pulmonary Artery/enlarged".
- `Problems` - problemi clinici, es. "Cardiomegaly;Pulmonary Artery".

Estrazione effettiva (split su `;`, base MeSH su `/`, lowercase, escluso
"normal"):

- **118 termini unici** Problems ∪ MeSH.
- Distribuzione sana: da ~1100 occorrenze (`opacity`, `lung`) a termini rari.
- Tutti chest-radiograph-specifici e ricorrenti: `cardiomegaly` (690),
  `pulmonary atelectasis` (660), `calcified granuloma` (552), `pleural effusion`
  (320), `airspace disease` (246), `nodule` (232), `scoliosis` (236).

### Confronto con RadLex

| | RadLex (attuale) | Gold standard MeSH/Problems |
|---|---|---|
| Termini | 508 | 118 |
| Natura | Ontologia generale | Etichette dal dataset |
| Distribuzione | Molti off-distribution | Solo ciò che c'e' nelle immagini |
| Allineamento | max cosine 0.29 | atteso piu' alto |
| Etichette per-immagine | No | **Si** |

### Perche' il gold standard e' quasi certamente superiore

CLIP e' addestrato per allineare testo e immagine. Termini semanticamente vicini
al **contenuto reale** delle immagini hanno embedding testuali vicini a quelli
delle immagini -> si allineano meglio alle direzioni del decoder SAE. RadLex
generico e' lontano dal dominio; MeSH/Problems *e'* il dominio. Atteso: max
cosine ben oltre 0.29.

## Approcci sistematizzati

| | Fonte | Pro | Contro |
|---|---|---|---|
| **A. RadLex** (attuale) | Ontologia esterna | Copertura strutturata | Off-distribution, molti irrilevanti |
| **B. Gold standard** | `MeSH`/`Problems` | In-distribution, controllato, +etichette | Solo ciò che i radiologi hanno menzionato |
| **C. Mining narrativo** | `findings`/`impression` via TF-IDF o NER medico (scispaCy, Med7) | Più ricco di MeSH, espressioni naturali | Rumore (negazioni, templated), serve normalizzazione |
| **D. Ibrido** | B ∪ C (+ A backbone) | Copertura + allineamento | Più engineering |
| **E. LLM-generated** | Prompt -> frasi cliniche encodate | Phrase-level (CLIP preferisce frasi ai termini singoli) | Bias del modello generativo |
| **F. SPLiCE** | *(metodo, non fonte)* | Naming come ottimizzazione sparsa sui pesi decoder invece di greedy cosine | Ortogonale alla scelta del vocab - migliora entrambi |

## Ablation aggiuntivi derivati

Questi sono i nuovi ablation che si sbloccano cambiando vocabolario/metodo.
Non hanno ancora un ID (a0-a6 sono presi; a3/a5 droppati) - l'assegnazione avviene
all'implementazione.

### 1. Naming comparison: RadLex vs gold standard vs ibrido
Ripetere il concept naming del modello baseline (seed 42) con 3 vocabolari
(RadLex, MeSH/Problems, B ∪ C) a parita' di SAE e metodo. Misurare:
- max / mean cosine decoder<->vocab.
- copertura: % feature con cosine > soglia.
- coerenza clinica qualitativa dei top-nominati.

**Atteso:** MeSH/Problems batte RadLex su max cosine e copertura. Costo basso:
analysis-only, no retrain (riusa il modello baseline gia' addestrato), serve
solo ri-encodare i termini del vocab con BiomedCLIP text encoder.

### 2. Faithfulness concept-attivazione <-> etichetta (la porta riaperta) — ✅ CONSEGNATO (a5)
> **Delivered come `05_faithfulness.ipynb` (a5).** Point-biserial correlation
> feature↔etichetta su 50 label prevalenti (MeSH/Problems), null calibrato
> per-feature (shuffle p95) + BH-FDR. ~10% delle feature live fedeli oltre il
> caso. Vedi `notebooks/autoencoder/ablation/REPORT.md` §Ablation 05.

L'handoff aveva droppato "concept <-> 14 patologie NIH" perche' quelle etichette
**non esistono** in IU X-Ray. Vero. Ma `MeSH`/`Problems` sono etichette
per-immagine -> riaprono una valutazione di faithfulness reale, con zero nuovi dati.

**Protocollo:**
- Join: `image_id` (basename PNG) -> `uid` via `indiana_projections.csv` ->
  `MeSH`/`Problems` via `indiana_reports.csv`. Dati gia' sul disco.
- Per ogni concetto X e etichetta Y: information_mutual / AUROC (l'attivazione
  di X predice la presenza di Y sull'immagine?).
- Concetti "fedeli" = alta MI con almeno un'etichetta. Metrica di sintesi:
  % di concetti fedeli.

**Perche' conta piu' del naming:** il naming e' cosmetico (stringa attribuita);
la faithfulness e' *validazione* (il concetto predice qualcosa di vero
sull'immagine). E' il tipo di risultato che distingue "concetti interpretabili"
da "etichette rumorose". Costo medio: analysis-only sul test set, ma richiede
il join e una metrica di MI/AUROC non ancora nel codice.

### 3. SPLiCE vs greedy cosine (metodo)
Confrontare il naming greedy-coseno attuale con SPLiCE (decomposizione sparsa
dei pesi del decoder contro il vocabolario). Ortogonale alla scelta del
vocabolario: testarlo su MeSH/Problems (fonte migliore). Costo medio: serve
l'ottimizzazione sparsa (es. L1 su pesi decoder), no retrain SAE.

## Raccomandazione + sequenza

- **Base: B (gold standard MeSH/Problems)** - 118 termini gia' pronti,
  in-distribution, + etichette per faithfulness. Investimento a piu' alto ritorno.
- **Opzionale: B ∪ C (ibrido)** se i 118 risultano troppo pochi per coprire i
  concetti vivi del SAE.
- **Metodo: SPLiCE (F)** invece del greedy cosine, qualunque sia la fonte.
- **Ablation da fare prima:** #1 (naming comparison) - e' il piu' a basso costo
  e quantifica direttamente il guadagno di allineamento. Poi #2 (faithfulness)
  come risultato di validazione piu' forte. #3 (SPLiCE) se resta tempo.

**Per il presente** (le 5 ablation in corso) il vocab resta fisso sul RadLex
committed - giustamente, non si varia due cose insieme. Questo e' materiale per
dopo, come estensione.

## Issue #7 risolta dal rebuild

Il rebuild del vocabolario e' anche l'occasione per risolvere l'issue #7
(schema mismatch): `build_vocabulary.save_vocabulary` scrive dict
`{term, similarity_score, source}`, mentre il `data/vocabulary.json` committed
e' una lista di 508 dict `{term,...}`. **Risolto a runtime**: `_vocab_term` in
`sae_module.py` normalizza dict→stringa `term` ovunque (CLI, notebook, naming),
e `concept_naming` e' verificato contro questo schema. Il rebuild del 2026-06-22
ha portato il conteggio a 508.

## Riferimenti

- MeSH - NLM/NIH Medical Subject Headings (ontologia controllata).
- Radford et al. (2021) CLIP - allineamento testo-immagine condiviso.
- Bhalla, Srinivas, Hsieh (2024) SPLiCE "Compositional Explanations" - naming
  via ottimizzazione sparsa sui pesi del decoder.
- Neumann et al. (2019) scispaCy - NER biomedicalo per mining narrativo.
- Marks et al. (2024) "Sparse Feature Circuits" - valutazione concept<->label.
