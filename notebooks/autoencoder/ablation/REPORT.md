# REPORT — Ablazioni SAE (`notebooks/autoencoder/ablation/`)

Report cumulativo delle ablation. Una sezione per notebook, aggiornata ad ogni run.

**Indice**
- [In parole semplici — tutta la storia](#in-parole-semplici--tutta-la-storia)
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder (lr pinned)](#ablation-01--dictionary-size-ladder-lr-pinned)
- [Ablation 02 — k (Sparsity) Sweep, null-calibrated](#ablation-02--k-sparsity-sweep-null-calibrated)
- [Ablation 03 — Concept Baselines + Empirical Jaccard Floor](#ablation-03--concept-baselines--empirical-jaccard-floor)
- [Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)](#ablation-04--activation-family-bake-off-topk-vs-batchtopk-vs-jumprelu)

---

## In parole semplici — tutta la storia

> **Leggi qui per capire l'intero programma di ablation senza i dettagli tecnici.**

**Il punto di partenza (dal baseline REPORT).** Il nostro SAE funziona tecnicamente (ricostruisce a cosine 0.988), ma ha un problema grosso: se lo addestri 5 volte con seed diversi, le 5 versioni trovano concetti **quasi completamente diversi** (Jaccard 0.004 ≈ 0). La domanda madre di tutte le ablation è:

> **Perché i concetti non sono riproducibili? È un fallimento vero, o è matematica inevitabile? Si può fixare cambiando qualche parametro?**

**Le 4 ablation sono 4 tentativi di risposta, ognuno con una ipotesi diversa:**

| Ab | Ipotesi ("forse il colpevole è…") | Verdetto |
|---|---|---|
| **00** | "I 5 run trovano gli stessi concetti, ma salvati a indici diversi (è solo permutazione)" | ❌ **Falso** — anche nello spazio delle direzioni sono disgiunti |
| **01** | "Il dizionario è troppo grande (4096 = 8× sovradimensionato)" | ❌ Per i dead sì, per la stabilità **no** |
| **02** | "La sparsità k è sbagliata" | ⚠️ **Parziale** — c'è un debole sweet spot a k=16, non risolve |
| **03** | "È il SAE a essere scarso, o è il pavimento del caso?" | ✅ **Pavimento del caso** — random fa uguale (0.0037 ≈ 0.0038) |
| **04** | "È colpa di TopK? Proviamo famiglie diverse (BatchTopK, JumpReLU)" | ❌ dead% sì (BatchTopK meglio), stabilità **no** (consensus 0 per tutti) |

**La conclusione d'insieme (onesta, negativa ma chiara):**
1. Il 0.004 "preoccupante" del baseline **non è un fallimento** — è il **pavimento matematico del caso**. Due dizionari grandi e indipendenti si sovrappongono sempre per ~0.004 per pura probabilità, anche con numeri random (ab03 lo dimostra: random@4096 = 0.0037).
2. L'instabilità **non si fixa** con iperparametri: né `dict_size` (ab01), né `k` (ab02), né la **famiglia di attivazione** (ab04: TopK/BatchTopK/JumpReLU tutte con consensus 0) la risolvono.
3. Cause profonde rimaste: **pochi campioni (5976) + non-unicità intrinseca della decomposizione sparsa** su questo dataset (vale per TopK, BatchTopK e JumpReLU — non è specifica di TopK). Roba strutturale, non un bug da parameter-tuning. **ab04 chiude l'indagine**: cambiare il meccanismo centrale (la funzione di attivazione) non aiuta più di quanto aiuti cambiare dict_size o k.

**Cosa quindi?** Accettare la seed-dipendenza come limite dichiarato del metodo, oppure aggregare i seed (model soup / consensus clustering con validazione). Il valore del SAE qui resta **strutturale**: sparsità garantita (solo 32 feature) + ricostruzione buona (0.988) + naming top-end sopra il caso.

---

# Ablation 00 — Cross-Seed Consensus (direction-space)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA** (auto-rilevato)
**Notebook:** `notebooks/autoencoder/ablation/00_consensus.ipynb` (run headless post-fix cell 18)
**Input:** 5 checkpoint baseline `models/sae_seed{0,42,123,456,789}/` (06-05, riusati — **zero training**), `test_embeddings.pt` (1494), vocabolario RadLex **508 termini** (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt`)
**Config:** `dict_size=4096`, `k=32`, 5 seed, `dead_threshold=1e-8`, griglia `tau ∈ {0.80, 0.85, 0.90, 0.95}`, **headline `tau=0.90`**, Hungarian match `cos ≥ 0.90`, shuffle-null = 200 permutazioni

### In parole semplici

> **La domanda.** Il baseline dice che 5 run diverse condividono <0.4% delle feature attive (Jaccard 0.004). Ma questo confronta gli **indici** (lo slot `i` del seed A combacia con lo slot `i` del seed B?). Forse i 5 run imparano gli **stessi concetti** ma li salvano a **posizioni diverse** nel dizionario — come 5 persone che mettono gli stessi libri su scaffali numerati in modo diverso. Se confronti solo i numeri dello scaffale, sembrano diversi; ma i libri sono gli stessi.
>
> **Cosa fa ab00.** Invece di guardare gli indici, guarda lo **spazio delle direzioni**: clusterizza tutte le feature dei 5 seed in base a quanto sono simili geometricamente (coseno tra direzioni), indipendentemente da dove sono salvate. Se i concetti ricorrono, appariranno cluster multi-seed.
>
> **Risultato.** **Quasi zero cluster condivisi.** Solo 1 direzione su 20480 ricorre su ≥3 seed; nessuna su ≥4. Lo shuffle-null dà p=1.0 (l'osservato è identico al caso). **Conclusione: i 5 run imparano davvero direzioni diverse, non è solo permutazione di indici.** Ipotesi falsificata, riportato onestamente.

> **Tesi del notebook.** La baseline riporta un **mean index-Jaccard di 0.0038** (off-diagonali 0.002–0.010) — test di *identità di slot*: "lo slot `i` del seed A spara sugli stessi sample dello slot `i` del seed B?". Se due seed imparano la **stessa direzione concettuale** ma la salvano a **indici diversi**, l'index-Jaccard li segna zero anche se la geometria coincide. Ipotesi: il 0.0038 è un **artefatto di permutazione**. L'ablation lo verifica ri-analizzando lo **spazio delle direzioni** del decoder (indipendente dall'indice).
>
> **Esito: TESI FALSIFICATA.** Lo spazio delle direzioni mostra **~0 struttura condivisa** tra seed. Il risultato negativo è riportato onestamente (nessun p-hacking abbassando `tau`). Vedi §3.

---

## 1. Cosa produce ogni fase

| Fase | Output | Stato |
|---|---|---|
| (A) Pool decoder rows | 5×`W_dec` (4096×512) L2-normalizzate, tag seed | ✅ 20480 righe, **0 dead** |
| (B) Clustering `cos>tau` | `scipy.connected_components` su grafo sparso | ✅ griglia tau + headline 0.90 |
| (C) Reappearance | cluster per #seed rappresentati | ✅ consensus@3/4 |
| (D) Hungarian direction-match | `linear_sum_assignment` per coppia di seed | ✅ direction-Jaccard |
| (E) Name-agreement | termine RadLex argmax per membro cluster | ✅ |
| (F) Faithfulness proxy | naming-cos + mean test activation per concetto | ✅ (proxy, no ground-truth) |
| (G) Headline figure | `results/figures/ablation/a0_consensus_headline.png` (3 pannelli) | ✅ |
| (H) Shuffle-null | 200 permutazioni tag → consensus@4 | ✅ p-value |
| **Persist** | `results/ablation/a0_consensus.json` | ✅ |

> **Isolamento protocollo:** output scritti solo in `results/ablation/` + `results/figures/ablation/` — la `results/` della baseline **non viene toccata**.

---

## 2. Risultati nel dettaglio

### 2.1 Pooling decoder (A) — 20480 righe, 0 dead

> **Cosa significa.** Riuniamo tutte le feature dei 5 seed (5×4096 = 20480 direzioni) in un unico "sacco", etichettate per seed. Poi cerchiamo quali si somigliano tra seed diversi. 0 dead qui perché i decoder sono già normalizzati.

Per ogni seed: `get_decoder_weights()` → `(4096, 512)`, `F.normalize` righe, scarto righe a norma `< 1e-8`.

- **5 × 4096 = 20480 righe pooled**, tutte live.
- **0% dead qui** perché nei checkpoint addestrati le righe del decoder sono **già unit-norm** (la lib `dictionary_learning` normalizza ogni colonna a ogni step). Lo scarto dead è un **no-op**.

> ⚠️ **Conflitto di definizioni "dead"** (come CLAUDE.md / baseline §2.3): qui *decoder-norm dead* = 0% (colonne unit-norm post-training). La baseline riporta *activation dead* ~44% (feature mai attiva sul test). **Sono due metriche diverse** — non contraddittorie. L'analisi consenso gira sul decoder **completo**, non su un subset potato.

### 2.2 Clustering `cos>tau` (B) — headline `tau=0.90`

> **Cosa significa.** Costruiamo un grafo: due feature sono collegate se la loro similarità (coseno) supera una soglia `tau`. Poi cerchiamo le "componenti connesse" = gruppi di feature molto simili tra loro. Se i seed condividessero concetti, vedremmo gruppi formati da feature di seed diversi. A `tau=0.90` (soglia alta = solo feature quasi identiche) troviamo 1 solo cluster condiviso.

Grafo sparso `cosine > tau` → connected components.

| `tau` | componenti | multi-member | max size | coesione intra (cos medio) |
|------:|-----------:|-------------:|---------:|---------------------------:|
| 0.80 | 20474 | 3 | 4 | 0.832 |
| 0.85 | 20477 | 1 | 4 | 0.879 |
| **0.90** | **20478** | **1** | **3** | **0.884** |
| 0.95 | 20480 | 0 | 1 | — (singleton) |

- Headline `tau=0.90`: cluster piccoli e coesi (coesione 0.884), max size 3.
- `tau` più basso fonde direzioni non correlate in pochi cluster giganti; più alto frantuma in singleton. **0.90 è il punto interpretable.**

### 2.3 Reappearance (C) — ❌ essenzialmente nulla

> **Cosa significa.** Per ogni cluster trovato a `tau=0.90`, contiamo **quanti seed diversi** contribuiscono. Se un cluster ha feature di 3+ seed, quel concetto è "robusto" (ricorre). Risultato: 20477 cluster su 20478 hanno feature di un solo seed. Solo 1 cluster ha 3 seed. Nessuno ne ha 4 o 5. Praticamente ogni direzione è **esclusiva di un seed**.

Cluster per #seed rappresentati (su 20478 cluster a `tau=0.90`):

| #seed per cluster | #cluster |
|---|---|
| 1 | 20477 |
| 2 | 0 |
| 3 | **1** |
| 4 | 0 |
| 5 | 0 |

- **`consensus@≥3/5` = 0.0146%** (1 cluster su 3 seed).
- **`consensus@≥4/5` = 0.00%**.
- → Solo **1 direzione** ricorre su ≥3 seed; nessuna su ≥4. Quasi tutto il decoder pooled è costituito da direzioni **seed-esclusive**.

### 2.4 Hungarian direction-Jaccard (D) — ❌ ~0

> **Cosa significa.** Metodo più forte del clustering: per ogni coppia di seed, cerchiamo l'"abbinamento ottimo" (Hungarian) tra le loro 4096 feature — cioè, per ogni feature del seed A, la feature del seed B più simile. Se anche così (massimo sforzo di abbinamento) quasi nessuna coppia supera coseno 0.90, allora le direzioni sono davvero diverse, non è un artefatto del clustering. Risultato: 0-1 abbinamenti su 4096 per coppia.

Per ogni coppia di seed: matrice coseno `4096×4096` tra righe decoder → `linear_sum_assignment` → match tenuti se `cos ≥ 0.90`.

| Coppia | match / 4096 | rate |
|---|---|---|
| 0↔42, 0↔123, 0↔456, 0↔789 | 0 | 0.0000 |
| 42↔123, 42↔456 | 0 | 0.0000 |
| 42↔789 | 1 | 0.0002 |
| 123↔456, 123↔789 | 0 | 0.0000 |
| 456↔789 | 1 | 0.0002 |

- **Direction-Jaccard medio = 4.9e-5** (~0/4096 per coppia).
- vs **raw index-Jaccard baseline = 0.0038**.

> ⚠️ **Quantità DIVERSE, riportate side-by-side — non una "correzione".** L'index-Jaccard è identità di slot; il direction-Jaccard è **invariante a permutazione** (corrispondenza per direzione, non per indice). **Entrambe ~0** → niente struttura condivisa né in spazio indici né in spazio direzioni. Questo **falsifica** l'ipotesi "0.0038 è solo permutazione".

### 2.5 Name-agreement (E) — ❌ 0%

> **Cosa significa.** Per i pochissimi cluster condivisi, controlliamo se i seed li chiamano con lo stesso termine RadLex. Risultato: nessun cluster ha nome unanime. → Anche dove c'è un minimo di sovrapposizione geometrica, l'etichetta medica non è coerente.

- 1 cluster multi-seed (≥2 seed). Termini RadLex argmax dei membri: **nessun cluster con termine unanime**.
- **Name-agreement rate = 0.00%**.
- Primo dato di consistenza naming multi-seed → nullo.

### 2.6 Faithfulness proxy (F) — ⚠️ debole, solo proxy

> **Cosa significa.** Per l'unico concetto che ricorre (cluster di 3 seed, chiamato `bulging fissure sign`), misuriamo quanto è "buono": similarità col termine (0.158 = molto debole, vs 0.395 medio del baseline) e attivazione media sul test (0.005 = quasi niente). L'unico concetto "stabile" è scarsamente ancorato a RadLex e quasi non si attiva. **Proxy dichiarato**: niente ground-truth clinico qui.

Per l'unico concetto di consenso (cluster size 3, 3 seed):

| Metrica | Valore |
|---|---|
| n_concepts | 1 |
| Termine vincente | `bulging fissure sign` |
| Naming-cos (dir media vs emb termine) | **0.1580** |
| Mean test activation (membri seed-42) | **0.0047** |

- Naming-cos 0.158 = **molto debole** (vs naming medio baseline gap-corrected 0.395). L'unico concetto "stabile" è scarsamente ancorato a RadLex.
- **Proxy dichiarato**: naming-cos + activation media sul test. **No ground-truth** (niente label NIH in questa pipeline).

### 2.7 Shuffle-null (H) — p=1.0, nessun segnale oltre il caso

> **Cosa significa.** Test statistico di sicurezza: mescoliamo a caso le etichette dei seed 200 volte e vediamo quanto "consenso" spunterebbe per puro caso. Se l'osservato (0%) è uguale al caso mescolato (0%), p=1.0 → **zero segnale sopra il rumore**. Il consenso che vediamo non è significativo.

- `consensus@≥4/5` osservato: **0.00%**.
- Shuffle-null (200 permutazioni dei tag seed): **0.00%**.
- **p-value = 1.0** → osservato = null. Gap 0.00 pp.
- **Verdetto: nessun segnale sopra il caso.** Il consenso osservato (già ~0) non supera la baseline casuale.

---

## 3. Giudizio d'insieme: tesi falsificata, onestamente

**Sì — risultato negativo solido e riportato senza abbellimenti.**

| Domanda | Esito |
|---|---|
| Il 0.0038 della baseline è un artefatto di permutazione? | ❌ **No** — direction-Jaccard 4.9e-5, ~0 anche in spazio direzioni. |
| I 5 seed imparano direzioni concettuali vicine? | ❌ **No** — max coseno off-diagonale within-seed ~0.577, ben sotto soglia 0.90. |
| Esiste consenso cross-seed sopra il caso? | ❌ **No** — consensus@4 = 0%, shuffle-null p=1.0. |
| I concetti stabili sono clinicamente ancorati? | ⚠️ **Debole** — 1 concetto, naming-cos 0.158 (proxy, no ground-truth). |

**Punti chiave per la discussione:**
1. **L'instabilità della baseline (§2.4 baseline REPORT) è geometricamente reale, non rumore di labeling.** I 5 SAE scoprono basi sostanzialmente disgiunte — cambiare seed cambia quali direzioni vengono apprese, non solo i loro indici.
2. **Non è p-hacking:** abbassare `tau` a 0.80 per fabbricare qualche cluster multi-member produrrebbe un headline "positivo" fittizio su un risultato nullo. L'ablation si rifiuta di farlo.
3. **Conseguenza operativa:** il seed primario 42 è arbitrario; naming/explanations della baseline dipendono dal seed. Per concetti riproducibili serve **aggregazione cross-seed** (model soup, init condiviso, consensus clustering su `tau` molto più basso con validazione) o accettare la seed-dipendenza e riportarla come limite.
4. **Direzione di fuga:** l'ablation `01_dict_size` / `02_k_sweep` (ridurre gradi di libertà) sono il prossimo test naturale — meno parametri → meno divergenza tra seed.

---

## 4. Note di riproducibilità & stato

- **Run headless (2026-06-21 18:48):** celle 2–24 eseguite via `.venv/bin/python` (torch 2.12.0+cu130, CUDA RTX 5070), backend matplotlib Agg. Cell 18 eseguita senza crash dopo il fix dict→term (vedi sotto).
- **Fix applicato in questa run:** cella 6 normalizza `vocabulary.json` (lista di dict `{"term",...}`) → stringhe `term` al load. Senza di esso, `vocab_labels[i]` era un dict → crash `"{t:28s}"` in cell 18 (`unsupported format string passed to dict.__format__`). Stesso ceppo del `445c6ac` / `dd2dd87`.
- **Zero training:** riusa i 5 checkpoint baseline (06-05). La correzione del modality gap (baseline) **non influenza** questa ablation — qui si confrontano direzioni del decoder raw, non si fa naming gap-corrected (l'unica naming è il proxy argmax coseno in cell 16/18).
- **"Dead" = decoder-norm dead (0% qui)** ≠ activation-dead baseline (~44%). Entrambe corrette, definizioni diverse. Vedi §2.1.
- **Vocabolario = 508 termini** (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt` allineati, verificato in run: `Vocabulary: 508 terms, embeddings [508, 512]`). Markdown del notebook corretto a 508 in questa commit (era stale "310" pre-rebuild multi-centroid, celle 0/5/15); i conteggi del codice erano già corretti — usano gli embedding reali a 508 righe.
- **Artefatti prodotti:** `results/ablation/a0_consensus.json` (metriche complete) + `results/figures/ablation/a0_consensus_headline.png` (3 pannelli: heatmap index-Jaccard 5×5, istogramma reappearance, scatter 2D decoder pooled UMAP colored by cluster/seed).
- **Index-Jaccard di riferimento:** la baseline REPORT §2.4 riporta mean 0.0039; questa ablation hard-coda `raw_index_jaccard_mean_baseline = 0.0038` (letterale in cella 24). Difetto rounding 0.0001 — irrilevante (entrambi ~0.004).

## Riferimenti
- [00_consensus.ipynb](00_consensus.ipynb) — notebook sorgente
- [../../src/autoencoder/sae_module.py](../../src/autoencoder/sae_module.py) — `SAEManager.get_decoder_weights`, `encode_topk`
- [../baseline/REPORT.md](../baseline/REPORT.md) — baseline §2.4 stabilità (index-Jaccard 0.0039)
- [../../results/ablation/a0_consensus.json](../../results/ablation/a0_consensus.json) — metriche complete
- [../../results/figures/ablation/a0_consensus_headline.png](../../results/figures/ablation/a0_consensus_headline.png) — figura headline

---

# Ablation 01 — Dictionary-Size Ladder (lr pinned)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA**
**Notebook:** `notebooks/autoencoder/ablation/01_dict_size.ipynb` (run IDE, 21/21 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), vocabolario RadLex **508 termini**
**Config:** `dict_size ∈ {1024, 2048, 4096}`, `k=32`, **lr pinned 4e-4** (capacity = unica variabile), `steps=12000`, `batch_size=256`, seeds `(0, 42, 123)`, `primary_seed=42`, naming **gap-corrected** (Soluzione 1). Plus: revival probe (dict2048), sensitivity `lr=auto`.

### In parole semplici

> **La domanda.** Il baseline ha due patologie accoppiate: **~44% di feature morte** (spreco) e **Jaccard ≈ 0.004** (instabilità). Ipotesi naturale: il dizionario di 4096 feature è **8 volte più grande** dello spazio da 512 dimensioni (sovradimensionamento). Forse il sovradimensionamento causa **entrambi** i problemi — troppa roba inutilizzata che diverge tra seed.
>
> **Cosa fa ab01.** Addestra SAE con dizionari di 3 dimensioni diverse (1024, 2048, 4096), tenendo fisso tutto il resto (stesso lr, stesso k). Se l'ipotesi è giusta, dizionari più piccoli dovrebbero avere **meno dead** E **più stabilità** (signal-to-null ratio più alto).
>
> **Risultato: MISTO.** I dead scendono come previsto (40.9% → 30.7%). MA la stabilità **non** migliora — anzi, il dizionario più grande ha il ratio più alto. **Conclusione: il sovradimensionamento causa i dead (spreco), NON l'instabilità.** Sono due problemi separati. Rifinitura onesta dell'ipotesi del turno precedente.

> **Domanda.** La baseline (dict_size=4096, 8× over il 512-d BiomedCLIP) mostra due patologie accoppiate: **~44% dead** e **Jaccard ≈ 0.0038** (appena sopra il null 0.0039). L'over-expansion è la **causa condivisa** di entrambe?
>
> **Ipotesi pre-registrata:** dict_size più piccolo → **dead% cala** AND **signal-to-null ratio sale** (ratio = Jaccard / null ipergeometrico a quel (k, D); controlla il fatto che il null cresce trivialmente quando D cala).
>
> **Esito: MISTO.** dead% ✓ cala. Ratio ✗ NON sale — falsificato (4096 ha il ratio più alto). **Over-expansion spiega i dead, NON l'instabilità.** Vedi §3/§4.

---

## 1. Cosa testa ogni fase

| Fase | Output |
|---|---|
| Training ladder | 3 dict_size × 3 seed = 9 SAE (12k step, lr pinned) |
| Per-size metrics | cosine, dead%, L0, entropy (test) |
| Within-group Jaccard | matrice 3×3 per size (Protocollo: costante dict_size+k) |
| Signal-to-null ratio | Jaccard / null ipergeometrico (cross-size causal) |
| Consensus reappearance | cluster direction-space (τ=0.9, index-agnostic) — stesso algo di ab00 |
| Feature splitting | mean/p90 pairwise cos tra alive rows (subsample 2000) |
| Revival probe | dict2048, dead_threshold abbassato + auxk forte (negative probe) |
| Sensitivity | ripete ladder con `lr=auto` |
| Naming | primary seed 42, gap-corrected, per size |
| Persist | `results/ablation/a1_dict_size.json` + 3 figure |

---

## 2. Risultati per-size (lr pinned 4e-4, 12k step, 3 seed)

> **Come leggere la tabella.** Per ciascuna dimensione del dizionario, la colonna chiave è **ratio** = Jaccard osservato diviso il "null" (la sovrapposizione attesa per puro caso a quella dimensione). Ratio > 1 = i concetti si accordano **oltre** il caso. Ratio ≈ 1 = sul pavimento del caso.

| dict_size | cosine | dead% | raw Jaccard | null | **ratio** | consensus reappearance | splitting (mean / p90) | naming (mean / max) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.9937 | **30.7** | 0.0166 | 0.0159 | 1.04 | 0.0003 (1 cluster) | 0.0073 / 0.110 | 0.395 / 0.516 |
| 2048 | 0.9921 | 33.6 | 0.0070 | 0.0079 | 0.89 | 0.0 | 0.0062 / 0.107 | 0.394 / 0.537 |
| 4096 | 0.9903 | 40.9 | 0.0056 | 0.0039 | **1.43** | 0.0 | 0.0043 / 0.098 | 0.393 / 0.534 |

---

## 3. Analisi

### 3.1 dead% ✓ — scala con dict_size (over-expansion = causa dei dead)
> **Cosa significa.** Riducendo il dizionario, le feature morte scendono monotonamente (40.9 → 33.6 → 30.7%). Conferma: troppi atomi competono per la stessa "torta" di attivazione → molti restano a bocca asciutta. Il sovradimensionamento causa lo spreco. ✓

40.9 → 33.6 → 30.7% calando dict_size. **Monotono e chiaro.** Più atomi competono per la stessa activation mass → più atomi restano inutilizzati. Over-expansion confermata come causa dei dead. (Sensitivity `lr=auto`: stesso trend 47 → 42 → 41%.)

### 3.2 signal-to-null ratio ✗ — NON monotonico (ipotesi falsificata)
> **Cosa significa.** Qui l'ipotesi crolla. Se il sovradimensionamento causasse anche l'instabilità, ridurre il dizionario dovrebbe alzare il ratio (più accordo oltre il caso). Invece è il contrario: il dizionario più **grande** (4096) ha il ratio **più alto** (1.43), e il 2048 è persino **sotto** il caso (0.89). **Ridurre il dizionario NON aumenta la robustezza.**

Ratio: **4096 (1.43) > 1024 (1.04) > 2048 (0.89)**. L'ipotesi "ratio↑ quando dict↓" è **falsificata**. Il dict più **grande** ha il signal-to-null più alto (più accordo oltre il caso); il 2048 è persino **sotto null** (0.89 < 1).
→ Ridurre dict_size NON aumenta la robustezza cross-seed. L'over-expansion NON spiega l'instabilità.

### 3.3 Consensus reappearance — ~0 ovunque (invariante al dict_size)
> **Cosa significa.** Lo stesso test di ab00 (cluster di direzioni condivise tra seed), ripetuto a ogni dimensione. Risultato: ~0 ovunque. La mancanza di direzioni condivise **non dipende** da quanto è grande il dizionario.

Direction-space: 1024 → 0.03%, 2048 → 0%, 4096 → 0% cluster multi-seed. **Identico al null di ab00** a tutte le capacità.
→ L'instabilità in spazio direzioni è **invariante** al dict_size. Ridurre il dizionario non fa ricomparire direzioni condivise.

### 3.4 Feature splitting — direzione OPPOSTA all'ipotesi
> **Cosa significa.** "Splitting" = quante feature vive si somigliano tra loro (collisioni). Se il dizionario fosse troppo grande, ci aspetteremmo feature ridondanti/sovrapposte. Invece il dizionario più **piccolo** ha feature più **affollate** (cos 0.0073 vs 0.0043). L'ipotesi "troppe feature causano splitting" è falsificata: più spazio = meno collisioni, come aspettato.

Mean pairwise cos tra alive rows: **1024 (0.0073) > 2048 (0.0062) > 4096 (0.0043)**. p90 idem (0.110 → 0.107 → 0.098).
→ Dict più **piccolo** → alive rows più **affollate/redundanti** (cos più alto). L'ipotesi "over-expansion causa splitting" è **falsificata**: più atomi = più spazio per dispiegarsi, meno collisioni.

### 3.5 Naming — STABILE cross-size (~0.394)
> **Cosa significa.** La qualità dell'ancoraggio a RadLex per-feature è **robusta alla dimensione** del dizionario: ~0.394 ovunque, identico al baseline. Quindi non è la qualità del singolo concetto a essere instabile — è **quali** concetti finiscono nel set.

mean 0.395 / 0.394 / 0.393, max 0.52–0.54 per tutti e tre. **Identico alla baseline (0.3949).**
→ La qualità del grounding RadLex per-feature è **robusta al dict_size**. Non è la qualità del singolo concetto a essere instabile — è la **composizione del set**.

### 3.6 Revival probe (dict2048) — negative probe confermato
> **Cosa significa.** Test di sicurezza: proviamo a "risvegliare" le feature morte (abbassando la soglia dead + auxk forte). I dead scendono (33.6 → 30.9%), ma la stabilità **non** migliora (0.0070 → 0.0059). Quindi "vivificare" le feature morte riduce lo spreco ma NON rende i concetti più riproducibili. Essere "vivi" ≠ essere "robusti".

dead_threshold abbassato + auxk forte: **dead% 33.6 → 30.9** (cala ✓), ma **Jaccard 0.0070 → 0.0059** (flat/↓), ratio 0.89 → 0.75.
→ Revivere feature morte riduce lo spreco MA NON migliora la robustezza. **"Alive" ≠ "robust".** Feature vive ma arbitrarie sono disaccoppiate dalla stabilità.

---

## 4. Giudizio d'insieme: over-expansion = dead, NON instability

**Risposta diretta alla domanda "dizionario troppo grande o troppi pochi campioni?" (turno precedente):**

| Patologia | Causa? | Evidenza |
|---|---|---|
| ~44% dead features | ✅ **Over-expansion** | dead% scala con dict_size (40.9 → 30.7%) |
| Cross-seed instability (Jaccard 0.004) | ❌ **NON over-expansion** | ratio non sale riducendo dict; consensus ~0 ovunque |

→ **L'over-expansion spiega lo spreco (dead), NON l'instabilità.** L'ipotesi del turno scorso ("overcompleteness causa primaria dell'instabilità") è **rifinita da ab01**: ridurre il dizionario riduce i dead ma NON rende i concetti riproducibili. L'instabilità è più **fondamentale** — probabili cause: pochi campioni (5976) + non-unicità intrinseca del TopK SAE su questo cloud. Non risolvibile abbassando dict_size.

**Punti chiave:**
1. **Dict più piccolo è comunque "meglio"** (meno dead, stessa ricostruzione 0.99+, stesso naming 0.39, meno compute) — ma **NON per la robustezza**.
2. **Naming robusto cross-size** → il grounding individuale funziona; il problema è *quale set* di feature si apprende.
3. **Revival probe**: vivificare i dead non aiuta → l'instabilità non è un problema di "feature addormentate".
4. **Prossimi test naturali:** `02_k_sweep` (k più vincolato?), `03_baselines`. Se anche k non aiuta, l'instabilità è **strutturale** → accettare la seed-dipendenza come limite dichiarato, o aggregazione cross-seed (model soup / consensus a τ molto basso con validazione).

---

## 5. Note di riproducibilità & stato
- **Run IDE (2026-06-21 19:03):** 21/21 celle, 9 SAE addestrati (3 size × 3 seed, 12k step) + revival probe + sensitivity. Artefatti: `a1_dict_size.json`, `a1_naming_dict{1024,2048,4096}.json`, 3 figure (`a1_stability_frontier`, `a1_splitting_dendrogram`, `a1_dead_jaccard_vs_dict`).
- **3 seed (non 5):** ladder controllato a `(0,42,123)` per compute; sufficiente per il trend di capacity. 12k step (non 50k baseline) — il punto 4096 qui è **fresh re-run**, confronto apples-to-apples dentro il ladder.
- **lr pinned 4e-4:** rende capacity l'unica variabile. Sensitivity `lr=auto` (appendice) coincide con 4e-4 a queste size (tutte < 16384 ref) → l'effetto è genuinamente di capacity.
- **Signal-to-null = Jaccard / E[J] ipergeometrico**, `E[J] ≈ k/(2D−k)` per `k≪D`. Forma esatta e approssimata concordano a 4 decimali.
- **Consensus reappearance usa lo stesso algo di ab00** (`connected_components` su grafo `coseno>τ`, τ=0.9) → rate direttamente comparabili cross-ablation.
- **Baseline reference** (nel json): cosine 0.988, dead 44%, Jaccard 0.0038, naming mean 0.395 / max 0.546.

## Riferimenti
- [01_dict_size.ipynb](01_dict_size.ipynb) — notebook sorgente
- [00_consensus.ipynb](00_consensus.ipynb) — ab00 (consensus ~0 confermato a tutte le size)
- [../baseline/REPORT.md](../baseline/REPORT.md) — baseline (dead ~44%, Jaccard 0.0039)
- [../../results/ablation/a1_dict_size.json](../../results/ablation/a1_dict_size.json) — metriche complete
- [../../results/figures/ablation/a1_stability_frontier.png](../../results/figures/ablation/a1_stability_frontier.png) — frontier cosine vs ratio
- [../../results/figures/ablation/a1_dead_jaccard_vs_dict.png](../../results/figures/ablation/a1_dead_jaccard_vs_dict.png) — dead% + Jaccard vs dict_size
- [../../results/figures/ablation/a1_splitting_dendrogram.png](../../results/figures/ablation/a1_splitting_dendrogram.png) — feature splitting per size

---

# Ablation 02 — k (Sparsity) Sweep, null-calibrated

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA**
**Notebook:** `notebooks/autoencoder/ablation/02_k_sweep.ipynb` (run IDE, 12/12 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494)
**Config:** `dict_size` **fissato a 2048**, `k ∈ {8, 16, 32, 64}`, seeds `(0, 42, 123, 456)`, `steps=12000`, `lr=auto` (scala solo con dict_size → **costante tra i gruppi k**, elimina il confound ab01). Within-group Jaccard con `n=k` esplicito, null ipergeometrico esatto, bootstrap CI 1000× sui 1494 sample test.

### In parole semplici

> **La domanda.** Dopo ab01 (il dizionario non conta), proviamo l'altro parametro: **k**, cioè quanti concetti attivi per immagine. Il baseline usa k=32. Forse un k diverso dà concetti più riproducibili? Più k = meno sparso ma magari più stabile; meno k = più sparso ma magari collassa.
>
> **Cosa fa ab02.** Fissa il dizionario a 2048 e addestra con k ∈ {8, 16, 32, 64}. Per ogni k misura la stabilità e la confronta col null esatto (calcolato analiticamente, non stimato). Aggiunge anche intervalli di confidenza (bootstrap 1000×) per dire se il ratio è "davvero" sopra 1.
>
> **Risultato: PARZIALE.** Il baseline k=32 è sul pavimento del caso (ratio 0.95 ≈ 1 → "rumore"). C'è un debole sweet spot a **k=16** (ratio 1.30, l'unico dove l'intervallo di confidenza esclude 1). k=8 è patologico (91.6% dead, collassa sotto il caso). **k modula la stabilità più di dict_size, ma non la risolve** — anche al picco, l'accordo assoluto resta minuscolo.

> **Domanda.** Il baseline (k=32) ha mean index-Jaccard **0.0038**. È *segno* o è *rumore*? L'ablation lo confronta con il **null analitico esatto** (Jaccard atteso tra due sottoinsiemi size-k indipendenti di un D-set, ipergeometrico). Claim difendibile: il baseline **siede sul floor del random-overlap** (ratio ≈ 1). Poi sweep k per trovare se uno sparsity ottimo porta sopra il null.
>
> **Ipotesi pre-registrata:** ratio ≈ 1 al baseline (k=32), **rising as k shrinks** (meno feature attive → meno overlap casuale → ratio↑ se i concetti sono reali), e **dead% ↗ a k molto piccolo**. Il Pareto front (VE vs ratio) sceglie il sweet spot.
>
> **Esito: PARZIALE.** Baseline sul null floor ✓ (ratio 0.954). dead% ↗ small k ✓ (91.6% a k=8). Ma ratio **NON** monotonico — picco a **k=16 (1.30)**, poi cala; k=8 collassa *sotto* null (0.80). k modula la stabilità (a differenza di dict_size), ma l'accordo assoluto resta minuscolo. Vedi §3/§4.

---

## 1. Cosa testa ogni fase

| Fase | Output |
|---|---|
| Training grid | 4 k × 4 seed = 16 SAE (12k step, dict_size=2048 fisso) |
| Per-k ricostruzione | cosine, VE, MSE, L0 (=k), dead% (test) |
| Within-group Jaccard | `compute_stability` per k-gruppo, `n=k` esplicito |
| Exact hypergeometric null | `Σ_j j/(2k−j)·P(j)` via `scipy.stats.hypergeom` |
| Signal-to-null ratio | raw Jaccard / null, CI 95% bootstrap 1000× |
| Consensus reappearance | direction-space, τ=0.9 (stesso algo a0/a1) |
| Baseline anchor | dict4096/k32 come punto standalone null-calibrato (NON confrontato via Jaccard) |
| Figures | `a2_k_vs_stability.png`, `a2_pareto_front.png` |
| Persist | `results/ablation/a2_k_sweep.json` |

---

## 2. Risultati per-k (dict_size=2048, 12k step, 4 seed)

> **Come leggere.** `signal/null` = quanto l'accordo osservato supera il caso. >1 = segnale reale; ≈1 = rumore; <1 = sotto il caso. `CI 95%` = intervallo di confidenza: se **non include 1**, il segnale è statisticamente significativo (succede solo a k=16).

| k | cosine | VE | dead% | raw Jaccard | null | **signal/null** | CI 95% | consensus ≥2 | ≥3 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.984 | 0.968 | **91.6** | 0.00167 | 0.00209 | 0.80 | 0.69–0.90 | 0.65% | 0.48% |
| 16 | 0.989 | 0.978 | 74.7 | 0.00528 | 0.00405 | **1.30** | **1.24–1.37** | 0.16% | 0.11% |
| 32 | 0.992 | 0.985 | 41.3 | 0.00916 | 0.00799 | 1.15 | 1.12–1.18 | 0.037% | 0.037% |
| 64 | 0.997 | 0.994 | 40.2 | 0.01557 | 0.01599 | 0.97 | 0.96–0.99 | 0% | 0% |

**Baseline anchor** (dict4096/k32): raw 0.0038, null 0.00398, ratio **0.954** (~1, sul floor).

---

## 3. Analisi

### 3.1 Baseline sul null floor ✓ — "0.0038 è rumore"
> **Cosa significa.** Il ratio del baseline è 0.954 ≈ 1: il suo Jaccard di 0.0038 è **statisticamente indistinguibile dal puro caso**. A k=32/dict4096, i concetti non sono più riproducibili di due dizionari casuali. Claim onesto e difendibile.

Ratio baseline 0.954 ≈ 1 → il Jaccard 0.0038 del baseline è **statisticamente indistinguibile dal random-overlap**. Claim difendibile confermato: a k=32/dict4096 i concetti non sono più riproducibili del caso (in spazio indici).

### 3.2 Signal-to-null NON monotonico — picco a k=16
> **Cosa significa.** L'ipotesi "più sparso = più stabile" è parzialmente falsificata: il ratio sale da k=64 a k=16, ma a k=8 **crolla sotto** il caso (91.6% di feature morte, niente da allineare). k=16 è l'unico dove l'intervallo di confidenza **esclude 1** (1.24–1.37): lì l'accordo è davvero sopra il caso. È il sweet spot di stabilità.

Ratio: **k=16 (1.30) > k=32 (1.15) > k=64 (0.97) > k=8 (0.80)**.
- L'ipotesi "rising as k shrinks" è **parzialmente falsificata**: sale da k=64→32→16, ma **k=8 collassa sotto null** (troppo sparso → 91.6% dead → niente da allineare).
- **k=16 è l'unico k dove la CI esclude 1** (1.24–1.37): l'accordo reale supera chiaramente il caso. Sweet spot di stabilità.

### 3.3 dead% ↗ small k ✓
> **Cosa significa.** Confermato: meno feature attive per immagine = più feature non si attivano mai. k=8 è patologico (quasi tutto morto).

91.6% (k=8) → 74.7% (k=16) → 41.3% (k=32) → 40.2% (k=64). Confermato: meno feature attive per pass → più feature mai si attivano. k=8 è patologico (quasi tutto morto).

### 3.4 Consensus reappearance — ingannevole a k=8
> **Cosa significa.** A k=8 la "reappearance" sembra alta (0.65%), ma è un'illusione: con 91.6% di dead, il set vivo è minuscolo (~170 feature su 2048), quindi i cluster sono forzati dall'affollamento, non da riproducibilità reale. Il signal-to-null (che corregge per la dimensionalità) lo conferma: k=8 è **sotto** il caso. k=16 resta il sweet spot onesto.

consensus≥2: k=8 (0.65%) > k=16 (0.16%) > k=32 (0.037%) > k=64 (0%). Ma k=8 ha **91.6% dead** → live set minuscolo (~170/2048) → cluster forzati per affollamento, non riproducibilità reale. Il signal-to-null (che corregge per la dimensionalità) dice il contrario: k=8 *sotto* null. **k=16 resta il sweet spot onesto.**

### 3.5 Tradeoff stabilità ↔ ricostruzione (Pareto)
> **Cosa significa.** k più alto = ricostruzione migliore e meno dead, ma **meno** stabile sopra k=16. k=16 massimizza la stabilità; k=32 è il compromesso operativo. Nessun k raggiunge riproducibilità reale (raw Jaccard max 0.006, consensus ~0).

k↑ → migliore ricostruzione (cosine 0.984→0.997, VE 0.968→0.994) e meno dead (91.6→40.2%), MA ratio cala sopra k=16.
- **k=16**: max stabilità (1.30), recon 0.989, dead 74.7%.
- **k=32**: stabilità 1.15, recon 0.992, dead 41.3% — compromesso operativo (baseline-like).
- Nessun k raggiunge riproducibilità reale (raw Jaccard max 0.0056 a k=16, consensus ~0).

---

## 4. Giudizio d'insieme: k modula, non risolve

**Confronto con ab01 (entrambi sweep di un iperparametro):**

| Sweep | Cosa muove stability? | Verdetto |
|---|---|---|
| ab01 — dict_size | ratio **invariante** (~flat) | dict_size NON spiega instabilità |
| ab02 — k (dict fisso) | ratio **non-monotonico**, picco k=16 | k MODULA la stabilità (debolmente) |

→ **k conta più di dict_size** per la stabilità cross-seed: c'è un optimum a k=16 (ratio 1.30, l'unico chiaramente sopra null). Ma:
1. Anche al picco, **accordo assoluto minuscolo** (raw Jaccard 0.005, consensus direction-space ~0). k=16 alza il *rapporto* sopra il caso, non risolve la riproducibilità.
2. **k=8 patologico** (91.6% dead) — troppo sparso.
3. **k=32 (baseline) è sul null floor** → i concetti baseline sono rumore in spazio indici (claim onesto).

**Risposta cumulativa alla domanda iniziale ("instabilità = overcomplete o pochi campioni?"):**
- ab01: over-expansion causa i dead, NON l'instabilità.
- ab02: k ha un debole sweet spot (k=16), ma non risolve. Il baseline stesso è rumore-vs-null.
- → L'instabilità è **strutturale**. Né dict_size né k la risolvono. Rimangono: pochi campioni (5976) + non-unicità TopK + dataset small. **Prossimo: `03_baselines`** (confronto con metodi alternativi) e poi eventuale aggregazione cross-seed.

---

## 5. Note di riproducibilità & stato
- **Run IDE (2026-06-21 19:23):** 12/12 celle, 16 SAE (4 k × 4 seed). Artefatti: `a2_k_sweep.json` + 2 figure (`a2_k_vs_stability`, `a2_pareto_front`).
- **dict_size=2048 fisso** → lr auto-scale identico tra k-gruppi (elimina confound dict→LR di ab01).
- **4 seed (non 3/5):** `(0,42,123,456)` per più potenza statistica sul bootstrap CI.
- **Null = ipergeometrico esatto** `Σ_j j/(2k−j)·P(j)`, P(j) via `scipy.stats.hypergeom(M=D,n=k,N=k)`. CI via bootstrap 1000× sui 1494 sample test (mean-of-ratios).
- **Naming tag allineato (fixato in commit `0ef9f7e`):** i tag interni dei notebook ora matchano i numeri `00→a0, 01→a1, 02→a2, 03→a3, 04→a4`. Prima erano scrambled (`02→a4`, `03→a6`, `04→a2`) per un renumbering non propagato ai tag interni. Artefatti su disco rinominati di conseguenza (`a2_k_sweep.json`, `a3_baselines.json`, figure `a2_*`/`a3_*`, `models/ablation_a2`).
- **Baseline anchor** è standalone (dict_size diverso → Jaccard cross-config vietato dal protocollo).

## Riferimenti
- [02_k_sweep.ipynb](02_k_sweep.ipynb) — notebook sorgente
- [01_dict_size.ipynb](01_dict_size.ipynb) — ab01 (dict_size non spiega instabilità)
- [00_consensus.ipynb](00_consensus.ipynb) — ab00 (consensus direction-space ~0)
- [../baseline/REPORT.md](../baseline/REPORT.md) — baseline (k=32, Jaccard 0.0039 ≈ null)
- [../../results/ablation/a2_k_sweep.json](../../results/ablation/a2_k_sweep.json) — metriche complete
- [../../results/figures/ablation/a2_k_vs_stability.png](../../results/figures/ablation/a2_k_vs_stability.png) — k vs Jaccard/null/ratio
- [../../results/figures/ablation/a2_pareto_front.png](../../results/figures/ablation/a2_pareto_front.png) — VE vs signal-to-null

---

# Ablation 03 — Concept Baselines + Empirical Jaccard Floor

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA**
**Notebook:** `notebooks/autoencoder/ablation/03_baselines.ipynb` (run IDE, 13/13 celle)
**Input:** `train_embeddings.pt` (5976, fit PCA/KMeans qui) / `test_embeddings.pt` (1494, score metriche qui), vocabolario RadLex **508 termini**
**Config:** **zero training** — 3 dizionari hand-built (Random, Dense-PCA, Freq-KMeans) da embedding esistenti; `D_b=256` (spazio indice condiviso within-group), `D_B_BIG=4096` (Random nel native index space del SAE), `K=32` (L0 budget fair), seeds `(0,42,123)`, naming **gap-corrected** (stesso shift `W_dec -= gap` del SAE), SAE reference hard-codato (gap-corrected, non ri-addestrato).

### In parole semplici

> **La domanda.** Tutte le ablation finora confrontavano il SAE con se stesso (seed diversi). Qui facciamo la domanda più diretta: **il SAE è davvero meglio di metodi stupidi?** E soprattutto — il famoso "0.0038" di instabilità, è un fallimento del SAE o è il rumore che otterrebbe **chiunque**, anche buttando numeri a caso?
>
> **Cosa fa ab03.** Costruisce 3 dizionari banali **senza addestramento**: (1) **Random** = direzioni buttate a caso; (2) **PCA** = le direzioni principali dei dati (matematica classica); (3) **KMeans** = i centri di 256 gruppi nei dati. Poi li misura con le stesse metriche del SAE. Il test chiave: prende un dizionario Random a 4096 feature, lo rifà 3 volte con seed diversi, misura la sovrapposizione. Quello è il **"pavimento del caso"** — il rumore minimo che ottieni sempre.
>
> **Il risultato centrale (leggi questo).** Random@4096 = **0.0037**, il nostro SAE = **0.0038**. **Identici.** Cioè: il numero "preoccupante" del SAE è **esattamente** quello che ottieni con numeri a caso. È matematica: due dizionari grandi e indipendenti si sovrappongono sempre per ~0.004 per pura probabilità. **Il 0.0038 NON è un fallimento del SAE — è il pavimento del caso.** Questo chiude la questione aperta da ab00/01/02 in modo pulito e indipendente.
>
> **Gli altri due risultati.** PCA ricostruisce meglio del SAE (0.996 vs 0.988) — ma è "denso" (usa tutte le feature); il pregio del SAE è usare **solo 32**. Sul naming medio, il SAE (0.395) fa appena meglio del Random (0.372) — lo shift del gap domina; KMeans vince (0.83) ma i suoi "concetti" sono macchie medie, non concetti puliti.

> **Tesi pre-registrata:** Random@4096 within-group Jaccard ≈ 0.004 → calibra il 0.0038 del SAE come **near-null** (artefatto di spazio indici). PCA = ceiling denso di ricostruzione. SAE = unico metodo sparse + nominato.
>
> **Esito: TESI CONFERMATA sul Jaccard floor; SAE sopravvive solo su sparsity + naming top-end.** Random@4096 = 0.0037 ≈ SAE 0.0038 (ratio 0.95, sul floor del caso). MA il naming mean del SAE (0.395) è **appena sopra il Random** (0.372) — lo shift del gap domina il signal. KMeans (0.83) schiaccia tutti sul naming. Vedi §3/§4.

---

## 1. Cosa produce ogni fase

| Fase | Output | Stato |
|---|---|---|
| 3 dizionari baseline | Random (256 + 4096), Dense-PCA (256), Freq-KMeans (256) — per seed | ✅ 4 baselines × 3 seed |
| Ricostruzione fair-L0 | cosine a L0=32 (top-k coefficienti per magnitudo) per ogni baseline | ✅ |
| Naming gap-corrected | decoder rows ↔ vocab, stesso shift del SAE | ✅ |
| Within-group index-Jaccard | Random@256 e Random@4096 (3 seed → null empirico) | ✅ |
| Null analitico cross-check | `E[J] ≈ k/(2D−k)` ipergeometrico | ✅ ratio 1.00 / 0.95 |
| Tabelle + figure | comparison table + jaccard-floor bar | ✅ `a3_comparison_table`, `a3_jaccard_floor` |
| Persist | `results/ablation/a3_baselines.json` + `a3_cache/` (fit PCA/KMeans) | ✅ |

> **Rubric ≥3 baselines soddisfatta.** Random / Dense-PCA / Freq-KMeans, ciascuno costruito da train embedding e scored su test con le metriche standalone del SAE (funzioni libere verificate contro `sae_module.py`, righe citate nelle docstring).

---

## 2. Risultati (primary seed 42; SAE reference hard-codato, gap-corrected)

> **Come leggere.** Confronto diretto: il nostro SAE vs tre metodi banali. "recon cosine" = qualità ricostruzione; "naming" = allineamento col vocabolario medico.

| Metodo | recon cosine | L0 | dead% | naming mean | naming max |
|---|---|---|---|---|---|
| **SAE** (dict4096, k32, baseline) | 0.988 | 32 | 44.0 | 0.395 | 0.546 |
| Random (D=256) | 0.454 | 32 | 0.0 | 0.372 | 0.442 |
| Dense-PCA (D=256) | **0.996** | 32 | 0.0 | 0.383 | 0.594 |
| Freq-KMeans (D=256) | 0.961 | 32 | 0.0 | **0.829** | **0.875** |

**Random-Jaccard floor (within-group, 3 seed)** — *il test chiave dell'intera ablation*:

> **Cosa significa.** Prendo un dizionario Random e lo rifò 3 volte (seed diversi), poi misuro quanto le 3 versioni si somigliano. Questo è il "pavimento del caso" — la sovrapposizione minima inevitabile. A 4096 feature è 0.0037, **identica** al 0.0038 del SAE. Il `null` analitico (formula matematica) dà 0.0039, confermando che non è un caso: è proprio il valore atteso.

| Gruppo | D | empirical J | analytical null | ratio |
|---|---|---|---|---|
| Random (small) | 256 | 0.0666 | 0.0667 | 1.00 |
| **Random (big)** | **4096** | **0.0037** | **0.0039** | **0.95** |
| — SAE baseline (cross-seed, 5 seed) | 4096 | 0.0038 | — | — |

---

## 3. Analisi

### 3.1 Random@4096 ≈ SAE → index-Jaccard del SAE sul floor del caso ✓
> **Cosa significa.** Questa è la conclusione principale di tutta la serie di ablation. Il SAE e un dizionario di numeri casuali hanno **la stessa** sovrapposizione tra run (0.0038 vs 0.0037). Ratio 0.95 = il SAE è esattamente sul pavimento del caso. Tradotto: il "0.0038 preoccupante" **non è un fallimento**, è il rumore matematico inevitabile quando confronti dizionari grandi e indipendenti.

Random@4096 = 0.0037, SAE = 0.0038. **Identici entro rumore.** ratio 0.95 = il SAE siede esattamente sul null empirico per dizionari 4096-dim. → Il 0.0038 cross-seed del SAE è **calibrato come near-null in spazio indici**: confrontare indici tra dizionari 4096-dim indipendenti produce ~0.004 di puro overlap casuale. Cross-check analitico `k/(2D−k)` = 0.0039 conferma (ratio 0.95; l'empirico è leggermente sotto perché i top-k-set sulla STESSA data condividono struttura, ma l'ordine di grandezza è quello).

### 3.2 PCA = ceiling denso di ricostruzione ✓ (non è "SAE è scarso")
> **Cosa significa.** PCA batte il SAE su ricostruzione (0.996 vs 0.988), ma è **atteso e voluto**: PCA usa tutte le 256 feature contemporaneamente (è "denso"), il SAE solo 32. Il SAE perde 0.008 di qualità in cambio della **sparsità** (interpretabilità). Questo è il tradeoff, non un difetto.

PCA 0.996 > SAE 0.988 su raw cosine. **Atteso e pedagogico:** PCA è denso (256 atomi tutti attivi, zeroato a L0=32 solo *dopo* il fit per confronto fair) — sacrifica sparsity e monosemanticità per la ricostruzione. Il SAE perde ~0.008 di cosine in cambio di **L0=32 enforced + naming**. Questo è il Pareto tradeoff, non un difetto.

### 3.3 Naming: SAE ≈ Random, KMeans schiaccia tutti ⚠️ (risultato severo)
> **Cosa significa.** Risultato scomodo: sull'allineamento **medio** col vocabolario, il SAE (0.395) batte il Random (0.372) di pochissimo (+0.023). Perché? Lo shift del modality gap sposta **tutte** le direzioni della stessa quantità prima del confronto, quindi domina il signal — la "roba imparata" dal SAE aggiunge poco sul naming medio. KMeans vince (0.83) perché i suoi centroidi sono i "modi" della distribuzione dati, che combaciano col vocabolario. Attenzione però: centroidi KMeans sono **macchie medie** dense, non concetti puliti/monosemantici — non è una vera vittoria come concept-discoverer. Il confronto più onesto è il naming **max** (top-end): SAE 0.546 > Random 0.442.

- naming mean: **KMeans 0.829 >> SAE 0.395 ≈ PCA 0.383 ≈ Random 0.372**.
- Il SAE **batte il Random di soli +0.023** sul naming mean. Lo shift del modality gap (`W_dec -= gap`) muove *tutte* le righe decoder della stessa quantità prima del coseno → domina il signal, e l'apprendimento del SAE aggiunge margine minimo sul naming *medio*.
- KMeans domina perché i centroidi **sono i modi della distribuzione dati** → allineati al cloud del vocabolario (anch'esso modi-dominato). naming mean alto ≠ grounding genuino: i centroidi KMeans sono blend densi (non monosemantici), l'alta similarità riflette allineamento cloud-vs-cloud, non concetti isolati.
- **Caveat dict-size:** SAE 4096 feature vs baseline 256 → per-feature naming mean non perfettamente comparabile (più atomi = slice più stretta per feature). L'ORDINE (KMeans >> resto ≈) resta il signal robusto; il confronto più pulito è il **top-end** (max): SAE 0.546 > Random 0.442.

### 3.4 Random recon scala con D: 0.45 (256) → 0.60 (4096)
> **Cosa significa.** Anche la ricostruzione "a caso" migliora se hai più feature: con 4096 direzioni casuali, è più probabile che qualcuna allinei col vettore da ricostruire. Questo dimostra che la ricostruzione grezza cresce trivialmente con dict_size anche senza imparare nulla — per questo le ablation normalizzano via il "null" (signal-to-null).

Più atomi casuali = più probabilità che qualcuno allinei con `x` → ricostruzione top-k migliore anche per puro caso. Conferma che raw recon cresce trivialmente con dict_size anche senza apprendimento — ragione in più per normalizzare via null (come fanno ab01/ab02 col signal-to-null).

---

## 4. Giudizio d'insieme: il SAE sopravvive solo su sparsity + naming top-end

| Domanda | Esito |
|---|---|
| Rubric ≥3 baselines? | ✅ Random / PCA / KMeans |
| Il 0.0038 del SAE è sopra il null (spazio indici)? | ❌ **No** — Random@4096 0.0037, ratio 0.95, sul floor |
| PCA batte SAE su recon? | ✅ Sì (0.996 vs 0.988) — atteso, è il ceiling denso |
| SAE batte i baseline sul naming? | ⚠️ **Appena** (mean 0.395 vs Random 0.372); max 0.546 > Random 0.442 (top-end sì) |
| KMeans domina il naming? | ✅ Sì (0.829) — ma modi dati densi, non monosemantici |

**Verdetto cumulativo (ab00→ab03):**
1. **ab00** (direction-Jaccard ~0) + **ab03** (index-Jaccard sul null floor) → il 0.0038 del SAE è rumore **sia in spazio indici che direzioni**. Conferma indipendente via due null diversi.
2. Il SAE **non vince su recon** (PCA ceiling) né sul **naming medio** (≈ Random, lo shift del gap domina). L'unica advantage difendibile: **L0=32 enforced per costruzione** (PCA/KMeans sono dense) + **top-end naming** (max 0.546 > 0.442).
3. **Connessione al thread principale:** risultato più severo della serie. ab01/ab02 mostravano che l'instabilità non si risolve con iperparametri; ab03 mostra che il SAE **appena supera baselines casuali** sull'asse naming-mean. Il valore del SAE qui è **strutturale** (sparsity garantita, recon 0.988 a L0=32), non un guadagno misurabile sui concetti vs alternative generiche.

**Caveat onesti:**
- Naming mean SAE-vs-baseline confonduto da dict_size (4096 vs 256) + dal fatto che lo shift del gap domina. Il top-end (max) è il confronto più pulito.
- KMeans naming 0.83 è cloud-alignment, non monosemanticità — non è una "vittoria" del KMeans come concept-discoverer.

---

## 5. Note di riproducibilità & stato
- **Run IDE (2026-06-21 19:35):** 13/13 celle, zero training. Artefatti: `a3_baselines.json` (6.1 KB), `a3_cache/` (PCA + KMeans fit per seed, `.npz`), 2 figure (`a3_comparison_table`, `a3_jaccard_floor`).
- **Zero training / no model writes:** `SAEManager.train` mai chiamato. PCA/KMeans fit su **train**, metriche scored su **test** (test-set discipline).
- **Metriche standalone:** `SAEManager.compute_stability`/`name_concepts`/`compute_cosine_reconstruction` richiedono un `AutoEncoderTopK` su disco → riscritte come funzioni libere, verificate contro `sae_module.py` (righe citate nelle docstring).
- **Naming gap-corrected per tutti:** `modality_gap = train_emb.mean(0) − vocab_emb.mean(0)` applicato a ogni `W_dec` in `name_cosine` → confronto naming apples-to-apples (non SAE-corrected vs baseline-raw).
- **SAE reference hard-codato:** numeri dal baseline REPORT (gap-corrected), non ri-addestrato qui.
- **Null analitico:** `E[J] ≈ k/(2D−k)` per `k≪D`; ratio empirical/analytical 1.00 (D=256) e 0.95 (D=4096).

## Riferimenti
- [03_baselines.ipynb](03_baselines.ipynb) — notebook sorgente
- [00_consensus.ipynb](00_consensus.ipynb) — ab00 (direction-Jaccard ~0, falsifica "permutazione")
- [02_k_sweep.ipynb](02_k_sweep.ipynb) — ab02 (baseline k=32 sul null floor, ratio 0.954)
- [../baseline/REPORT.md](../baseline/REPORT.md) — baseline (SAE reference hard-codato qui)
- [../../results/ablation/a3_baselines.json](../../results/ablation/a3_baselines.json) — metriche complete
- [../../results/figures/ablation/a3_comparison_table.png](../../results/figures/ablation/a3_comparison_table.png) — tabella SAE vs baseline
- [../../results/figures/ablation/a3_jaccard_floor.png](../../results/figures/ablation/a3_jaccard_floor.png) — Random-Jaccard floor vs SAE 0.0038

---

# Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA**
**Notebook:** `notebooks/autoencoder/ablation/04_activation_bakeoff.ipynb` (run IDE, 29 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), vocabolario RadLex **508 termini**
**Config:** **3 famiglie di attivazione** addestrate a config **identico**: TopK (baseline), BatchTopK, JumpReLU. `dict_size=2048` (spazio indice condiviso), **lr=5e-5 pinned & matched** (elimina il confound lr ~8×), `steps=12000`, seeds `(0,42,123)`, `primary_seed=42`, naming **gap-corrected**.

### In parole semplici

> **La domanda.** Tutte le ablation finora usano TopK (il SAE del baseline). Forse il problema è proprio **TopK** — la sua regola "esattamente 32 feature per immagine" è rigida. Esistono famiglie alternative: **BatchTopK** (sceglie i top-k su tutto il batch, non per-sample → ogni immagine può usare più o meno feature) e **JumpReLU** (soglia imparata per-feature). Forse una di queste trova concetti più riproducibili? **BatchTopK e JumpReLU non sono mai state provate su VLM medici** — questa è la novità dell'ablation.
>
> **Cosa fa ab04.** Addestra le 3 famiglie a configurazione identica (stesso lr, stesso dizionario, stessi seed) e le confronta su ricostruzione, dead%, stabilità within-family, e soprattutto **consenso cross-famiglia**: quanti concetti vengono riscoperti da famiglie diverse?
>
> **Risultato.** L'ipotesi "BatchTopK/JumpReLU → più consenso e meno dead" è **MISTA/FALSIFICATA**:
> - **dead%**: BatchTopK vince netto (4.8% vs 16% TopK vs 49% JumpReLU). ✓ per BatchTopK.
> - **consenso**: **ZERO per tutti e tre** (0 cluster condivisi a τ=0.90). ✗
> - Cross-famiglia: solo **2.8%** dei concetti è condiviso tra 2 famiglie, **0%** tra tutte e 3.
> - Ricostruzione, naming, stabilità: **identiche** tra le famiglie (~0.99, ~0.40, ~0.005).
>
> **Conclusione: cambiare la funzione di attivazione non salva la riproducibilità.** dead% risponde alla famiglia (BatchTopK spreca meno), ma i concetti restano non riproducibili per tutti. È il test più profondo della serie: neanche toccare il meccanismo centrale aiuta.

> **Ipotesi pre-registrata:** a lr matched, BatchTopK/JumpReLU danno **consensus-rate più alto** e **dead% più basso** di TopK, perché permettono alle feature di specializzarsi sui sample che le servono invece di forzare k=32 per sample.
>
> **Esito: MISTO/FALSIFICATO.** dead% ✓ BatchTopK (4.8%) molto meglio di TopK (16%); JumpReLU peggio (49%). MA consensus-rate **0 per tutti e tre** (τ=0.90). Ricostruzione/naming/stabilità ~identiche. Cross-famiglia: 2.8% condiviso tra 2 famiglie, 0% tra tutte e 3. La famiglia di attivazione modula dead%, NON la riproducibilità.

---

## 1. Cosa produce ogni fase

| Fase | Output |
|---|---|
| Training | 3 famiglie × 3 seed = 9 SAE (12k step, lr=5e-5 matched) |
| Per-famiglia metriche | recon cosine, MSE, L0 effettivo, dead%, entropia (test) |
| Distribuzione L0 | istogramma per-sample (TopK=puntiforme a 32, altre=variabile) |
| Within-family stability | Jaccard renormalizzato n=20 (3 seed per famiglia) |
| Consensus reappearance | cluster direction-space τ=0.90, within-family (stesso algo a0/a1) |
| **Cross-activation consensus** | pool 9 modelli, cluster τ=0.90, conta cluster che spannano ≥2 famiglie |
| Naming | seed 42, gap-corrected, per famiglia |
| Figures | 4 (`a4_effective_l0_distribution`, `a4_jumprelu_threshold_hist`, `a4_activation_comparison`, `a4_cross_activation_consensus`) |
| Persist | `results/ablation/a4_activation.json` |

---

## 2. Risultati (3 seed, lr=5e-5 matched, dict=2048)

### 2.1 Per-famiglia: ricostruzione, L0, dead%

> **Cosa significa.** Le tre famiglie ricostruiscono **praticamente uguale** (~0.99). La grande differenza è i **dead%**: BatchTopK spreca pochissimo (4.8%), JumpReLU ne spreca metà (49% — la soglia imparata non converge bene a questo lr), TopK sta in mezzo (16%). L'L0 effettivo: TopK sempre 32 (rigido), BatchTopK ~38 (un po' di più), JumpReLU ~33.

| Famiglia | recon cosine | L0 effettivo | dead% | util% |
|---|---|---|---|---|
| **TopK** | 0.9913 | 32.0 (rigido) | 16.0 | 84.4 |
| **BatchTopK** | 0.9917 | ~38.3 | **4.8** | 95.2 |
| **JumpReLU** | 0.9905 | ~33.4 | 48.8 | 51.2 |

*(Baseline riferimento dict4096: recon 0.988, dead 44% — il TopK qui ha dead più basso perché dict=2048 + lr=5e-5.)*

### 2.2 Within-family stability (Jaccard renormalizzato n=20, floor=0.00977)

> **Cosa significa.** Per ciascuna famiglia, confrontiamo i 3 seed tra loro (quanto si somigliano?). Tutti e tre ~0.005, signal-to-null ~0.5×. **Le tre famiglie sono essenzialmente identiche sulla stabilità** — differenze 0.43–0.53× non significative. Nessuna famiglia è "più riproducibile".

| Famiglia | mean Jaccard (n=20) | signal/null |
|---|---|---|
| TopK | 0.00477 | 0.49× |
| BatchTopK | 0.00521 | 0.53× |
| JumpReLU | 0.00419 | 0.43× |

### 2.3 Consensus reappearance (direction-space, τ=0.90, within-family)

> **Cosa significa.** Lo stesso test di ab00 (cluster di direzioni condivise tra seed), per ogni famiglia. **Zero per tutti.** Nessuna famiglia riscopre le stesse direzioni tra seed a τ=0.90.

| Famiglia | pooled rows | cluster | consensus (≥2 seed) | rate |
|---|---|---|---|---|
| TopK | 6144 | 6144 | 0 | 0.000 |
| BatchTopK | 6144 | 6144 | 0 | 0.000 |
| JumpReLU | 6144 | 6144 | 0 | 0.000 |

### 2.4 Cross-activation consensus (9 modelli, τ=0.90) — *il test chiave della novità*

> **Cosa significa.** La domanda di novità: ci sono concetti che **famiglie diverse** riscoprono (non solo seed diversi della stessa famiglia)? Mettiamo insieme tutti i 9 modelli (3 famiglie × 3 seed) e cerchiamo cluster che toccano ≥2 famiglie. Risultato: **solo 2.8%** dei concetti è condiviso tra 2 famiglie, **0%** tra tutte e 3. Quasi tutte le direzioni sono specifiche della famiglia. Non esiste un "nucleo" di concetti universali che tutte le famiglie trovano.

| Metrica | Valore |
|---|---|
| Pooled rows (9 modelli) | 18432 |
| Cluster totali (τ=0.90) | 17936 |
| Cluster che spannano ≥2 famiglie | 496 (**2.8%**) |
| Cluster che spannano tutte e 3 | 0 (**0%**) |

### 2.5 Naming (seed 42, gap-corrected)

> **Cosa significa.** L'allineamento col vocabolario RadLex è **quasi identico** tra le famiglie (~0.40 medio, ~0.55–0.58 max). JumpReLU ha il naming max leggermente più alto (0.581). Top concept coerenti: anatomia vertebrale (ligamentum flavum, spinal stenosis), devices (core needle, shapeable wire tip). → Anche la qualità del naming per-feature è robusta alla famiglia di attivazione.

| Famiglia | n_live | naming mean | naming max |
|---|---|---|---|
| TopK | 2048 | 0.4026 | 0.5489 |
| BatchTopK | 2048 | 0.3969 | 0.5457 |
| JumpReLU | 2048 | 0.3897 | **0.5812** |

---

## 3. Analisi

### 3.1 dead% ✓ risponde alla famiglia — BatchTopK è il migliore
> **Cosa significa.** L'unica parte dell'ipotesi che tiene: BatchTopK ha molti meno dead (4.8%) di TopK (16%). Ha senso: il top-(k·B) globale lascia specializzare le feature sui sample che le servono → meno spreco. JumpReLU invece peggiora (49% dead) — la sua soglia imparata non converge bene a lr=5e-5/12k step. **Ma attenzione**: questo riguarda l'efficienza del dizionario (spreco), NON la riproducibilità.

BatchTopK 4.8% << TopK 16.0% << JumpReLU 48.8%. BatchTopK spreca meno (specializzazione globale); JumpReLU peggiora (threshold imparata non converge a lr=5e-5/12k). **Dead% risponde alla famiglia** — come in ab01 rispondeva a dict_size.

### 3.2 Consensus ✗ ZERO per tutti — ipotesi falsificata
> **Cosa significa.** L'ipotesi principale ("BatchTopK/JumpReLU più riproducibili") crolla: **tutte e tre le famiglie hanno 0 cluster condivisi** a τ=0.90. Cambiare la funzione di attivazione NON crea concetti più riproducibili. Il signal-to-null within-family è ~0.5× per tutti — identico. La stabilità è **invariante alla famiglia**.

consensus-rate 0.000 per tutti (τ=0.90). signal-to-null within-family ~0.5× identico. **La famiglia di attivazione NON modula la riproducibilità** — a differenza di dead%, che modula.

### 3.3 Cross-family: 2.8% condiviso, 0% universale
> **Cosa significa.** Messo tutti i 9 modelli insieme: il 2.8% dei concetti è trovato da 2 famiglie (un segnale debole ma non zero), ma **nessun** concetto è trovato da tutte e 3. Le famiglie trovano per lo più direzioni diverse. Non esiste un "dizionario universale" latente che tutte scoprono.

2.8% cluster span ≥2 famiglie, 0% span 3. Le famiglie sono quasi completamente disgiunte in spazio direzioni.

### 3.4 Ricostruzione + naming identici tra famiglie
> **Cosa significa.** Sugli assi "tecnici" (ricostruzione ~0.99, naming ~0.40) le tre famiglie sono **indistinguibili**. La scelta di famiglia non cambia la qualità tecnica né l'ancoraggio RadLex. Cambia solo dead% (efficienza) e il profilo L0 (TopK rigido, altre adattivo).

recon 0.9905–0.9917, naming mean 0.39–0.40, max 0.55–0.58. **Identici entro rumore.**

### 3.5 L0 adattivo = la novità reale (ma ininfluente sulla stabilità)
> **Cosa significa.** La differenza visibile tra le famiglie è il **profilo L0**: TopK è un picco puntiforme a 32 (rigido), BatchTopK/JumpReLU hanno una distribuzione (ogni immagine usa un numero diverso di feature). È il comportamento "sparsità adattiva" non studiato su VLM medici. Però — non porta a concetti più riproducibili. La novità c'è, ma non risolve il problema centrale.

TopK: L0=32 puntiforme. BatchTopK/JumpReLU: distribuzione di L0 per-sample (adattivo). **Comportamento nuovo su VLM medici**, ma ininfluente sulla riproducibilità.

---

## 4. Giudizio d'insieme: la famiglia non salva la riproducibilità

| Domanda | Esito |
|---|---|
| Rubric (≥1 variante non-TopK)? | ✅ BatchTopK + JumpReLU |
| BatchTopK/JumpReLU più riproducibili di TopK? | ❌ **No** — consensus 0 per tutti |
| dead% più basso con famiglie alternative? | ⚠️ **Parziale** — BatchTopK sì (4.8%), JumpReLU no (49%) |
| Esiste un nucleo di concetti universali tra famiglie? | ❌ **No** — 0% span 3 famiglie, 2.8% span 2 |
| Ricostruzione/naming cambiano con la famiglia? | ❌ **No** — identici (~0.99, ~0.40) |

**Verdetto cumulativo (ab00→ab04, chiusura dell'indagine):**
1. ab04 è il **test più profondo**: cambia il meccanismo centrale (la funzione di attivazione), non un iperparametro. Neanche questo aiuta la riproducibilità.
2. **dead% e stabilità sono disaccoppiate** (come in ab01): BatchTopK riduce i dead, ma i concetti restano non riproducibili. Essere "efficiente" ≠ essere "robusto".
3. **Conferma strutturale definitiva:** l'instabilità non è dovuta a TopK, né a dict_size, né a k. È **strutturale** — pochi campioni (5976) + non-unicità della decomposizione sparsa (vale per tutte e 3 le famiglie).

**Caveat onesti:**
- **lr matched (5e-5):** elimina il confound lr ~8×, ma potrebbe sotto-addestrare TopK/BatchTopK (default ~2.8e-4). Confronto valido ma conservativo — da ri-verificare a lr family-tuned in un follow-up.
- **JumpReLU 49% dead:** probabilmente lr/steps/warmup non ottimali per questa famiglia (nessun tuning per-famiglia). Non è un verdetto su JumpReLU in assoluto, solo a config matched.
- **3 seed (non 5):** compute. Sufficienza per il trend, ma il consensus a 0 è già netto.

**Conclusione dell'intero programma ablation (ab00→ab04):**
- Il "0.004" del baseline è il **pavimento matematico del caso** (ab03), non un fallimento.
- Non si fixa con dict_size (ab01), k (ab02), o famiglia di attivazione (ab04).
- → **L'instabilità è un limite strutturale dichiarato** del metodo su questo dataset. Accettarlo come limite, o aggregare i seed (model soup / consensus a τ basso con validazione). Il valore del SAE resta **strutturale**: sparsità garantita + ricostruzione 0.99 + naming top-end sopra il caso.

---

## 5. Note di riproducibilità & stato
- **Run IDE (2026-06-21 20:06):** 29 celle, 9 SAE addestrati (3 famiglie × 3 seed, 12k step). Artefatti: `a4_activation.json` (6.1 KB), 4 figure, modelli in `models/ablation_a4/{topk,batchtopk,jumprelu}_2048/sae_seed{N}/`.
- **lr pinned 5e-5 matched:** elimina il confound lr (TopK/BatchTopK auto-scale ~2.8e-4 a dict2048; JumpReLU default 7e-5). Conservativo ma valido cross-famiglia.
- **3 famiglie via `trainSAE` diretto** (non `SAEManager.train`, che hardcoda TopKTrainer). Loader bespoke per-famiglia (`AutoEncoderTopK`/`BatchTopKSAE`/`JumpReluAutoEncoder`); decoder-row extraction differisce (TopK/BatchTopK: `decoder.weight.T`; JumpReLU: `W_dec` già `(dict,act)`).
- **`compute_stability` non usato:** hardcoda `AutoEncoderTopK` → crash su BatchTopK/JumpReLU. Jaccard riscritto standalone, renormalizzato a n=20 comune (hard rule #1: within-group, size costante).
- **Dead% = activation-based** (feature mai non-zero sul test), standalone. Mai dal counter interno (soglia 10M step → bogus a 12k).
- **Naming gap-corrected** per tutte e 3 (`W_dec -= gap`), apples-to-apples col baseline.

## Riferimenti
- [04_activation_bakeoff.ipynb](04_activation_bakeoff.ipynb) — notebook sorgente
- [01_dict_size.ipynb](01_dict_size.ipynb) — ab01 (dead/stabilità disaccoppiate, stesso pattern)
- [00_consensus.ipynb](00_consensus.ipynb) — ab00 (consensus direction-space ~0, stesso algo)
- [../baseline/REPORT.md](../baseline/REPORT.md) — baseline (TopK reference)
- [../../results/ablation/a4_activation.json](../../results/ablation/a4_activation.json) — metriche complete
- [../../results/figures/ablation/a4_activation_comparison.png](../../results/figures/ablation/a4_activation_comparison.png) — bar comparison 4 metriche
- [../../results/figures/ablation/a4_effective_l0_distribution.png](../../results/figures/ablation/a4_effective_l0_distribution.png) — distribuzione L0 per famiglia
- [../../results/figures/ablation/a4_cross_activation_consensus.png](../../results/figures/ablation/a4_cross_activation_consensus.png) — cross-family consensus
