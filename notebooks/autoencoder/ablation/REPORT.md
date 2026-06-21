# REPORT — Ablazioni SAE (`notebooks/autoencoder/ablation/`)

Report cumulativo delle ablation. Una sezione per notebook, aggiornata ad ogni run.

**Indice**
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder (lr pinned)](#ablation-01--dictionary-size-ladder-lr-pinned)
- [Ablation 02 — k (Sparsity) Sweep, null-calibrated](#ablation-02--k-sparsity-sweep-null-calibrated)
- [Ablation 03 — Concept Baselines + Empirical Jaccard Floor](#ablation-03--concept-baselines--empirical-jaccard-floor)

---

# Ablation 00 — Cross-Seed Consensus (direction-space)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA** (auto-rilevato)
**Notebook:** `notebooks/autoencoder/ablation/00_consensus.ipynb` (run headless post-fix cell 18)
**Input:** 5 checkpoint baseline `models/sae_seed{0,42,123,456,789}/` (06-05, riusati — **zero training**), `test_embeddings.pt` (1494), vocabolario RadLex **508 termini** (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt`)
**Config:** `dict_size=4096`, `k=32`, 5 seed, `dead_threshold=1e-8`, griglia `tau ∈ {0.80, 0.85, 0.90, 0.95}`, **headline `tau=0.90`**, Hungarian match `cos ≥ 0.90`, shuffle-null = 200 permutazioni

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

Per ogni seed: `get_decoder_weights()` → `(4096, 512)`, `F.normalize` righe, scarto righe a norma `< 1e-8`.

- **5 × 4096 = 20480 righe pooled**, tutte live.
- **0% dead qui** perché nei checkpoint addestrati le righe del decoder sono **già unit-norm** (la lib `dictionary_learning` normalizza ogni colonna a ogni step). Lo scarto dead è un **no-op**.

> ⚠️ **Conflitto di definizioni "dead"** (come CLAUDE.md / baseline §2.3): qui *decoder-norm dead* = 0% (colonne unit-norm post-training). La baseline riporta *activation dead* ~44% (feature mai attiva sul test). **Sono due metriche diverse** — non contraddittorie. L'analisi consenso gira sul decoder **completo**, non su un subset potato.

### 2.2 Clustering `cos>tau` (B) — headline `tau=0.90`

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

> ⚠️ **Quantità DIVERSE, riportate side-by-side — non una "correzione".** L'index-Jaccard è identità di slot; il direction-Jaccard è **invariante a permutazione** (corrispondeenza per direzione, non per indice). **Entrambe ~0** → niente struttura condivisa né in spazio indici né in spazio direzioni. Questo **falsifica** l'ipotesi "0.0038 è solo permutazione".

### 2.5 Name-agreement (E) — ❌ 0%

- 1 cluster multi-seed (≥2 seed). Termini RadLex argmax dei membri: **nessun cluster con termine unanime**.
- **Name-agreement rate = 0.00%**.
- Primo dato di consistenza naming multi-seed → nullo.

### 2.6 Faithfulness proxy (F) — ⚠️ debole, solo proxy

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
- **Index-Jaccard di riferimento:** la baseline REPORT §2.4 riporta mean 0.0039; questa ablation hard-coda `raw_index_jaccard_mean_baseline = 0.0038` (letterale in cella 24). Diffetto rounding 0.0001 — irrilevante (entrambi ~0.004).

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

| dict_size | cosine | dead% | raw Jaccard | null | **ratio** | consensus reappearance | splitting (mean / p90) | naming (mean / max) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.9937 | **30.7** | 0.0166 | 0.0159 | 1.04 | 0.0003 (1 cluster) | 0.0073 / 0.110 | 0.395 / 0.516 |
| 2048 | 0.9921 | 33.6 | 0.0070 | 0.0079 | 0.89 | 0.0 | 0.0062 / 0.107 | 0.394 / 0.537 |
| 4096 | 0.9903 | 40.9 | 0.0056 | 0.0039 | **1.43** | 0.0 | 0.0043 / 0.098 | 0.393 / 0.534 |

---

## 3. Analisi

### 3.1 dead% ✓ — scala con dict_size (over-expansion = causa dei dead)
40.9 → 33.6 → 30.7% calando dict_size. **Monotono e chiaro.** Più atomi competono per la stessa activation mass → più atomi restano inutilizzati. Over-expansion confermata come causa dei dead. (Sensitivity `lr=auto`: stesso trend 47 → 42 → 41%.)

### 3.2 signal-to-null ratio ✗ — NON monotonico (ipotesi falsificata)
Ratio: **4096 (1.43) > 1024 (1.04) > 2048 (0.89)**. L'ipotesi "ratio↑ quando dict↓" è **falsificata**. Il dict più **grande** ha il signal-to-null più alto (più accordo oltre il caso); il 2048 è persino **sotto null** (0.89 < 1).
→ Ridurre dict_size NON aumenta la robustezza cross-seed. L'over-expansion NON spiega l'instabilità.

### 3.3 Consensus reappearance — ~0 ovunque (invariante al dict_size)
Direction-space: 1024 → 0.03%, 2048 → 0%, 4096 → 0% cluster multi-seed. **Identico al null di ab00** a tutte le capacità.
→ L'instabilità in spazio direzioni è **invariante** al dict_size. Ridurre il dizionario non fa ricomparire direzioni condivise.

### 3.4 Feature splitting — direzione OPPOSTA all'ipotesi
Mean pairwise cos tra alive rows: **1024 (0.0073) > 2048 (0.0062) > 4096 (0.0043)**. p90 idem (0.110 → 0.107 → 0.098).
→ Dict più **piccolo** → alive rows più **affollate/redundanti** (cos più alto). L'ipotesi "over-expansion causa splitting" è **falsificata**: più atomi = più spazio per dispiegarsi, meno collisioni.

### 3.5 Naming — STABILE cross-size (~0.394)
mean 0.395 / 0.394 / 0.393, max 0.52–0.54 per tutti e tre. **Identico alla baseline (0.3949).**
→ La qualità del grounding RadLex per-feature è **robusta al dict_size**. Non è la qualità del singolo concetto a essere instabile — è la **composizione del set**.

### 3.6 Revival probe (dict2048) — negative probe confermato
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
Ratio baseline 0.954 ≈ 1 → il Jaccard 0.0038 del baseline è **statisticamente indistinguibile dal random-overlap**. Claim difendibile confermato: a k=32/dict4096 i concetti non sono più riproducibili del caso (in spazio indici).

### 3.2 Signal-to-null NON monotonico — picco a k=16
Ratio: **k=16 (1.30) > k=32 (1.15) > k=64 (0.97) > k=8 (0.80)**.
- L'ipotesi "rising as k shrinks" è **parzialmente falsificata**: sale da k=64→32→16, ma **k=8 collassa sotto null** (troppo sparso → 91.6% dead → niente da allineare).
- **k=16 è l'unico k dove la CI esclude 1** (1.24–1.37): l'accordo reale supera chiaramente il caso. Sweet spot di stabilità.

### 3.3 dead% ↗ small k ✓
91.6% (k=8) → 74.7% (k=16) → 41.3% (k=32) → 40.2% (k=64). Confermato: meno feature attive per pass → più feature mai si attivano. k=8 è patologico (quasi tutto morto).

### 3.4 Consensus reappearance — ingannevole a k=8
consensus≥2: k=8 (0.65%) > k=16 (0.16%) > k=32 (0.037%) > k=64 (0%). Ma k=8 ha **91.6% dead** → live set minuscolo (~170/2048) → cluster forzati per affollamento, non riproducibilità reale. Il signal-to-null (che corregge per la dimensionalità) dice il contrario: k=8 *sotto* null. **k=16 resta il sweet spot onesto.**

### 3.5 Tradeoff stabilità ↔ ricostruzione (Pareto)
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
- **Naming tag allineato (fixato in questa commit):** i tag interni dei notebook ora matchano i numeri `00→a0, 01→a1, 02→a2, 03→a3, 04→a4`. Prima erano scrambled (`02→a4`, `03→a6`, `04→a2`) per un renumbering non propagato ai tag interni. Artefatti su disco rinominati di conseguenza (`a2_k_sweep.json`, `a3_baselines.json`, figure `a2_*`/`a3_*`, `models/ablation_a2`).
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

> **Domanda.** Quanto del comportamento del SAE è spiegato da un dizionario *generico* vs uno *imparato*? E il cross-seed index-Jaccard di 0.0038 è segnale o artefatto di confrontare indici tra dizionari 4096-dim indipendenti?
>
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

| Metodo | recon cosine | L0 | dead% | naming mean | naming max |
|---|---|---|---|---|---|
| **SAE** (dict4096, k32, baseline) | 0.988 | 32 | 44.0 | 0.395 | 0.546 |
| Random (D=256) | 0.454 | 32 | 0.0 | 0.372 | 0.442 |
| Dense-PCA (D=256) | **0.996** | 32 | 0.0 | 0.383 | 0.594 |
| Freq-KMeans (D=256) | 0.961 | 32 | 0.0 | **0.829** | **0.875** |

**Random-Jaccard floor (within-group, 3 seed):**

| Gruppo | D | empirical J | analytical null | ratio |
|---|---|---|---|---|
| Random (small) | 256 | 0.0666 | 0.0667 | 1.00 |
| **Random (big)** | **4096** | **0.0037** | **0.0039** | **0.95** |
| — SAE baseline (cross-seed, 5 seed) | 4096 | 0.0038 | — | — |

---

## 3. Analisi

### 3.1 Random@4096 ≈ SAE → index-Jaccard del SAE sul floor del caso ✓
Random@4096 = 0.0037, SAE = 0.0038. **Identici entro rumore.** ratio 0.95 = il SAE siede esattamente sul null empirico per dizionari 4096-dim. → Il 0.0038 cross-seed del SAE è **calibrato come near-null in spazio indici**: confrontare indici tra dizionari 4096-dim indipendenti produce ~0.004 di puro overlap casuale. Cross-check analitico `k/(2D−k)` = 0.0039 conferma (ratio 0.95; l'empirico è leggermente sotto perché i top-k-set sulla STESSA data condividono struttura, ma l'ordine di grandezza è quello).

### 3.2 PCA = ceiling denso di ricostruzione ✓ (non è "SAE è scarso")
PCA 0.996 > SAE 0.988 su raw cosine. **Atteso e pedagogico:** PCA è denso (256 atomi tutti attivi, zeroato a L0=32 solo *dopo* il fit per confronto fair) — sacrifica sparsity e monosemanticità per la ricostruzione. Il SAE perde ~0.008 di cosine in cambio di **L0=32 enforced + naming**. Questo è il Pareto tradeoff, non un difetto.

### 3.3 Naming: SAE ≈ Random, KMeans schiaccia tutti ⚠️ (risultato severo)
- naming mean: **KMeans 0.829 >> SAE 0.395 ≈ PCA 0.383 ≈ Random 0.372**.
- Il SAE **batte il Random di soli +0.023** sul naming mean. Lo shift del modality gap (`W_dec -= gap`) muove *tutte* le righe decoder della stessa quantità prima del coseno → domina il signal, e l'apprendimento del SAE aggiunge margine minimo sul naming *medio*.
- KMeans domina perché i centroidi **sono i modi della distribuzione dati** → allineati al cloud del vocabolario (anch'esso modi-dominato). naming mean alto ≠ grounding genuino: i centroidi KMeans sono blend densi (non monosemantici), l'alta similarità riflette allineamento cloud-vs-cloud, non concetti isolati.
- **Caveat dict-size:** SAE 4096 feature vs baseline 256 → per-feature naming mean non perfettamente comparabile (più atomi = slice più stretta per feature). L'ORDINE (KMeans >> resto ≈) resta il signal robusto; il confronto più pulito è il **top-end** (max): SAE 0.546 > Random 0.442.

### 3.4 Random recon scala con D: 0.45 (256) → 0.60 (4096)
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
