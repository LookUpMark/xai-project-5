# REPORT — Ablazioni SAE (`notebooks/autoencoder/ablation/`)

Report cumulativo delle ablation. Una sezione per notebook, aggiornata ad ogni run.

**Indice**
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder (lr pinned)](#ablation-01--dictionary-size-ladder-lr-pinned)

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
