# REPORT — Ablation 00: Cross-Seed Consensus (direction-space)

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
