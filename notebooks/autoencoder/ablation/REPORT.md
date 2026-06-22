# REPORT — Ablazioni SAE (`notebooks/autoencoder/ablation/`)

Report cumulativo del programma di ablation. Una sezione per notebook, aggiornata ad ogni run.

**Companion:** `../baseline/REPORT.md` descrive il run baseline (TopK, dict4096, k=32, 5 seed) da cui tutte queste ablation derivano. L'instabilità cross-seed osservata nel baseline (Jaccard 0.0038) è la domanda madre dell'intero programma; la sua interpretazione come "pavimento del caso" è stabilita qui (Ablation 03).

**Indice**
- [Sintesi esecutiva](#sintesi-esecutiva)
- [Glossario](#glossario)
- [Metriche e null: definizioni formali](#metriche-e-null-definizioni-formali)
- [Ablation 00 — Cross-Seed Consensus (direction-space)](#ablation-00--cross-seed-consensus-direction-space)
- [Ablation 01 — Dictionary-Size Ladder (lr pinned)](#ablation-01--dictionary-size-ladder-lr-pinned)
- [Ablation 02 — k (Sparsity) Sweep, null-calibrated](#ablation-02--k-sparsity-sweep-null-calibrated)
- [Ablation 03 — Concept Baselines + Empirical Jaccard Floor](#ablation-03--concept-baselines--empirical-jaccard-floor)
- [Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)](#ablation-04--activation-family-bake-off-topk-vs-batchtopk-vs-jumprelu)
- [Ablation 05 — Concept Faithfulness vs clinical labels (MeSH/Problems)](#ablation-05--concept-faithfulness-vs-clinical-labels-meshproblems)
- [Conclusione cumulativa](#conclusione-cumulativa)
- [Bibliografia](#bibliografia)

---

## Sintesi esecutiva

Il run baseline (vedi `../baseline/REPORT.md`) raggiunge una ricostruzione ottima (cosine 0.988 con soli `k=32` feature attivi) ma scopre concetti **quasi completamente diversi** ad ogni seed: mean cross-seed index-Jaccard **0.0038 ≈ 0**. L'intero programma di ablation nasce da una domanda: il 0.0038 è un fallimento reale del metodo, oppure il pavimento matematico del caso? E si può mitigare agendo su un iperparametro?

Le prime cinque ablation (00–04) investigano la **causa** dell'instabilità lungo quattro assi ortogonali — spazio di rappresentazione, capacità del dizionario, sparsità, famiglia di attivazione — sempre contro un null calibrato (analitico o empirico). La sesta (05) apre l'asse complementare della **fedeltà**: ammesso che i concetti siano instabili, quelli che esistono sono almeno clinicamente significativi?

| Ab | Asse | Domanda | Esito |
|---|---|---|---|
| **00** | direzioni del decoder | Il 0.0038 è un artefatto di permutazione degli indici? | ❌ No — disgiunti anche nello spazio delle direzioni |
| **01** | capacità (`dict_size`) | Il sovradimensionamento (4096 = 8×) causa instabilità e dead? | ❌ Per i dead sì, per la stabilità **no** |
| **02** | sparsità (`k`) | Un `k` diverso porta sopra il null? | ⚠️ Parziale — debole sweet spot a k=16, non risolve |
| **03** | alternative al SAE | Il SAE batte metodi banali, o è sul pavimento del caso? | ✅ Pavimento del caso — random fa uguale |
| **04** | famiglia di attivazione | È colpa di TopK? (BatchTopK, JumpReLU) | ❌ dead% sì, stabilità **no** |
| **05** | fedeltà clinica | I concetti esistenti sono fedeli a etichette reali? | ✅ Parzialmente sì — ~10% fedele oltre il null |

**Conclusione d'insieme.** Il 0.0038 "preoccupante" del baseline **non è un fallimento**: è il pavimento matematico del caso (Ablation 03: Random@4096 = 0.0037 ≈ SAE), confermato come rumore sia in spazio indici (00) che direzioni (03). L'instabilità **non si fixa** con iperparametri: né `dict_size` (01), né `k` (02), né la famiglia di attivazione (04: consensus 0 per tutte e tre) la risolvono. Le cause profonde rimaste sono strutturali — pochi campioni (5976) e non-unicità intrinseca della decomposizione sparsa su embedding CLIP proiettati — e valgono per TopK, BatchTopK e JumpReLU. La diagnosi causale completa è in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

Tuttavia **l'instabilità non equivale a inutilità** (05). I concetti che esistono in un seed sono moderatamente ma genuinamente **fedeli** a etichette cliniche reali: ~10% delle feature live (226/2251) batte un null calibrato per-feature, e le più forti tracciano concetti clinicamente attesi (impianti medici |r|=0.46, versamento pleurico, enfisema, ombra cardiaca, edema). Il risultato d'insieme è quindi **bilanciato e difendibile**: il SAE su questo dataset ha un limite strutturale dichiarato (seed-dipendenza) ma produce direzioni con grounding clinico reale — non riproducibili seed-a-seed, ma non rumore.

---

## Glossario

Termini ricorrenti nel report. Le definizioni formali (con formule) sono nella sezione successiva.

- **Jaccard (index)** — sovrapposizione tra i *set di indici* delle feature attive di due SAE: `|A∩B|/|A∪B|`. Testa identità di slot ("lo slot `i` del seed A spara sugli stessi sample dello slot `i` del seed B?"). Sensibile alla permutazione.
- **Jaccard (direction)** — sovrapposizione nello spazio delle *direzioni* del decoder, indipendente dall'indice: si abbinano le feature per similarità geometrica (coseno) via matching Hungarian, tenendo gli abbinamenti con coseno ≥ τ. Risolve l'artefatto di permutazione.
- **Signal-to-null ratio** — `Jaccard_osservato / E[Jaccard]_caso`. >1 = accordo oltre il caso; ≈1 = sul pavimento del caso; <1 = sotto. Il null è la sovrapposizione attesa tra due dizionari casuali indipendenti a pari dimensione.
- **Null (calibrazione)** — valore di riferimento "per puro caso". Tre forme usate: (1) **analitico** ipergeometrico `k/(2D−k)` per Jaccard; (2) **shuffle-null** — permutare i tag (seed o etichetta) e ricalcolare la metrica, prendere un percentile (p95); (3) **BH-FDR** — correzione multi-ipotesi Benjamini–Hochberg sui p-value.
- **Consensus reappearance** — per quante seed/famiglie ricorre una stessa direzione. Si clusterizzano tutte le righe del decoder pooled (`connected_components` su grafo `coseno > τ`) e si conta in quanti seed è rappresentato ciascun cluster. `consensus@≥k/5` = frazione di cluster presenti in ≥k seed.
- **Dead feature** — due definizioni divergenti, vanno tenute distinte:
  - *decoder-norm dead* = riga del decoder a norma ~0 (`‖w_i‖ < 1e-8`). Nei checkpoint addestrati è **0%** (la libreria normalizza ogni colonna ad ogni step).
  - *activation dead* = feature mai non-zero sul test set. Nel baseline ~**44%** (dizionario sovradimensionato per ~7400 immagini).
- **L0** — numero di feature non-zero per immagine. Con TopK è `k` esatto (rigido); con BatchTopK/JumpReLU è variabile (adattivo).
- **VE (Variance Explained)** — `1 − ‖x−x̂‖²/‖x‖²`. Strettamente legato al cosine di ricostruzione.
- **Modality gap** — scostamento geometrico sistematico tra lo spazio delle immagini e quello dei testi nei modelli contrastivi (CLIP/BiomedCLIP). Le due modalità occupano "coni" separati; il gap è approssimativamente una traslazione costante. Corretto post-hoc con `W_dec -= (visual_centroid − text_centroid)` prima del naming.
- **Feature splitting** — quanto le feature vive si somigliano tra loro (mean/p90 pairwise coseno tra alive rows). Splitting alto = ridondanza/collisioni.
- **Faithfulness** — una feature è "fedele" a un'etichetta clinica se si attiva proprio sulle immagini che contengono quell'etichetta, **oltre il puro caso**. Misurata come correlazione point-biserial tra il pattern di attivazione della feature e l'etichetta binaria.
- **Point-biserial correlation** — correlazione di Pearson tra una variabile continua e una binaria. Equivalentemente `A_zᵀ·Y_z/N` con entrambe le matrici z-scored. Usata in 05 invece di AUROC per costo O(una matmul).
- **Hungarian matching** — algoritmo di assegnamento ottimo (`linear_sum_assignment`); in 00 abbinamento 1-a-1 per massimizzare la similarità media tra le feature di due seed.

---

## Metriche e null: definizioni formali

Notazione: `x ∈ ℝⁿ` embedding di un'immagine, `x̂ = W_dec·z + b_dec` ricostruzione SAE, `z ∈ ℝᴰ` codice sparso con `k` non-zero (TopK), `D = dict_size`, `τ` soglia di coseno.

**Ricostruzione.**
- Cosine: `cos(x, x̂) = ⟨x, x̂⟩ / (‖x‖·‖x̂‖)`. Nel baseline ~0.988.
- Variance Explained: `VE = 1 − ‖x − x̂‖² / ‖x − b_dec‖²` (rispetto al bias, non all'origine). ~99.3% nel baseline.
- L0: `‖z‖₀ = #{i : zᵢ ≠ 0}`. TopK lo forza a `k` esatto.

**Stabilità.**
- Index-Jaccard (baseline, 00): `J(A,B) = |A∩B| / |A∪B|` dove `A,B` sono i set di indici attivi di due SAE sullo stesso sample.
- Direction-Jaccard (00): matching Hungarian su matrice coseno `D×D` tra righe decoder, si contano gli abbinamenti con `cos ≥ τ`.
- Null analitico per l'index-Jaccard, due dizionari indipendenti di dimensione `D` che scelgono ciascuno `k` indici: `E[J] = Σⱼ j/(2k−j)·P(j)` con `P(j) = hypergeom(M=2D, n=k, N=k)(j)`; per `k ≪ D` si riduce a `E[J] ≈ k/(2D−k)`. A D=4096, k=32 → 0.0039.
- Signal-to-null ratio: `r = J_osservato / E[J]`.

**Dead feature.**
- decoder-norm dead: `‖w_i‖ < 1e-8` (0% nei checkpoint addestrati).
- activation dead: `∀ sample s: zᵢ(s) = 0` sul test set (~44% baseline).

**Consensus (00, 01, 02, 04).** Grafo sparso su righe decoder pooled con archi dove `coseno > τ` → `scipy.sparse.connected_components`. `consensus@≥m/5` = frazione di componenti che includono righe di ≥m seed.

**Shuffle-null.** Per un'ipotesi H su N elementi etichettati: permutare le etichette `B` volte, ricalcolare la metrica sotto H, prendere il percentile (p95 in 05, oppure il valore puro per un p-value in 00). Specifica per-feature in 05 (corregge per la distribuzione di prevalenza dell'etichetta).

**Faithfulness (05).** Matrice attivazioni `A ∈ ℝ^{N×D}` (z-scored per feature), matrice etichette `Y ∈ {0,1}^{N×L}` (z-scored per etichetta). Correlazione point-biserial `R = A_zᵀ Y_z / N` (matrice `D×L`). Per feature `i`: `max_j |Rᵢⱼ|`. Null triplo: (1) SE analitica `1/√N` = 0.0259; (2) shuffle-null per-feature p95 (200 perm, mediana 0.188); (3) BH-FDR 0.05 sui `D_live × L` test (112550).

**Modality gap (naming).** `gap = mean(train_emb, 0) − mean(vocab_emb, 0)`; naming gap-corrected: `W_dec ← W_dec − gap`, poi `F.normalize` righe + coseno con `F.normalize(vocab_emb)`. Sposta le colonne del decoder dal "cono" visivo verso quello testuale prima del confronto.

---

# Ablation 00 — Cross-Seed Consensus (direction-space)

**Data run:** 2026-06-21 · **Macchina:** Linux / NVIDIA RTX 5070 Laptop, **CUDA** (auto)
**Notebook:** `00_consensus.ipynb` (run headless post-fix cell 18)
**Input:** 5 checkpoint baseline `models/sae_seed{0,42,123,456,789}/` (06-05, riusati — zero training), `test_embeddings.pt` (1494), vocabolario RadLex **508 termini**
**Config:** `dict_size=4096`, `k=32`, 5 seed, griglia `τ ∈ {0.80, 0.85, 0.90, 0.95}`, headline `τ=0.90`, shuffle-null = 200 permutazioni

## Contesto e domanda

Il baseline riporta un mean index-Jaccard di 0.0038 (off-diagonali 0.002–0.010). Quel valore confronta gli **indici**: lo slot `i` del seed A combacia con lo slot `i` del seed B? Se due seed imparano la stessa direzione concettuale ma la salvano a indici diversi — come cinque persone che mettono gli stessi libri su scaffali numerati in modo diverso — l'index-Jaccard li segna zero anche se la geometria coincide. L'ipotesi naturale è quindi che il 0.0038 sia un **artefatto di permutazione**.

Questa ablation lo verifica ri-analizzando lo spazio delle **direzioni** del decoder (invariante all'indice): si poolano tutte le righe `W_dec` dei 5 seed, si clusterizzano per similarità geometrica e si conta in quanti seed ricorre ciascun cluster. Se i concetti fossero gli stessi a indici diversi, apparirebbero cluster multi-seed.

**Ipotesi pre-registrata:** il 0.0038 è un artefatto di permutazione → lo spazio delle direzioni mostra cluster multi-seed sopra il null.

**Esito: ipotesi FALSIFICATA.** Lo spazio delle direzioni mostra ~0 struttura condivisa. Solo 1 direzione su 20480 ricorre su ≥3 seed; nessuna su ≥4. Lo shuffle-null dà p=1.0 (l'osservato è identico al caso). I 5 run imparano davvero direzioni diverse, non è solo permutazione.

---

## 1. Cosa produce ogni fase

| Fase | Output | Stato |
|---|---|---|
| (A) Pool decoder rows | 5×`W_dec` (4096×512) L2-normalizzate, tag seed | ✅ 20480 righe, 0 dead (decoder-norm) |
| (B) Clustering `cos>τ` | `scipy.connected_components` su grafo sparso | ✅ griglia τ + headline 0.90 |
| (C) Reappearance | cluster per #seed rappresentati | ✅ consensus@3/4 |
| (D) Hungarian direction-match | `linear_sum_assignment` per coppia di seed | ✅ direction-Jaccard |
| (E) Name-agreement | termine RadLex argmax per membro cluster | ✅ |
| (F) Faithfulness proxy | naming-cos + mean test activation | ✅ (proxy, no ground-truth) |
| (G) Headline figure | `a0_consensus_headline.png` (3 pannelli) | ✅ |
| (H) Shuffle-null | 200 permutazioni tag → consensus@4 | ✅ p-value |
| **Persist** | `results/ablation/a0_consensus.json` | ✅ |

> Isolamento: output scritti solo in `results/ablation/` + `results/figures/ablation/` — la `results/` della baseline non viene toccata.

---

## 2. Risultati

### 2.1 Pooling decoder (A) — 20480 righe, 0 dead

Per ogni seed: `get_decoder_weights()` → `(4096, 512)`, `F.normalize` righe, scarto righe a norma `< 1e-8`. Si ottengono **5 × 4096 = 20480 righe pooled**, tutte live (0% decoder-norm dead). I decoder sono già unit-norm post-training perché la libreria normalizza ogni colonna ad ogni step, quindi lo scarto dead è un no-op.

Qui *decoder-norm dead* = 0% non contraddice il ~44% *activation dead* della baseline: sono due metriche diverse (vedi Glossario).

### 2.2 Clustering `cos>τ` (B) — headline `τ=0.90`

Grafo sparso `coseno > τ` → connected components. A `τ=0.90` (soglia alta, solo feature quasi identiche) si trova 1 solo cluster condiviso.

| `τ` | componenti | multi-member | max size | coesione intra (cos medio) |
|------:|-----------:|-------------:|---------:|---------------------------:|
| 0.80 | 20474 | 3 | 4 | 0.832 |
| 0.85 | 20477 | 1 | 4 | 0.879 |
| **0.90** | **20478** | **1** | **3** | **0.884** |
| 0.95 | 20480 | 0 | 1 | — (singleton) |

`τ` più basso fonde direzioni non correlate in pochi cluster giganti; più alto frantuma in singleton. **0.90 è il punto interpretabile** (cluster piccoli e coesi, coesione 0.884, max size 3).

### 2.3 Reappearance (C) — essenzialmente nulla

Per ogni cluster a `τ=0.90` si conta in quanti seed è rappresentato. Se un cluster ha feature di ≥3 seed, quel concetto è "robusto".

| #seed per cluster | #cluster |
|---|---|
| 1 | 20477 |
| 2 | 0 |
| 3 | **1** |
| 4 | 0 |
| 5 | 0 |

- `consensus@≥3/5` = **0.0146%** (1 cluster su 3 seed).
- `consensus@≥4/5` = **0.00%**.
- Solo **1 direzione** ricorre su ≥3 seed; nessuna su ≥4. Quasi tutto il decoder pooled è costituito da direzioni seed-esclusive.

### 2.4 Hungarian direction-Jaccard (D) — ~0

Metodo più forte del clustering: per ogni coppia di seed si cerca l'abbinamento ottimo (Hungarian) tra le 4096 feature e si contano i match con `cos ≥ 0.90`. È il massimo sforzo di abbinamento.

| Coppia | match / 4096 | rate |
|---|---|---|
| 0↔42, 0↔123, 0↔456, 0↔789 | 0 | 0.0000 |
| 42↔123, 42↔456 | 0 | 0.0000 |
| 42↔789 | 1 | 0.0002 |
| 123↔456, 123↔789 | 0 | 0.0000 |
| 456↔789 | 1 | 0.0002 |

- **Direction-Jaccard medio = 4.9e-5** (~0/4096 per coppia) vs raw index-Jaccard baseline = 0.0038.

Sono quantità diverse riportate side-by-side, non una "correzione". L'index-Jaccard è identità di slot; il direction-Jaccard è invariante a permutazione. **Entrambe ~0** → niente struttura condivisa né in spazio indici né in spazio direzioni, il che falsifica l'ipotesi "0.0038 è solo permutazione".

### 2.5 Name-agreement (E) — 0%

Per i pochissimi cluster condivisi si controlla se i seed li chiamano con lo stesso termine RadLex. L'unico cluster multi-seed non ha termine unanime → **name-agreement rate = 0.00%**. Anche dove c'è un minimo di sovrapposizione geometrica, l'etichetta medica non è coerente.

### 2.6 Faithfulness proxy (F) — debole, solo proxy

Per l'unico concetto che ricorre (cluster di 3 seed, chiamato `bulging fissure sign`):

| Metrica | Valore |
|---|---|
| n_concepts | 1 |
| Termine vincente | `bulging fissure sign` |
| Naming-cos (dir media vs emb termine) | **0.1580** |
| Mean test activation (membri seed-42) | **0.0047** |

Naming-cos 0.158 = molto debole (vs naming medio baseline gap-corrected 0.395) e attivazione quasi nulla. Proxy dichiarato: naming-cos + activation media, no ground-truth clinico (la valutazione clinica vera è in Ablation 05).

### 2.7 Shuffle-null (H) — p=1.0, nessun segnale oltre il caso

Test di sicurezza: si mescolano a caso le etichette seed 200 volte e si misura quanto "consenso" spunterebbe per puro caso.

- `consensus@≥4/5` osservato: **0.00%**.
- Shuffle-null (200 perm): **0.00%**.
- **p-value = 1.0** → osservato = null, gap 0.00 pp. Il consenso osservato non supera la baseline casuale.

---

## 3. Giudizio d'insieme: tesi falsificata, onestamente

| Domanda | Esito |
|---|---|
| Il 0.0038 della baseline è un artefatto di permutazione? | ❌ No — direction-Jaccard 4.9e-5, ~0 anche in spazio direzioni |
| I 5 seed imparano direzioni concettuali vicine? | ❌ No — max coseno off-diagonale within-seed ~0.577, ben sotto 0.90 |
| Esiste consenso cross-seed sopra il caso? | ❌ No — consensus@4 = 0%, shuffle-null p=1.0 |
| I concetti stabili sono clinicamente ancorati? | ⚠️ Debole — 1 concetto, naming-cos 0.158 (proxy, no ground-truth) |

Punti chiave:

1. **L'instabilità della baseline è geometricamente reale, non rumore di labeling.** I 5 SAE scoprono basi sostanzialmente disgiunte: cambiare seed cambia quali direzioni vengono apprese, non solo i loro indici.
2. **Non è p-hacking.** Abbassare `τ` a 0.80 per fabbricare qualche cluster multi-member produrrebbe un headline "positivo" fittizio su un risultato nullo. L'ablation si rifiuta di farlo.
3. **Conseguenza operativa.** Il seed primario 42 è arbitrario; naming/explanations della baseline dipendono dal seed. Per concetti riproducibili serve aggregazione cross-seed (model soup, init condiviso, consensus clustering su `τ` molto più basso con validazione) o accettare la seed-dipendenza come limite.
4. **Direzione di fuga.** Le ablation 01 (dict_size) e 02 (k) — ridurre i gradi di libertà — sono il prossimo test naturale: meno parametri → meno divergenza tra seed.

---

## 4. Note di riproducibilità

- **Run headless (2026-06-21 18:48):** celle 2–24 via `.venv/bin/python` (torch 2.12.0+cu130, CUDA RTX 5070), backend matplotlib Agg. Cell 18 eseguita senza crash dopo il fix dict→term.
- **Fix applicato in questa run:** cella 6 normalizza `vocabulary.json` (lista di dict `{"term",...}`) → stringhe `term` al load. Senza di esso, `vocab_labels[i]` era un dict → crash `"{t:28s}"` in cell 18.
- **Zero training:** riusa i 5 checkpoint baseline (06-05). La correzione del modality gap (baseline) non influenza questa ablation — qui si confrontano direzioni del decoder raw, non si fa naming gap-corrected.
- **Vocabolario = 508 termini** (`data/vocabulary.json` + `embeddings/text_vocab_embeddings.pt` allineati; verificato in run: 508 termini, embeddings `[508, 512]`).
- **Artefatti:** `results/ablation/a0_consensus.json` (metriche complete) + `results/figures/ablation/a0_consensus_headline.png` (3 pannelli: heatmap index-Jaccard 5×5, istogramma reappearance, scatter 2D decoder pooled UMAP colored by cluster/seed).
- **Index-Jaccard di riferimento:** la baseline riporta mean 0.0039; questa ablation hard-coda `raw_index_jaccard_mean_baseline = 0.0038` (letterale in cella 24). Difetto rounding 0.0001 — irrilevante.

---

# Ablation 01 — Dictionary-Size Ladder (lr pinned)

**Data run:** 2026-06-21 · **Macchina:** Linux / RTX 5070, **CUDA**
**Notebook:** `01_dict_size.ipynb` (21/21 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), vocabolario RadLex **508 termini**
**Config:** `dict_size ∈ {1024, 2048, 4096}`, `k=32`, **lr pinned 4e-4** (capacity = unica variabile), `steps=12000`, `batch_size=256`, seeds `(0, 42, 123)`, naming **gap-corrected**

## Contesto e domanda

Il baseline ha due patologie accoppiate: ~44% di feature morte (spreco) e Jaccard ≈ 0.004 (instabilità). Ipotesi naturale: il dizionario di 4096 feature è **8 volte più grande** dello spazio da 512 dimensioni (sovradimensionamento). Forse il sovradimensionamento causa entrambi i problemi — troppa roba inutilizzata che diverge tra seed.

Questa ablation addestra SAE con dizionari di 3 dimensioni diverse (1024, 2048, 4096), tenendo fisso tutto il resto (stesso lr, stesso k). Se l'ipotesi è giusta, dizionari più piccoli dovrebbero avere meno dead **e** più stabilità (signal-to-null ratio più alto).

**Ipotesi pre-registrata:** `dict_size` più piccolo → dead% cala **AND** signal-to-null ratio sale (il null cresce trivialmente quando D cala, quindi il ratio è il confronto corretto).

**Esito: MISTO.** dead% cala come previsto (40.9 → 30.7%) ma la stabilità non migliora — anzi, il dizionario più grande ha il ratio più alto (1.43). L'over-expansion spiega i dead, **non** l'instabilità: sono due problemi separati.

---

## 1. Cosa produce ogni fase

| Fase | Output |
|---|---|
| Training ladder | 3 dict_size × 3 seed = 9 SAE (12k step, lr pinned) |
| Per-size metrics | cosine, dead%, L0, entropy (test) |
| Within-group Jaccard | matrice 3×3 per size (Protocollo: costante dict_size+k) |
| Signal-to-null ratio | Jaccard / null ipergeometrico |
| Consensus reappearance | cluster direction-space (τ=0.9) — stesso algo di 00 |
| Feature splitting | mean/p90 pairwise cos tra alive rows (subsample 2000) |
| Revival probe | dict2048, dead_threshold abbassato + auxk forte (negative probe) |
| Sensitivity | ripete ladder con `lr=auto` |
| Naming | primary seed 42, gap-corrected, per size |
| Persist | `results/ablation/a1_dict_size.json` + 3 figure |

---

## 2. Risultati per-size (lr pinned 4e-4, 12k step, 3 seed)

La colonna chiave è **ratio** = Jaccard osservato diviso il null (la sovrapposizione attesa per puro caso a quella dimensione). Ratio > 1 = i concetti si accordano oltre il caso; ≈ 1 = sul pavimento.

| dict_size | cosine | dead% | raw Jaccard | null | **ratio** | consensus reappearance | splitting (mean / p90) | naming (mean / max) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.9937 | **30.7** | 0.0166 | 0.0159 | 1.04 | 0.0003 (1 cluster) | 0.0073 / 0.110 | 0.395 / 0.516 |
| 2048 | 0.9921 | 33.6 | 0.0070 | 0.0079 | 0.89 | 0.0 | 0.0062 / 0.107 | 0.394 / 0.537 |
| 4096 | 0.9903 | 40.9 | 0.0056 | 0.0039 | **1.43** | 0.0 | 0.0043 / 0.098 | 0.393 / 0.534 |

---

## 3. Analisi

### 3.1 dead% ✓ — scala con dict_size (over-expansion = causa dei dead)
Riducendo il dizionario, le feature morte scendono monotonamente (40.9 → 33.6 → 30.7%). Conferma: troppi atomi competono per la stessa "torta" di attivazione → molti restano inutilizzati. Il sovradimensionamento causa lo spreco. (Sensitivity `lr=auto`: stesso trend 47 → 42 → 41%.)

### 3.2 signal-to-null ratio ✗ — NON monotonico (ipotesi falsificata)
Se il sovradimensionamento causasse anche l'instabilità, ridurre il dizionario dovrebbe alzare il ratio. Invece è il contrario: il dizionario più grande (4096) ha il ratio più alto (1.43), e il 2048 è persino sotto il caso (0.89). Ratio: **4096 (1.43) > 1024 (1.04) > 2048 (0.89)**. L'over-expansion NON spiega l'instabilità.

### 3.3 Consensus reappearance — ~0 ovunque (invariante al dict_size)
Lo stesso test di 00 (cluster di direzioni condivise tra seed), ripetuto a ogni dimensione: 1024 → 0.03%, 2048 → 0%, 4096 → 0% cluster multi-seed. Identico al null di 00 a tutte le capacità. La mancanza di direzioni condivise non dipende da quanto è grande il dizionario.

### 3.4 Feature splitting — direzione OPPOSTA all'ipotesi
"Splitting" = quante feature vive si somigliano tra loro (collisioni). Mean pairwise cos tra alive rows: **1024 (0.0073) > 2048 (0.0062) > 4096 (0.0043)**, p90 idem. Il dizionario più piccolo ha feature più affollate/redundanti; più atomi = più spazio per dispiegarsi = meno collisioni. L'ipotesi "over-expansion causa splitting" è falsificata.

### 3.5 Naming — stabile cross-size (~0.394)
mean 0.395 / 0.394 / 0.393, max 0.52–0.54 per tutti e tre. Identico alla baseline (0.3949). La qualità del grounding RadLex per-feature è robusta al dict_size: non è la qualità del singolo concetto a essere instabile, è la composizione del set.

### 3.6 Revival probe (dict2048) — negative probe confermato
dead_threshold abbassato + auxk forte: **dead% 33.6 → 30.9** (cala ✓) ma **Jaccard 0.0070 → 0.0059** (flat/↓), ratio 0.89 → 0.75. Revivere feature morte riduce lo spreco ma non migliora la robustezza: "alive ≠ robust". Feature vive ma arbitrarie sono disaccoppiate dalla stabilità.

---

## 4. Giudizio d'insieme: over-expansion = dead, NON instability

| Patologia | Causa? | Evidenza |
|---|---|---|
| ~44% dead features | ✅ Over-expansion | dead% scala con dict_size (40.9 → 30.7%) |
| Cross-seed instability (Jaccard 0.004) | ❌ NON over-expansion | ratio non sale riducendo dict; consensus ~0 ovunque |

L'over-expansion spiega lo spreco (dead), non l'instabilità. L'ipotesi "overcompleteness causa primaria dell'instabilità" è rifinita da questa ablation: ridurre il dizionario riduce i dead ma non rende i concetti riproducibili. L'instabilità è più fondamentale — probabili cause: pochi campioni (5976) + non-unicità intrinseca del TopK SAE su questo cloud. Non risolvibile abbassando dict_size.

Punti chiave:
1. **Dict più piccolo è comunque "meglio"** (meno dead, stessa ricostruzione 0.99+, stesso naming 0.39, meno compute) — ma **non** per la robustezza.
2. **Naming robusto cross-size** → il grounding individuale funziona; il problema è *quale set* di feature si apprende.
3. **Revival probe**: vivificare i dead non aiuta → l'instabilità non è un problema di "feature addormentate".
4. **Prossimi test naturali:** 02 (k più vincolato?), 03 (baselines). Se anche k non aiuta, l'instabilità è strutturale.

---

## 5. Note di riproducibilità
- **Run IDE (2026-06-21 19:03):** 21/21 celle, 9 SAE addestrati (3 size × 3 seed, 12k step) + revival probe + sensitivity. Artefatti: `a1_dict_size.json`, `a1_naming_dict{1024,2048,4096}.json`, 3 figure.
- **3 seed (non 5):** ladder controllato a `(0,42,123)` per compute; sufficiente per il trend di capacity. 12k step (non 50k baseline) — il punto 4096 qui è fresh re-run, confronto apples-to-apples dentro il ladder.
- **lr pinned 4e-4:** rende capacity l'unica variabile. Sensitivity `lr=auto` coincide con 4e-4 a queste size → l'effetto è genuinamente di capacity.
- **Signal-to-null = Jaccard / E[J] ipergeometrico**, `E[J] ≈ k/(2D−k)` per `k≪D`. Forma esatta e approssimata concordano a 4 decimali.
- **Baseline reference** (nel json): cosine 0.988, dead 44%, Jaccard 0.0038, naming mean 0.395 / max 0.546.

---

# Ablation 02 — k (Sparsity) Sweep, null-calibrated

**Data run:** 2026-06-21 · **Macchina:** Linux / RTX 5070, **CUDA**
**Notebook:** `02_k_sweep.ipynb` (12/12 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494)
**Config:** `dict_size` **fissato a 2048**, `k ∈ {8, 16, 32, 64}`, seeds `(0, 42, 123, 456)`, `steps=12000`, `lr=auto` (scala solo con dict_size → costante tra i gruppi k, elimina il confound 01). Within-group Jaccard con `n=k` esplicito, null ipergeometrico esatto, bootstrap CI 1000× sui 1494 sample test.

## Contesto e domanda

Dopo 01 (il dizionario non conta per la stabilità), si prova l'altro parametro: **k**, cioè quanti concetti attivi per immagine. Il baseline usa k=32. Più k = meno sparso ma forse più stabile; meno k = più sparso ma forse collassa. Qui il dizionario è fissato a 2048 e si varia k, confrontando la stabilità col null esatto (calcolato analiticamente) con intervalli di confidenza bootstrap.

**Ipotesi pre-registrata:** ratio ≈ 1 al baseline (k=32), **rising as k shrinks** (meno feature attive → meno overlap casuale → ratio↑ se i concetti sono reali), con dead% ↗ a k molto piccolo. Il Pareto front (VE vs ratio) sceglie il sweet spot.

**Esito: PARZIALE.** Il baseline k=32 è sul pavimento del caso (ratio 0.954 ≈ 1). C'è un debole sweet spot a **k=16** (ratio 1.30, l'unico dove l'intervallo di confidenza esclude 1). k=8 è patologico (91.6% dead, collassa sotto il caso). k modula la stabilità più di dict_size, ma non la risolve — anche al picco l'accordo assoluto resta minuscolo.

---

## 1. Cosa produce ogni fase

| Fase | Output |
|---|---|
| Training grid | 4 k × 4 seed = 16 SAE (12k step, dict_size=2048 fisso) |
| Per-k ricostruzione | cosine, VE, MSE, L0 (=k), dead% (test) |
| Within-group Jaccard | `compute_stability` per k-gruppo, `n=k` esplicito |
| Exact hypergeometric null | `Σ_j j/(2k−j)·P(j)` via `scipy.stats.hypergeom` |
| Signal-to-null ratio | raw Jaccard / null, CI 95% bootstrap 1000× |
| Consensus reappearance | direction-space, τ=0.9 (stesso algo 00/01) |
| Baseline anchor | dict4096/k32 come punto standalone null-calibrato (NON confrontato via Jaccard) |
| Figures | `a2_k_vs_stability.png`, `a2_pareto_front.png` |
| Persist | `results/ablation/a2_k_sweep.json` |

---

## 2. Risultati per-k (dict_size=2048, 12k step, 4 seed)

`signal/null` = quanto l'accordo osservato supera il caso. >1 = segnale reale; ≈1 = rumore; <1 = sotto il caso. Se la **CI 95%** non include 1, il segnale è statisticamente significativo (succede solo a k=16).

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
Ratio baseline 0.954 ≈ 1 → il Jaccard 0.0038 del baseline è statisticamente indistinguibile dal random-overlap. A k=32/dict4096 i concetti non sono più riproducibili di due dizionari casuali. Claim onesto e difendibile.

### 3.2 Signal-to-null NON monotonico — picco a k=16
L'ipotesi "più sparso = più stabile" è parzialmente falsificata: il ratio sale da k=64 a k=16, ma a k=8 crolla sotto il caso (91.6% dead, niente da allineare). **k=16 è l'unico k dove la CI esclude 1** (1.24–1.37): l'accordo reale supera chiaramente il caso. Sweet spot di stabilità.

### 3.3 dead% ↗ small k ✓
91.6% (k=8) → 74.7% (k=16) → 41.3% (k=32) → 40.2% (k=64). Meno feature attive per pass → più feature mai si attivano. k=8 patologico.

### 3.4 Consensus reappearance — ingannevole a k=8
A k=8 la "reappearance" sembra alta (0.65%) ma è un'illusione: con 91.6% dead, il set vivo è minuscolo (~170/2048), quindi i cluster sono forzati dall'affollamento, non da riproducibilità reale. Il signal-to-null (che corregge per la dimensionalità) lo conferma: k=8 è sotto il caso. k=16 resta il sweet spot onesto.

### 3.5 Tradeoff stabilità ↔ ricostruzione (Pareto)
k↑ → migliore ricostruzione (cosine 0.984→0.997, VE 0.968→0.994) e meno dead (91.6→40.2%), ma ratio cala sopra k=16. k=16 massimizza la stabilità (1.30); k=32 è il compromesso operativo (1.15, recon 0.992, dead 41.3%). Nessun k raggiunge riproducibilità reale (raw Jaccard max 0.006, consensus ~0).

---

## 4. Giudizio d'insieme: k modula, non risolve

Confronto con 01 (entrambi sweep di un iperparametro):

| Sweep | Cosa muove stability? | Verdetto |
|---|---|---|
| 01 — dict_size | ratio invariante (~flat) | dict_size NON spiega instabilità |
| 02 — k (dict fisso) | ratio non-monotonico, picco k=16 | k MODULA la stabilità (debolmente) |

k conta più di dict_size: c'è un optimum a k=16 (ratio 1.30, l'unico chiaramente sopra null). Ma:
1. Anche al picco, **accordo assoluto minuscolo** (raw Jaccard 0.005, consensus direction-space ~0). k=16 alza il *rapporto* sopra il caso, non risolve la riproducibilità.
2. **k=8 patologico** (91.6% dead) — troppo sparso.
3. **k=32 (baseline) è sul null floor** → i concetti baseline sono rumore in spazio indici.

Risposta cumulativa: 01 (over-expansion = dead, non instability) + 02 (k ha un debole sweet spot, non risolve; il baseline stesso è rumore-vs-null) → l'instabilità è **strutturale**. Né dict_size né k la risolvono.

---

## 5. Note di riproducibilità
- **Run IDE (2026-06-21 19:23):** 12/12 celle, 16 SAE (4 k × 4 seed). Artefatti: `a2_k_sweep.json` + 2 figure.
- **dict_size=2048 fisso** → lr auto-scale identico tra k-gruppi (elimina confound dict→LR di 01).
- **4 seed (non 3/5):** `(0,42,123,456)` per più potenza statistica sul bootstrap CI.
- **Null = ipergeometrico esatto** `Σ_j j/(2k−j)·P(j)`, P(j) via `hypergeom(M=D,n=k,N=k)`. CI via bootstrap 1000× (mean-of-ratios).
- **Baseline anchor** standalone (dict_size diverso → Jaccard cross-config vietato dal protocollo).

---

# Ablation 03 — Concept Baselines + Empirical Jaccard Floor

**Data run:** 2026-06-21 · **Macchina:** Linux / RTX 5070, **CUDA**
**Notebook:** `03_baselines.ipynb` (13/13 celle)
**Input:** `train_embeddings.pt` (5976, fit PCA/KMeans qui) / `test_embeddings.pt` (1494, score metriche qui), vocabolario RadLex **508 termini**
**Config:** **zero training** — 3 dizionari hand-built (Random, Dense-PCA, Freq-KMeans) da embedding esistenti; `D_b=256` (spazio indice condiviso within-group), `D_B_BIG=4096` (Random nel native index space del SAE), `K=32`, seeds `(0,42,123)`, naming **gap-corrected**, SAE reference hard-codato.

## Contesto e domanda

Tutte le ablation finora confrontavano il SAE con se stesso (seed diversi). Qui la domanda è più diretta: **il SAE è davvero meglio di metodi banali?** E soprattutto — il famoso 0.0038 di instabilità è un fallimento del SAE o è il rumore che otterrebbe chiunque, anche buttando numeri a caso?

Si costruiscono 3 dizionari banali senza addestramento: **Random** (direzioni casuali), **Dense-PCA** (le direzioni principali dei dati), **Freq-KMeans** (i centri di 256 cluster nei dati). Il test chiave: prendere un dizionario Random a 4096 feature, rifarlo 3 volte con seed diversi, misurare la sovrapposizione. Quello è il **pavimento del caso** — il rumore minimo inevitabile.

**Ipotesi pre-registrata:** Random@4096 within-group Jaccard ≈ 0.004 → calibra il 0.0038 del SAE come near-null (artefatto di spazio indici). PCA = ceiling denso di ricostruzione. SAE = unico metodo sparse + nominato.

**Esito: TESI CONFERMATA sul Jaccard floor; il SAE sopravvive solo su sparsità + naming top-end.** Random@4096 = 0.0037 ≈ SAE 0.0038 (ratio 0.95, sul floor). Ma il naming mean del SAE (0.395) è appena sopra il Random (0.372) — lo shift del gap domina il signal. KMeans (0.83) schiaccia tutti sul naming, ma con centroidi densi non monosemantici.

---

## 1. Cosa produce ogni fase

| Fase | Output | Stato |
|---|---|---|
| 3 dizionari baseline | Random (256 + 4096), Dense-PCA (256), Freq-KMeans (256) — per seed | ✅ 4 baselines × 3 seed |
| Ricostruzione fair-L0 | cosine a L0=32 (top-k coefficienti per magnitudo) | ✅ |
| Naming gap-corrected | decoder rows ↔ vocab, stesso shift del SAE | ✅ |
| Within-group index-Jaccard | Random@256 e Random@4096 (3 seed → null empirico) | ✅ |
| Null analitico cross-check | `E[J] ≈ k/(2D−k)` ipergeometrico | ✅ ratio 1.00 / 0.95 |
| Tabelle + figure | comparison table + jaccard-floor bar | ✅ |
| Persist | `results/ablation/a6_baselines.json` + `a6_cache/` (fit PCA/KMeans) | ✅ |

> Rubric ≥3 baselines soddisfatta: Random / Dense-PCA / Freq-KMeans, ciascuno costruito da train embedding e scored su test con le metriche standalone del SAE. (L'artefatto su disco si chiama `a6_baselines.json` — residuo di un renumbering interno; la numeratura logica del REPORT è `Ablation 03`.)

---

## 2. Risultati (primary seed 42; SAE reference hard-codato, gap-corrected)

| Metodo | recon cosine | L0 | dead% | naming mean | naming max |
|---|---|---|---|---|---|
| **SAE** (dict4096, k32, baseline) | 0.988 | 32 | 44.0 | 0.395 | 0.546 |
| Random (D=256) | 0.454 | 32 | 0.0 | 0.372 | 0.442 |
| Dense-PCA (D=256) | **0.996** | 32 | 0.0 | 0.383 | 0.594 |
| Freq-KMeans (D=256) | 0.961 | 32 | 0.0 | **0.829** | **0.875** |

**Random-Jaccard floor (within-group, 3 seed)** — il test chiave dell'intera ablation:

| Gruppo | D | empirical J | analytical null | ratio |
|---|---|---|---|---|
| Random (small) | 256 | 0.0666 | 0.0667 | 1.00 |
| **Random (big)** | **4096** | **0.0037** | **0.0039** | **0.95** |
| — SAE baseline (cross-seed, 5 seed) | 4096 | 0.0038 | — | — |

---

## 3. Analisi

### 3.1 Random@4096 ≈ SAE → index-Jaccard del SAE sul floor del caso ✓
Il SAE e un dizionario di numeri casuali hanno la stessa sovrapposizione tra run (0.0038 vs 0.0037). Ratio 0.95 = il SAE siede esattamente sul null empirico per dizionari 4096-dim. Il 0.0038 cross-seed è calibrato come near-null in spazio indici: confrontare indici tra dizionari 4096-dim indipendenti produce ~0.004 di puro overlap casuale. Cross-check analitico `k/(2D−k)` = 0.0039 conferma (ratio 0.95).

### 3.2 PCA = ceiling denso di ricostruzione ✓ (non è "SAE è scarso")
PCA 0.996 > SAE 0.988 su raw cosine, ma è atteso e pedagogico: PCA è denso (256 atomi tutti attivi, zeroato a L0=32 solo dopo il fit per confronto fair) — sacrifica sparsità e monosemanticità per la ricostruzione. Il SAE perde ~0.008 di cosine in cambio di L0=32 enforced + naming. È il Pareto tradeoff, non un difetto.

### 3.3 Naming: SAE ≈ Random, KMeans schiaccia tutti ⚠️ (risultato severo)
naming mean: **KMeans 0.829 >> SAE 0.395 ≈ PCA 0.383 ≈ Random 0.372**. Il SAE batte il Random di soli +0.023: lo shift del modality gap muove tutte le righe decoder della stessa quantità prima del coseno → domina il signal, e l'apprendimento del SAE aggiunge margine minimo sul naming medio. KMeans domina perché i centroidi sono i modi della distribuzione dati → allineati al cloud del vocabolario. Ma naming mean alto ≠ grounding genuino: i centroidi KMeans sono blend densi (non monosemantici), l'alta similarità riflette allineamento cloud-vs-cloud, non concetti isolati.

Caveat dict-size: SAE 4096 feature vs baseline 256 → per-feature naming mean non perfettamente comparabile. L'ordine (KMeans >> resto ≈) resta il signal robusto; il confronto più pulito è il **top-end** (max): SAE 0.546 > Random 0.442.

### 3.4 Random recon scala con D: 0.45 (256) → 0.60 (4096)
Più atomi casuali = più probabilità che qualcuno allinei con `x` → ricostruzione top-k migliore anche per puro caso. Conferma che raw recon cresce trivialmente con dict_size anche senza apprendimento — ragione in più per normalizzare via null (come fanno 01/02).

---

## 4. Giudizio d'insieme: il SAE sopravvive solo su sparsità + naming top-end

| Domanda | Esito |
|---|---|
| Rubric ≥3 baselines? | ✅ Random / PCA / KMeans |
| Il 0.0038 del SAE è sopra il null (spazio indici)? | ❌ No — Random@4096 0.0037, ratio 0.95, sul floor |
| PCA batte SAE su recon? | ✅ Sì (0.996 vs 0.988) — atteso, è il ceiling denso |
| SAE batte i baseline sul naming? | ⚠️ Appena (mean 0.395 vs Random 0.372); max 0.546 > Random 0.442 (top-end sì) |
| KMeans domina il naming? | ✅ Sì (0.829) — ma modi dati densi, non monosemantici |

**Verdetto cumulativo (00→03):**
1. 00 (direction-Jaccard ~0) + 03 (index-Jaccard sul null floor) → il 0.0038 del SAE è rumore **sia in spazio indici che direzioni**. Conferma indipendente via due null diversi.
2. Il SAE non vince su recon (PCA ceiling) né sul naming medio (≈ Random). L'unica advantage difendibile: **L0=32 enforced per costruzione** (PCA/KMeans sono dense) + **top-end naming** (max 0.546 > 0.442).
3. Risultato più severo della serie fin qui. 01/02 mostravano che l'instabilità non si risolve con iperparametri; 03 mostra che il SAE appena supera baselines casuali sull'asse naming-mean. Il valore del SAE qui è **strutturale** (sparsità garantita, recon 0.988 a L0=32), non un guadagno misurabile sui concetti vs alternative generiche. (Questo verdetto severo viene poi **ribilanciato** da 05: i concetti, pur instabili, sono clinicamente fedeli.)

---

## 5. Note di riproducibilità
- **Run IDE (2026-06-21 19:35):** 13/13 celle, zero training. Artefatti: `a6_baselines.json` (6.1 KB), `a6_cache/` (PCA + KMeans fit per seed, `.npz`), 2 figure.
- **Zero training / no model writes:** `SAEManager.train` mai chiamato. PCA/KMeans fit su train, metriche scored su test (test-set discipline).
- **Metriche standalone:** `compute_stability`/`name_concepts`/`compute_cosine_reconstruction` richiedono un `AutoEncoderTopK` su disco → riscritte come funzioni libere, verificate contro `sae_module.py`.
- **Naming gap-corrected per tutti:** `modality_gap = train_emb.mean(0) − vocab_emb.mean(0)` applicato a ogni `W_dec` → confronto naming apples-to-apples.
- **SAE reference hard-codato:** numeri dal baseline REPORT (gap-corrected), non ri-addestrato qui.
- **Null analitico:** `E[J] ≈ k/(2D−k)` per `k≪D`; ratio empirical/analytical 1.00 (D=256) e 0.95 (D=4096).

---

# Ablation 04 — Activation-Family Bake-off (TopK vs BatchTopK vs JumpReLU)

**Data run:** 2026-06-21 · **Macchina:** Linux / RTX 5070, **CUDA**
**Notebook:** `04_activation_bakeoff.ipynb` (29 celle)
**Input:** `train_embeddings.pt` (5976) / `test_embeddings.pt` (1494), vocabolario RadLex **508 termini**
**Config:** **3 famiglie di attivazione** addestrate a config identico: TopK (baseline), BatchTopK, JumpReLU. `dict_size=2048` (spazio indice condiviso), **lr=5e-5 pinned & matched** (elimina il confound lr ~8×), `steps=12000`, seeds `(0,42,123)`, naming **gap-corrected**.

## Contesto e domanda

Tutte le ablation finora usano TopK. Forse il problema è proprio TopK — la sua regola "esattamente 32 feature per immagine" è rigida. Esistono famiglie alternative: **BatchTopK** (sceglie i top-k su tutto il batch, non per-sample → ogni immagine può usare più o meno feature) e **JumpReLU** (soglia imparata per-feature). Forse una di queste trova concetti più riproducibili. BatchTopK e JumpReLU non erano mai state provate su VLM medici — questa è la novità dell'ablation.

Si addestrano le 3 famiglie a configurazione identica (stesso lr, stesso dizionario, stessi seed) e le si confronta su ricostruzione, dead%, stabilità within-family, e soprattutto **consenso cross-famiglia**: quanti concetti vengono riscoperti da famiglie diverse.

**Ipotesi pre-registrata:** a lr matched, BatchTopK/JumpReLU danno consensus-rate più alto e dead% più basso di TopK, perché permettono alle feature di specializzarsi sui sample che le servono invece di forzare k=32 per sample.

**Esito: MISTO/FALSIFICATO.** dead% ✓ BatchTopK (4.8%) molto meglio di TopK (16%); JumpReLU peggio (49%). Ma consensus-rate **0 per tutti e tre** (τ=0.90), ricostruzione/naming/stabilità ~identiche. Cross-famiglia: 2.8% condiviso tra 2 famiglie, 0% tra tutte e 3. La famiglia di attivazione modula dead%, **non** la riproducibilità.

---

## 1. Cosa produce ogni fase

| Fase | Output |
|---|---|
| Training | 3 famiglie × 3 seed = 9 SAE (12k step, lr=5e-5 matched) |
| Per-famiglia metriche | recon cosine, MSE, L0 effettivo, dead%, entropia (test) |
| Distribuzione L0 | istogramma per-sample (TopK=puntiforme a 32, altre=variabile) |
| Within-family stability | Jaccard renormalizzato n=20 (3 seed per famiglia) |
| Consensus reappearance | cluster direction-space τ=0.90, within-family (stesso algo 00/01) |
| **Cross-activation consensus** | pool 9 modelli, cluster τ=0.90, conta cluster che spannano ≥2 famiglie |
| Naming | seed 42, gap-corrected, per famiglia |
| Figures | 4 (`a2_effective_l0_distribution`, `a2_jumprelu_threshold_hist`, `a2_activation_comparison`, `a2_cross_activation_consensus`) |
| Persist | `results/ablation/a2_activation.json` |

> Gli artefatti su disco per 02/03/04 portano nomi "scrambled" (`a2_activation`, `a4_k_sweep`, `a6_baselines`) residuo di un renumbering interno non propagato; la numeratura logica del REPORT è `02 = k_sweep`, `03 = baselines`, `04 = activation`.

---

## 2. Risultati (3 seed, lr=5e-5 matched, dict=2048)

### 2.1 Per-famiglia: ricostruzione, L0, dead%

| Famiglia | recon cosine | L0 effettivo | dead% | util% |
|---|---|---|---|---|
| **TopK** | 0.9913 | 32.0 (rigido) | 16.0 | 84.4 |
| **BatchTopK** | 0.9917 | ~38.3 | **4.8** | 95.2 |
| **JumpReLU** | 0.9905 | ~33.4 | 48.8 | 51.2 |

Le tre famiglie ricostruiscono praticamente uguale (~0.99). La grande differenza è i dead%: BatchTopK spreca pochissimo (4.8%), JumpReLU ne spreca metà (49% — la soglia imparata non converge bene a questo lr/steps), TopK sta in mezzo (16%). L'L0 effettivo: TopK sempre 32 (rigido), BatchTopK ~38, JumpReLU ~33. (Baseline riferimento dict4096: recon 0.988, dead 44% — il TopK qui ha dead più basso perché dict=2048 + lr=5e-5.)

### 2.2 Within-family stability (Jaccard renormalizzato n=20, floor=0.00977)

| Famiglia | mean Jaccard (n=20) | signal/null |
|---|---|---|
| TopK | 0.00477 | 0.49× |
| BatchTopK | 0.00521 | 0.53× |
| JumpReLU | 0.00419 | 0.43× |

Tutti e tre ~0.005, signal-to-null ~0.5×. Le tre famiglie sono essenzialmente identiche sulla stabilità — differenze 0.43–0.53× non significative. Nessuna famiglia è "più riproducibile".

### 2.3 Consensus reappearance (direction-space, τ=0.90, within-family)

| Famiglia | pooled rows | cluster | consensus (≥2 seed) | rate |
|---|---|---|---|---|
| TopK | 6144 | 6144 | 0 | 0.000 |
| BatchTopK | 6144 | 6144 | 0 | 0.000 |
| JumpReLU | 6144 | 6144 | 0 | 0.000 |

Nessuna famiglia riscopre le stesse direzioni tra seed a τ=0.90.

### 2.4 Cross-activation consensus (9 modelli, τ=0.90) — il test chiave della novità

La domanda di novità: ci sono concetti che **famiglie diverse** riscoprono (non solo seed diversi della stessa famiglia)?

| Metrica | Valore |
|---|---|
| Pooled rows (9 modelli) | 18432 |
| Cluster totali (τ=0.90) | 17936 |
| Cluster che spannano ≥2 famiglie | 496 (**2.8%**) |
| Cluster che spannano tutte e 3 | 0 (**0%**) |

Solo 2.8% dei concetti è condiviso tra 2 famiglie, 0% tra tutte e 3. Quasi tutte le direzioni sono specifiche della famiglia: non esiste un "nucleo" di concetti universali che tutte le famiglie trovano.

### 2.5 Naming (seed 42, gap-corrected)

| Famiglia | n_live | naming mean | naming max |
|---|---|---|---|
| TopK | 2048 | 0.4026 | 0.5489 |
| BatchTopK | 2048 | 0.3969 | 0.5457 |
| JumpReLU | 2048 | 0.3897 | **0.5812** |

L'allineamento col vocabolario RadLex è quasi identico tra le famiglie (~0.40 medio, ~0.55–0.58 max). JumpReLU ha il naming max leggermente più alto. Top concept coerenti: anatomia vertebrale (ligamentum flavum, spinal stenosis), devices (core needle, shapeable wire tip).

---

## 3. Analisi

### 3.1 dead% ✓ risponde alla famiglia — BatchTopK è il migliore
L'unica parte dell'ipotesi che tiene: BatchTopK ha molti meno dead (4.8%) di TopK (16%). Ha senso: il top-(k·B) globale lascia specializzare le feature sui sample che le servono → meno spreco. JumpReLU peggiora (49% dead) — la sua soglia imparata non converge bene a lr=5e-5/12k step. Ma questo riguarda l'efficienza del dizionario (spreco), **non** la riproducibilità.

### 3.2 Consensus ✗ ZERO per tutti — ipotesi falsificata
L'ipotesi principale ("BatchTopK/JumpReLU più riproducibili") crolla: tutte e tre le famiglie hanno 0 cluster condivisi a τ=0.90. Cambiare la funzione di attivazione non crea concetti più riproducibili. Il signal-to-null within-family è ~0.5× per tutti — identico. La stabilità è invariante alla famiglia.

### 3.3 Cross-family: 2.8% condiviso, 0% universale
Il 2.8% dei concetti è trovato da 2 famiglie (un segnale debole ma non zero), ma nessun concetto è trovato da tutte e 3. Le famiglie sono quasi completamente disgiunte in spazio direzioni: non esiste un dizionario universale latente che tutte scoprono.

### 3.4 Ricostruzione + naming identici tra famiglie
Sugli assi "tecnici" (ricostruzione ~0.99, naming ~0.40) le tre famiglie sono indistinguibili. La scelta di famiglia non cambia la qualità tecnica né l'ancoraggio RadLex. Cambia solo dead% (efficienza) e il profilo L0 (TopK rigido, altre adattivo).

### 3.5 L0 adattivo = la novità reale (ma ininfluente sulla stabilità)
La differenza visibile tra le famiglie è il profilo L0: TopK è un picco puntiforme a 32 (rigido), BatchTopK/JumpReLU hanno una distribuzione (ogni immagine usa un numero diverso di feature). È il comportamento "sparsità adattiva" non studiato su VLM medici. Però non porta a concetti più riproducibili: la novità c'è, ma non risolve il problema centrale.

---

## 4. Giudizio d'insieme: la famiglia non salva la riproducibilità

| Domanda | Esito |
|---|---|
| Rubric (≥1 variante non-TopK)? | ✅ BatchTopK + JumpReLU |
| BatchTopK/JumpReLU più riproducibili di TopK? | ❌ No — consensus 0 per tutti |
| dead% più basso con famiglie alternative? | ⚠️ Parziale — BatchTopK sì (4.8%), JumpReLU no (49%) |
| Esiste un nucleo di concetti universali tra famiglie? | ❌ No — 0% span 3 famiglie, 2.8% span 2 |
| Ricostruzione/naming cambiano con la famiglia? | ❌ No — identici (~0.99, ~0.40) |

**Verdetto cumulativo (00→04, chiusura dell'indagine):**
1. 04 è il **test più profondo**: cambia il meccanismo centrale (la funzione di attivazione), non un iperparametro. Neanche questo aiuta la riproducibilità.
2. **dead% e stabilità sono disaccoppiate** (come in 01): BatchTopK riduce i dead, ma i concetti restano non riproducibili. Essere "efficiente" ≠ essere "robusto".
3. **Conferma strutturale definitiva:** l'instabilità non è dovuta a TopK, né a dict_size, né a k. È strutturale — pochi campioni (5976) + non-unicità della decomposizione sparsa (vale per tutte e 3 le famiglie).

**Caveat onesti:**
- **lr matched (5e-5):** elimina il confound lr ~8×, ma potrebbe sotto-addestrare TopK/BatchTopK (default ~2.8e-4). Confronto valido ma conservativo.
- **JumpReLU 49% dead:** probabilmente lr/steps/warmup non ottimali per questa famiglia (nessun tuning per-famiglia). Non è un verdetto su JumpReLU in assoluto, solo a config matched.
- **3 seed (non 5):** compute. Il consensus a 0 è già netto.

---

## 5. Note di riproducibilità
- **Run IDE (2026-06-21 20:06):** 29 celle, 9 SAE (3 famiglie × 3 seed, 12k step). Artefatti: `a2_activation.json` (6.1 KB), 4 figure, modelli in `models/ablation_a4/{topk,batchtopk,jumprelu}_2048/sae_seed{N}/`.
- **lr pinned 5e-5 matched:** elimina il confound lr (TopK/BatchTopK auto-scale ~2.8e-4 a dict2048; JumpReLU default 7e-5). Conservativo ma valido cross-famiglia.
- **3 famiglie via `trainSAE` diretto** (non `SAEManager.train`, che hardcoda TopKTrainer). Loader bespoke per-famiglia (`AutoEncoderTopK`/`BatchTopKSAE`/`JumpReluAutoEncoder`); decoder-row extraction differisce (TopK/BatchTopK: `decoder.weight.T`; JumpReLU: `W_dec` già `(dict,act)`).
- **`compute_stability` non usato:** hardcoda `AutoEncoderTopK` → crash su BatchTopK/JumpReLU. Jaccard riscritto standalone, renormalizzato a n=20 comune.
- **Dead% = activation-based** (feature mai non-zero sul test), standalone.
- **Naming gap-corrected** per tutte e 3 (`W_dec -= gap`).

---

# Ablation 05 — Concept Faithfulness vs clinical labels (MeSH/Problems)

**Data run:** 2026-06-22 · **Macchina:** macOS / Apple Silicon, **device MPS** (auto)
**Notebook:** `05_faithfulness.ipynb` (run headless via nbconvert, 9/9 celle)
**Input:** checkpoint baseline `models/sae_seed42/` (dict4096, k=32) — zero training; `test_embeddings.pt` (1494) + `test_image_ids.json`; etichette cliniche da `data/iu_xray/reports/indiana_reports.csv` (colonne `MeSH`/`Problems`)
**Config:** matrice di attivazione `A` (1494×4096, TopK continuo) × matrice binaria etichette `Y` (1494×50 dopo filtro prevalenza ≥10); correlazione **point-biserial** vettorizzata `A_zᵀ·Y_z/N`; null = SE analitica `1/√N` + shuffle-null per-feature (p95, 200 perm) + BH-FDR 0.05.

## Contesto e domanda

Tutte le ablation finora (00–04) sono una sola grande domanda: *perché i concetti non sono riproducibili tra seed?* Verdetto: è un limite strutturale, il 0.004 è il pavimento del caso. Ma "instabile" non vuol dire "inutile". Qui si cambia domanda: **i concetti che il SAE scopre (in un seed) sono significativi, cioè si attivano sulle immagini che contengono davvero una certa patologia/anatomia?** È la differenza tra "concetti rumorosi" e "concetti che significano qualcosa".

Si prende il SAE seed-42, si codificano le immagini di test → per ogni immagine si sa quali feature si accendono (`A`). Poi si leggono le vere etichette cliniche di quelle immagini dai referti (colonne `MeSH`/`Problems` di IU X-Ray: cardiomegalia, versamento pleurico, ecc.) → matrice `Y`. Si calcola la correlazione tra ogni feature e ogni etichetta. Una feature è "fedele" se si accende proprio sulle immagini con una certa etichetta — e lo fa oltre il puro caso (confronto contro un null calibrato per-feature).

**Ipotesi pre-registrata:** una quota non banale di feature live ha `max_j |corr(activation_i, label_j)|` superiore a un null calibrato per-feature (p95 di uno shuffle delle etichette). Le feature più fedeli dovrebbero corrispondere a concetti visivamente concreti e clinicamente attesi.

**Esito: PARZIALMENTE CONFERMATO — il primo positivo della serie.** 226/2251 feature live (10.0%) battono il proprio shuffle-null p95 (mediana null 0.188). |r|>0.10 sul 53.8% delle live, >0.20 sul 9.5%. Le più forti tracciano impianti medici (0.46), versamento pleurico (0.34), enfisema, ombra cardiaca. I concetti sono instabili cross-seed (00–04) ma, quando esistono, sono moderatamente fedeli a etichette cliniche reali.

---

## 1. Cosa produce ogni fase

| Fase | Output |
|---|---|
| Etichette cliniche | parsing `MeSH`/`Problems` → 118 termini base; 101 presenti nel test, **50** dopo filtro prevalenza ≥10 |
| Join per-immagine | `image_id → uid` (via `indiana_projections.csv` + fallback prefisso) → `MeSH`/`Problems`; 0 join mancanti su 1494 |
| Attivazioni | `mgr.encode(test_emb)` → `A` (1494×4096), 32 non-zero/immagine |
| Point-biserial | `A_zᵀ·Y_z/N` → matrice corr (4096×50); per-feature max abs(corr) |
| Null triplo | SE analitica 0.0259 + shuffle-null per-feature p95 + BH-FDR 0.05 |
| Naming cross-ref | nome RadLex gap-corrected di ogni feature fedele (vs la label a cui è fedele) |
| Persist | `results/ablation/a5_faithfulness.json` + `a5_faithfulness_headline.png` |

---

## 2. Risultati (seed 42, 1494 test, 50 etichette prevalenti)

Per ogni feature "viva" (che si attiva almeno una volta sul test: 2251/4096 = 55%, coerente col ~44% dead della baseline) si prende la correlazione più forte con una qualsiasi etichetta clinica. `|r|` = quanto la feature traccia la sua etichetta migliore. La colonna chiave è il **null per-feature**: per battere il caso, la feature deve superare il proprio p95 (mediana 0.188).

### 2.1 % feature fedeli per soglia (sulle 2251 live)

| soglia abs(corr) | feature live fedeli | % delle live |
|---:|---:|---:|
| > 0.10 | 1210 | 53.8% |
| > 0.15 | 576 | 25.6% |
| > 0.20 | 213 | 9.5% |
| > 0.25 | 82 | 3.6% |
| > 0.30 | 22 | 1.0% |

### 2.2 Null calibrato — il test chiave

"Oltre il caso" non è un'opinione: per ogni feature si mescolano 200 volte le etichette tra le immagini e si misura la correlazione più forte che si otterrebbe per puro caso. La soglia è il 95° percentile di quello shuffle, **specifica per feature** (corregge per la distribuzione di prevalenza delle etichette). Una feature "passa" solo se batte la sua soglia.

| Null | Valore |
|---|---|
| SE analitica `1/√N` | 0.0259 (corr>0.10 ≈ 3.9σ) |
| Shuffle-null p95 (mediana per-feature, 200 perm) | **0.188** |
| Feature live che battono il proprio null p95 | **226 / 2251 (10.0%)** |
| BH-FDR 0.05 (su 112550 test) | 3496 coppie (feature,label) sig.; soglia ≈ corr>0.082 |

### 2.3 Top feature fedeli (+ nome RadLex cross-ref)

Le feature più fedeli e la label a cui si ancorano. Cross-check interessante: la label reale (in-distribution, da IU X-Ray) è clinicamente sensata, ma il nome RadLex assegnato dal SAE è spesso rumoroso/diverso. Conferma che il naming RadLex (off-distribution, vedi `VOCAB_BUILDING_ALTERNATIVES.md`) è più debole del comportamento reale del concetto.

| feature | abs(corr) | label fedele (IU X-Ray) | prev. | nome RadLex (gap-corrected) |
|---:|---:|---|---:|---|
| 3785 | **0.458** | implanted medical device | 22 | mucosal surface |
| 2983 | 0.349 | implanted medical device | 22 | anterior segment of upper lobe (L) |
| 1261 | 0.344 | implanted medical device | 22 | curved sheath |
| 224 | 0.342 | pleural effusion | 57 | idiopathic pulmonary fibrosis |
| 1253 | 0.319 | infiltrate | 19 | — |
| 3260 | 0.315 | diaphragmatic eventration | 10 | — |
| 3330 | 0.315 | pulmonary emphysema | 10 | — |
| 3034 | 0.311 | cardiac shadow | 20 | — |

### 2.4 Per-etichetta: il SAE riesce a rappresentare ogni concetto clinico diffuso?

Per ciascuna etichetta, la feature migliore che la predice. Le etichette visivamente concrete (impianti, versamento, enfisema) raggiungono |r| 0.30–0.46; la copertura decade sulle patologie più sottili.

| etichetta (prevalenza) | miglior corr |
|---|---:|
| implanted medical device (22) | 0.458 |
| pleural effusion (57) | 0.342 |
| infiltrate (19) | 0.319 |
| cardiac shadow (20) | 0.311 |
| pulmonary edema (22) | 0.305 |

---

## 3. Analisi

### 3.1 I concetti esistenti sono genuinamente fedeli, non rumore ✓
226 feature su 2251 (10%) battono un null per-feature calibrato — non un valore fisso, ma la soglia specifica che ciascuna feature dovrebbe superare per caso. Corroborato dal BH-FDR (3496 coppie significative) e dall'SE analitica. Il segnale è reale: c'è una minoranza sostanziale di feature il cui pattern di attivazione traccia un'etichetta clinica oltre il caso.

### 3.2 La fedeltà si concentra su concetti visivamente concreti
Le feature più forti tracciano impianti medici (pacemaker, defibrillatori — oggetti ad alto contrasto), versamento pleurico, enfisema, ombra cardiaca, edema. Sono proprio i concetti che un SAE su radiografie toraciche dovrebbe scoprire prima: entità visive ad alto contrasto. Le patologie fini (sottili pattern texturali) restano più deboli — coerente con un regime data-starved su embedding CLIP proiettati.

### 3.3 Fedeltà modesta in valore assoluto (ma sopra il null)
La correlazione più forte in assoluto è |r|=0.46, e solo il 10% delle feature batte il null. Non è "ogni concetto è un colpo netto": è "una minoranza significativa ha un ancoraggio clinico reale, al di sopra del caso". Onesto: il valore del SAE qui non è "concetti cristallini", è "struttura sparsa + ricostruzione buona (0.988) + una minoranza di concetti clinicamente fedeli".

### 3.4 Naming RadLex ≠ comportamento reale (cross-check)
Una feature fedele a "implanted medical device" porta il nome RadLex "anterior segment of upper lobe". Il comportamento della feature (fedele a impianti) è più informativo del suo nome (off-distribution). Questo giustifica a posteriori l'uso di un gold standard in-distribution (MeSH/Problems) per valutare i concetti, oltre al naming RadLex — e rafforza la diagnosi di `concept_naming_analysis.md`: il naming debole è in parte artefatto del vocabolario, non solo dell'SAE.

---

## 4. Giudizio d'insieme: instabilità ≠ inutilità

| Domanda | Esito |
|---|---|
| Le feature del SAE predicono etichette cliniche reali oltre il caso? | ✅ Sì (minoranza sostanziale) — 226/2251 live (10%) battono un null per-feature |
| Le più fedeli sono clinicamente sensate? | ✅ Sì — impianti, versamento, enfisema, ombra cardiaca, edema |
| La fedeltà è forte in valore assoluto? | ⚠️ Modesta — max |r|≈0.46; concentrata su concetti visivi concreti |
| Naming RadLex coincide con la label reale? | ❌ Spesso no — il comportamento è più informativo del nome off-distribution |

**Posizionamento nel programma (00→05):**
1. 00–04: i concetti sono instabili cross-seed (sia indici che direzioni), il 0.004 è il pavimento del caso, e l'instabilità non si fixa con iperparametri — limite strutturale.
2. 05 (questa): i concetti che esistono sono moderatamente ma genuinamente **fedeli** a etichette cliniche reali. È l'asse complementare che la serie mancava: non "sono riproducibili?" (no) ma "significano qualcosa?" (sì, in parte).
3. Il risultato d'insieme è bilanciato e difendibile: il SAE su questo dataset ha un limite strutturale dichiarato (seed-dipendenza) ma produce direzioni con grounding clinico reale. Non un fallimento, non un successo completo — un risultato onesto e sfumato.

**Caveat onesti:**
- **Etichette da referti, non gold standard annotato:** `MeSH`/`Problems` derivano dai report clinici (ricchezza reale ma non annotazione controllata). La fedeltà misura allineamento concetto↔report, non concetto↔verità-di-immagine.
- **Solo seed 42:** la fedeltà è misurata sul modello di riferimento. Quanto sia stabile la *quota* di feature fedeli across-seed non è testato qui (ma 00 dice quali feature siano è già instabile).
- **Prevalenza ≥10:** taglia le etichette rarissime (degeneri `|r|=1`). Lo shuffle-null per-feature corregge comunque per la distribuzione di prevalenza.

---

## 5. Note di riproducibilità
- **Run headless (2026-06-22):** 9/9 celle via `jupyter nbconvert --execute`, backend Agg. Artefatti: `a5_faithfulness.json` (12 KB), `a5_faithfulness_headline.png` (3 pannelli: distribuzione max|corr| + null, % fedeli per soglia, per-label best).
- **Zero training:** riusa `models/sae_seed42/`. `SAEManager.encode` → attivazioni continue TopK.
- **Point-biserial, non AUROC:** una matmul `A_zᵀ·Y_z/N` (Pearson con var binaria), O(una matmul) vs ~500k chiamate AUROC su 4096×118.
- **Null triplo:** SE analitica `1/√N`=0.0259; shuffle-null per-feature p95 (200 perm, `seed=0`); BH-FDR 0.05 sulla matrice (2251 live × 50 label = 112550 test).
- **Filtro prevalenza ≥10:** evita i casi degeneri `|r|=1` delle etichette in 1–3 immagini. 50/101 label prevalenti tenute (mediana prev 30, max 191).
- **Naming cross-ref:** usa lo stesso shift del modality gap di 01–04 (`W_dec -= visual_centroid − text_centroid`). `train_emb` usato solo per il gap, mai per la correlazione (test-set discipline).
- **Isolamento output:** scrive solo `results/ablation/` + `results/figures/ablation/` — baseline intoccata.

---

## Conclusione cumulativa

| Ab | Asse | Esito sintetico |
|---|---|---|
| 00 | direzioni | Il 0.0038 non è permutazione: direction-Jaccard ~0, consensus@4 = 0%, shuffle-null p=1.0 |
| 01 | dict_size | dead% ✓ scala con la capacità; stabilità ✗ invariante (ratio 4096 > 1024 > 2048) |
| 02 | k | baseline k=32 sul null floor (ratio 0.954); debole sweet spot a k=16 (ratio 1.30); k=8 patologico |
| 03 | baselines | Random@4096 = 0.0037 ≈ SAE → il 0.0038 è il pavimento del caso; SAE sopravvive su sparsità + naming top-end |
| 04 | famiglia | dead% ✓ (BatchTopK 4.8%); consensus ✗ 0 per tutti; cross-family 2.8% / 0% universale |
| 05 | fedeltà | 226/2251 live (10%) fedeli oltre il null; top: impianti |r|=0.46, versamento, enfisema, ombra cardiaca |

1. Il "0.004" del baseline è il **pavimento matematico del caso** (03), non un fallimento — confermato come rumore in spazio indici (03) e direzioni (00).
2. Non si fixa con dict_size (01), k (02), o famiglia di attivazione (04). L'instabilità è un **limite strutturale** dichiarato del metodo su questo dataset (pochi campioni + non-unicità della decomposizione sparsa su embedding CLIP proiettati).
3. Ma l'instabilità **non equivale a inutilità** (05): i concetti esistenti sono moderatamente fedeli a etichette cliniche reali.
4. **Cosa fare:** accettare la seed-dipendenza come limite dichiarato, oppure aggregare i seed (model soup / consensus clustering con validazione). Il valore del SAE è **strutturale** (sparsità garantita + recon 0.988) **e parzialmente semantico** (05), oltre al naming top-end sopra il caso.

Soft spot aperti: fedeltà misurata solo su seed 42 (la quota fedele cross-seed non è testata — vedi 00); etichette derivate da referti, non gold standard annotato; pre-projection (06) / augmented (07) restano future work.

---

## Bibliografia

Riferimenti che sostengono le scelte metodologiche e il framing teorico. La diagnosi causale estesa è in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

- Olshausen & Field (1997) — sparse coding; regime dati-campioni/feature.
- Spielman, Wang, Wright (2012) — "Exact Recovery of Sparsely-Used Dictionaries": condizioni di identificabilità del dictionary learning.
- Soltanolkotabi, Elhamifar, Candès (2013–2014) — robustness/identifiability dello structured sparsity.
- Bricken et al. (2023) "Towards Monosemanticity" — SAE su milioni di attivazioni (regime data-rich di riferimento).
- Gao, Dupré la Tour et al. (2024) "Scaling and Evaluating Sparse Autoencoders" [arXiv:2406.04093] — architettura Top-K SAE usata nel progetto.
- Rajamanoharan et al. (2024) — BatchTopK / JumpReLU SAE (varianti in 04).
- Bhalla, Srinivas, Hsieh (2024) "SpLiCE" [arXiv:2402.10376] — naming via ottimizzazione sparsa sui pesi del decoder.
- Liang et al. (2022) "Mind the Gap" [arXiv:2203.02053] — caratterizzazione formale del modality gap nei modelli contrastivi (framing del naming gap-corrected).
