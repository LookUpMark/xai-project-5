# REPORT — Run del notebook autoencoder (`pipeline.ipynb`)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA** (auto-rilevato)
**Input:** embedding BiomedCLIP (512-d) di IU X-Ray — `train_embeddings.pt` (5976), `test_embeddings.pt` (1494), vocabolario RadLex **508 termini** (`text_vocab_embeddings.pt` + `data/vocabulary.json`)
**Config SAE:** Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4), `batch_size=256`, **5 seed** = `(0, 42, 123, 456, 789)`, `primary_seed=42`

**Companion:** `../ablation/REPORT.md` estende questo run con il programma di ablation (00–05). L'instabilità cross-seed osservata qui (Jaccard 0.0039) è la domanda madre di quel programma; la sua interpretazione come "pavimento del caso" è stabilita dall'Ablation 03, e la fedeltà clinica dei concetti dall'Ablation 05.

**Indice**
- [Sintesi esecutiva](#sintesi-esecutiva)
- [Glossario](#glossario)
- [Metriche: definizioni formali](#metriche-definizioni-formali)
- [1. Cosa ha prodotto ogni fase](#1-cosa-ha-prodotto-ogni-fase)
- [2. Risultati nel dettaglio](#2-risultati-nel-dettaglio)
- [3. Giudizio d'insieme](#3-giudizio-dinsieme)
- [4. Direzioni successive (gia' coperte dalle ablation)](#4-direzioni-successive-gia-coperte-dalle-ablation)
- [5. Note di riproducibilità & stato](#5-note-di-riproducibilita--stato)

---

## Sintesi esecutiva

La pipeline trasforma le radiografie in concetti interpretabili. BiomedCLIP converte ogni immagine in un vettore opaco di 512 dimensioni; lo **Sparse Autoencoder (SAE)** decompone ciascun vettore in `k=32` feature (concetti) prese da un dizionario di 4096. L'obiettivo è sostituire un vettore denso illeggibile con una lista corta di concetti medici (es. "tubo endotracheale", "anatomia vertebrale").

Tre assi di valutazione:

| Asse | Risultato |
|---|---|
| **Ricostruisce bene?** (qualità tecnica) | ✅ Cosine 0.988 con soli 32 concetti su 4096 — quasi perfetto |
| **I concetti hanno senso medico?** (interpretabilità) | ✅ Dopo il fix del modality gap: allineamento RadLex mean 0.40 / max 0.55 (era 0.12 / 0.29) |
| **Sono riproducibili?** (robustezza) | ❌ 5 run scoprono concetti quasi completamente diversi (Jaccard 0.004). Limite strutturale, non un bug |

**La novità di questa run — modality gap corretto (Soluzione 1).** BiomedCLIP ha un modality gap: lo spazio dei vettori visivi e quello dei vettori testuali non coincidono, sono "traslati" l'uno rispetto all'altro (img↔text coseno ~0.27 vs intra-modale ~0.79). Senza correzione si confrontavano le colonne del decoder (spazio visivo) con gli embedding del vocabolario (spazio testuale) — come misurare la distanza tra due città su mappe con coordinate sfalsate. La correzione `W_dec -= (visual_centroid − text_centroid)` riallinea le mappe prima del confronto. **Risultato: naming score da mean 0.117 / max 0.29 a mean 0.395 / max 0.55** (~3.4× medio).

Nota storica: i modelli precedenti erano **toy a 100 step** (dead 97.6%, cosine 0.14), poi sostituiti con **5 modelli reali a 50k step**. I 5 SAE attuali sono validi: la correzione del modality gap non richiede riaddestramento (è uno shift locale su `W_dec` dentro `name_concepts`, non tocca i pesi salvati). Ricostruzione e stabilità sono identiche pre/post fix.

**Sull'instabilità (Jaccard 0.004).** Questo report la documenta come il limite principale, ma non la risolve: è materiale per le ablation successive. Il risultato chiave, stabilito in `../ablation/REPORT.md` (Ablation 03), è che il 0.0039 **non è un fallimento del SAE** ma il pavimento matematico del caso — due dizionari grandi e indipendenti si sovrappongono sempre per ~0.004 per pura probabilità (Random@4096 = 0.0037 ≈ SAE). E l'Ablation 05 mostra che i concetti, pur instabili, sono moderatamente **fedeli** a etichette cliniche reali (~10% delle feature live oltre il null). L'instabilità non equivale a inutilità.

---

## Glossario

- **SAE (Sparse Autoencoder)** — ricostruisce un embedding `x` come `x̂ = W_dec·z + b_dec`, dove `z` è un codice **sparso** (pochi non-zero). TopK forza esattamente `k` non-zero. Le colonne di `W_dec` sono le "direzioni" dei concetti.
- **Cosine reconstruction** — `cos(x, x̂)`: quanto la ricostruzione è parallela all'originale. 0.988 = perdita minima.
- **L0** — numero di feature attive (non-zero) per immagine. Qui `k=32` esatto per costruzione (TopK).
- **Dead feature** — due definizioni divergenti (vedi Metriche):
  - *naming dead* = colonna del decoder a norma zero. **0** qui (la libreria normalizza ogni colonna ad ogni step).
  - *activation dead* = feature mai attiva sul test. **~44%** qui (dizionario sovradimensionato per ~7400 immagini).
- **Jaccard cross-seed** — sovrapposizione tra i set di indici attivi di due SAE: `|A∩B|/|A∪B|`. 0.004 = concetti quasi completamente diversi tra seed. Sensibile alla permutazione degli indici (l'Ablation 00 lo verifica nello spazio delle direzioni).
- **Concept naming** — per ogni feature, il termine RadLex più simile (coseno tra la direzione della feature e l'embedding del termine). Score alto = concetto ancorato a un termine reale.
- **Modality gap** — scostamento geometrico sistematico tra spazio immagini e spazio testi nei modelli contrastivi. Corretto post-hoc con `W_dec -= (visual_centroid − text_centroid)`. Analisi completa in `docs/suggestions/concept_naming_analysis.md`.
- **Explanations (pseudo-report)** — per ogni immagine di test, i suoi top-k concetti attivi assemblati in una descrizione testuale. È l'input che il giudice LLM (MedGemma) valuterà.

---

## Metriche: definizioni formali

**Ricostruzione (TopK SAE).** `x̂ = W_dec·z + b_dec`, con `z = topk(ReLU(W_enc·(x − b_enc) + b_enc), k)`. TopK azzera tutti tranne i `k` valori più alti.
- Cosine: `cos(x, x̂) = ⟨x, x̂⟩/(‖x‖·‖x̂‖)`.
- Variance Explained: `VE = 1 − ‖x − x̂‖²/‖x − b_dec‖²` (~99.3%).
- L0: `‖z‖₀ = k = 32` (rigido, garantito da TopK).
- Entropia: `H(p) = −Σᵢ pᵢ log pᵢ` sulla distribuzione di utilizzo delle feature sul test (~6.3 nats → ~540 feature usate).

**Dead feature.**
- naming dead: `‖w_i‖ ≈ 0` (0 qui, colonne unit-norm post-training).
- activation dead: `∀ s : zᵢ(s) = 0` sul test (~44% qui).

**Jaccard cross-seed.** `J(A,B) = |A∩B|/|A∪B|` dove `A,B` = set di indici attivi di due SAE su un sample. Mean sulle 10 coppie di seed. Null analitico `k/(2D−k)` = 0.0039 a D=4096, k=32 → ratio ~1 (sul pavimento del caso).

**Modality gap (naming gap-corrected).** `gap = mean(train_emb, 0) − mean(vocab_emb, 0)`; `W_dec ← W_dec − gap`, poi `F.normalize` righe + coseno con `F.normalize(vocab_emb)`. Corrisponde a *Soluzione 1* di `docs/suggestions/concept_naming_analysis.md`.

---

## 1. Cosa ha prodotto ogni fase

| Fase | Output | Stato |
|---|---|---|
| Split train/test | `train/test_embeddings.pt` | ⏭️ saltato (già currente) |
| **Training SAE** | `models/sae_seed{0,42,123,456,789}/` (50k step) | ✅ 5 modelli (riusati, 06-05) |
| Modality gap | `models/modality_gap.pt` (512-d) | ✅ pre-computato, riusato |
| Loss curve | `figures/loss_curve.png` + 50 checkpoint | ✅ rigenerata (17:41) |
| Concept naming | `results/concept_names.json` (4096 feature) | ✅ gap-corrected |
| Explanations | `results/sample_explanations.json` (1494 record) | ✅ ⚠️ vedi image_id |
| Stability | `results/stability_analysis.json` (Jaccard 5×5 + per-seed) | ✅ |
| Figure | `concept_scores_dist`, `per_seed_metrics`, `jaccard_heatmap`, `sparsity_summary`, `concept_activations_heatmap`, `loss_curve` | ✅ rigenerate |

---

## 2. Risultati nel dettaglio

### 2.1 Qualità di ricostruzione — ✅ OTTIMA

La ricostruzione misura se, ricombinando i 32 concetti attivi per un'immagine, si ottiene di nuovo il vettore originale. Un cosine di 0.988 significa che il vettore ricostruito è quasi parallelo all'originale: la decomposizione perde pochissimo. Questa metrica è **gap-indipendente** (usa il path encoder, non tocca `W_dec`).

| Seed | MSE (raw) | Cosine | Dead % | Dict util % | Entropy |
|------|-----------|--------|--------|-------------|---------|
| 0 | 4.6e-5 | 0.9882 | 41.5 | 58.5 | 6.3955 |
| 42 | 4.4e-5 | **0.9888** | 44.3 | 55.7 | 6.3219 |
| 123 | 4.6e-5 | 0.9882 | 44.9 | 55.1 | 6.3660 |
| 456 | 4.4e-5 | 0.9887 | 45.7 | 54.3 | 6.3529 |
| 789 | 4.5e-5 | 0.9886 | 43.0 | 57.0 | 6.3258 |

- **Cosine medio ~0.988** → vettore ricostruito quasi parallelo all'originale.
- **L0 = 32.0** su tutti i seed = esattamente `k`: il vincolo Top-K è rispettato alla perfezione (ogni immagine usa esattamente 32 concetti, mai uno di più).
- I 5 seed sono coerenti tra loro (MSE 4.4–4.6e-5, cosine 0.9882–0.9888) → riproducibile a livello di *qualità*.

*(La cella "loss curve" riporta MSE su scala di attivazione normalizzata, non confrontabile col MSE raw; il segnale corretto di convergenza è cosine ~0.988 + plateau della curva.)*

### 2.2 Sparsity — ✅ CORRETTA

La sparsità è il punto di un SAE: un vettore denso (tutti i 512 numeri) è illeggibile, 32 concetti sono interpretabili. L0=32 e entropia ~6.3 confermano che la sparsità funziona come voluto.

- **L0 medio = 32.0** = `k`. Vincolo Top-K rispettato.
- **Entropia ~6.3 nats** → le attivazioni sono distribuite su ~`e^6.3 ≈ 540` feature diverse nel test set (uso diffuso, non concentrato su poche feature).

### 2.3 Dead features — ⚠️ MODERATA

Le feature "dead" non si attivano mai, su nessuna immagine: sono spreco. ~44% significa che su 4096 feature, ~1800 sono inutilizzate. Succede perché il dizionario (4096) è sovradimensionato rispetto al numero di immagini (~7400). Atteso per dataset piccoli — non fatale, ma è spreco.

- **~41–46% delle 4096 feature non si attiva mai** sul test (dict utilization ~54–59%).
- ~1800 feature "morte" → dizionario (4096) sovradimensionato per ~7400 immagini.
- ⚠️ **Due definizioni di "dead" divergenti** (come da CLAUDE.md) — importante non confonderle:
  - *Naming dead* (decoder a norma zero, in `concept_names.json`): **0** (nessuna — la lib `dictionary_learning` normalizza a unit-norm ogni colonna ad ogni step, quindi non esistono colonne a norma zero post-training).
  - *Activation dead* (mai attiva sul test, in stability): **~44%**.

### 2.4 Stabilità cross-seed — ❌ LA CRITICA PRINCIPALE

Questa è la metrica più importante per la fidatezza dei concetti. Addestrando 5 volte lo stesso modello (cambiando solo il seed, cioè il punto di partenza casuale), i 5 risultati trovano gli stessi concetti? La risposta è quasi per niente: condividono <0.4% delle feature attive. In pratica i "concetti" estratti dipendono fortemente da quale delle 5 run si sceglie. Il seed primario 42 è arbitrario; cambiandolo, naming ed explanations cambiano. Questo è il limite strutturale del progetto, dichiarato apertamente.

**Mean Jaccard = 0.0039** (matrice 5×5, off-diagonal ~0.002–0.009). **Gap-indipendente.**

| | 0 | 42 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.004 | 0.009 | 0.003 | 0.003 |
| 42 | 0.004 | 1.00 | 0.004 | 0.003 | 0.004 |
| 123 | 0.009 | 0.004 | 1.00 | 0.003 | 0.002 |
| 456 | 0.003 | 0.003 | 0.003 | 1.00 | 0.003 |
| 789 | 0.003 | 0.004 | 0.002 | 0.003 | 1.00 |

- I 5 SAE ricostruiscono ugualmente bene ma con feature quasi completamente diverse. Condividono <0.4% delle feature attive.
- → I "concetti" scoperti non sono robusti/riproducibili: dipendono fortemente dal seed.
- Questo è il problema aperto "scarsa robustezza dei concetti" che il progetto cita esplicitamente — risultato atteso ma significativo da discutere.
- Le ablation 00–05 investigano se questo 0.004 sia un vero fallimento o il "pavimento del caso" matematico. **Spoiler: è sul pavimento del caso** (Ablation 03: Random@4096 = 0.0037 ≈ SAE), e i concetti esistenti sono comunque clinicamente fedeli (Ablation 05). Vedi `../ablation/REPORT.md`.

### 2.5 Concept naming — ✅ MIGLIORATO (gap-corrected)

Il naming assegna a ogni feature il termine medico RadLex più simile (coseno tra la direzione della feature e l'embedding del termine). Score alto = concetto ancorato a un termine reale → interpretabile. Prima del fix era ~0.12 (debolissimo, i nomi erano quasi casuali); dopo il fix ~0.40 medio con picchi a 0.55. I concetti top sono clinicamente plausibili: tubi/devices endocavitari, anatomia vertebrale, vie spinale.

Headline di questa run: la correzione del **modality gap** ha risolto il principale limite del naming.

| Metrica | Pre-fix (stale) | **Post-fix (questa run)** | Δ |
|---|---|---|---|
| Mean score | 0.117 | **0.3949** | ×3.4 |
| Max score | 0.291 | **0.5457** | ×1.9 |
| Min score | −0.063 | **0.2815** | — |

- 4096 feature nominate, **0 marcate `DEAD_FEATURE`**.
- **Score medio 0.395, max 0.55** → allineamento decoder↔vocabolario ora solido (prima 0.29 era weakly-grounded). La correzione `W_dec -= gap` porta le colonne del decoder nello spazio testuale prima del coseno.
- **Top-8 per score (clinicamente plausibili e ora fortemente ancorati):**

  | Feat | Nome | Score |
  |---|---|---|
  | 690 | cricothyroid tube | 0.5457 |
  | 1090 | ligamentum flavum | 0.5230 |
  | 1806 | moderate central spinal stenosis | 0.5192 |
  | 3824 | sacral segment of spinal epidural space | 0.5172 |
  | 1239 | right spinotectal tract of spinal cord | 0.5159 |
  | 2059 | endotracheal tube | 0.5117 |
  | 2977 | Foramina vertebralia | 0.5114 |
  | 1172 | brachytherapy catheter | 0.5114 |

- I concetti top sono coerenti tra loro (tubi/devices endocavitari, anatomia vertebrale, vie spinale) → il dizionario cattura pattern reali dello spazio di embedding.
- **Come è implementato:** `train_sae.compute_and_save_modality_gap()` precomputa `gap = train_emb.mean(0) − vocab_emb.mean(0)` → `models/modality_gap.pt`; `sae_module.name_concepts(..., modality_gap=gap)` fa `W_dec = W_dec − gap` poi `F.normalize` + coseno. Corrisponde a *Soluzione 1* di `docs/suggestions/concept_naming_analysis.md`.
- **Fix collaterale di questa run:** `name_concepts` ora coerce i label dict → stringa (`_vocab_term`), così i consumer che caricano `vocabulary.json` come lista di dict (notebook baseline) ottengono comunque `name` stringa invece di far crashare `generate_explanation`.
- **Cross-check con la fedeltà (Ablation 05):** il naming RadLex è *off-distribution* e talvolta rumoroso. L'Ablation 05 mostra che una feature può essere fedele a "implanted medical device" pur portando il nome RadLex "anterior segment of upper lobe" — il comportamento è più informativo del nome. Vedi `../ablation/REPORT.md` §Ablation 05.

### 2.6 Explanations — ✅ STRUTTURALMENTE CORRETTE, ✅ image_id RIPRISTINATI

Per ogni immagine di test si prendono i suoi top-k concetti attivi e li si assemblano in uno pseudo-report (una descrizione testuale generata dai concetti). È l'output finale che il giudice LLM (MedGemma) valuterà: "questo pseudo-report è allineato col referto clinico reale dell'immagine?". Ora pronti: 1494 record, schema corretto, image_id reali (niente più placeholder), referti joinati.

- **1494 record** (uno per immagine di test). Schema del contract judge verificato: `{image_id, top_k_concepts[].{feature_id,name,activation}, pseudo_report}`.
- Attivazioni: mean 0.1423, max 0.3865 (coerenti col naming gap-corrected).
- **`image_id`: 1494/1494 basename reali** (e.s. `3222_IM-1522-2001.dcm.png`) — sidecar `embeddings/{visual,train,test}_image_ids.json` ricostruiti (vedi §5). Prima erano tutti fallback `sample_N` (sidecar mancanti dalla run 06-05); ora joinabili via `indiana_projections.csv` (filename→uid) → `indiana_reports.csv` (uid→findings).
  - ✅ **Judge-ready:** `data/iu_xray/reports.csv` generato (7466 righe, colonne `image_id`+`combined_text`, join filename→uid→findings). Lookup end-to-end verificato: **1493/1494** test image_id hanno report non-vuoto (1 PNG orfano senza voce in `indiana_projections.csv` → judge lo salta).
- Esempio (`3222_IM-1522-2001.dcm.png`): `intervertebral foramen`, `progressive massive fibrosis`, `left coronary artery`, `ligamentum flavum`… — pseudo-report template-based.

---

## 3. Giudizio d'insieme

Risultato sensato, e nettamente migliorato rispetto alla run pre-fix per il naming.

| Obiettivo | Esito |
|---|---|
| Il SAE impara a decomporre gli embedding? | ✅ Sì, ricostruisce a cosine ~0.988 con k=32 |
| I concetti sono *sparsi e monosemantici*? | ✅ Sparsi (L0=32); la monosemantia è ora più plausibile (naming mean 0.395) |
| I concetti sono *robusti*? | ❌ No — Jaccard 0.004, dipendono dal seed (non risolto dal gap fix; ma è il pavimento del caso, vedi Ablation 03) |
| I concetti sono *clinicamente ancorati*? | ✅ Migliorato — allineamento RadLex da max 0.29 a max 0.55 (mean 0.117→0.395) |
| La pipeline produce output judge-ready? | ✅ Schema corretto + image_id reali + reports.csv generato (1493/1494 coperti) |

Punti chiave per la discussione:
1. Il **modality gap era il colpevole** del naming debole: corretto, +3.4× medio.
2. Il SAE *funziona tecnicamente* ma la scoperta dei concetti non è stabile cross-seed (Jaccard 0.004) — materiale per "failure cases / limiti", non un bug. Il programma di ablation ha poi chiarito che è il pavimento del caso (03) e che i concetti esistenti sono clinicamente fedeli (05).
3. **Bloccante operativo:** ripristinare i sidecar image-id prima di rilanciare il judge.

### 3.1 Rapporto con MedConcept (deviazioni dichiarate)

La pipeline è un'istanza **MedConcept-ispirata**, non una replica fedele (Haque et al., arXiv:2604.11868). Lo scheletro è fedele — SAE su un VLM medico → attivazioni sparse → grounding in vocabolario clinico → pseudo-report → giudice LLM Aligned/Unaligned/Uncertain — ma con tre deviazioni materiali sul metodo, dichiarate per onestà:

- **D1 — SAE TopK invece di ReLU+L1.** MedConcept (Eq. 2) impone la sparsità con penalità L1 (λ₁=2e-3); la pipeline usa TopK (selezione hard top-k nell'encoder, niente L1, auxk per i dead). Meccanismo di sparsità diverso (k fisso vs data-dependent), stessa idea di decomposizione sparse.
- **D2 — `dict_size` scollegato dal vocabolario.** MedConcept lega `k = |vocabolario|` (un neurone = un concetto); la pipeline usa `dict_size=4096` contro un vocabolario RadLex di 508 termini → naming many-to-one, ~44% di neuroni mai attivi. Il vincolo 1:1 di MedConcept è assente.
- **D3 — correzione del modality gap.** MedConcept (Eq. 3) usa cosine puro e accetta il vision-text gap come limitazione; la pipeline sottrae un vettore di modality gap (visual_centroid − text_centroid) dalle decoder rows prima del cosine (Mind the Gap, Liang et al.) — passo extra che alza il naming mean da 0.117 a 0.395.

> **Nuance di reporting:** il "mean naming score" (0.395) è mediato su tutti i 4096 feature, inclusi i ~44% *dead-by-activation* (mai attivi sul test), che ricevono etichette poco significative e abbassano la media. Non corrompe le spiegazioni (il judge filtra `activation>0`) ma va dichiarato quando si cita il mean score.

---

## 4. Direzioni successive (gia' coperte dalle ablation)

Le ipotesi di miglioramento elencate qui sotto sono state **verificate dal programma di ablation** (`../ablation/REPORT.md`). Si riportano i verdetti per chiusura.

- **Ridurre `dict_size`** (4096 → 2048/1024): Ablation 01 → riduce i dead (40.9 → 30.7%) ma **NON** aumenta la stabilità cross-seed (ratio non monotonico).
- **Aggregazione cross-seed / consensus**: Ablation 00 → il consensus nello spazio delle direzioni è ~0 (nessuna direzione condivisa tra ≥4 seed su 5); l'aggregazione richiede validazione su τ molto più bassi.
- **LR più basso (`5e-5`)**: rimane non testato come singolo lever; le ablation usano lr pinned per controllare i confound.
- **Variazione di `k`**: Ablation 02 → debole sweet spot a k=16 (ratio 1.30, l'unico sopra null), ma l'accordo assoluto resta minuscolo; k=32 (baseline) è sul pavimento del caso.
- **Famiglia di attivazione alternativa**: Ablation 04 → BatchTopK riduce i dead (4.8%) ma consensus 0 per tutte e tre le famiglie (TopK/BatchTopK/JumpReLU).
- **Naming oltre il coseno (SPLiCE)** e **vocab più ampio/curato**: future work (`docs/suggestions/VOCAB_BUILDING_ALTERNATIVES.md`).
- **Validazione qualitativa / faithfulness**: Ablation 05 → ~10% delle feature live è fedele a etichette cliniche reali oltre il null (impianti, versamento, enfisema).

L'instabilità è un **limite strutturale** del metodo su questo dataset (pochi campioni + non-unicità della decomposizione sparsa su embedding CLIP proiettati). La diagnosi causale completa è in `docs/suggestions/CONCEPT_INSTABILITY_DIAGNOSIS.md`.

---

## 5. Note di riproducibilità & stato

- **CUDA (RTX 5070)**: metriche di questa run (cosine 0.988, dead ~44%) coerenti con la run MPS di riferimento (06-15) → il cambio device non ha alterato i risultati (cross-device riproducibile).
- **Modality gap è cached** (`models/modality_gap.pt`): `compute_and_save_modality_gap()` ha guardia skip-if-exists senza check di contenuto. Se gli embedding vengono rigenerati con split diverso, il gap va **cancellato a mano** (`rm models/modality_gap.pt`) prima di rilanciare, altrimenti resta stale.
- **`vocabulary.json` = 508 dict** `{"term","similarity_score","source"}` (output del builder multi-centroid). I consumer (CLI `concept_naming.py` e notebook) normalizzano a `term`-stringa; `name_concepts` inoltre coerce via `_vocab_term` come safety net. Il vecchio "schema #7 aperto" è risolto.
- **Sidecar image-id ricostruiti** (18:12): `visual/train/test_image_ids.json` rigenerati da `sorted(glob("*.png"))` dei 7470 PNG reali (`chest-xrays-indiana-university/images/images_normalized/`) + split `sklearn(random_state=42, ratio=0.8)`; allineamento righe verificato via `torch.equal` contro i tensor esistenti (match esatto su train + test). `sample_explanations.json` ora ha 1494/1494 basename reali. Niente re-extraction né riaddestramento.
- **`data/iu_xray/reports.csv` generato**: join `indiana_projections`(filename→uid) + `indiana_reports`(uid→findings+impression), colonne `image_id`+`combined_text` (schema richiesto da `evaluate_llm_judge.py`: `zip(image_id, combined_text)`). Lookup judge verificato: **1493/1494** test coperti (1 PNG orfano senza voce in projections). Il judge è ora eseguibile end-to-end.
- **5 SAE non riaddestrati**: la correzione del modality gap è uno shift locale su `W_dec` in `name_concepts`, non persistito nei pesi. I modelli 06-05 restano validi.
