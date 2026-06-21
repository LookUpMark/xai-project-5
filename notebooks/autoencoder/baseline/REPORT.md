# REPORT — Run del notebook autoencoder (`pipeline.ipynb`)

**Data run:** 2026-06-21
**Macchina:** Linux / NVIDIA RTX 5070 Laptop, **device CUDA** (auto-rilevato)
**Input:** embedding BiomedCLIP (512-d) di IU X-Ray — `train_embeddings.pt` (5976), `test_embeddings.pt` (1494), vocabolario RadLex **508 termini** (`text_vocab_embeddings.pt` + `data/vocabulary.json`)
**Config SAE:** Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4), `batch_size=256`, **5 seed** = `(0, 42, 123, 456, 789)`, `primary_seed=42`

> **Novità di questa run — modality gap corretto (Soluzione 1).** BiomedCLIP ha un *modality gap* tra lo spazio visivo e quello testuale (img↔text coseno ~0.27 vs intra-modale ~0.79). Senza correzione, il `concept_naming` confrontava le colonne del decoder (spazio visivo) con gli embedding del vocabolario (spazio testuale) e otteneva allineamenti deboli. La correzione `W_dec -= (visual_centroid − text_centroid)` prima di normalizzare+coseno **ha portato il naming score da mean 0.117 / max 0.29 a mean 0.395 / max 0.55** (~3.4× medio, ~1.9× max). Vedi §2.5.

> Nota storica: i modelli precedenti erano **toy a 100 step** (dead 97.6%, cosine 0.14), poi **sostituiti con 5 modelli reali a 50k step**. I 5 SAE attuali (06-05) sono validi: la correzione del modality gap **non richiede riaddestramento** (è uno shift locale su `W_dec` dentro `name_concepts`, non tocca i pesi salvati). Le metriche di ricostruzione/stabilità sono identiche pre/post fix (path encoder, gap-indipendenti).

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

Il SAE ricostruisce quasi perfettamente gli embedding con soli `k=32` feature attive. **Gap-indipendente** (path encoder, non tocca `W_dec`).

| Seed | MSE (raw) | Cosine | Dead % | Dict util % | Entropy |
|------|-----------|--------|--------|-------------|---------|
| 0 | 4.6e-5 | 0.9882 | 41.5 | 58.5 | 6.3955 |
| 42 | 4.4e-5 | **0.9888** | 44.3 | 55.7 | 6.3219 |
| 123 | 4.6e-5 | 0.9882 | 44.9 | 55.1 | 6.3660 |
| 456 | 4.4e-5 | 0.9887 | 45.7 | 54.3 | 6.3529 |
| 789 | 4.5e-5 | 0.9886 | 43.0 | 57.0 | 6.3258 |

- **Cosine medio ~0.988** → il vettore ricostruito è quasi parallelo all'originale.
- **L0 = 32.0** su tutti i seed = esattamente `k`: il vincolo Top-K è rispettato alla perfezione.
- I 5 seed sono **coerenti tra loro** (MSE 4.4–4.6e-5, cosine 0.9882–0.9888) → riproducibile.

*(La cella "loss curve" riporta MSE su scala di attivazione normalizzata, non confrontabile col MSE raw; il segnale corretto di convergenza è cosine ~0.988 + plateau della curva, rigenerata oggi.)*

### 2.2 Sparsity — ✅ CORRETTA

- **L0 medio = 32.0** = `k`. Vincolo Top-K rispettato.
- **Entropia ~6.3 nats** → attivazioni distribuite su ~`e^6.3 ≈ 540` feature nel test set (uso diffuso, non concentrato).

### 2.3 Dead features — ⚠️ MODERATA

- **~41–46% delle 4096 feature non si attiva mai** sul test (dict utilization ~54–59%).
- ~1800 feature "morte" → dizionario (4096) **sovradimensionato** per ~7400 immagini. Atteso per dataset piccoli.
- ⚠️ **Due definizioni di "dead" divergenti** (come da CLAUDE.md):
  - *Naming dead* (decoder a norma zero, in `concept_names.json`): **0** (nessuna — la lib `dictionary_learning` normalizza a unit-norm ogni colonna ad ogni step, quindi non esistono colonne a norma zero post-training).
  - *Activation dead* (mai attiva sul test, in stability): **~44%**.

### 2.4 Stabilità cross-seed — ❌ LA CRITICA PRINCIPALE

**Mean Jaccard = 0.0039** (matrice 5×5, off-diagonal ~0.002–0.009). **Gap-indipendente.**

| | 0 | 42 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.004 | 0.009 | 0.003 | 0.003 |
| 42 | 0.004 | 1.00 | 0.004 | 0.003 | 0.004 |
| 123 | 0.009 | 0.004 | 1.00 | 0.003 | 0.002 |
| 456 | 0.003 | 0.003 | 0.003 | 1.00 | 0.003 |
| 789 | 0.003 | 0.004 | 0.002 | 0.003 | 1.00 |

- I 5 SAE **ricostruiscono ugualmente bene ma con feature quasi completamente diverse**. Condividono <0.4% delle feature attive.
- → I "concetti" scoperti **non sono robusti/riproducibili**: dipendono fortemente dal seed. Il seed primario 42 è arbitrario; cambiandolo, naming/explanations cambiano.
- **Questo è il problema aperto "scarsa robustezza dei concetti" che il progetto cita esplicitamente** — risultato atteso ma significativo da discutere.

### 2.5 Concept naming — ✅ MIGLIORATO (gap-corrected)

> Headline di questa run: la correzione del **modality gap** ha risolto il principale limite del naming.

| Metrica | Pre-fix (stale) | **Post-fix (questa run)** | Δ |
|---|---|---|---|
| Mean score | 0.117 | **0.3949** | ×3.4 |
| Max score | 0.291 | **0.5457** | ×1.9 |
| Min score | −0.063 | **0.2815** | — |

- 4096 feature nominate, **0 marcate `DEAD_FEATURE`**.
- **Score medio 0.395, max 0.55** → allineamento decoder↔vocabolario ora **solido** (prima 0.29 era weakly-grounded). La correzione `W_dec -= gap` porta le colonne del decoder nello spazio testuale prima del coseno.
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

### 2.6 Explanations — ✅ STRUTTURALMENTE CORRETTE, ⚠️ image_id DA RIPRISTINARE

- **1494 record** (uno per immagine di test). Schema del contract judge verificato: `{image_id, top_k_concepts[].{feature_id,name,activation}, pseudo_report}`.
- Attivazioni: mean 0.1423, max 0.3865 (coerenti col naming ora gap-corrected).
- ⚠️ **`image_id`: TUTTI e 1494 sono fallback `sample_{0..1493}`** — i sidecar `embeddings/{visual,train,test}_image_ids.json` **mancano** sul disco. La vecchia run (06-15) li aveva e produceva basename PNG reali; ora l'estrazione degli embedding del 06-05 non li ha scritti (o sono stati persi).
  - **Impatto:** il judge LLM joina su `reports.csv` via `image_id`. Con `sample_N` il join **fallisce** → output del judge non allineabile ai report reali.
  - **Fix richiesto:** ri-estrarre gli embedding con propagazione degli image-id (richiede dataset IU X-Ray locale, attente assenti: `data/iu_xray/images/images_normalized` vuoto), oppure ricostruire i sidecar dall'ordine di `reports.csv` se l'estrazione rispetta quell'ordine.
- Esempio (`sample_0`): `intervertebral foramen`, `progressive massive fibrosis`, `left coronary artery`, `ligamentum flavum`… — pseudo-report template-based.

---

## 3. Giudizio d'insieme: è un risultato sensato?

**Sì, e nettamente migliorato rispetto alla run pre-fix per il naming.**

| Obiettivo | Esito |
|---|---|
| Il SAE impara a decomporre gli embedding? | ✅ Sì, ricostruisce a cosine ~0.988 con k=32. |
| I concetti sono *sparsi e monosemantici*? | ✅ Sparsi (L0=32); la monosemantia è ora più plausibile (naming mean 0.395). |
| I concetti sono *robusti*? | ❌ **No** — Jaccard 0.004, dipendono dal seed. (Non risolto dal gap fix.) |
| I concetti sono *clinicamente ancorati*? | ✅ **Sì, migliorato** — allineamento RadLex da max 0.29 a **max 0.55** (mean 0.117→0.395). |
| La pipeline produce output judge-ready? | ⚠️ Parzialmente — schema corretto, ma `image_id` tutti fallback → **judge non allineabile finché non si ripristinano i sidecar**. |

**Punti chiave per la discussione:**
1. Il **modality gap era il colpevole** del naming debole: corretto, +3.4× medio.
2. Il SAE *funziona tecnicamente* ma la **scoperta dei concetti non è stabile cross-seed** (Jaccard 0.004) — materiale per "failure cases / limiti", non un bug.
3. **Bloccante operativo:** ripristinare i sidecar image-id prima di rilanciare il judge.

---

## 4. Cosa provare a cambiare nelle prossime run (in ordine di priorità)

### 🔴 Priorità alta

1. **Ripristinare i sidecar image-id** (blocca il judge). Ri-estrarre gli embedding con dataset IU X-Ray locale, o ricostruire `*_image_ids.json` dall'ordine di `reports.csv`.
2. **Stabilità cross-seed (Jaccard 0.004)** — il problema aperto principale, non toccato dal gap fix:
   - **Ridurre `dict_size`** (4096 → 2048/1024): meno gradi di libertà per divergere tra seed (preset "Conservative" suggerisce 2048). Atteso: Jaccard più alto, dead più bassi.
   - **LR più basso** (`5e-5`): convergenza più stabile, meno optima locali.
   - **Aggregazione cross-seed**: clusterizzare le feature dei 5 seed nello spazio del decoder (consensus) e usare i cluster come "concetti stabili" — mitiga senza riaddestrare (vedi `ablation/00_consensus.ipynb`).
   - **Model soup / init condiviso** tra seed.

### 🟡 Priorità media

3. **Resampling dead feature** durante il training o `dict_size` minore → riduce il ~44% di spreco.
4. **Naming oltre il coseno**: provare SPLiCE (ottimizzazione sparsa sui pesi del decoder) invece del coseno greedy, ora che la baseline gap-corrected è solida (0.395).
5. **Validazione qualitativa**: i top-concept (endotracheal tube, ligamentum flavum, ecc.) sono plausibili — campionare immagini attivanti e verificare visivamente.

### 🟢 Priorità bassa

6. **Determinismo completo** (`PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG`).
7. **W&B abilitato** per tracciare curve sulle ablation `dict_size`/`k`/`lr`.

### Ablation consigliata (prossima sessione)

Matrice 2×2: `dict_size ∈ {1024, 2048}` × `lr ∈ {auto, 5e-5}`, seed 42 singolo, ~10k step. Misurare **Jaccard vs altri seed** e **dead%**. Tenere il config che massimizza stabilità, poi rilanciare a 50k × 5 seed.

---

## 5. Note di riproducibilità & stato

- **CUDA (RTX 5070)**: metriche di questa run (cosine 0.988, dead ~44%) coerenti con la run MPS di riferimento (06-15) → il cambio device non ha alterato i risultati (cross-device riproducibile).
- **Modality gap è cached** (`models/modality_gap.pt`): `compute_and_save_modality_gap()` ha guardia skip-if-exists senza check di contenuto. Se gli embedding vengono rigenerati con split diverso, il gap va **cancellato a mano** (`rm models/modality_gap.pt`) prima di rilanciare, altrimenti resta stale.
- **`vocabulary.json` = 508 dict** `{"term","similarity_score","source"}` (output del builder multi-centroid). I consumer (CLI `concept_naming.py` e notebook) normalizzano a `term`-stringa; `name_concepts` inoltre coerce via `_vocab_term` come safety net. Il vecchio "schema #7 aperto" è **risolto**.
- **Schema judge rispettato** tranne `image_id`: `sample_explanations.json` ha struttura corretta ma `image_id` tutti `sample_N` (sidecar mancanti) → judge non allineabile finché non ripristinati (§2.6, §4.1).
- **5 SAE non riaddestrati**: la correzione del modality gap è uno shift locale su `W_dec` in `name_concepts`, non persistito nei pesi. I modelli 06-05 restano validi.
