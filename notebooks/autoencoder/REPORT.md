# REPORT — Run del notebook autoencoder (`pipeline.ipynb`)

**Data run:** 2026-06-15
**Macchina:** macOS / Apple Silicon, **device MPS** (auto-rilevato)
**Input:** embedding BiomedCLIP (512-d) di IU X-Ray — `train_embeddings.pt` (5976), `test_embeddings.pt` (1494), vocabolario RadLex 310 termini (`text_vocab_embeddings.pt` + `data/vocabulary.json`)
**Config SAE:** Top-K, `k=32`, `dict_size=4096`, `steps=50000`, `lr=auto` (~4e-4), `batch_size=256`, **5 seed** = `(0, 42, 123, 456, 789)`, `primary_seed=42`

> Nota storica: i modelli locali precedenti erano **toy a 100 step** (dead 97.6%, cosine 0.14). Questa run li ha **sostituiti con 5 modelli reali a 50k step**. Le metriche ora coincidono con la run CUDA di riferimento → buona riproducibilità cross-device.

---

## 1. Cosa ha prodotto ogni fase

| Fase | Output | Stato |
|---|---|---|
| Split train/test | `train/test_embeddings.pt` + sidecar `*_image_ids.json` | ⏭️ saltato (già currente del 14 giu) |
| **Training SAE** | `models/sae_seed{0,42,123,456,789}/` (50k step ciascuno) | ✅ 5 modelli |
| Loss curve | `figures/loss_curve.png` + checkpoint | ✅ convergenza 0.993 |
| Concept naming | `results/concept_names.json` (4096 feature) | ✅ |
| Explanations | `results/sample_explanations.json` (1494 record) | ✅ |
| Stability | `results/stability_analysis.json` (Jaccard 5×5 + per-seed) | ✅ |
| Figure | `concept_scores_dist`, `per_seed_metrics`, `jaccard_heatmap`, `sparsity_summary`, `concept_activations_heatmap`, `loss_curve` | ✅ |

---

## 2. Risultati nel dettaglio

### 2.1 Qualità di ricostruzione — ✅ OTTIMA

Il SAE ricostruisce quasi perfettamente gli embedding con soli `k=32` feature attive:

| Seed | MSE (raw) | Cosine | Var. spiegata |
|------|-----------|--------|---------------|
| 0 | 4.63e-5 | 0.9882 | — |
| 42 | 4.45e-5 | **0.9887** | 0.993 |
| 123 | 4.53e-5 | 0.9885 | — |
| 456 | 4.58e-5 | 0.9884 | — |
| 789 | 4.59e-5 | 0.9883 | — |

- **Cosine medio ~0.988** → il vettore ricostruito è quasi parallelo all'originale.
- **frac_variance_explained = 0.993** (99.3% della varianza spiegata) → convergenza raggiunta e stabilizzata (la loss curve mostra plateau: +58% train / +51% test MSE reduction da step 0 a 50000).
- I 5 seed sono **coerenti tra loro** (MSE entro 4.4–4.6e-5, cosine entro 0.9882–0.9887) → riproducibile.

*(La cella "loss curve" riporta MSE ~0.0008: è su scala di attivazione normalizzata, non confrontabile col MSE raw 4.5e-5; il segnale corretto di convergenza è `frac_variance_explained` 0.993.)*

### 2.2 Sparsity — ✅ CORRETTA

- **L0 medio = 32.0** su tutti i seed = esattamente `k`. Il vincolo Top-K è rispettato alla perfezione (ogni sample attiva 32 feature su 4096).
- **Entropia di attivazione ~6.3** (in nats) → le attivazioni sono distribuite su ~`e^6.3 ≈ 540` feature diverse nel test set (uso abbastanza diffuso, non concentrato su poche feature).

### 2.3 Dead features — ⚠️ MODERATA

- **~43–45% delle 4096 feature non si attiva mai** sul test set (dict utilization ~55–57%).
- Cioè ~1800 feature "morte" → il dizionario (4096) è **sovradimensionato** per ~7400 immagini. Atteso per dataset piccoli.
- ⚠️ Attenzione: ci sono **due definizioni di "dead"** divergenti (come da CLAUDE.md):
  - *Naming dead* (decoder a norma zero, in `concept_names.json`): **0%** (nessuna).
  - *Activation dead* (mai attiva sul test, in stability): **~44%**.

### 2.4 Stabilità cross-seed — ❌ LA CRITICA PRINCIPALE

**Mean Jaccard = 0.0038** (matrice 5×5, off-diagonal tutte ~0.003–0.010).

| | 0 | 42 | 123 | 456 | 789 |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.003 | 0.010 | 0.003 | 0.003 |
| 42 | 0.003 | 1.00 | 0.004 | 0.003 | 0.003 |
| 123 | 0.010 | 0.004 | 1.00 | 0.004 | 0.002 |
| 456 | 0.003 | 0.003 | 0.004 | 1.00 | 0.003 |
| 789 | 0.003 | 0.003 | 0.002 | 0.003 | 1.00 |

- I 5 SAE **ricostruiscono ugualmente bene ma con feature quasi completamente diverse**. Condividono <0.4% delle feature attive.
- → I "concetti" scoperti **non sono robusti/riproducibili**: dipendono fortemente dal seed. Il seed primario 42 è arbitrario; cambiandolo, naming/explanations cambiano.
- **Questo è esattamente il problema aperto "scarsa robustezza dei concetti" che il progetto cita esplicitamente.** È quindi un risultato *atteso ma significativo* da discutere.

### 2.5 Concept naming — ⚠️ ALLINEAMENTO DEBOLE

- 4096 feature nominate (0 marcate `DEAD_FEATURE`).
- **Score (coseno decoder↔vocab) basso: mean 0.117, max 0.291, min −0.063.** Un coseno 0.29 = allineamento debole → i nomi RadLex sono "best effort", non fortemente ancorati.
- Top-8 per score (clinicamente plausibili ma weakly-grounded): `cluster-of-grapes sign of lung` (0.29), `air embolism` (0.29), `ischemic cardiomyopathy` (0.28), `vertebral fusion` (0.28), `airway ectasia` (0.28)…
- → Le feature catturano pattern dello spazio di embedding che il vocabolario RadLex (310 termini) copre solo parzialmente.

### 2.6 Explanations — ✅ STRUTTURALMENTE CORRETTE

- **1494 record** (uno per immagine di test), **0 fallback `sample_`** → `image_id` reali = basename PNG (pronti per il join del judge).
- Schema del contract judge verificato: `{image_id, top_k_concepts[].{feature_id,name,activation}, pseudo_report}`.
- Attivazioni modeste (mean 0.14, max 0.386) → coerenti con l'allineamento debole del naming.
- Esempio (`3222_IM-1522-2001.dcm.png`): `crazy-paving sign` (0.218), `ischemic cardiomyopathy` (0.164), `Pleuritis` (0.158)… — pseudo-report template-based.

---

## 3. Giudizio d'insieme: è un risultato sensato?

**Sì, per la meccanica del SAE; preoccupante per l'interpretabilità.**

| Obiettivo | Esito |
|---|---|
| Il SAE impara a decomporre gli embedding? | ✅ Sì, ricostruisce al 99.3% con k=32. |
| I concetti sono *sparsi e monosemantici*? | ✅ Sparsi sì (L0=32); la monosemantia è dubbia (naming debole). |
| I concetti sono *robusti*? | ❌ **No** — Jaccard 0.004, dipendono dal seed. |
| I concetti sono *clinicamente ancorati*? | ⚠️ Parzialmente — allineamento RadLex debole (max 0.29). |
| La pipeline produce output judge-ready? | ✅ Sì — `sample_explanations.json` con `image_id` reali + schema corretto. |

**Il punto chiave da portare nella discussione del progetto:** il SAE *funziona tecnicamente* ma la **scoperta dei concetti non è stabile** né fortemente ancorata al vocabolario. Questo è materiale per la sezione "failure cases / limiti" richiesta dal progetto, non un bug.

---

## 4. Cosa provare a cambiare nelle prossime run (in ordine di priorità)

### 🔴 Priorità alta — stabilità (il problema principale)

1. **Ridurre `dict_size`** (4096 → 2048 o 1024). Meno ridondanza = meno gradi di libertà per divergere tra seed. Il preset "Conservative" del docstring SAE suggerisce `dict_size=2048`. Atteso: Jaccard più alto, dead features più bassi.
2. **LR più basso** (`lr=None`→`5e-5`). CLAUDE.md lo raccomanda per dataset piccoli (~7400). Convergenza più stabile → meno optima locali diversi.
3. **Aggregazione cross-seed**: invece di confrontare feature grezze, **clusterizzare** le feature dei 5 seed nello spazio del decoder (`stability_analysis` ha già un concept-clustering) e usare i cluster come "concetti stabili". Mitiga la non-robustezza senza riaddestrare.
4. **Media dei decoder** (model soup) o **init condiviso** tra seed per forzare feature comuni.

### 🟡 Priorità media — dead features & allineamento

5. **Resampling dei dead feature** durante il training (se supportato dalla lib `dictionary_learning`) o dict_size minore → riduce il 44% di spreco.
6. **Vocabolario più ampio/curato**: 310 termini RadLex coprono male lo spazio (max coseno 0.29). Provare più termini o un naming tipo SPLiCE (ottimizzazione sparsa sui pesi del decoder invece del coseno greedy).
7. **Più step / più dati**: 50k step su 5976 sample = molte epoche; potrebbe esserci overfit. Ma il dataset è fisso; provare `dict_size` minore è più promettente.

### 🟢 Priorità bassa — config/ripetibilità

8. **Determinismo completo**: impostare `PYTHONHASHSEED=0 OMP_NUM_THREADS=1` (e su CUDA `CUBLAS_WORKSPACE_CONFIG`). Su MPS la riproducibilità non è garantita al 100%.
9. **W&B abilitato** (`WANDB_ENABLED=True`) per tracciare curve su più run quando si fanno ablation su `dict_size`/`k`/`lr`.

### Ablation consigliata (prossima sessione)

Matrice 2×2: `dict_size ∈ {1024, 2048}` × `lr ∈ {auto, 5e-5}`, seed 42 singolo, ~10k step (veloce). Misurare **Jaccard vs seed 0/123** e **dead%**. Tenere il config che massimizza stabilità, poi rilanciare a 50k × 5 seed.

---

## 5. Note di riproducibilità & stato

- **MPS ≈ CUDA**: le metriche di questa run (cosine 0.988, dead 44%) coincidono con la run CUDA di riferimento nel notebook in cache → il cambio device non ha alterato i risultati.
- **Schema judge rispettato**: `sample_explanations.json` ha `image_id` reali (basename PNG), propagati dal sidecar `test_image_ids.json`. Il judge (membro 3) può joinare su `reports.csv` senza modifiche al producer.
- **`vocabulary.json` NON va toccato**: è una lista di 310 stringhe; il builder scriverebbe dict (schema #7 aperto). Per ora è corretto e allineato a `text_vocab_embeddings.pt`.
