# FINDINGS — log dei risultati utili per paper e presentazione

**Scopo:** raccogliere in un unico posto i finding empirici/metodologici che
**motivano o spiegano i risultati** del progetto, da citare nel report (2–3 pag)
e nelle slide. Documento *living*: aggiungere un nuovo finding man mano che emerge,
seguendo lo schema (Osservazione / Evidenza / Implicazione per il paper).

**Stato del finding:** ✅ verificato empiricamente · 🔍 ipotesi con evidenza parziale ·
📄 da riconfermare sui run finali.

> I numeri esatti (~) provengono dalle analisi citate; vanno riconfermati sui run
> finali prima di andare in print.

---

## A. Dati & dataset

### A1 — IU X-Ray è data-starved per un SAE ✅
- **Osservazione:** ~7.470 immagini totali (~5.800 nel train split) per un SAE con
  `dict_size=2048` ⇒ **~2.8 campioni per feature**.
- **Evidenza:** diagnosi in `docs/suggestions/PRE_PROJECTION_EMBEDDINGS.md`,
  `medconcept_alignments.md`.
- **Implicazione (paper):** è la **causa root** dei feature-rumore. Giustifica perché
  serva un dataset più grande (PadChest) — la Fase 2 è costruita su questo finding.
  Frame: *"small-data regime in cui la sparse factorization è non-identificabile"*.

### A2 — PadChest: report sistematicamente anonimizzati/troncati ✅
- **Osservazione:** il CSV PadChest disponibile ha report con **ogni parola troncata
  delle ultime 1–2 lettere** (es. *"condensacion atelectasi lm context clinic pacient
  sugier neumoni ."* per *"condensación, atelectasia, [lóbulo medio], contexto clínico,
  paciente, sugiere neumonía"*). Il PadChest originale (BIMCV) ha spagnolo completo.
- **Evidenza:** ispezione diretta `data/padchest/PADCHEST_…_160K_….csv` (sessione
  2026-07-04, 1645 immagini / 160.861 righe).
- **Implicazione (paper):** **limitazione dichiarata**. Il judge multilingua deve
  *inferire* le parole troncate ⇒ introduce rumore e **deprime le metriche
  Aligned/Unaligned/Uncertain** su PadChest. Va discusso esplicitamente (la traccia
  chiede failure cases + sensitività). È un punto a favore dell'onestà metodologica.
- **Mitigazione possibile:** colonna `Labels` (pulita, non troncata) e `labelCUIS`
  disponibili come ground-truth complementare.

### A3 — Augmentation genera embedding near-duplicate ✅
- **Osservazione:** BiomedCLIP è ~invariante alle perturbazioni lievi (rotazione 5°,
  crop 0.95) ⇒ le copie augmentate hanno embedding quasi identico all'originale.
- **Evidenza:** ablation augmented (notebooks/autoencoder): dead% sale, Jaccard
  cross-seed invariato — l'aumento di campioni è "fittizio".
- **Implicazione (paper):** **failure case** del "più dati = meglio". L'augmentation
  classica su un VLM robusto non risolve la data-starvation. Da presentare come
  negative result interessante.

---

## B. Metodologia SAE

### B1 — Non-identificabilità della sparse factorization in regime data-starved ✅
- **Osservazione:** con ~2.8 campioni/feature, la decomposizione che minimizza la loss
  **non è unica** e non corrisponde a direzioni "vere" → le feature apprese sono rumore.
- **Evidenza:** feature con max activation ~0.29 (non si accendono su nulla), naming
  con garbage concepts (es. "pipestem ureter", label tedesche RadLex).
- **Implicazione (paper):** collega il risultato empirico alla **teoria** (identificabilità
  della sparse coding). Dà spessore metodologico: non è "il SAE non funziona", è
  *"il SAE non è identificabile sotto questa soglia di dati"*.

### B2 — Instabilità cross-seed al chance floor ✅
- **Osservazione:** Jaccard cross-seed ≈ `k/(2D−k)` (il valore atteso da feature
  **casuali**), anche sull'ablation index-agnostic (a0 = consenso sulle *direzioni* del
  decoder) = 0.0.
- **Evidenza:** `results/iu_xray/baseline/stability_analysis.json` — mean Jaccard **0.0084** vs
  chance floor `k/(2D−k)` = 0.0079 (k=32, D=2048) → al caso; matched-cosine (permutation-invariant)
  0.299 vs null 0.151, frac_matched@0.7 = 0.002 → le direzioni non allineano tra seed. Ricostruzione
  buona (cosine 0.991, mse 3.6e-5) → il SAE fitta i dati ma le **feature sono arbitrarie**.
  Report: `results/iu_xray/failure_cases/REPORT.md`.
- **Implicazione (paper):** **metrica quantitativa** del fallimento. Dimostra che i
  concetti scoperti non sono riproducibili → non sono "reali".failure case chiave.

### B3 — Naming ≈ random ✅
- **Osservazione:** top-1 cosine tra feature e vocabolario ~0.35 (vicino all'atteso
  casuale su un vocab CXR-filtrato); **97.8%** delle feature < 0.5, max 0.556.
- **Evidenza:** `results/iu_xray/baseline/concept_names.json` — mean 0.354, 16% dead;
  i 15 concetti "migliori" (score più alto) sono anatomia **non-toracica**:
  *ligamentum flavum, stapedius nerve, nephroureteral stent, coccygeal segment of spinal
  cord, filum terminale* — il cosine-argmax è rumore. Report: `results/iu_xray/failure_cases/REPORT.md`.
- **Implicazione (paper):** l'allineamento visuale↔testuale è rumore. Conferma B1/B2
  da un'angolazione diversa (grounding semantico).

### B4 — Vocabolario derivato dalle label = circolare (lesson learned) ✅
- **Osservazione:** una prima versione del vocab ROCOv2 derivava i termini dai CUI del
  dataset stesso. Questo era **circolare**: il pool di candidati del naming = il pool
  delle label gold → ogni metrica CUI-based sarebbe stata biased (il naming sceglie
  tra le risposte), e il vocab era dipendente dal dataset, non esterno (inconsistente
  col paradigma RadLex, che è ontologia esterna filtrata al dominio).
- **Risultato:** pivoted a **MeSH esterno** (libero, senza licenza UMLS) filtrato ai
  rami radiologici (alberi A/C/E) + ranking per anchor radiologici → vocab indipendente
  dalle label → caption-judge rigoroso. **CUI-matching rimosso** (niente crosswalk CUI
  libero; match per stringa troppo rumoroso).
- **Implicazione (paper):** **rigore metodologico** — il vocabolario di naming deve
  essere **esterno e indipendente** dal ground-truth di valutazione (come RadLex lo è
  dai referti IU). Da citare in Method / Threats-to-validity. (B1/B2/B3 vanno rifatti
  su ROCOv2 con il vocab MeSH per essere comparabili.)

### B5 — Più dati (ROCOv2, ~10×) NON risolve l'instabilità; rafforza B1 ✅
- **Osservazione:** ROCOv2 (~80k immagini, **~31 campioni/feature** nel train vs ~3 di
  IU) risolve in larga parte la data-starvation (A1), **eppure**:
  - **Stabilità cross-seed ancora al chance floor**: mean Jaccard **0.0077** vs chance
    0.0079 (IU 0.0084) → al caso. Matched (permutation-invariant) best-cosine **0.327**
    vs null 0.151 (IU 0.299); frac_matched@0.7 = **1.4%** (IU 0.2%) → ~7× meglio, ma
    ancora ~98.6% delle feature senza controparte stabile tra seed.
  - **Naming migliorato (netto)**: mean top-1 cosine **0.48** vs IU 0.354; **29%** delle
    feature ≥ 0.5 (IU 2.6%, ~11×); max 0.63 vs 0.56; feature morte **3** (IU 328, 16%).
    I top-concept sono radiologicamente rilevanti (*Heart Valve Prosthesis, Skull,
    Carcinoma Non-Small-Cell Lung, Spinal Cord, Liver, EEG*) vs IU che era anatomia
    non-toracica (rumore).
  - **Ricostruzione peggiorata**: cosine 0.967 vs IU 0.991 (atteso: dominio multimodale
    tutto-corpo più ampio da comprimere nello stesso SAE).
- **Evidenza:** `results/rocov2/baseline/{stability_analysis,stability_matched,concept_names}.json`,
  `results/rocov2/failure_cases/REPORT.md`.
- **Implicazione (paper):** **risultato chiave** — la quantità di dati **non è** la
  (sole) causa dell'instabilità. A1 (data-starved) era una spiegazione parziale, non la
  causa root; **B1 (non-identificabilità della sparse factorization nello spazio proiettato
  di BiomedCLIP) ne esce rafforzata** come storia reale. Confound da dichiarare: ROCOv2 è
  anche un dominio più vasto/difficile (multimodale, tutto-corpo), non "più IU" — per cui
  **PadChest** (stesso dominio chest, scala maggiore) resta il test controllato decisivo:
  se PadChest resta al chance floor → B1 confermata; se si sblocca → la larghezza del
  dominio ROCOv2 era il fattore. Rinforza la narrativa "failure case → scale test →
  generalizzazione" con un **negative result difendibile** (più data non basta).

### B6 — PadChest (stesso dominio chest) conferma B1; seed 123 = non-identificabilità empirica ✅
- **Osservazione:** run PadChest a ~17k campioni (train **13.433**, **~6,5 campioni/feature** —
  scala intermedia, non full; referti spagnoli **troncati**, vedi A2). È lo stesso dominio chest di
  IU → il test controllato che B5 indicava come decisivo. **Risultato: negativo.**
  - **Stabilità ancora al chance floor**: mean Jaccard **0,0077** vs chance 0,0079; matched
    best-cosine **0,345** (null 0,151); **frac_matched@0,7 = 0,078%** — il **più basso** dei tre
    dataset (IU 0,2%, ROCOv2 1,4%): le direzioni non allineano tra seed quasi mai.
  - **Naming debole**: mean **0,44** (IU 0,354; ROCOv2 0,48); **6,2%** ≥ 0,5; top-concept rumorosi
    — label **RadLex tedesche** + fuori dominio (*endometritis, sialodochitis fibrinosa,
    "Persistierende fetale Zirkulation", "High-flow vaskuläre Malformation"*) come IU; pochi morti
    (5, naming `is_dead`).
  - **Ricostruzione buona** (cosine 0,977, tra IU 0,991 e ROCOv2 0,967).
  - **Seed 123 collassato**: **~48 feature vive** su 2048 (97,7% "dead" per frequenza) ma
    ricostruzione **identica o migliore** (mse 9,06e-5, il minimo dei 5 seed). I
    `training_manifest.json` di tutti i seed sono **identici a parte il seed** (stesso hash dati,
    lr 5e-5, stessa config, GTX 1650) → il collasso dipende **solo dall'init**.
- **Confronto tre dataset:**

  | | IU X-Ray | PadChest | ROCOv2 |
  |---|---|---|---|
  | dominio | chest (EN) | chest (ES, troncato) | multimodale (EN) |
  | train sample/feature | ~3,0 | ~6,5 | ~31 |
  | mean Jaccard | 0,0084 | 0,0077 | 0,0077 |
  | (chance = 0,0079) | al caso | al caso | al caso |
  | matched frac@0,7 | 0,20% | 0,078% | 1,37% |
  | naming mean cosine | 0,354 | 0,441 | 0,481 |
  | ricostruzione cosine | 0,991 | 0,977 | 0,967 |

- **Evidenza:** `results/padchest/baseline/{stability_analysis,stability_matched,concept_names}.json`,
  `models/padchest/sae_seed*/training_manifest.json`, `results/padchest/failure_cases/REPORT.md`.
- **Implicazione (paper):** **B1 confermata dal test decisivo.** Tre dataset, tre scale
  (3→6,5→31 sample/feature), due domini, **tutti al chance floor** ⇒ l'instabilità non è little-data
  (IU) né larghezza del dominio (ROCOv2): è la **non-identificabilità** della sparse factorization
  nello spazio proiettato di BiomedCLIP. **Seed 123 ne è la dimostrazione empirica**: ~48 feature
  ricostruiscono come ~1500 → l'obiettivo di ricostruzione ammette minimi degeneri equivalenti, non
  vincola la decomposizione (non è un bug, è il fenomeno stesso). Caveat: scala intermedia (full
  PadChest ~42/feature resterebbe quasi certamente al chance floor, vedi ROCOv2 a 31); referti
  troncati (A2) confondono il naming; dead-feature revival/resampling ridurrebbe il collasso ma
  **non** l'instabilità cross-seed (gli altri 4 seed, meno collassati, sono comunque al chance
  floor). **Negative result robusto e difendibile** — il cuore metodologico del lavoro.

---

## C. Valutazione & LLM-judge

### C1 — Disaccordo inter-modello del giudice ✅
- **Osservazione:** tre modelli giudice danno verdetti **opposti** sugli stessi pseudo-report:
  Llama-3.1-8B ~95% Uncertain, MedGemma-4B ~84% Aligned, Gemma-4-26B ~82% Unaligned.
- **Evidenza:** `results/iu_xray/*judge_checkpoint*.json` (run parziali per-modello);
  tabella + figura in `results/iu_xray/failure_cases/REPORT.md`.
- **Implicazione (paper):** **criticità dell'LLM-as-a-judge** (la traccia cita il paper
  sul position bias dei judge, ref [13]). Frame: la metrica è sensibile al modello
  giudice ⇒ serve reporting multi-judge o almeno dichiarare il modello. Failure case
  di valutazione.

### C2 — MedConcept stesso ottiene Aligned < 0.1 su un dataset ✅
- **Osservazione:** il paper di riferimento riporta median Aligned <0.1 (Unaligned >0.8)
  su AbdomenAtlas, attribuendolo alla densità/stile dei referti.
- **Evidenza:** `docs/requirements/MedConcept.md` (Results + Limitations).
- **Implicazione (paper):** **ridimensiona il "fallimento"** del nostro progetto —
  alignment basso è un fenomeno documentato anche nel riferimento, non solo nostro.
  Difende i risultati dal "è tutto sbagliato".

---

## D. Cornice & framework

### D1 — La traccia è domain-agnostic; chest/RadLex erano scelte del team ✅
- **Osservazione:** la traccia non menziona "chest"; cita **UMLS**; chiede metriche
  "adapting" + failure cases. MedConcept riferimento = **TC addominale 3D + Merlin + UMLS**.
- **Evidenza:** `docs/requirements/PROJECT-BRIEF.md`, `docs/requirements/MedConcept.md`.
- **Implicazione (paper):** giustifica il **reframe multi-dataset** (IU→PadChest→ROCOv2)
  e l'uso di UMLS: non è una deviazione, è un ritorno più fedele al riferimento.
  Da dire esplicitamente in Introduction/Method.

### D2 — Architettura multi-dataset resa modulare (DatasetSpec) ✅
- **Osservazione:** refactor Fase 0–3 introduce `DatasetSpec` (un'unica descrizione per
  dataset) + routing per-dataset (`embeddings/<dataset>/…`, `models/<dataset>/`,
  `results/<dataset>/`, `vocabulary.json` incluso) + `select_dataset` / `--dataset`.
  IU X-Ray migrato senza cambiare gli artifact (sha256 byte-identical). Fase 3:
  ROCOv2 wired (vocab **MeSH** esterno + caption judge; vedi B4 sul perché non
  UMLS/CUI). 3 dataset registrati: `IU_XRAY_SPEC`, `PADCHEST_SPEC`, `ROCOV2_SPEC`.
- **Evidenza:** `xai_datasets/spec.py`, `scripts/verify_byte_identical.py`, ~150 test verdi.
- **Implicazione (paper):** **contributo ingegneristico/riproducibilità**: la pipeline è
  dataset-agnostic, ogni esperimento è isolato per-dataset. Da citare in Method/Reproducibility.

---

## Come usarlo nel paper (suggerimento)

- **Introduction / Motivation:** A1, D1.
- **Method:** D2, B1 (teoria).
- **Results:** B2, B3, C2 (su IU); **B5 (ROCOv2)** + **B6 (PadChest)** — tre dataset, tre scale, due domini, **tutti al chance floor**; naming migliora con scala + aderenza del vocab.
- **Failure cases & limitations (la traccia li chiede):** A2, A3, C1, B1, **B5, B6** (seed 123 = minimo degenere equivalente).
- **Discussion:** C2 (ridimensiona il fallimento), **B5+B6** (quantità di dati e dominio *non* sono la causa root → **B1 non-identificabilità** confermata come tesi centrale); PadChest full-scale come colpo di grazia opzionale.

---

## Findings da aggiungere (quando disponibili)

- [x] **IU X-Ray failure-case report** → `results/iu_xray/failure_cases/REPORT.md`, generato da
      `scripts/run_failure_case_analysis.py --dataset iu_xray` (naming, stability, judge disagreement).
- [x] **PadChest baseline run (~17k)** → `results/padchest/baseline/` + `results/padchest/failure_cases/REPORT.md` (finding **B6**). Stabilità al chance floor come IU/ROCOv2 → **B1 confermata**; seed 123 collassato (~48 feature vive, ricostruzione identica) = non-identificabilità empirica.
- [ ] PadChest: LLM-judge (`python src/evaluate_llm_judge.py --dataset padchest`) per la sezione C1 su PadChest (richiede il fix dell'env giudice — vedi errore `ModelWrapper`).
- [ ] (Opzionale, test definitivo) PadChest **full scale** (~109k, ~42 sample/feature) — quasi certamente conferma B6, ma chiude il caveat sulla scala intermedia del run a 17k.
- [x] **ROCOv2 baseline run** → `results/rocov2/baseline/` + `results/rocov2/failure_cases/REPORT.md` (finding **B5**). Naming migliorato (mean 0.48, top-concept radiologici, 3 feature morte) ma stabilità ancora al chance floor → **rafforza B1**.
- [ ] ROCOv2: LLM-judge (`python src/evaluate_llm_judge.py --dataset rocov2`) per popolare la sezione C1 su ROCOv2.
- [ ] Confronto baseline (512-d) vs Path A (768-d hidden) su PadChest.
