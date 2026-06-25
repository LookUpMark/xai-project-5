# Reframe del progetto: metodi principali, baseline e "estensione"

Proposta strategica per massimizzare il punteggio (16 pt) riformulando la
narrativa e i ruoli dei componenti della pipeline, **senza** modificare la
correttezza tecnica di quanto gia' implementato. Da leggersi insieme a:

- `docs/audits/ML-AUDIT-2026-06-25.md` (audit metodologico — finding M-001..M-008),
- `docs/requirements/progetto_5_descrizione_completa.md` (rubric e attivita' richieste),
- `docs/requirements/indicazioni_progetto_xai.md` (criteri di valutazione),
- `CONCEPT_INSTABILITY_DIAGNOSIS.md` e `ADDITIONAL_ABLATION_STUDIES.md` (diagnosi e ablation derivate).

> **Stato: proposta (non implementata).** Nessun codice modificato. Il documento
> definisce ruoli, motivazione sul punteggio, piano implementativo dettagliato,
> sequencing e risk register. La decisione di esecuzione e' del team.

---

## 1. Motivazione

L'audit metodologico (`ML-AUDIT-2026-06-25.md`) conclude che:

- l'implementazione attuale e' **meccanicamente corretta** (post split-fix
  `b5a7b2e`), ma
- il risultato principale (Jaccard cross-seed 0.0038 = pavimento analitico del
  caso; dead-feature 40-60%; naming SAE 0.395 ~ random) e' il **sintomo atteso**
  di una scelta metodologica debole (M-001): si allena un SAE *seed-dipendente*
  sull'embedding **proiettato a 512-d**, dove la fattorizzazione sparsa e'
  **non-identificabile** a dati scarsi.

Invece di consegnare questo come "il risultato del progetto" (negativo e debole
sull'asse *originalita'/novita'*), lo si **riformula** sfruttando tre fatti:

1. Il **rubric** del corso premia esplicitamente: *literature review, research
   gaps, methodology+assessment, originalita'/novita', discussione/analisi,
   failure-case analysis*.
2. Il requisito (§3 "Implementazione") elenca il SAE come **una** direzione tra
   diverse ("Le possibili direzioni indicate dal testo sono diverse") e parla di
   "pipeline **ispirata** ai framework piu' recenti" — non vincola a un singolo
   metodo.
3. Il gap di ricerca #1 del progetto stesso e' *"scarsa robustezza dei concetti
   scoperti"*: il progetto che **diagnostica e risolve** quel gap fa piu' punti
   di quello che si limita a *citare* il problema.

La proposta: **due metodi principali** (A: SAE sullo hidden state 768 — il SAE
MedConcept-allineato fatto *correttamente*; B: SPLiCE — decomposizione diretta,
robusta su dataset piccolo), l'implementazione 512-d **degradata a baseline /
failure case documentato**, e una **vera "estensione"** (organizzazione
strutturata dei concetti) cosi' come definita letteralmente dal §3.

---

## 2. Mappatura requisiti -> punteggio

Criteri di valutazione (da `indicazioni_progetto_xai.md` §Valutazione) e come il
reframe li serve:

| Criterio del rubric | Come il reframe lo massimizza |
|---|---|
| Literature review | Review puo' presentare **entrambi i paradimi** verificati: SPLiCE (su embedding proiettato) e SAE-interpretability (su hidden state). Copertura piu' ampia e piu' accurata di "SAE soltanto". |
| Research gaps | Gap #1 (stabilita' concetti) e' **addressato direttamente**: il baseline lo manifesta, i metodi principali lo risolvono/comparano. |
| Methodology + assessment | Tre metodi a confronto (A/B/baseline) + valutazione quantitativa (judge) su tutti. Metodologia piu' ricca di un singolo run. |
| **Originalita' / novita'** | Confronto sistematico *SAE-su-proiettato vs SAE-su-hidden vs SPLiCE* su dataset medico reale = contributo metodologico originale, non esecuzione didattica. |
| Discussione e analisi | L'audit `ML-AUDIT-2026-06-25.md` e' gia' l'analisi critica approfondita (root-cause, ipotesi verificate, refutate). |
| Failure cases (richiesto esplicitamente in §4) | Il baseline 512-d e' un **failure case completo con root-cause**: concetti spurii, instabilita' al pavimento del caso, naming ~ random. |
| Chiarezza | Ruoli espliciti (main/baseline/extension) rendono la narrativa lineare. |

---

## 3. Ruoli dei componenti (matrice)

| Ruolo | Componente | Spazio | Stato | Rubric axis |
|---|---|---|---|---|
| **Main 1** | **A — SAE su hidden state 768** (pre-projection CLS) | 768-d | da implementare | Methodology + originalita' |
| **Main 2** | **B — SPLiCE** decomposizione diretta su dizionario RadLex | 512-d (shared) | da implementare | Methodology + originalita' (paradigma alternativo) |
| **Baseline / failure case** | SAE TopK su embedding 512 proiettato (pipeline attuale) | 512-d | **gia' fatto** | Research gap + failure-case |
| **Estensione** (§3 letterale) | Organizzazione strutturata dei concetti: clustering + mappatura a gerarchia RadLex | — | da implementare | Estensione §3 |
| **Evaluation** | LLM judge MedGemma (Aligned/Unaligned/Uncertain) su A e B | — | richiede fix M-007 prima | Assessment quantitativo |
| **Stabilita'** | Jaccard cross-seed (A) / determinismo (B) | — | A: re-run; B: banale | Research gap #1 |

### 3.1 Perche' "baseline" e non "estensione" per la pipeline 512-d

Il §3 definisce **estensione** = *"filtraggio, clustering o organizzazione
strutturata dei concetti scoperti"*. Chiamare l'intera pipeline SAE-512
"estensione" **non corrisponde** a quella definizione e sarebbe meno premiato.
Il framing corretto (e piu' forte) e': la pipeline 512-d e' il **baseline
naive**, il cui fallimento (documentato in audit) **motiva** i metodi principali
A e B. Il failure-case analysis e' esplicitamente richiesto (§4) e valued.

### 3.2 Perche' SPLiCE e' difendibile come "main" nonostante MedConcept sia SAE-based

Il framework riferimento (MedConcept) usa SAE, ma:

- il requisito dice "pipeline **ispirata** ai framework piu' recenti" e lista il
  SAE come *una* delle direzioni;
- SPLiCE e' uno dei paper della literature review richiesta (e' in
  `docs/literature/03_SPLiCE_NeurIPS2024.pdf`);
- implementare **entrambi** i paradigmi e confrontarli e' metodologicamente
  piu' ricco di un singolo paradigma.

Frame narrativo: *"implementiamo sia il paradigma SAE (MedConcept) sia la
decomposizione sparsa diretta (SPLiCE), e mostriamo quale produce concetti
stabili e fedeli sullo stesso dataset clinico."*

---

## 4. Implementazione dettagliata

### 4.1 Baseline (zero costo) — riposizionamento narrativo

Nessun nuovo codice. Lavoro solo di scrittura:

1. Riformulare `notebooks/autoencoder/ablation/REPORT*.md` presentando la
   pipeline 512-d come **baseline**, non come risultato principale.
2. Spostare il finding "Jaccard = pavimento del caso" dal contributo positivo al
   **failure case motivante** (con link a `ML-AUDIT-2026-06-25.md` M-001).
3. Esplicitare che il baseline implementa *fedelmente* MedConcept (SAE su
   embedding condiviso) e che proprio per questo isola il limite metodologico.

**Verifica:** narrativa coerente tra recap, slide e report.

### 4.2 Estensione (§3) — organizzazione strutturata dei concetti

Sopra il metodo principale vincente (A o B). Letteralmente l'estensione richiesta.

1. **Ridondanza concettuale:** clustering delle direzioni/concetti scoperti
   (agglomerative clustering su similarita' del coseno dei concetti nello spazio
   testuale RadLex). Output: gruppi di concetti affini.
2. **Organizzazione strutturata:** se RadLex fornisce relazioni
   anatomiche/gerarchiche, mappare i cluster a quella struttura; altrimenti
   costruire una gerarchia indotta dal clustering.
3. **Spiegazione strutturata per-sample:** invece di "top-k concetti piatti",
   presentare "quali cluster/famiglie concettuali si attivano" per immagine.
4. Modulo nuovo: `src/concept_discovery/organize.py` (+ notebook
   `notebooks/autoencoder/06_concept_organization.ipynb`).

**Verifica:** cluster interpretabili; spiegazione strutturata non peggiore di
quella piatta su coerenza; riduzione della ridondanza misurabile.

### 4.3 Fix prerequisito — LLM judge (M-007)

Prerequisito per qualunque numero di valutazione quantitativa. Tracciato in
`docs/audits/ML-AUDIT-2026-06-24.md` (fix plan). Sintesi:

- verb/label map nel prompt (F-001),
- retry dei pair in errore su `--resume` (F-002),
- `temperature=0.0` + seed per riproducibilita' (F-003),
- `JudgeConfig` in `config.py` (F-007).

**File:** `src/evaluate_llm_judge.py`, `src/config.py`, test.

### 4.4 Path B — SPLiCE (main 2, **safety net**, implementare per primo)

Decomposizione sparsa **deterministica** dell'embedding immagine (512-d, spazio
condiviso) su un dizionario **fisso** di concetti testuali (RadLex). Nessun SAE
appreso, nessun seed, nessuna instabilita'. Robusto su dataset piccolo perche' il
dizionario **non** e' imparato dai 6k campioni.

**Perche' primo:** garantisce un risultato positivo dimostrabile alle slide in
tempo predeterminato, indipendentemente dall'esito di A.

1. **Modulo nuovo** `src/concept_discovery/spliece.py`:
   - input: `image_emb` (512-d, L2-normalato), dizionario `D = vocab_emb`
     (RadLex, 508 x 512, colonne L2-normalate),
   - **correzione modality gap** prima della regressione (sottrarre il gap
     all'embedding immagine, stesso trick del naming SAE attuale),
   - risoluzione: `min ||x - D c||_2` con `c >= 0` e budget di sparsita'.
     Opzioni: `sklearn.linear_model.OrthogonalMatchingPursuit` (L0 esatto, k
     termini) oppure `Lasso(positive=True)` (L1). SPLiCE originale usa L1 non-negativo;
   - output per immagine: `c` (508,) coefficienti; concetti = top-k per
     coefficiente.
2. **Naming intrinseco:** i coefficienti **sono** gia' i concetti (nessun bridge
   separato). Riusa `data/vocabulary.json` + `text_vocab_embeddings.pt`.
3. **Stabilita':** banale (deterministica) — niente Jaccard, niente seed.
4. **Faithfulness:** judge (post-M-007) sulle spiegazioni SPLiCE.
5. **Notebook** `notebooks/autoencoder/07_spliece.ipynb` + risultati in
   `results/spliece/`.

**Verifica:** coerenza concettuale su immagini campione (top-k concetti
sensati); faithfulness >= baseline SAE; confronto diretto copertura concetti
SPLiCE vs feature-live del SAE.

**Risk:** assume che RadLex **copra** il manifold delle immagini (se il vocab e'
incompleto, SPLiCE non puo' esprimere cio' che vede). Mitigazione: confrontare
copertura e considerare vocab piu' ampio (vedi `VOCAB_BUILDING_ALTERNATIVES.md`).

### 4.5 Path A — SAE su hidden state 768 (main 1, **centerpiece metodologico**)

Sposta il SAE dall'embedding proiettato 512-d al **CLS token pre-projection 768-d**
(cio' che letteralmente alimenta la proiezione). Attacca M-001 alla radice ed e'
allineato alla linea SAE-interpretability (SAE su residual/hidden stream, non su
proiezione finale).

**Relazione esatta con l'attuale codice:** `model.get_image_features()` fa
internamente `vision_model -> last_hidden_state[:,0,:] -> projection(768->512)`.
Il 768-d CLS e' **esattamente** il pre-projection. Estrarlo e' una variante
minima dell'esistente.

1. **Nuova modalita' di estrazione** in `src/embedding_extraction/extract_embeddings.py`
   (+ notebook `notebooks/vlm/extract_embeddings.ipynb`):
   ```python
   out = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
   hidden = out.last_hidden_state[:, 0, :]          # (B, 768) CLS pre-projection
   # opzionale: out.hidden_states[-2][:, 0, :] per un residual stream piu' profondo
   ```
   **Verificare al load** l'attributo esatto su `chuhac/BiomedCLIP-vit-bert-hf`
   (`vision_model` e' un `ViTModel` HF; `last_hidden_state` e' (B, 197, 768)).
2. **Config:** `SAEConfig.activation_dim: 768`; ridefinire `dict_size` (8x768 =
   6144, o mantenere 4096 ~ 5.3x); applicare igiene Path C (**4.6**): `steps`
   ~5k-10k, `lr=5e-5` esplicito. Aggiungere `EmbeddingConfig` mode hidden/standard.
3. **Persistere** in dir nuova (`embeddings/standard_hidden/`,
   `augmented_hidden/`) per mantenere comparabilita' con il baseline 512.
   Rigenerare `modality_gap.pt` per il nuovo spazio (centroidi su 768-d; nota:
   il vocab testuale Resta 512-d, quindi il gap e' definito solo dopo il bridge).
4. **Retrain 5 seed** (`python src/autoencoder/train_sae.py`).
5. **Bridge di naming** (punto critico, vedi 4.5.bis).
6. **Re-run** `concept_naming.py`, `stability_analysis.py`,
   `generate_explanations.py`.

**File:** `extract_embeddings.py`, `notebooks/vlm/extract_embeddings.ipynb`,
`config.py`, `train_sae.py`, `sae_module.py`, `concept_naming.py`.

#### 4.5.bis Bridge di naming 768 -> spazio testuale 512

I decoder row del SAE vivono in 768-d; gli embedding RadLex in 512-d. Opzioni,
in ordine di costo:

- **(a) Proiezione congelata (consigliata, basso costo, principiata):** riusare
  il layer di proiezione **congelato** di BiomedCLIP (`W_proj`: 768 -> 512).
  `v_text = d_768 @ W_proj`, poi coseno vs RadLex 512. Usa la **stessa**
  proiezione che il modello apprende per allineare testo/immagine. Da verificare
  come accedervi (`model.visual_projection` o `model.vision_model.projection` /
  attributo HF effettivo).
- **(b) Probe lineare appreso (fallback):** allenare `768 -> 512` supervisionato
  dai concetti attivi per immagine. Piu' flessibile ma richiede supervisione e
  reintroduce una fonte di variabilita'.
- **(c) Sparse-text projection** in stile letteratura SAE-CLIP (piu' costoso).

**Verifica Path A:** Jaccard cross-seed si solleva **materialmente** sopra
`k/(2D-k)`; dead-feature scende; naming mean cresce sopra il random di margine
chiaro; ricostruzione resta alta (atteso). **Honest expectation:** se 7470
immagini sono comunque troppo poche, A potrebbe non risolvere completamente —
per questo B e' il safety net.

### 4.6 Path C — igiene iperparametri (complementare, su A e opz. baseline)

Non risolve M-001 da solo (le ablation 01/02 mostrano che dict_size e k non
muovono il Jaccard), ma va fatto comunque perche' gli iperparametri attuali sono
mis-sized per il dataset:

- `n_training_steps` 50.000 -> ~5.000-10.000 (50k x batch256 / 5976 ~ 2140 epoche,
  overfit),
- `lr` auto ~4e-4 -> `5e-5` esplicito (per `SAE_TRAINING_SMALL_DATASET.md`),
- shared init / model soup across seed (ablation A03),
- `dict_size` 4096 -> 2048 (riduce dead%, non fissa Jaccard).

**File:** `config.py` (`SAEConfig`, `TrainingConfig`).

---

## 5. Sequencing raccomandato

1. **Fix judge M-007** (prerequisito eval) — in parallelo, indipendente.
2. **Path B (SPLiCE) per primo** — safety net, deterministico, garantisce demo
   positiva. Output: concetti coerenti su campioni + faithfulness.
3. **Estensione organizzazione concetti** sopra B (cluster/gerarchia) — valore
   rubric immediato, basso costo.
4. **Path A (SAE 768)** — centerpiece metodologico; piu' costoso (nuova
   estrazione + retrain + bridge). Priorita' alta ma schedulata dopo B.
5. **Path C (igiene)** applicato ad A (e, se tempo, ri-run baseline con
   iperparametri corretti per una comparazione onesta).
6. **Consolidamento:** recap 2-3 pagine, slide 15 min, repo.

### Dipendenze temporali

- A e B sono **indipendenti** tra loro (possono parallelizzare su piu' persone:
  il gruppo e' da 3).
- Estensione dipende dal metodo vincente (A o B) ma puo' prototipare su B.
- Eval (judge) blocca i numeri di faithfulness di entrambi.

---

## 6. Risk register

| Rischio | Likelihood | Impatto | Mitigazione |
|---|---|---|---|
| Path A non solleva il Jaccard (7470 img ancora poche) | Media | Alto (perde il centerpiece) | B come safety net garantisce risultato positivo; A riportato come confronto metodologico comunque valido |
| Bridge di naming 768->512 distorce (proiezione off-manifold) | Media | Medio | (a) prima; fallback (b) probe appreso; riportare entrambi |
| SPLiCE limitato dalla copertura RadLex | Media | Medio | Confrontare copertura; considerare vocab piu' ampio (`VOCAB_BUILDING_ALTERNATIVES.md`) |
| Tempo/compute: nuova estrazione 7470x768 + retrain 5 seed | Alto | Medio | B prima a costo basso; A schedulata; GPU/CPU stimare in anticipo |
| Judge non fissato in tempo → nessun numero di faithfulness | Media | Alto | M-007 come prerequisito, sequenziato per primo/in parallelo |
| Cambio narrativa percepito come "post-hoc" dai docenti | Bassa | Medio | Audit datato e cronologico (già esiste 06-01..06-25): la diagnosi *precede* la correzione, narrativa onesta e lineare |

---

## 7. Deliverable mappati agli output richiesti

Output richiesti (`progetto_5_descrizione_completa.md` §"In sintesi operativa" +
`indicazioni_progetto_xai.md`):

| Output richiesto | Contenuto nel reframe |
|---|---|
| Review critica | Due paradigmi (SAE-interp + SPLiCE) + MetConcept, 9 paper in `docs/literature/` |
| Gaps motivati | Stabilita' concetti (gap #1) addressato, non solo citato |
| Pipeline funzionante | A (SAE 768) + B (SPLiCE), entrambi end-to-end |
| Metriche + failure cases | Judge su A/B + baseline 512 come failure case con root-cause |
| Contributo originale | Confronto A vs B vs baseline su dataset clinico reale |
| Recap 2-3 pag | Struttura 6 sezioni (intro/related/gaps/method/results/conclusion) |
| Slide 15 min | Stessa struttura; demo live di B (deterministica) |
| Repo GitHub | Moduli esistenti + `src/concept_discovery/{spliece,organize}.py` |

### Bozza struttura report/slide (6 sezioni consigliate dal rubric)

1. **Intro** — problema concetti in medical VLM, obiettivo.
2. **Related work** — CBM, TCAV, SAE-on-CLIP (hidden), SPLiCE (proiettato), MedConcept, judge.
3. **Research gaps** — stabilita', validita' clinica, dipendenza ontologie/LLM; focus su stabilita'.
4. **Methodology** — baseline 512 (naive) + A (SAE hidden) + B (SPLiCE) + estensione organizzazione; judge.
5. **Results** — comparazione metriche (Jaccard, dead%, naming, faithfulness); baseline come failure case; A/B come risposte.
6. **Conclusion** — cosa rende i concetti stabili/fedeli; limiti; work futuro.

---

## 8. Domande aperte / decisioni da prendere

1. **Deadline reale** (scritto / appello / presentazione) per schedulare A+B entro tempo. Da confermare con il gruppo.
2. **B come co-main o come fallback** — la proposta assume co-main (confronto); se il tempo stringe, B fallback di A e' accettabile ma riduce l'asse originalita'.
3. **Vocabolario** — RadLex (508) corrente, o passare a vocab piu' ampio/in-distribution prima di B (condiziona la copertura).
4. **Estensione** — confermare cluster + gerarchia, o altra estensione del §3 (filtraggio concetti spurii, ranking per fedelta').
5. **Layer per Path A** — CLS ultimo layer (default) vs residual stream di layer piu' profondo (sub-ablation, in linea con SAE-CLIP che prova per-layer).

---

## 9. Riferimenti

- Interni: `docs/audits/ML-AUDIT-2026-06-25.md` (M-001..M-008),
  `CONCEPT_INSTABILITY_DIAGNOSIS.md`, `ADDITIONAL_ABLATION_STUDIES.md`,
  `SAE_TRAINING_SMALL_DATASET.md`, `VOCAB_BUILDING_ALTERNATIVES.md`,
  `notebooks/autoencoder/ablation/REPORT*.md`.
- Requisiti: `docs/requirements/progetto_5_descrizione_completa.md`,
  `docs/requirements/indicazioni_progetto_xai.md`.
- Letteratura verificata: SPLiCE (NeurIPS 2024, arXiv 2402.10376), Steering CLIP
  ViT with SAEs (arXiv 2504.08729), clip-topk-sae (HF lasgroup), Scaling SAEs
  (OpenAI).
