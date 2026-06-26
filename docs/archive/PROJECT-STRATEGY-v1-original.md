# Progetto 5 — Strategia di Implementazione Ottimale: Unsupervised Concept Discovery per Medical VLM

## Sintesi Strategica

L'obiettivo di questo documento è tradurre i requisiti del progetto 5 in una strategia concreta, ad alto punteggio e a basso overhead implementativo. La strategia si basa su tre principi: (1) usare modelli e framework già esistenti piuttosto che implementare da zero; (2) massimizzare il numero di requisiti coperti con pipeline riutilizzabili; (3) distinguere chiaramente i contributi "necessari" da quelli "opzionali" per ottenere il massimo punteggio sui criteri di valutazione.

I criteri di valutazione del progetto sono: **literature review**, **research gaps**, **methodology and assessment**, **originality/novelty**, **discussion and analysis**, e **clarity**. Tutti e sei influenzano il punteggio fino a 16 punti. La strategia che segue massimizza ciascuna di queste dimensioni.[^1]

***

## Contesto: Cosa Valuta la Commissione

```mermaid
pie title Distribuzione Criteri di Valutazione (max 16 pt)
    "Literature Review" : 3
    "Research Gaps" : 2
    "Methodology & Assessment" : 4
    "Originality/Novelty" : 3
    "Discussion & Analysis" : 2
    "Clarity" : 2
```

Per ottenere il massimo punteggio è essenziale capire cosa si intende per ciascun criterio.[^1]

- **Literature review**: copertura sistematica e critica della letteratura, non semplice elenco di paper.
- **Research gaps**: identificazione chiara e motivata di limiti aperti non risolti dagli approcci attuali.
- **Methodology and assessment**: pipeline implementata con valutazione seria e rigorosa dei risultati.
- **Originality/novelty**: almeno un contributo che vada oltre la semplice replica, anche piccolo.
- **Discussion and analysis**: analisi profonda dei risultati, failure cases, limiti della propria pipeline.
- **Clarity**: presentazione chiara e ben strutturata sia del documento sia delle slide.

***

## Stato dell'Arte: Letteratura Chiave

### Il Framework di Riferimento: MedConcept

Il paper da cui il progetto parte direttamente è **MedConcept** (Haque et al., 2026). MedConcept introduce una pipeline completamente non supervisionata per estrarre concetti medici latenti da VLM pretrained e allinearli a semantica testuale clinicamente verificabile tramite report radiologici. Il framework:[^2]

1. Estrae attivazioni sparse neuronale da VLM pretrained.
2. Allinea le feature latenti con un vocabolario medico derivato da UMLS.[^2]
3. Traduce le attivazioni in pseudo-report-style summaries ispezionabili da medici.
4. Valuta l'allineamento semantico con tre metriche quantitative: **Aligned**, **Unaligned**, **Uncertain**, usando un LLM medico frozen come giudice esterno.[^2]

Il codice è annunciato come "to be released on acceptance" — quindi **non è ancora pubblico al momento della stesura**. Questo è un elemento cruciale per la strategia: il gruppo dovrà re-implementare una versione semplificata, non copiare il codice originale.[^2]

### Sparse Autoencoder su VLM: SOTA

Il lavoro più rilevante e direttamente utilizzabile è **"Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models"** (Pach et al., NeurIPS 2025), che estende esplicitamente gli SAE ai VLM come CLIP e ne dimostra l'efficacia nel decomporsi in feature monosemantiche. Il repository GitHub pubblico (`ExplainableML/sae-for-vlm`) fornisce codice completo, script di training e monosemanticity score. Questo è il punto di partenza tecnico ideale.[^3][^4]

Complementare a questo è **SPLiCE** (Bhalla et al., NeurIPS 2024), che propone un approccio alternativo senza training: decompone i CLIP embedding in combinazioni lineari sparse di concetti da un vocabolario testuale, senza addestrare alcun autoencoder. SPLiCE è quindi ancora più leggero in termini computazionali. Il codice è pubblico.[^5]

**DN-CBM** (Rao et al., ECCV 2024) inverte il paradigma classico dei Concept Bottleneck Models: prima usa un SAE per scoprire concetti latenti in CLIP, poi li nomina automaticamente e ci allena sopra probe lineari per classificazione. Anche questo ha codice pubblico su GitHub (`neuroexplicit-saar/discover-then-name`).[^6][^7]

### Modelli VLM Medici Disponibili

| Modello | Tipo | Accesso | Note |
|---------|------|---------|------|
| **BiomedCLIP** (Microsoft) | Vision-Language | HuggingFace open[^8] | SOTA su benchmark biomedical, 15M coppie PMC |
| **RAD-DINO** (Microsoft) | Vision-only encoder | HuggingFace open[^9] | Self-supervised DINOv2 su chest X-ray, eccellente per CXR |
| **BioViL / BioViL-T** (Microsoft) | Vision-Language | HuggingFace open[^10] | Specializzato su CXR con MIMIC |
| **CXR-BERT** (Microsoft) | Language encoder | HuggingFace open[^11] | Domain-specific per radiology reports |
| **CheXagent** (Stanford AIMI) | VLM multimodale | HuggingFace open[^12] | Foundation model per CXR interpretation |

**Raccomandazione**: usare **BiomedCLIP** come backbone principale, poiché combina encoder visivo e testuale in un unico spazio condiviso su dati biomedici, ed è ben documentato con API semplici.[^13][^8]

### Dataset Disponibili

| Dataset | Immagini | Report | Note |
|---------|----------|--------|------|
| **IU X-ray (OpenI)** | ~7.470 | ~3.955 | Leggero, no registrazione, ideale per sviluppo[^14] |
| **NIH ChestX-ray14** | 112.120 | No (label estratte) | Solo label, no report testuali liberi[^15] |
| **MIMIC-CXR** | 377.110 | 227.835 | Massimo coverage, richiede PhysioNet access[^16] |
| **CheXpert** | 224.316 | No (label estratte) | Label automatic, no free-text[^17] |

**Raccomandazione**: IU X-ray per prototipazione e sviluppo rapido; MIMIC-CXR per valutazione finale (se si ottiene accesso PhysioNet, che è gratuito ma richiede verifica identità). IU X-ray è scaricabile immediatamente, ha report con sezioni `Findings` e `Impression` ben strutturate, ed è la scelta ottimale per un progetto universitario.[^14]

### LLM come Giudice per la Valutazione Semantica

MedConcept usa un LLM frozen come valutatore semantico. Le opzioni pratiche per replicarlo senza costi eccessivi sono:[^2]

- **BioMedLM / BioGPT**: modelli aperti per NLP biomedico, deployabili localmente.
- **Llama 3.1 / Mistral-7B-Instruct** con prompt medici: zero-shot, localmente, costo zero.
- **API OpenAI / Gemini**: più costoso, ma output di qualità per valutazione semantica LLM-as-a-Judge.[^18][^19]

Per un progetto universitario, **Llama 3.1 8B o Mistral-7B via Ollama o HuggingFace Inference** è la scelta pragmatica: nessun costo, nessuna dipendenza da chiavi API, reproducibile.

***

## Research Gaps da Identificare (Sezione 2 del Progetto)

```mermaid
flowchart LR
    subgraph GAPS["Research Gaps Identificati"]
        direction TB
        G1["🔄 Gap 1\nInstabilità concetti\ntra seed/run"]
        G2["📚 Gap 2\nDipendenza UMLS\nnon portabile"]
        G3["⚠️ Gap 3\nBias LLM Judge\nposition/verbosity"]
        G4["📝 Gap 4\nReport incompleti\nfalsi Unaligned"]
        G5["🔗 Gap 5\nConcetti indipendenti\nno relazioni"]
        G6["📈 Gap 6\nValutazione non scala\nsolo qualitativa"]
    end

    subgraph EXT["Nostre Estensioni"]
        direction TB
        E1["6A: Stability\nAnalysis"]
        E2["Text-encoder\napproach"]
        E3["6D: Bias\nAnalysis"]
        E5["6B: Clustering\nGerarchico"]
    end

    G1 -.->|"risolto da"| E1
    G2 -.->|"risolto da"| E2
    G3 -.->|"risolto da"| E3
    G5 -.->|"risolto da"| E5

    style GAPS fill:#fff8e1,stroke:#FF8F00
    style EXT fill:#e8f5e9,stroke:#2E7D32
```

Questa sezione del progetto richiede identificazione critica dei limiti attuali. I gap più solidi su cui costruire sono:[^4][^5][^2]

1. **Mancanza di garanzie di stabilità**: concetti scoperti da SAE cambiano a seconda di seed, iperparametri e layer scelto. Nessun approccio attuale fornisce barre di confidenza sui concetti.
2. **Dipendenza da ontologie esterne (UMLS)**: il mapping verso UMLS richiede accesso a terminologie curate e non è direttamente adattabile a nuovi domini o lingue.
3. **Bias del LLM giudice**: l'uso di un LLM come valutatore introduce position bias, verbosity bias e hallucination. MedConcept riconosce questo limite ma non lo risolve.[^20]
4. **Report clinici incompleti**: i Findings e Impression di MIMIC o IU X-ray possono omettere informazioni, causando falsi Unaligned.[^2]
5. **Trattamento indipendente dei concetti**: nessun approccio corrente modella relazioni strutturate tra concetti (gerarchie anatomiche, dipendenze causali).
6. **Scalabilità della valutazione qualitativa**: l'ispezione manuale dei concetti non scala. Le metriche Aligned/Unaligned/Uncertain sono quantitative ma non coprono coerenza intra-sample.

***

## Architettura della Pipeline Raccomandata

La pipeline raccomandata è una versione semplificata e strumentata di MedConcept, costruita interamente su componenti open-source già esistenti. Si articola in 5 moduli.

```mermaid
flowchart TD
    subgraph M1["Modulo 1 — Feature Extraction"]
        CXR["🖼️ CXR Images"] --> CLIP["BiomedCLIP\n(frozen, 512-dim)"]
    end

    subgraph M2["Modulo 2 — SAE Training"]
        SAE["Sparse Autoencoder\nTop-K (k=32)\n512 → 4096"]
    end

    subgraph M3["Modulo 3 — Concept Naming"]
        VOCAB["Medical Vocabulary\n(RadLex ~400 terms)"] --> MATCH["Cosine Similarity\nnearest neighbor"]
    end

    subgraph M4["Modulo 4 — Explanation Generation"]
        PSEUDO["Top-K activated features\n→ pseudo-report"]
    end

    subgraph M5["Modulo 5 — LLM-as-Judge"]
        JUDGE["Gemma 4 E4B\n(Ollama locale)"] --> SCORE["Aligned / Unaligned\n/ Uncertain"]
    end

    CLIP -->|"z ∈ ℝ⁵¹²"| SAE
    SAE -->|"W_dec features"| MATCH
    MATCH -->|"concept_names"| PSEUDO
    PSEUDO -->|"pseudo-report + report reale"| JUDGE

    style M1 fill:#e3f2fd,stroke:#1565C0
    style M2 fill:#fce4ec,stroke:#C62828
    style M3 fill:#e8f5e9,stroke:#2E7D32
    style M4 fill:#fff3e0,stroke:#E65100
    style M5 fill:#f3e5f5,stroke:#6A1B9A
```

### Modulo 1 — Feature Extraction da VLM Pretrained

```
Input: CXR image
Model: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
Output: embedding visivo z ∈ R^512 per ogni immagine
```

Usare `open_clip` o HuggingFace `AutoModel` per estrarre le rappresentazioni del visual encoder. Questo modulo non richiede training: è pura inferenza forward-pass. Per il batch completo di IU X-ray (~5.000 immagini) bastano 10-20 minuti su Colab GPU T4.[^13]

### Modulo 2 — Sparse Autoencoder Training

```
Input: dataset di embedding z (N × 512)
Architecture: Encoder lineare → Top-K activation → Decoder lineare
Output: dizionario di feature W ∈ R^(d_hidden × 512), d_hidden >> 512 (es. 4096)
Loss: reconstruction MSE + L1 sparsity penalty
```

Usare il codice del repository `ExplainableML/sae-for-vlm` come base. Il training è leggero: un SAE top-k su embedding 512-dim converge in 10-30 minuti su GPU anche modesta. Iperparametri chiave: dimensione hidden (4096 o 8192), k nel top-k (es. k=20-50).[^4]

**Alternativa più leggera**: usare **SPLiCE** invece dell'SAE, che non richiede training ma ottimizzazione LASSO sui vettori testo. Adatto se le risorse computazionali sono limitate.[^21][^5]

### Modulo 3 — Concept Naming (Grounding Testuale)

```
Input: feature attive W_i (i-esimo neurone SAE)
Method: nearest neighbor nel CLIP text embedding space su vocabolario medico
Output: nome concetto c_i (es. "cardiomegaly", "pleural effusion", "atelectasis")
```

Per costruire il vocabolario medico: estrarre lista di termini clinici rilevanti da UMLS (subset CXR) o usare direttamente i label di NIH ChestX-ray14 (14 condizioni) espansi con sinonimi via ClinicalBERT / BiomedBERT. Per ogni feature SAE, il nome viene assegnato per massima similarity coseno con l'encoder testuale di BiomedCLIP.[^11]

Questo approccio replica la grounding step di MedConcept senza richiedere accesso all'ontologia UMLS completa, usando invece la struttura semantica già codificata nel text encoder di BiomedCLIP.

### Modulo 4 — Concept-Based Explanation Generation

```
Input: nuova immagine x, SAE trained
Process: forward pass BiomedCLIP → SAE encode → top-k features attivate → lookup nomi
Output: lista ordinata di concetti con activation score, pseudo-report
```

Per ogni campione, si recuperano i concetti più attivati (es. top-5) e si costruisce una spiegazione pseudo-report in linguaggio naturale concatenando i concetti ordinati per activation. Questo è il punto in cui si può aggiungere novità: per esempio, organizzando i concetti in gruppi semantici (anatomia, patologia, qualità immagine) oppure usando un template strutturato.

### Modulo 5 — Valutazione Quantitativa (Aligned/Unaligned/Uncertain)

```
Input: pseudo-report generato, report radiologico reale (Findings + Impression)
Method: LLM-as-judge con prompt strutturato
Output: score Aligned / Unaligned / Uncertain per ogni concetto e campione
```

Il prompt da usare con il LLM giudice (Mistral o Llama-3 locale):

```
Given the radiology report: "{report}"
And the concept discovered by the model: "{concept}"
Determine if the report: 
(A) SUPPORTS the concept (Aligned)
(B) CONTRADICTS the concept (Unaligned)  
(C) is AMBIGUOUS or does not mention it (Uncertain)
Answer with exactly one word: Aligned, Unaligned, or Uncertain.
```

Calcolare le statistiche aggregate: percentuale di concetti Aligned sul totale, per ogni classe di patologia, per ogni layer del modello. Confrontare i risultati con e senza SAE (feature dense vs feature sparse) per mostrare il beneficio della decomposizione.[^2]

***

## Estensione Originale: Come Ottenere Punti di Novità

```mermaid
quadrantChart
    title Estensioni — Effort vs Impact
    x-axis "Basso Sforzo" --> "Alto Sforzo"
    y-axis "Basso Impatto" --> "Alto Impatto"
    quadrant-1 "Sweet spot"
    quadrant-2 "Quick wins"
    quadrant-3 "Evitare"
    quadrant-4 "Se c'è tempo"
    "Stability Analysis (6A)": [0.25, 0.85]
    "Layer Comparison (6C)": [0.35, 0.65]
    "Clustering Gerarchico (6B)": [0.55, 0.60]
    "Bias Analysis (6D)": [0.60, 0.75]
```

La valutazione richiede almeno un contributo originale. Le estensioni raccomandate per massimizzare il punteggio di novelty con sforzo contenuto sono, in ordine di difficoltà crescente:

### Opzione A — Analisi di Stabilità dei Concetti (basso sforzo, alto impatto)
Ripetere il training del SAE con seed diversi (5 run) e misurare la variabilità nei concetti scoperti: quanti concetti sono consistenti tra run, quanti cambiano? Questo risponde direttamente al research gap #1 e non è fatto in MedConcept. Implementazione: loop su seed con seed diversi, calcolo Jaccard similarity sui top-k concetti per run.[^2]

### Opzione B — Clustering Gerarchico dei Concetti (medio sforzo)
Invece di trattare i concetti come entità indipendenti, raggrupparli in categorie (anatomia, patologia, aspetto qualitativo dell'immagine) tramite clustering (K-Means o agglomerativo) sugli embedding dei nomi dei concetti. Questo risponde al research gap #5 e aggiunge struttura alle spiegazioni.

### Opzione C — Layer Comparison (basso sforzo, medio impatto)
Addestrare l'SAE su layer diversi di BiomedCLIP (early vs late layers) e confrontare la qualità dei concetti estratti via Aligned score. Questo analizza come la profondità del layer influenza la semanticità dei concetti, replicando un'analisi simile a quella del paper sae-for-vlm.[^3]

**Raccomandazione**: implementare almeno l'**Opzione A** (stabilità) e la **Layer Comparison** (Opzione C). Entrambe richiedono poche decine di righe di codice aggiuntivo, ma producono risultati quantitativi originali e figure pubblicabili nel report.

***

## Framework e Librerie da Usare

| Componente | Libreria/Repo | Note |
|-----------|--------------|------|
| VLM Backbone | `open_clip` + HuggingFace `transformers` | BiomedCLIP caricabile con 2 righe[^13] |
| SAE Training | `ExplainableML/sae-for-vlm` (GitHub)[^4] | Codice pubblico, NeurIPS 2025 |
| SAE alternativo | `saprmarks/dictionary_learning` | Usato internamente da sae-for-vlm |
| SAE leggero | `SPLiCE` (NeurIPS 2024)[^5] | No training, solo ottimizzazione |
| Dataset CXR | IU X-ray (OpenI) / MIMIC-CXR (PhysioNet) | IU X-ray no-registration[^14] |
| LLM Giudice | Ollama + Mistral-7B / Llama-3 8B | Locale, gratuito, reproducibile |
| Metriche XAI | `Quantus` (Hedstrom et al., JMLR 2023) | Libreria per evaluation faithfulness |
| Similarity | `sentence-transformers` | Per concept-report matching |
| Clustering | `scikit-learn` | KMeans / agglomerative per Opzione B |

Tutto il codice può girare su **Google Colab Pro o una GPU universitaria**: l'inference di BiomedCLIP su ~5000 immagini richiede ~1 GB VRAM, il training del SAE ~2 GB VRAM.[^4]

***

## Piano di Lavoro Settimana per Settimana

Assumendo di avere ~4-6 settimane prima dell'appello target.

| Settimana | Attività | Deliverable |
|-----------|----------|-------------|
| **1** | Literature review sistematica (MedConcept, SAE-VLM, SPLiCE, DN-CBM, Concept Bottleneck, TCAV) | Sezione "Related Work" bozza |
| **2** | Setup ambiente, download IU X-ray, caricamento BiomedCLIP, estrazione embedding | Feature matrix salvata |
| **3** | Training SAE, concept naming, generazione pseudo-report su IU X-ray subset | Pipeline end-to-end funzionante |
| **4** | Implementazione valutazione LLM-as-judge (Aligned/Unaligned/Uncertain), calcolo statistiche | Risultati quantitativi |
| **5** | Estensione originale (stabilità / layer comparison), failure case analysis | Grafici e tabelle finali |
| **6** | Scrittura recap document, preparazione slide, revisione | Submission-ready |

***

## Struttura Consigliata del Recap Document (2-3 pagine)

1. **Introduction** (0.3 pag): motivazione clinica, opacità dei medical VLM.
2. **Related Work** (0.5 pag): MedConcept, SAE su VLM, SPLiCE, DN-CBM — con focus critico sui limiti.
3. **Research Gaps** (0.3 pag): i 3-4 gap più forti, motivati dalla letteratura.
4. **Methodology** (0.6 pag): pipeline in 5 moduli, con schema grafico semplice.
5. **Results and Analysis** (0.8 pag): tabella Aligned/Unaligned/Uncertain, grafici, failure cases, confronto layer/seed.
6. **Conclusion** (0.2 pag): sintesi e limitazioni della propria pipeline.

***

## Tabella Riassuntiva: Requisiti del Progetto vs Strategia Proposta

| Requisito ufficiale | Come viene coperto | Sforzo |
|--------------------|--------------------|--------|
| Literature review su CBM, TCAV, SAE | Review critica di 8-10 paper chiave | Medio |
| Identification of research gaps | 4-6 gap ben motivati da letteratura | Basso |
| Pipeline di concept discovery | SAE su BiomedCLIP + concept naming | Medio |
| Allineamento con vocabolario testuale | Nearest neighbor nel text embedding space | Basso |
| Concept-based explanations per sample | Top-k features attivate → pseudo-report | Basso |
| Valutazione su dataset medico | IU X-ray o MIMIC-CXR | Basso |
| Metriche Aligned/Unaligned/Uncertain | LLM-as-judge con prompt strutturato | Medio |
| Analisi failure cases | Analisi concetti ambigui, spurii, instabili | Basso |
| Contributo originale | Analisi stabilità + layer comparison | Basso-Medio |

***

## Avvertenze e Rischi

- **Accesso a MIMIC-CXR**: richiede registrazione PhysioNet e training CITI. Procedere subito se si vuole usarlo, oppure usare IU X-ray come alternativa immediata.[^16]
- **MedConcept non ha codice pubblico**: il codice sarà rilasciato "on acceptance" — non è disponibile ora. Non basarsi su di esso. Reimplementare la pipeline ispirandosi all'architettura descritta nel paper.[^2]
- **Costo LLM API**: se si usa GPT-4 come giudice, stimare ~0.02$ per chiamata × numero campioni valutati. Con IU X-ray (500-1000 campioni di test) il costo è gestibile (<20€). Con Mistral locale il costo è zero.
- **Compute**: BiomedCLIP + SAE training si esegue comodamente su Colab T4. Non servono cluster HPC per questa pipeline.[^13][^4]

---

## References

1. [XAI_00b_project_presentation.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_c055d2c8-7204-4532-860c-713e4b47a5ba/e4cd6e00-4edf-4032-969f-7433edc224fd/XAI_00b_project_presentation.pdf?AWSAccessKeyId=ASIA2F3EMEYE455DHQKT&Signature=k%2F3jNa48%2BP3Sx5rOi%2FLkphSYuAo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEG0aCXVzLWVhc3QtMSJIMEYCIQDtU9y4VSPCdSHXZcnk1mtktmEj3ig%2BIhHE7Fckrmfv9wIhAOnmIxuUtUNifcwPkF%2FQmwINi4xj34fbOlab3lDsgNePKvMECDYQARoMNjk5NzUzMzA5NzA1Igx%2FKJ8z8fVXgY1oeqwq0ASSbTPmtXEz4YeZY%2F1IhDOGm8dGMyTrAVttWrM8su8DI7ixX0vGoxGyRDn35fTSn%2BoZwz3K2fqMUNwSVTbCNN%2FjzZpDKPRYQxGcOCyGQefBazLZdKQnDk4cKnCCYWCSBkQ9thm5CeoncWFttYqEa3GlTuUCWfu0LV1w8O%2Bbkqsi1QOu8yOVpD4Kbz%2Fg82N4AoL%2BYGM3hQ3aGkwcxmy7Mrj9StJk7lSaTDXji9XLvuTA2b4os6qcofkKVsQ8zXEeO73r8724%2FRW%2FwJgCJ9g91SbSDThJ8izSGy5DcpjyuEfxWJ6NNwI8NgPq8k1MhEL1Kl43n6ODbA6eFrIOVCdQ3htzZ1LV%2BRnFKHW6v3SAfAgHlqKzCD5z59zT%2FEyRkF0cWE7R%2FbuSYIitZ0bQrHRyt4sE4TlFG2Cl3Qd195ZnIDNEsA07qXJobWHWWXDI%2FbwF3k5AbU%2BPJv9F6wtu%2FsOO8n9TB%2BXLJgq4mQvEWJ0yu3QFNUnP%2FGmZxyTrCbWWSnR8UoT6dJF8T%2FnOEsjBPGySC3AhcVLxG5Oav6%2F0zKobpSA2j99y8rcvHfnGKayAx9nu6KMQ6KY%2BBo77ruLqrDvVqxVZeq0B0FcUnff3513fHG8Q9qJjro%2FMVLW1IxaBM5J75ikmJ9S5NuFRVWaueLYIve%2FdA6ZsUdkk08PQVYtNCofa%2F2%2B58HEHQPjPxFDvGiG7yvQif%2BT9dVNuI0RBJ4A8OQVmt48DJcSLkRR%2B2eMTAX5lPVqbQ1klPzTzE5dAtU6VtURvj3adzCHUxIbgnXLzacZhMKDJxtAGOpcBJUE6FYrI6z68h21CtwJJpm0FME9Jecv8UTADajQrTDubct%2FrDbW52TBrn0DlluveurZm9%2Fy%2BizVgLRTWud%2F5%2BrO3KZ3U6Hb%2BwMtw8AH7Wpqoltsjsx30VXk2HTMD5RQQRiAj4yx7%2B%2BoO4iXs9q8FXfKJPciuH56aEvTmR9JcfWfBY%2F923s3wsgNMzhLZj5dWdQmNNy%2BryQ%3D%3D&Expires=1779544691) - Projects presentation Projects presentation Explainable and Trustworthy AI 2025-2026 Eliana Pastor P...

2. [Unsupervised Concept Discovery for Interpretability in Medical VLMs](https://arxiv.org/abs/2604.11868) - Abstract page for arXiv paper 2604.11868: MedConcept: Unsupervised Concept Discovery for Interpretab...

3. [Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](https://arxiv.org/abs/2504.02821) - Sparse Autoencoders (SAEs) have recently gained attention as a means to improve the interpretability...

4. [GitHub - ExplainableML/sae-for-vlm: [NeurIPS 2025] Sparse ...](https://github.com/ExplainableML/sae-for-vlm) - Sparse Autoencoders (SAEs) have recently gained attention as a means to improve the interpretability...

5. [Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)](https://arxiv.org/abs/2402.10376) - CLIP embeddings have demonstrated remarkable performance across a wide range of multimodal applicati...

6. [Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery](https://www.arxiv.org/abs/2407.14499) - Concept Bottleneck Models (CBMs) have recently been proposed to address the 'black-box' problem of d...

7. [Discover-then-Name: Task-Agnostic Concept Bottlenecks via ... - arXiv](https://arxiv.org/abs/2407.14499) - We use sparse autoencoders to first discover concepts learnt by the model, and then name them and tr...

8. [BiomedCLIP: a multimodal biomedical foundation model ...](https://www.microsoft.com/en-us/research/publication/biomedclip-a-multimodal-biomedical-foundation-model-pretrained-from-fifteen-million-scientific-image-text-pairs/) - Biomedical data is inherently multimodal, comprising physical measurements and natural language narr...

9. [RAD-DINO: Exploring Scalable Medical Image Encoders ...](https://huggingface.co/papers/2401.10815) - We introduce RAD-DINO, a biomedical image encoder pre-trained solely on unimodal biomedical imaging ...

10. [microsoft/BiomedVLP-BioViL-T - Hugging Face](https://huggingface.co/microsoft/BiomedVLP-BioViL-T) - In the second stage, BioViL-T is continually pretrained from CXR ... The radiology notes are accompa...

11. [CXR-BERT-general - AI Model Catalog | Microsoft Foundry Models](https://ai.azure.com/catalog/models/microsoft-biomedvlp-cxr-bert-general) - CXR-BERT is a chest X-ray (CXR) domain-specific language model that makes use of an improved vocabul...

12. [Stanford-AIMI/CheXagent - GitHub](https://github.com/Stanford-AIMI/CheXagent) - [Arxiv-2024] CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation - Stanford-AIMI/Ch...

13. [ZiyueWang/biomedclip - Hugging Face](https://huggingface.co/ZiyueWang/biomedclip) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

14. [Awesome-Medical-Dataset/resources/IU-Xray.md at main - GitHub](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/IU-Xray.md) - The IU-XRay (Indiana University Chest X-Rays) dataset contains a set of chest X-ray images and their...

15. [Table 1.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10195007/table/Tab1/) - Recent advances in deep learning have shown great potential for the automatic generation of medical ...

16. [MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports - Scientific Data](https://www.nature.com/articles/s41597-019-0322-0) - Measurement(s)Radiograph • investigation results reportTechnology Type(s)Chest Radiography • digital...

17. [CheXpert: A Large Dataset of Chest X-Rays and Competition for ...](https://stanfordmlgroup.github.io/competitions/chexpert/) - CheXpert is a large dataset of chest x-rays and competition for automated chest x-ray interpretation...

18. [LLM-as-a-Judge: automated evaluation of search query parsing ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12319771/) - Recent research (Schneider et al., 2024) has demonstrated that LLMs can effectively evaluate semanti...

19. [LLM-as-a-Judge - Wikipedia](https://en.wikipedia.org/wiki/LLM-as-a-Judge) - The LLM-based evaluations likely incorporate deeper semantic understanding, but at the cost of inter...

20. [[PDF] AUTOMATED CONCEPT DISCOVERY FOR LLM-AS-A- JUDGE ...](https://openreview.net/pdf?id=Xk0jEx8CO0) - To create a setting where LLM judge preferences can be analyzed consistently across a variety of inp...

21. [Interpreting CLIP with SpLiCE](https://www.emergentmind.com/papers/2402.10376) - This paper presents SpLiCE, a method that transforms CLIP embeddings into sparse, interpretable line...

