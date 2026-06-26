# Indicazioni per la consegna del progetto

```mermaid
flowchart TD
    FORM["Compilazione form\n(entro 4 maggio)"] --> CONF["Conferma assegnazione\n(entro 7 maggio)"]
    CONF --> WORK["Sviluppo progetto\n(3-5 settimane)"]
    WORK --> SCRITTO["Esame scritto\n(appello)"]
    SCRITTO --> SUB["Submission .zip\n(entro 5gg dopo scritto)"]
    SUB --> PRES["Presentazione orale\n(~7gg dopo scritto)\n15 min + 15 min Q&A"]
    PRES --> VERB["Verbalizzazione\n(scritto + progetto OK)"]

    WORK -.->|"alternativa"| PRES2["Presentazione in\nappello successivo"]

    style FORM fill:#e3f2fd,stroke:#1565C0
    style SUB fill:#fff3e0,stroke:#E65100
    style PRES fill:#fce4ec,stroke:#C62828
    style VERB fill:#e8f5e9,stroke:#2E7D32
```

Le modalità di consegna, le scadenze e i principali vincoli del progetto del corso di Explainable and Trustworthy AI sono comuni ai progetti proposti, incluso il progetto 5.[file:12]

## Gruppo e scelta del progetto

- Occorre selezionare uno dei progetti proposti dal corso e formare un gruppo di 3 persone.[file:12]
- La scelta del progetto va comunicata compilando un form online una sola volta per gruppo, inserendo matricola, nome e cognome di tutti i componenti.[file:12]
- La compilazione del form deve avvenire entro il 4 maggio.[file:12]
- La conferma ufficiale dell'assegnazione del progetto viene pubblicata entro il 7 maggio.[file:12]

## Modalità di consegna

- Il materiale del progetto deve essere caricato sul Portale della Didattica, nella sezione “Elaborati”.[file:12]
- La consegna deve avvenire sotto forma di un unico archivio `.zip`.[file:12]
- Lo `.zip` deve includere le slide della presentazione, il documento di supporto, il codice in un repository pubblico e i dati utilizzati.[file:12]

## Materiale richiesto

- È richiesto un breve documento di recap di circa 2–3 pagine, che servirà da supporto alla discussione e sarà letto dai docenti prima della presentazione.[file:12]
- Per questo documento verrà fornito un template da seguire.[file:12]
- Va inoltre preparata una presentazione orale della durata di 15 minuti, seguita da 15 minuti di discussione.[file:12]

## Struttura consigliata

```mermaid
flowchart LR
    subgraph DOC["Struttura Slide & Report"]
        direction TB
        S1["1. Introduzione"]
        S2["2. Related Work"]
        S3["3. Research Gaps"]
        S4["4. Methodology"]
        S5["5. Results & Analysis"]
        S6["6. Conclusione"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    style DOC fill:#f5f5f5,stroke:#616161
```

La struttura consigliata sia per le slide sia per il brief recap report è la seguente:[file:12]

1. Introduzione.[file:12]
2. Related work / Literature review.[file:12]
3. Research gap discussion e identificazione dei gap.[file:12]
4. Methodology and implementation.[file:12]
5. Results and analysis.[file:12]
6. Conclusione breve.[file:12]

## Deadline e presentazione

- La submission del progetto, comprensiva di recap document e slide, deve avvenire al massimo 5 giorni dopo l'esame scritto dell'appello in cui si intende far valere il progetto.[file:12]
- La presentazione si svolge approssimativamente 7 giorni dopo lo scritto, secondo slot compatibili fissati dai docenti.[file:12]
- Il progetto resta valido per l'intero anno accademico.[file:12]

## Vincoli e flessibilità

- Non è obbligatorio sostenere progetto e scritto nello stesso appello.[file:12]
- È possibile sostenere prima lo scritto e presentare il progetto in un appello successivo, oppure viceversa.[file:12]
- La verbalizzazione finale avviene quando risultano superati sia il progetto sia l'esame scritto.[file:12]

## Valutazione

```mermaid
pie title Composizione Voto Finale (max 32 pt)
    "Progetto (max 16 pt)" : 16
    "Esame Scritto (max 16 pt)" : 16
```

- Il progetto vale fino a 16 punti.[file:12]
- La valutazione del progetto considera literature review, research gaps, metodologia e assessment, originalità o novità, discussione e analisi, e chiarezza.[file:12]
- L'esame scritto vale anch'esso fino a 16 punti ed è basato sugli argomenti del corso.[file:12]
