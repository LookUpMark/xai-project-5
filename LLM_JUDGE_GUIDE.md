# Guida all'uso del LLM Judge (CLI)

Il modulo `evaluate_llm_judge.py` è responsabile della valutazione autonoma (tramite un LLM "giudice") delle spiegazioni/concetti generati rispetto ai referti radiologici reali (ground truth).

Questo file usa l'architettura **LangGraph** in background per gestire un agent con passaggi di ragionamento, garantendo estrazioni rigorose e un rating coerente. Di seguito viene illustrato come pilotare lo script da terminale tramite i vari argomenti disponibili (CLI).

---

## 🚀 Uso di base

Il caso d'uso più semplice (legge i dati della baseline dal dataset di default):
```powershell
python src\evaluate_llm_judge.py
```

Se vuoi specificare un dataset specifico (ad esempio `iu_xray`):
```powershell
python src\evaluate_llm_judge.py --dataset iu_xray
```

---

## 📂 1. Impostare la Sorgente dei Risultati (`--source`)
L'argomento più importante se stai testando metodologie diverse o modelli multipli. 
Di default, lo script cerca in `results/<dataset>/baseline/sample_explanations.json`.

**Sintassi:** `--source <nome_cartella_o_metodo>`

**Esempi:**
```powershell
# Valutare il metodo SPLiCE
python src\evaluate_llm_judge.py --source spliece

# Valutare il metodo SAE-Hidden
python src\evaluate_llm_judge.py --source sae_hidden

# Valutare una tua cartella custom "miei_test"
python src\evaluate_llm_judge.py --source miei_test
```
**Cosa fa sotto il cofano:**
1. Cerca l'input in `results/iu_xray/miei_test/sample_explanations.json`
2. Scrive il CSV di output (i voti) su `results/iu_xray/aligned_scores_miei_test.csv`
3. Salva checkpoint e metriche usando il prefisso `_miei_test`

---

## 🔄 2. Ripristino e Checkpointing (`--resume`, `--checkpoint-every`)
L'inferenza di migliaia di coppie richiede tempo. Il Judge implementa un robusto sistema di checkpointing.

**Sintassi:**
- `--resume`: Riavvia un'esecuzione interrotta. Lo script cercherà automaticamente un file `.ckpt` temporaneo e salterà le coppie già analizzate con successo.
- `--checkpoint-every N`: Ogni quanti campioni aggiornare il file di salvataggio su disco (default: 25).

**Esempio di esecuzione lunga con ripristino:**
```powershell
python src\evaluate_llm_judge.py --source sae_hidden --resume --checkpoint-every 10
```
*Se ti si spegne il computer, rilanciando lo stesso comando riprenderai esattamente da dove si era fermato!*

---

## 🤖 3. Usare LM Studio (API Locale OpenAI Compatibile)
Se non vuoi/puoi caricare il modello direttamente in memoria (HuggingFace/Transformers) ma preferisci usare **LM Studio** in background.

**Sintassi:**
- `--lm-studio`: Abilita l'utilizzo del server locale.
- `--lm-studio-url`: L'indirizzo del server (default: `http://localhost:1234/v1`).
- `--model`: Modifica il nome del modello richiesto all'API.

**Esempio:**
```powershell
# Usare LM Studio sulla porta di default
python src\evaluate_llm_judge.py --lm-studio

# Usare LM Studio specificando un nome modello custom:
python src\evaluate_llm_judge.py --lm-studio --model unsloth/medgemma-4b-it
```
*Nota: Con `--lm-studio` la GPU non verrà allocata dallo script Python, ma da LM Studio stesso.*

---

## 📊 4. Specificare il Dataset (`--dataset`)
Se hai dataset diversi (`iu_xray`, `padchest`, ecc.), puoi specificare su quale lavorare. Modifica automaticamente il percorso radice per la lettura e il salvataggio in `results/<dataset>/...`.

**Sintassi:**
- `--dataset <nome_dataset>` (deve esistere in `config.py`/`xai_datasets`)

**Esempio:**
```powershell
python src\evaluate_llm_judge.py --dataset padchest --source sae_hidden
```

---

## 📝 Riepilogo Comandi Utili (Cheat Sheet)

| Operazione | Comando |
| :--- | :--- |
| **Run Standard** | `python src\evaluate_llm_judge.py --dataset iu_xray` |
| **Run Metodo Custom** | `python src\evaluate_llm_judge.py --dataset iu_xray --source <metodo>` |
| **Riprendi Crash** | `python src\evaluate_llm_judge.py --source <metodo> --resume` |
| **Usa LM Studio** | `python src\evaluate_llm_judge.py --lm-studio` |
| **Run Sicuro e Lento**| `python src\evaluate_llm_judge.py --resume --checkpoint-every 5` |

**Nota Bene:** I risultati aggregati in JSON (con le metriche calcolate di rating, accuratezza, errori, ecc.) e il file CSV di riga per riga verranno sempre depositati automaticamente dentro `results/<dataset>/`. Non devi mai muovere o rinominare file manualmente!
