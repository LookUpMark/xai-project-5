# Ablation aggiuntive derivate dalla diagnosi di instabilita'

Estensioni del programma di ablation gia' in `notebooks/autoencoder/ablation/`
(a0 consensus, a1 dict_size, a2 k_sweep, a3/a6 baselines, a4 activation bakeoff).
Queste non attaccano i *sintomi* (k, dict_size, variante di attivazione - gia'
coperti) ma le **cause alla radice** identificate in
`CONCEPT_INSTABILITY_DIAGNOSIS.md`: lo spazio di attivazione sbagliato
(embedding CLIP proiettati) e il regime data-starved.

Principio guida: **consolidare prima i 5 notebook esistenti + REPORT**, poi
valutare le estensioni in base al tempo/GPU residui. Aggiungere ablation a
pioggia diluisce il finding; aggiungerne selettive lo rafforza.

> **Stato (aggiornato 2026-06-22).** I 5 notebook originali (a0–a4) + `REPORT.md`
> sono **consolidati e committati**. L'asse della *fedelta'* (validazione, non
> causa) e' stato aggiunto come **a5** (`05_faithfulness.ipynb`): ~10% delle
> feature live sono fedeli a etichette cliniche reali oltre un null per-feature.
> Di queste estensioni **nessuna e' ancora consegnata** — sono *future work*:
> #1 (pre-projection) e #2 (augmentation) sono i "big swing" ad alto costo
> GPU/MPS (nuova estrazione embedding + retrain); #4 (step-sweep) e #3 (shared
> init) sono i candidati low-cost se resta tempo.

## Candidati (rapporto impatto/costo)

| # | Ablation | Attacca | Costo | Priorita' |
|---|---|---|---|---|
| 1 | Pre-projection hidden states | Spazio di attivazione sbagliato (causa 1b) | Alto | Alta |
| 4 | Step-sweep / early stopping | Overfit vs strutturale (causa 3) | Bassissimo | Alta |
| 3 | Shared init / model soup | Init randomico come causa | Basso | Media |
| 2 | Augmentation pre-embedding | Dataset tiny (causa 1a) | Alto | Bassa* |
| 5 | SPLiCE naming (post-hoc) | Naming debole (causa 2) | Basso | Bassa** |
| 6 | Vocab dal gold standard / piu' ampio | Naming debole (causa 2) | Basso | Bassa** |

\* Gia' in `SAE_TRAINING_SMALL_DATASET.md`; richiede la stessa nuova estrazione di #1.
\*\* Vedi `VOCAB_BUILDING_ALTERNATIVES.md` (se scritto): il naming debole e' in
parte *conseguenza* dell'instabilita', dunque va affrontato insieme a dict_size/steps.

## #1 - Spazio di attivazione pre-projection (il "big swing")

L'estrazione attuale usa `model.get_image_features()` -> `(B, 512)`, cioe' lo
spazio di *proiezione* testo-immagine di CLIP, ottimizzato per essere liscio e
regolarizzato. E' lo spazio sbagliato per un SAE: la struttura sparse
interpretabile sta nei hidden states pre-projection dell'encoder.

**Cosa cambiare in `src/extract_embeddings.py`:**

```python
# Prima (proiettato, 512-d):
outputs = model.get_image_features(**inputs)            # (B, 512)

# Dopo (pre-projection):
vision = model.vision_model(pixel_values=inputs["pixel_values"])
outputs = vision.last_hidden_state[:, 0, :]             # token CLS, (B, 768) su ViT-B/32
# in alternativa: mean-pool su last_hidden_state
```

**Cosa implica:**
- `activation_dim` del SAE cambia (512 -> 768 su ViT-B/32); tutti i config
  SAE vanno adattati. I modelli baseline non sono piu' confrontabili direttamente
  (spazio diverso) -> e' un setting a se' stante, va confrontato via
  signal-to-null ratio e consensus, non via Jaccard cross-config.
- Riestrarre train/test embeddings (BiomedCLIP su 7470 immagini, GPU/MPS).
- Rieseguire un mini-programma SAE (1-2 dict_size, 3 seed).

**Perche' vale la pena (narrativamente win-win):**
- Se l'instabilita' *crolla* -> pivot positivo forte: "abbiamo individuato la
  causa nello spazio di attivazione e la risolviamo".
- Se *resta* -> conferma che il collo di bottiglia e' il dataset (causa 1a), e
  giustifica l'augmentation (#2).
In entrambi i casi il risultato e' piu' informativo del finding attuale.

## #4 - Step-sweep / early stopping (low-hanging fruit)

Quasi gratis se integrato nei training gia' pianificati. Testa se l'instabilita'
e' **strutturale** (emerge subito, causa 1) o **da overfit** (cresce col
training, causa 3).

**Cosa cambiare:** salvare checkpoint del SAE a step fissi (5k, 10k, 20k) oltre
a quello finale, poi calcolare Jaccard cross-seed a ciascun checkpoint.

**Letto atteso:**
- Jaccard gia' ~0 a 5k step e costante -> instabilita' strutturale (dataset):
  la causa e' la 1, non il numero di step. Conferma la tesi principale.
- Jaccard che *peggiora* col training -> overfit: few-steps + early stopping
  su validation reconstruction migliora la robustezza. Aprirebbe un lever
  azionabile a basso costo.

**Costo:** modificare il loop di training per dumpare checkpoint intermedi;
la valutazione e' analysis-only (come a0).

## #3 - Shared init / model soup cross-seed

Inizializzare tutti i seed dallo stesso punto (o pesarli in un model soup) per
forzare feature comuni. Testa l'ipotesi "l'instabilita' nasce dall'init randomico".

- Se lo shared init *fissa* l'instabilita' -> la causa e' l'init, non i dati:
  contributo pulito e controllabile.
- Se *non la fissa* -> la causa e' il landscape (dati/spazio): conferma la 1.
Costo basso (retrain con init condiviso, riusa la meccanica esistente).

## Sequenza consigliata

1. **Ora**: finire a1/a2/a4 (in corso) + `REPORT.md` delle ablation. Consolidare.
2. **Intanto, gratis**: attivare #4 (checkpoint intermedi) nei training in corso.
3. **Se resta tempo/GPU**: #1 (pre-projection) come big swing opzionale.
4. **Solo se residuo**: #3 (shared init). Evitare #2/#5/#6 se il tempo e' tirato
   (costosi o marginali rispetto al finding principale).

## Gia' considerate e droppate (non riproporre)

- **auxk / dead-resampling full ablation**: null-by-construction al budget di
  12k step (`TopKTrainer.dead_feature_threshold` hardcoded a 10M token).
  Risuscitato come probe interno ad a1 con threshold abbassato.
- **Faithfulness vs 14 patologie NIH**: le 14 etichette ChestX-ray14 *non*
  esistono in IU X-Ray. NOTA: esiste pero' un gold standard alternativo
  (colonne `MeSH`/`Problems` di `indiana_reports.csv`) che riapre una valutazione
  di faithfulness concept-attivazione <-> etichetta per-immagine - vedi
  `VOCAB_BUILDING_ALTERNATIVES.md`.
