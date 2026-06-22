# Suggerimenti per il training SAE su IU X-Ray (7400 campioni)

> **Stato (aggiornato 2026-06-22).** Le raccomandazioni qui sotto sono ora **verificate dal programma di ablation** (`notebooks/autoencoder/ablation/REPORT.md`). Sintesi dei verdetti:
> - **Ridurre `dict_size`** (Suggerimento 1) → **Ablation 01**: riduce i dead (40.9 → 30.7%) ma **NON** aumenta la stabilità cross-seed (l'over-expansion causa i dead, non l'instabilità — che è strutturale).
> - **Ridurre `k`** (Suggerimento 3) → **Ablation 02**: c'è un debole sweet spot a k=16 (signal-to-null 1.30), ma l'accordo assoluto resta minuscolo e k=32 è sul pavimento del caso.
> - **Augmentation** (Suggerimento 2) e **step-sweep** (Suggerimento 4) restano **future work** (`ADDITIONAL_ABLATION_STUDIES.md`).
>
> In breve: le raccomandazioni valgono per ridurre i dead e migliorare l'efficienza del dizionario, ma **non risolvono la non-riproducibilità cross-seed** — diagnosticata in `CONCEPT_INSTABILITY_DIAGNOSIS.md` come limite strutturale (pochi campioni + non-unicità della decomposizione sparsa su embedding CLIP proiettati).

## Problema

Il dataset IU X-Ray contiene ~7400 coppie immagine-testo. Con `dict_size=4096`,
il rapporto campioni/feature e' solo 1.8x. La letteratura suggerisce almeno 10x.

Conseguenze attese con la configurazione attuale:
- Alta percentuale di dead features (>30-50%)
- Overfitting latente (bassa loss ma concetti poco generalizzabili)
- Concept naming meno preciso per mancanza di varieta' nei dati

---

## Suggerimento 1: Ridurre dict_size

L'intervento piu' impattante e a costo zero.

| dict_size | Rapporto campioni/feature | Note |
|-----------|--------------------------|------|
| 4096 | 1.8x | Attuale, troppo basso |
| 2048 | 3.6x | Borderline, accettabile per progetto dimostrativo |
| 1024 | 7.2x | Raccomandato per dataset piccoli |

Meno concetti nel dizionario ma ognuno supportato da piu' dati. La granularita'
diminuisce ma l'interpretabilita' e la robustezza migliorano.

---

## Suggerimento 2: Augmentation pre-embedding

Generare varianti delle immagini prima dell'estrazione con BiomedCLIP.
Gli embedding risultanti saranno simili ma non identici, fornendo
maggiore diversita' al SAE.

Augmentazioni sicure per radiografie toraciche:
- Flip orizzontale (anatomia approssimativamente simmetrica)
- Crop random leggero (90-95% dell'immagine)
- Rotazione +/- 5 gradi
- Gaussian noise leggero

Augmentazioni da evitare:
- Color jitter (altera il contrasto diagnostico)
- Cutout/erasing (rimuove strutture anatomiche)
- Rotazioni eccessive (non realistiche per CXR)

Effetto: 3-5 augmentazioni per immagine -> 22k-37k embedding diversi.

---

## Suggerimento 3: Ridurre k (sparsita')

Da k=32 a k=16: con meno dati, forzare meno feature attive per campione
produce concetti meno ridondanti.

- 16 concetti per immagine sono sufficienti per radiografie toraciche
  (tipicamente 5-10 findings clinici per referto)
- Riduce il rischio che concetti diversi catturino la stessa informazione

---

## Suggerimento 4: Ridurre training steps

Con 7400 campioni e batch_size=256, un'epoca completa = 28 step.
50000 step = ~1730 epoche sul dataset.

Rischio: dopo un certo punto il modello memorizza i dati anziché
apprendere feature generalizzabili.

Suggerimento: 20000-30000 step con early stopping basato sulla
reconstruction loss su un validation set (10% hold-out).

---

## Configurazioni raccomandate

### Opzione A - Conservativa (minimo cambio)

```python
dict_size = 2048
k = 32
steps = 30_000
```

Rapporto 3.6x, mantiene la granularita' ragionevole, riduce dead features.

### Opzione B - Aggressiva (massima qualita' per campione)

```python
dict_size = 1024
k = 16
steps = 20_000
```

Rapporto 7.2x, concetti piu' robusti e stabili cross-seed.

### Opzione C - Augmentation + conservativa

```python
# Prima: generare 3 augmentazioni per immagine -> ~22k embedding
dict_size = 2048
k = 32
steps = 40_000
```

Rapporto effettivo 10.7x con augmentation 3x.

---

## Metriche da monitorare

Per scegliere la configurazione migliore, confrontare:

1. **dead_features_pct**: obiettivo < 20%
2. **Reconstruction MSE**: deve convergere senza oscillazioni
3. **Jaccard stability (cross-seed)**: obiettivo > 0.4 per top-k features
4. **Concept naming quality**: i top candidates devono avere score > 0.3

---

## Riferimenti

- Bricken et al. (2023) "Towards Monosemanticity" - SAE su milioni di attivazioni
- Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Directions"
- Per dataset piccoli: rapporto minimo 5-10x campioni/features e' prassi comune
  nella letteratura su dictionary learning (Olshausen & Field, 1997)
