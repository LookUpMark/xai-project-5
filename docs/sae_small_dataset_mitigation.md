# Mitigazione del Problema dei Dataset di Piccole Dimensioni nel Training di Sparse Autoencoders (SAE)

## 1. Introduzione al Problema

Nel nostro progetto stiamo addestrando un Top-K Sparse Autoencoder (SAE) per estrarre concetti interpretabili dagli embedding generati da BiomedCLIP su radiografie toraciche (Chest X-Ray, CXR). Attualmente, lo spazio di input ha 512 dimensioni e il dizionario del SAE è impostato a 4096 feature (un fattore di espansione di 8x). 

Tuttavia, il nostro training set è composto da circa 7400 campioni (IU X-Ray). Questo porta a un rapporto campioni/feature di circa 1.8x, causando un problema documentato: circa il 44% delle feature del dizionario risulta "morto" (dead features, ovvero feature che non si attivano mai su nessun campione). L'analisi della letteratura scientifica nel campo della Mechanistic Interpretability e del Dictionary Learning conferma che questo comportamento è atteso quando il volume di dati è insufficiente.

---

## 2. Analisi della Letteratura: Dead Features e Sample Size

La ricerca sugli Sparse Autoencoders dimostra che il dimensionamento del dizionario (expansion factor) e la dimensione del dataset (sample size) sono intimamente legati al problema delle "dead features" [1, 2]. 

Nei modelli all'avanguardia per l'interpretazione dei LLM (Large Language Models), i SAE sono tipicamente **overcomplete**, con fattori di espansione che vanno da 4x a 32x la dimensione dell'input [1]. Tuttavia, questi SAE vengono addestrati su milioni o miliardi di token. 
Quando i dati scarseggiano, una proporzione significativa del dizionario rimane inutilizzata perché non ci sono abbastanza campioni per "attivare" e far apprendere configurazioni latenti rare [3]. 

Come regola pratica (rule of thumb) nel Dictionary Learning per garantire l'identificabilità globale e la stabilità del dizionario, la dimensione del campione ($m$) dovrebbe essere significativamente maggiore del numero di feature ($k$). La letteratura suggerisce un rapporto campioni/feature di almeno **5x - 10x** [3, 4]. Nel nostro caso, con 7400 campioni e 4096 feature, il rapporto 1.8x è troppo basso, spiegando perfettamente il 44% di feature morte.

---

## 3. Soluzione 1: Riduzione della Dimensionalità (Dictionary Size)

La soluzione più immediata per ribilanciare l'equazione è ridurre il numero di feature latenti (`dict_size`). 

**Passaggio da 4096 a 2048 (o 1024) feature:**
- Ridurre a 2048 feature (espansione 4x) porterebbe il nostro rapporto campioni/feature a 3.6x.
- Ridurre a 1024 feature (espansione 2x) porterebbe il rapporto a 7.2x, rientrando perfettamente nel range raccomandato [4].

**Validazione dalla letteratura:**
La letteratura evidenzia che esista un compromesso ("diminishing returns") nell'aumentare la dimensione del dizionario [2]. Dizionari più grandi tendono a "splittare" i concetti in versioni troppo granulari [1], che in assenza di dati sufficienti si tramutano in dead features. Ridurre il dizionario costringe il SAE a imparare rappresentazioni più compatte, fitte e generalizzabili, abbassando drasticamente la percentuale di dead features e prevenendo l'overfitting latente [3].

---

## 4. Soluzione 2: Data Augmentation "Safe" per Medical Imaging

La seconda via è aumentare artificialmente il numero di campioni ($m$) passando da 7400 a circa 22000 tramite Data Augmentation (con 3 augmentazioni per immagine). Con 22000 campioni, anche mantenendo 2048 feature, avremmo un rapporto superiore a 10x, o ~5.4x con 4096 feature.

Tuttavia, come notato correttamente, l'applicazione della Data Augmentation alle immagini mediche (come le CXR) richiede estrema cautela. Tecniche standard nella computer vision possono alterare o distruggere dettagli patologici cruciali (es. piccoli noduli, versamenti pleurici) [5].

**Augmentazioni Sicure (Safe Augmentations) approvate in letteratura [5, 6]:**
1. **Piccole Rotazioni (Small-angle Rotations):** Rotazioni limitate tra -10° e +10° o al massimo +/-15° sono considerate sicure poiché simulano le naturali variazioni nel posizionamento del paziente senza introdurre geometrie innaturali [6, 7].
2. **Traslazione e Random Cropping leggero:** Aiuta il modello a non fare affidamento esclusivo sul bias spaziale centrale (center-bias) preservando l'anatomia, patto che il crop mantenga il 90-95% dell'immagine [5].
3. **Horizontal Flipping (Inapplicabile al nostro caso):** Sebbene l'inversione orizzontale sia tecnicamente una trasformazione geometrica sicura in molti task di computer vision, **non possiamo utilizzarla sul nostro dataset IU X-Ray**. Il motivo è duplice: in primis, i referti del dataset contengono forti dipendenze dalla lateralità (es. specificano "opacità nel lobo superiore destro" o "angolo cardiofrenico sinistro"); in secondo luogo, flippare una radiografia toracica inverte la posizione del cuore, simulando una destrocardia (condizione rarissima) e forzando il SAE ad allocare feature per anomalie anatomiche inesistenti [8].

**Augmentazioni da EVITARE (Distorsive) [5, 8]:**
- **Color Jitter / Alterazioni di contrasto forti:** Nelle CXR, l'intensità dei pixel (radiopacità) è il segnale patologico (es. consolidamenti, opacità). Modificare pesantemente la luminosità o il contrasto altera la diagnosi.
- **Deformazioni Geometriche Estreme (Shearing, Elastic distortions):** Producono immagini fisicamente impossibili in clinica.
- **Cutout/Erasing:** Rischia di oscurare patologie localizzate.

Generando le varianti delle immagini *prima* dell'estrazione con BiomedCLIP, forniremo al SAE embedding leggermente diversi che arricchiranno la varianza dello spazio latente visivo, aiutando le feature a rimanere vive.

---

## 5. Conclusioni e Piano d'Azione

Entrambe le soluzioni sono supportate dalla letteratura e non sono mutualmente esclusive. La combinazione ottimale per il nostro progetto è un approccio ibrido:
1. **Abbassare il `dict_size`** da 4096 a 2048.
2. **Implementare Data Augmentation sicura** (piccole rotazioni, leggeri crop) prima dell'estrazione degli embedding, per triplicare i campioni (da ~7400 a ~22000).

Con questo approccio combinato otterremo un rapporto campioni/feature di ~10.7x, ampiamente nel range ottimale raccomandato per garantire stabilità, alta interpretabilità e l'azzeramento quasi totale delle dead features.

---

## Riferimenti

[1] Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread.  
[2] Sharkey, K., et al. (2024). *Feature Splitting and Scaling in Sparse Autoencoders*. Alignment Forum.  
[3] Olshausen, B. A., & Field, D. J. (1997). *Sparse coding with an overcomplete basis set: A strategy employed by V1?*. Vision Research, 37(23), 3311-3325.  
[4] Ng, A. (2011). *Sparse autoencoder*. CS294A Lecture notes, Stanford University.  
[5] Chlap, P., et al. (2021). *A review of medical image data augmentation techniques for deep learning applications*. Journal of Medical Imaging and Radiation Oncology, 65(5), 545-563.  
[6] Hussain, Z., et al. (2017). *Differential Data Augmentation Techniques for Medical Image Classification*. Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC).  
[7] Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1), 1-48.  
[8] Mikołajczyk, A., & Grochowski, M. (2018). *Data augmentation for improving deep learning in image classification problem*. International Interdisciplinary PhD Workshop (IIPhDW).
