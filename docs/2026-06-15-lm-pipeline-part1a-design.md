# Design — Pipeline LM modulare (Parte 1.A)

Data: 2026-06-15
Stato: design approvato, in attesa di review prima del piano di implementazione
Scope di questo spec: infrastruttura della pipeline + Parte 1.A (GPT2 from scratch).
Fuori scope (brainstorming successivo): Parte 1.B (LoRA manuale) — la pipeline è
progettata per accoglierla senza modifiche strutturali.

---

## 1. Contesto e obiettivo

Progetto del corso NLU (UniTN). La Parte 1 (LM) va consegnata con una struttura di
file rigida (`main.py`, `functions.py`, `utils.py`, `model.py` per `part_A` e
`part_B`), niente notebook, codice che gira standalone senza bug.

La Parte 1.A richiede di partire dal GPT2 baseline e aggiungere migliorie **una alla
volta**, riportando la PPL di ogni esperimento, con obiettivo **PPL < 250** sul test
set di Penn Treebank. Le migliorie da provare incrementalmente:

0. baseline + ricerca del learning rate (altri iperparametri fissi);
1. hyperparameter tuning incrementale (`d_model`, `n_heads`, `num_layers`, `ff_dim`);
2. dropout in 4 punti (post-embedding, sui pesi di attention, dopo la output
   projection della MultiHeadAttention, dopo l'ultimo linear del Feed Forward);
3. weight tying (condivisione pesi `token_embed` ↔ `lm_head`).

Serve una pipeline **modulare e personalizzabile** che permetta di: lanciare molti
test con valori diversi degli iperparametri; scegliere una **frazione/dimensione del
dataset** per iterare in fretta; selezionare la configurazione migliore ed eseguire
una run finale sull'intero dataset per le metriche da riportare; **documentare ogni
run** in forma direttamente riutilizzabile nel report LaTeX (template IEEE).

## 2. Obiettivi e non-obiettivi

Obiettivi:
- una sola codebase ("verità") riusata sia in sviluppo sia in consegna;
- esperimenti definiti da config YAML, con sweep opzionale (prodotto cartesiano);
- subset del dataset come parametro di primo livello;
- logging locale che alimenta tabelle LaTeX + grafici PDF (+ TensorBoard opzionale);
- export ripetibile verso la struttura di consegna, senza riscrittura a mano;
- device-agnostico (Mac MPS ↔ Colab/CUDA), riproducibile (seed, git hash).

Non-obiettivi (per ora):
- Parte 1.B / LoRA (estensione successiva);
- Parte 2 (NLU/ATIS);
- dashboard cloud (W&B) o tracking online;
- ottimizzazione iperparametri automatica (Optuna/bayesian) — sweep a griglia basta.

## 3. Struttura delle directory

Tutto sotto `NLU_project_development/`.

```
NLU_project_development/
  lm_pipeline/              # package core (riusato anche nella consegna)
    __init__.py
    config.py               # [orchestrazione] load YAML, base+override, espansione sweep
    data.py                 # [puro] PTB: vocab, dataloader, subset, padding
    model.py                # [puro] GPT2 baseline parametrico
    train.py                # [puro] loop train/eval, PPL, early stopping, device
    experiment.py           # [orchestrazione] Runner: 1 config -> 1 run dir
    tracking.py             # [orchestrazione] run dir, seed, git hash, timing
    aggregate.py            # [orchestrazione] runs/ -> tabella + grafici + LaTeX/MD
  configs/
    base_partA.yaml
    experiments/            # config singole o sweep
  runs/                     # output: 1 cartella per run
  reports/                  # tabelle .tex/.md + figures/
  docs/                     # questo spec e i successivi
  run.py                    # CLI unico: run | sweep | aggregate | finalize | export
```

### 3.1 Frontiera di import (chiave per l'export)

Lo strato **puro** (`data.py`, `model.py`, `train.py`) NON importa mai lo strato di
**orchestrazione** (`config.py`, `experiment.py`, `tracking.py`, `aggregate.py`).
L'orchestrazione sta sopra e importa il puro, mai il contrario. Questo rende i moduli
puri già auto-contenuti e quindi copiabili tal quali nella consegna.

## 4. Moduli: responsabilità e interfacce

- `data.py` (puro): caricamento Penn Treebank, costruzione vocabolario, tokenizzazione,
  `DataLoader` per train/valid/test, applicazione del subset. Funzione chiave tipo
  `build_dataloaders(cfg_data) -> (train_dl, valid_dl, test_dl, vocab)`.
- `model.py` (puro): classe `GPT2` parametrica su `d_model`, `n_heads`, `num_layers`,
  `ff_dim`, `dropout` (con i 4 punti di dropout opzionali) e flag `weight_tying`.
  Multi-head masked self-attention, feed-forward con GELU, layer norm pre-blocco.
- `train.py` (puro): `train_one_epoch`, `evaluate` (ritorna loss e PPL), `fit` con
  early stopping su PPL di validazione, gradient clipping, scelta device
  (`cuda`/`mps`/`cpu` auto). Nessuna dipendenza da config/tracking: riceve oggetti già
  costruiti e iperparametri come argomenti.
- `config.py` (orchestrazione): carica YAML, applica `overrides`, espande la sezione
  `sweep` nel prodotto cartesiano, deriva il nome della run, valida i campi.
- `tracking.py` (orchestrazione): crea la run dir, fissa il seed globale e i flag
  deterministici, registra git hash, tempo wall-clock, device, scrive `metrics.json`
  e `curves.csv`.
- `experiment.py` (orchestrazione): orchestra una run end-to-end (config -> data ->
  model -> fit -> salvataggio artefatti). Resumable: salta le run già completate.
- `aggregate.py` (orchestrazione): scansiona `runs/`, costruisce la tabella
  riassuntiva ordinata per PPL, emette LaTeX + Markdown + figure matplotlib.

## 5. Schema della config (YAML)

```yaml
experiment:
  name: partA_baseline        # usato per nominare la run dir
  seed: 42
  device: auto                # auto | cuda | mps | cpu
  tensorboard: false
data:
  fraction: 1.0               # 0<f<=1, alternativa a max_samples
  max_samples: null           # int oppure null
  subset_seed: 42
  batch_size: 32
  seq_len: 128
model:
  d_model: 256
  n_heads: 4
  num_layers: 4
  ff_dim: 1024
  weight_tying: false
  dropout:
    enabled: false
    p: 0.1                     # applicato ai 4 punti quando enabled
optim:
  lr: 1.0e-3
  optimizer: adamw            # configurabile (adamw default; sgd/altri possibili)
  epochs: 30
  grad_clip: 1.0
  patience: 5                 # early stopping su valid PPL
  scheduler:
    enabled: true             # nel baseline: warmup + cosine decay
    type: warmup_cosine
    warmup_steps: 200
mode: dev                     # dev -> valuta su valid | final -> full + test

# opzionale: genera N run come prodotto cartesiano
sweep:
  optim.lr: [1.0e-3, 5.0e-4, 1.0e-4]
```

Convenzioni: `base_partA.yaml` contiene i default; una config di esperimento può
ridefinire solo i campi che cambiano. Le chiavi dotted nello `sweep` puntano a campi
annidati. Ogni combinazione di sweep produce una run con nome derivato (es.
`partA_baseline__lr5e-4`).

## 6. Subset del dataset

`data.fraction` (frazione) oppure `data.max_samples` (numero assoluto), applicati con
`subset_seed` fisso per riproducibilità. Il subset opera **per frasi (righe)** di PTB,
non per token: PTB ha una frase per riga e nel LM la predizione avviene dentro la
sequenza, quindi mantenere le frasi intere evita tagli arbitrari ed è più pulito da
spiegare nel report. La frazione effettiva e il numero di frasi usate vengono
registrati in `metrics.json` ("tuning su X% di PTB"). In `mode: final` il subset viene
ignorato (dataset intero).

## 7. Ciclo di vita di una run (CLI `run.py`)

1. `run.py run --config configs/experiments/X.yaml`
   risolve la config, fissa il seed, costruisce data/model/optim, allena con early
   stopping su valid-PPL, salva la run dir.
2. `run.py sweep --config configs/experiments/X.yaml`
   espande lo `sweep` ed esegue tutte le combinazioni in sequenza; resumable.
3. `run.py aggregate`
   legge tutte le run dir, costruisce la tabella ordinata per PPL, emette i report.
4. `run.py finalize [--run NOME]`
   default: seleziona automaticamente la run con valid-PPL minima (stampa quale e
   perché); `--run` la forza manualmente. Ri-allena quella config in `mode: final`
   (dataset intero), valuta su test, salva il `.pt` e stampa la PPL finale richiesta.
5. `run.py export`
   genera la struttura di consegna (vedi §9).

## 8. Tracking e output per il report

Ogni run dir contiene:
- `config.json` — snapshot completo della config risolta;
- `metrics.json` — best PPL, PPL per epoca, loss, n° parametri, tempo, device, seed,
  git hash, frazione/dimensione dataset;
- `curves.csv` — serie per epoca (train loss, valid loss, valid PPL);
- `curves.pdf` — curva PPL della run;
- `best_model.pt` — pesi del miglior checkpoint;
- `log.txt` — log testuale della run.

`aggregate` produce automaticamente in `reports/`:
- `partA_summary.tex` — tabella booktabs (`esperimento | modifica | lr | best PPL |
  #params | tempo`), mappata 1:1 sul requisito "una modifica alla volta";
- `figures/*.pdf` — curve per run + una figura comparativa, pronte per il template;
- `partA_summary.md` — stessa tabella per consultazione rapida.

TensorBoard è opzionale (`experiment.tensorboard: true`) per ispezione live delle
curve; non sostituisce gli output sopra, che restano la fonte per il report.

## 9. Export verso la struttura di consegna

`run.py export` (o per `part_A`) produce file standalone:

| File di consegna     | Origine                                      |
|----------------------|----------------------------------------------|
| `part_A/model.py`    | copia di `lm_pipeline/model.py`              |
| `part_A/utils.py`    | copia di `lm_pipeline/data.py`               |
| `part_A/functions.py`| copia di `lm_pipeline/train.py` (+ helper)   |
| `part_A/main.py`     | generato: iperparametri della config migliore congelati inline, carica il `.pt` da `bin/`, stampa la PPL |
| `part_A/bin/best_model.pt` | copia del modello finale               |
| `part_A/dataset/`    | copia di PTB                                 |

Lo strato di orchestrazione non viene mai copiato (non serve alla consegna). Poiché lo
strato puro non lo importa, i file copiati girano standalone, soddisfacendo il vincolo
"deve girare senza bug, niente notebook". L'export è ripetibile: niente divergenze tra
codice sperimentale e codice consegnato.

## 10. Riproducibilità e rigore

Seed globale (Python/NumPy/PyTorch), flag deterministici dove possibile, git hash e
config archiviati a ogni run, device registrato. Le run sono idempotenti rispetto alla
loro dir: rilanciare salta quelle già completate.

## 11. Mappatura sugli esperimenti di 1.A

L'approccio incrementale si traduce in una sequenza di config in
`configs/experiments/`, ciascuna che attiva una modifica in più rispetto alla migliore
precedente:

1. `00_baseline` — sweep su `lr`, resto fisso (scheduler warmup+cosine attivo).
2. `01_arch` — partendo dal miglior lr, sweep/incrementi su `d_model`, poi `n_heads`,
   poi `num_layers`, poi `ff_dim` (una alla volta).
3. `02_dropout` — aggiunge i 4 dropout sopra la migliore architettura.
4. `03_weight_tying` — aggiunge il weight tying.
5. `04_no_scheduler` (ablazione) — disattiva lo scheduler sulla migliore config per
   documentarne il contributo nel report.

Le modifiche che peggiorano la PPL si tengono comunque tracciate (richiesta del
report: commentare anche gli esperimenti non riusciti). La tabella di `aggregate`
documenta l'intera progressione.

## 12. Decisioni prese e punti ancora aperti

Decise:
- ottimizzatore di default AdamW, configurabile da YAML;
- scheduler warmup + cosine decay attivo nel baseline, con una run di ablazione
  (scheduler off) per documentarne il contributo;
- subset per frasi/righe di PTB (non per token).

Ancora aperto (da rifinire in fase di piano):
- formato esatto della tabella LaTeX (colonne) — da rifinire guardando il template;
- valori di default di `warmup_steps` ed epoche, da tarare sui primi run.

## 13. Nota: gradient clipping (cos'è e perché conta come regolarizzatore)

`optim.grad_clip` (default **1.0**) limita la dimensione del passo di aggiornamento.

**Cos'è.** Durante il training: forward → loss → `backward` (calcola il gradiente, cioè
di quanto/in che direzione muovere ogni peso) → l'optimizer aggiorna i pesi di un passo
proporzionale al gradiente. A volte il gradiente diventa **enorme** (gli "exploding
gradients", frequenti nei Transformer e soprattutto a inizio training con pesi casuali):
un gradiente enorme → aggiornamento enorme → il modello salta in una zona pessima e la
loss diverge (NaN). Il clipping mette un tetto.

**Come (clip-by-norm globale, in `train.py`).** Si concatenano i gradienti di tutti i
pesi in un unico vettore `g`; se la sua lunghezza `‖g‖ = √(Σ gᵢ²)` supera la soglia `c`,
si riscala l'intero vettore: `g ← g·(c/‖g‖)` (così `‖g‖ = c`). Si taglia la **lunghezza**,
non la **direzione**: vai comunque dove indica il gradiente, solo senza fare un passo
troppo lungo (come un limitatore di velocità: lo sterzo resta, il top speed è capato).
Nel codice: `nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` tra `backward()` e
`optimizer.step()`. Nato per la **stabilità**; va in coppia col **warmup** dello scheduler
(entrambi domano l'instabilità iniziale).

**Finding rilevante per il report.** La soglia agisce anche come **regolarizzatore**.
Una soglia stretta (1.0) impone passi piccoli → il modello si adatta in modo più
conservativo → overfitta meno → il dropout utile è **basso** (nei nostri sweep p=0.1;
oltre, peggiora). Una soglia più larga (es. 5) lascia fittare più aggressivamente →
overfitta di più → un dropout più alto (es. 0.3) può aiutare. Quindi il dropout ottimale
**non è universale**: dipende dall'interazione clip ↔ dropout ↔ overfitting. È questo (non
il caso) a spiegare perché due implementazioni dello stesso esercizio trovano punti di
dropout diversi — da dichiarare nel report quando si motiva la scelta del dropout.
