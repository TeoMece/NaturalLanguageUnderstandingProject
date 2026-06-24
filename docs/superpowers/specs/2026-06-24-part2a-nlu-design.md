# Design — Parte 2.A (NLU: intent classification + slot filling) + scaffold pipeline NLU

Data: 2026-06-24
Branch: `feat/part2-nlu` (ramificato da `feat/part1b-lora`)
Ciclo: **1 di 2**. Questo spec copre lo *scaffold* della pipeline NLU + la **Parte 2.A**.
La **Parte 2.B** (fine-tuning GPT2+BERT pre-addestrati, multi-task, sub-tokenizzazione)
avrà spec/plan separati in un secondo ciclo.

## Obiettivo

Costruire da zero una pipeline per il task NLU del lab 5 su **ATIS**: **intent
classification** (una label per frase) + **slot filling** (una label IOB per token),
in **multi-task**. La Parte 2.A parte dal **GPT2 della Parte 1** (decoder-only,
causale) e lo migliora **incrementalmente** (lr → architettura → dropout). Metriche:
**intent = accuracy**, **slot = F1 con lo script conll** del corso.

## Vincoli e approccio

- **Pipeline parallela e auto-contenuta**: mirror di `lm_pipeline`, ma **duplicando**
  il codice comune (config, tracking, backbone GPT2) invece di importarlo — così
  `nlu_pipeline` gira da sola e si esporta pulita nella consegna (come per la Parte 1).
- Non si tocca nulla della Parte 1 (LM): cartelle e branch separati.
- Chiarezza prima di tutto: ogni file commentato in modo esaustivo + una guida MD.

## Struttura

```
nlu_pipeline/
  __init__.py
  data.py          [nuovo]   ATIS + Lang + Dataset + collate + dataloaders
  model.py         [dup+mod] backbone GPT2 (Parte 1) + teste slot_out / intent_out
  train.py         [nuovo]   loop multi-task + eval (conll F1 + accuracy) + fit; pick_device (dup)
  conll.py         [riuso]   copiato verbatim dal lab (scorer F1 a chunk)
  config.py        [dup]     load YAML + sweep (identico a lm_pipeline)
  tracking.py      [dup]     seed, run dir, metrics.json, curve
  experiment.py    [nuovo]   una run NLU completa
  aggregate.py     [dup+mod] tabelle con intent_acc + slot_f1
run_nlu.py             [nuovo]  CLI (artefatti in runs_nlu/)
nlu_pipeline_export.py [nuovo]  genera NLU/part_A standalone
configs/nlu/base_partA_nlu.yaml + configs/nlu/experiments/{00_baseline,01_arch,02_dropout,_smoke}.yaml
NLU/part_A/   model.py functions.py utils.py main.py README.md bin/ dataset/
tests/test_nlu_data.py  tests/test_nlu_model.py  tests/test_nlu_train.py
docs/2026-06-24-part2a-nlu.md   guida dettagliata
runs_nlu/   (gitignored)
```

## Data layer (`nlu_pipeline/data.py`) — nuovo

Sorgente: `dataset/ATIS/{train.json,test.json}` (oggetti `{utterance, slots, intent}`).

- **`load_atis(dataset_dir)`**: legge train/test json.
- **dev split stratificato**: dal train, 10% stratificato sull'intent (sklearn
  `train_test_split`, `random_state` fisso). Gli intent con una sola occorrenza
  restano nel train (come nel lab) per non rompere la stratificazione.
- **`Lang`**: costruisce `word2id` (con `pad`, `unk`, `cls`), `slot2id` (con `pad` e
  `cls`→PAD, ignorato negli slot), `intent2id`, e gli inversi `id2*`. Le parole solo
  dal train; le label di slot/intent da tutto il corpus (niente label `unk`).
- **`IntentsAndSlots(Dataset)`**: mappa utterance/slots/intent in id; **appende un
  token CLS in coda** a ogni frase (la testa intent legge l'ultimo token, che con la
  maschera causale è l'unico a "vedere" tutta la frase).
- **`collate_fn`**: right-padding di utterance e slot a `max_len` (PAD=0), `intent`
  come vettore, ritorna `{utterances, intents, y_slots, slots_len}`. Tensori su CPU;
  spostamento su device nel train loop.
- **`build_dataloaders(cfg_data, dataset_dir)`**: ritorna `(train_dl, dev_dl, test_dl,
  lang)`; shuffle solo sul train.

## Modello (`nlu_pipeline/model.py`) — duplicato + modificato

Riusa i mattoni della Parte 1 (`MultiHeadAttention`, `FeedForward`, `TransformerBlock`,
embedding token+posizionali, maschera causale, `ln_f`, `init_weights`) **copiati** da
`lm_pipeline/model.py`. Differenza rispetto al LM: **niente `lm_head`**, ma due teste:

- `slot_out = nn.Linear(d_model, num_slots)` — applicata a **ogni** posizione (slot per
  token).
- `intent_out = nn.Linear(d_model, num_intents)` — applicata all'hidden dell'ultimo
  token (CLS) di ogni frase.
- `dropout` opzionale **prima delle teste** (è lo step 2 di 2.A).

`forward(idx, seq_lens) -> (slot_logits [B,L,num_slots], intent_logits [B,num_intents])`.

## Training/eval (`nlu_pipeline/train.py`) — nuovo

- **loss multi-task** = `CrossEntropy(intent) + CrossEntropy(slot)` (pesi uguali, come
  il lab; lo slot usa `ignore_index=PAD` così pad e CLS non contano).
- **`evaluate`**: produce le coppie (parola, slot) per ref/hyp e chiama
  `conll.evaluate(ref, hyp)` → **slot F1**; e l'**intent accuracy** (confronto argmax).
- **`fit`**: early stopping sulla **slot F1 di dev** (massimizzazione), con `patience`;
  ripristina i pesi migliori. Riusa `pick_device`, `_build_optimizer` (duplicati).
- Storico salvato: `train_loss`, `dev_slot_f1`, `dev_intent_acc` per epoca.

## Orchestrazione / config / tracking

- **`config.py`, `tracking.py`**: duplicati da `lm_pipeline` (config identico; tracking
  con `save_curves` adattato a salvare slot_f1/intent_acc per epoca).
- **`experiment.py`**: `run_experiment_nlu(cfg, runs_root, dataset_dir)`: seed/device →
  dataloaders+lang → modello → `init_weights` → `fit` → in `mode: final` valuta su test
  → salva curve, `best_model.pt`, `lang` (vocabolari) e `metrics.json`
  (`best_dev_slot_f1`, `dev_intent_acc`, `test_slot_f1`, `test_intent_acc`, n_params,
  config, ...).
- **`aggregate.py`**: duplicato, con colonne `slot_f1` e `intent_acc`; scrive
  `partA_nlu_summary.{tex,md}` + figura comparativa.
- **`run_nlu.py`**: CLI `run|sweep|aggregate|finalize|export` su `runs_nlu/`.

## Config (`configs/nlu/`)

`base_partA_nlu.yaml` (d_model/n_heads/num_layers/ff_dim ridotti, lr, epoche, patience,
batch_size, dropout off, `mode: dev`) + esperimenti incrementali:
`00_baseline` (sweep lr), `01_arch` (sweep d_model×num_layers, n_heads/ff_dim derivati),
`02_dropout` (sweep p prima delle teste), `_smoke` (pochi esempi/1 epoca).

## Consegna (`NLU/part_A/`) — `nlu_pipeline_export.py`

Mappa come la Parte 1: `model.py` (modello joint), `utils.py` (= data.py), `functions.py`
(= train.py con `pick_device` inline), `main.py` generato (ricostruisce modello +
`lang` salvato, carica `bin/best_model.pt`, stampa intent acc + slot F1 su test),
`README.md`, copia `dataset/ATIS` e `conll.py`.

## Test

- `test_nlu_data.py`: Lang/vocab coerenti; collate paddinga e allinea utterance/slot;
  dev split stratificato riproducibile.
- `test_nlu_model.py`: forward ritorna shape `(B,L,num_slots)` e `(B,num_intents)`.
- `test_nlu_train.py`: **smoke su CPU** (vocab/modello minimi, loader finto) — un passo
  gira, loss finita; eval ritorna F1/acc in [0,1].

## Documentazione

`docs/2026-06-24-part2a-nlu.md` (stile GUIDA/part1b): task, glossario (intent vs slot,
IOB, CLS, conll F1, multi-task), spiegazione file-per-file, scelte, e tabella
riuso/dal-lab/nuovo.

## Non-obiettivi (YAGNI / per il Ciclo 2)

- Parte 2.B (GPT2/BERT pre-addestrati, sub-tokenizzazione, multi-task fine-tuning).
- Pesatura della loss multi-task (baseline a pesi uguali; eventuale esperimento dopo).
- Training reale (serve la VM): qui codice + struttura + smoke CPU.

## Dipendenze

`scikit-learn` (stratified split + `classification_report`), `conll.py` (dal lab,
copiato). Tutto già presente nell'env `nlu26`.
