# Design — Parte 2.B (NLU): fine-tuning di GPT2 e BERT pre-addestrati (multi-task)

Data: 2026-06-24
Branch: `feat/part2-nlu` (stesso della 2.A — la 2.A+2.B formano l'unico report NLU)
Ciclo: 2 di 2 della Parte 2.

## Obiettivo

Fine-tuning di modelli **pre-addestrati** su ATIS in **multi-task** (intent + slot),
gestendo la **sub-tokenizzazione**:
- **GPT2** (decoder-only) e **BERT** (encoder-only), confrontati;
- il tokenizer del modello (BPE/WordPiece) spezza le parole in sub-token → bisogna
  **allineare** le label di slot ai sub-token (rif. Chen et al. 2019, arXiv 1902.10909);
- l'intent si legge diversamente: **BERT da `[CLS]`** (token 0, bidirezionale), **GPT2
  dall'ultimo token reale** (causale).
- Metriche: intent **accuracy**, slot **F1 (conll)**.

## Approccio

Pipeline parallela **`nlu_pipeline_b/`**, auto-contenuta: **duplica** il codice comune
(config, tracking, conll, i loader ATIS `load_atis`/`stratified_dev_split` di
`nlu_pipeline.data`). Nuovo: data layer sub-tokenizzato, modello joint che avvolge un
backbone HuggingFace, training di fine-tuning. Non si tocca 2.A (`nlu_pipeline/`).

## Struttura

```
nlu_pipeline_b/
  __init__.py
  data.py          [nuovo]   ATIS + tokenizer HF + allineamento label ai sub-token + collate
  model.py         [nuovo]   JointNLUTransformer (backbone HF + teste slot/intent)
  train.py         [nuovo]   fine-tuning multi-task + eval (conll F1 + accuracy) + fit
  conll.py         [copia]   dal lab (riuso)
  config.py        [dup]     da nlu_pipeline
  tracking.py      [dup]     da nlu_pipeline (save_curves su slot_f1/intent_acc)
  experiment.py    [nuovo]   una run completa; auto-tag del modello nel nome run
  aggregate.py     [dup]     da nlu_pipeline (colonne slot_f1/intent_acc)
run_nlu_b.py             [nuovo]  CLI (artefatti in runs_nlu_b/)
nlu_pipeline_b_export.py [nuovo]  genera NLU/part_B
configs/nlu_b/base_partB_nlu.yaml + experiments/{gpt2, bert, _smoke}.yaml
NLU/part_B/   model.py functions.py utils.py main.py conll.py README.md bin/ dataset/
tests/test_nlu_b_data.py  test_nlu_b_model.py  test_nlu_b_train.py
docs/2026-06-24-part2b-nlu.md
runs_nlu_b/  (gitignored)
```

## Modello (`model.py`) — nuovo

`JointNLUTransformer`:
- `backbone = AutoModel.from_pretrained(model_name)` → hidden states `(B, L, H)`.
- `slot_out = Linear(H, num_slots)` su ogni sub-token.
- `intent_out = Linear(H, num_intents)` sul token di frase:
  - `intent_pool="cls_first"` (BERT): hidden della posizione 0 (`[CLS]`);
  - `intent_pool="last"` (GPT2): hidden dell'ultimo token reale (da attention_mask).
- `intent_pool` derivato dal tipo di modello (config), così un solo modello copre GPT2
  e BERT.
- forward(input_ids, attention_mask) → (slot_logits `(B,L,num_slots)`, intent_logits
  `(B,num_intents)`).

> Nota GPT2: il tokenizer GPT2 non ha pad nativo → `pad_token = eos_token`; l'intent si
> legge dall'ultimo token non-pad (via attention_mask), non da un CLS.

## Data layer (`data.py`) — nuovo (cuore di 2.B)

- Riusa `load_atis`/`stratified_dev_split` (importati o duplicati) per gli esempi grezzi
  (parole + slot per parola + intent).
- `slot2id`/`intent2id` costruiti dal corpus (come 2.A); niente word2id (la
  tokenizzazione la fa HF).
- **Sub-tokenizzazione + allineamento** (per esempio): si tokenizza la frase
  `is_split_into_words=True`; tramite `word_ids()` si mappa ogni sub-token alla parola.
  La label di slot della parola va al **primo** sub-token; gli **altri sub-token** e i
  **token speciali** (`[CLS]/[SEP]/pad`) → **-100** (ignorati dalla CrossEntropy).
- `collate`: padding di `input_ids`, `attention_mask`, `slot_labels` (con -100), `intent`;
  si conserva `word_ids` per ricostruire gli slot a livello di parola in eval.
- `build_dataloaders(cfg_data, dataset_dir, tokenizer)` → `(train, dev, test, label_maps)`.

## Training/eval (`train.py`) — nuovo

- **Fine-tuning completo** (tutti i parametri trainabili), **AdamW** lr piccolo
  (~5e-5/3e-5/2e-5), poche epoche (3–5), gradient clipping, early stopping su **dev slot
  F1**. Riusa `pick_device`.
- **loss** = `CE(slot, ignore_index=-100) + CE(intent)`.
- **eval**: per ogni frase si prendono le predizioni di slot ai **primi sub-token** di
  ogni parola → sequenza a livello di parola → `conll.evaluate` (slot F1); intent =
  accuracy (argmax). Entrambe in [0,1], mostrate come %.

## Orchestrazione, CLI, scelta del modello

- **`experiment.py`**: costruisce tokenizer+modello da `cfg.model.name`, fa fit, in
  `mode final` valuta su test. **Auto-tag del modello nel nome run**: la run dir si
  chiama `<experiment.name>__<tag>` con `tag = model.name.split('/')[-1]` (es.
  `partB_nlu__gpt2`, `partB_nlu__bert-base-uncased`, `partB_nlu__bert-large-uncased`).
  Così cambiando `model.name` nel config la run è **automaticamente distinta** dalla base
  (no collisioni / no resumability skip). Salva metrics, curve, `best_model.pt`,
  `label_maps.json`, e il `model.name`/`intent_pool` (servono all'inference).
- **`run_nlu_b.py`**: CLI `run|sweep|aggregate|finalize|export` su `runs_nlu_b/`.
- **`aggregate.py`**: duplicato (colonne slot_f1/intent_acc), report `partB_nlu_summary.*`.

## Config (`configs/nlu_b/`)

`base_partB_nlu.yaml` (`model.name`, `intent_pool` o derivato, lr 3e-5, epochs 5,
patience 3, batch_size, dev_portion 0.10) + esperimenti:
`gpt2.yaml` (model openai-community/gpt2, sweep lr), `bert.yaml`
(google-bert/bert-base-uncased, sweep lr), `_smoke.yaml`. (gpt2-medium / bert-large:
basta cambiare `model.name`, la run si auto-tagga.)

## Consegna (`NLU/part_B/`)

Come 2.A: `model.py`/`functions.py`(train con `pick_device` inline)/`utils.py`(data)/
`conll.py`, `main.py` generato (ricostruisce tokenizer+modello dal `model.name` salvato,
carica `bin/best_model.pt` + `label_maps.json`, stampa test slot F1 + intent acc),
`README.md`. **Pesi grandi** (~500MB, full fine-tuning): `*.pt` gitignored → vanno nello
zip di consegna, non in git.

## Test

- `test_nlu_b_data.py`: l'allineamento mette la label sul **primo** sub-token e -100 sul
  resto / sui token speciali; collate paddinga coerentemente.
- `test_nlu_b_model.py`: forward shapes per **entrambi** i backbone, costruiti da config
  minima (BertConfig/GPT2Config piccoli, no download); intent dal token giusto
  (CLS vs last).
- `test_nlu_b_train.py`: smoke CPU (modello minimo, loader finto) — un passo gira, loss
  finita, eval ritorna F1/acc in [0,1].

## Documentazione

`docs/2026-06-24-part2b-nlu.md`: cosa cambia rispetto a 2.A (pre-addestrato vs da zero,
encoder vs decoder, sub-tokenizzazione), glossario, file-per-file, tabella riuso/nuovo,
workflow CLI.

## Non-obiettivi (YAGNI)

- LoRA / adapter (qui è **full fine-tuning**, come chiede il lab; LoRA era 1.B).
- gpt2-medium / bert-large come default (opzionali: basta `model.name`).
- Training reale (serve la VM): qui codice + struttura + smoke CPU.

## Dipendenze

`transformers==4.38.0` (GPT2/BERT), `scikit-learn` (split, gia' in requirements). BERT
si scarica al primo uso sulla VM.
