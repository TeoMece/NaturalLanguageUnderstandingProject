# Design — Parte 1.B (LoRA) e separazione 1.A/1.B

Data: 2026-06-22
Branch: `feat/part1b-lora`

## Obiettivo

Il progetto LM ha la **Parte 1.A** completa (GPT2 da zero su Penn Treebank, PPL < 250,
con ricerca lr / sweep architettura / dropout / weight tying). Manca la **Parte 1.B**
descritta nel lab 4 del corso (`NLU-2026-Labs/labs/04_LM_with_transformers.ipynb`):

> Partire da GPT2 **pre-addestrato** (HuggingFace) e fare fine-tuning con **LoRA
> implementato a mano** (no PEFT), applicato alle matrici **Q, K, V**. Allenare
> **solo** gli adapter. Sweep di `rank` e `alpha`. Requisito: **PPL < 250** e
> **inferiore** a quella di 1.A.

Due richieste dell'utente:
1. **Riorganizzare le cartelle** per separare nettamente 1.A da 1.B.
2. **Implementare 1.B** riusando tutto il riusabile, reimplementando solo il
   mancante, seguendo il lab 4 e usando il loro codice dove serve; commentare in
   modo esaustivo le aggiunte. Infine, una doc dettagliata.

## Vincolo trasversale

La Parte 1.A è coperta da 32 test verdi e **non va toccata**. Approccio scelto:
**parallelo a basso rischio** — `lm_pipeline/` (1.A) resta invariato; 1.B vive in
moduli/cartelle nuovi e *riusa* 1.A per import.

## Struttura finale delle cartelle

```
lm_pipeline/              # 1.A — INVARIATO
  data.py model.py train.py            [puro]
  config.py tracking.py experiment.py aggregate.py   [orchestrazione]
lm_pipeline_b/            # NUOVO
  __init__.py
  model_lora.py           # LoRALinear, CustomGPT2Attention, GPT2_LoRA, freeze, param_stats
  train_lora.py           # train/eval (loss interna HF) + fit (early stopping)
  experiment_lora.py      # una run LoRA completa (riusa data/tracking/config)
  report_b.py             # report B (riusa collect_runs/to_latex di aggregate.py)
run.py                    # 1.A — INVARIATO
run_b.py                  # NUOVO — CLI di B (run|sweep|aggregate|finalize|export)
lm_pipeline_export.py     # 1.A — INVARIATO
lm_pipeline_b_export.py   # NUOVO — genera LM/part_B standalone
configs/
  base_partA.yaml experiments/         # 1.A — INVARIATO
  base_partB.yaml                       # NUOVO
  experiments_b/00_baseline.yaml 01_rank_sweep.yaml 02_alpha_sweep.yaml _smoke.yaml  # NUOVO
LM/
  part_A/                 # consegna 1.A — INVARIATO
  part_B/                 # NUOVO standalone: model.py functions.py utils.py main.py bin/ README.md
tests/
  ...(esistenti)          # INVARIATO
  test_model_lora.py      # NUOVO
  test_train_lora.py      # NUOVO (incl. smoke su CPU)
runs_b/                   # NUOVO — artefatti run di B (separato da runs/)
docs/
  2026-06-22-part1b-lora.md   # NUOVO — doc dettagliata in italiano (stile GUIDA.md)
```

Decisioni di default confermate dall'utente:
- `run_b.py` separato (non subcommand di `run.py`) → non tocca `run.py`/`test_cli.py`.
- LoRA come **tre adapter Q/K/V separati**.
- `runs_b/` separata da `runs/`.

## Implementazione LoRA (`lm_pipeline_b/model_lora.py`)

Si parte dallo scaffold del lab 4 (`CustomGPT2Attention(GPT2Attention)` e
`GPT2_LoRA(GPT2LMHeadModel)`), riusando il loro `forward` **verbatim** e
completando i `pass`.

- **`LoRALinear(in_features, out_features, rank, alpha)`**: due `nn.Linear(bias=False)`,
  `A` (in→rank) e `B` (rank→out). `A` init `N(0, 0.02)` (o kaiming), **`B` init a 0**
  → a inizializzazione `ΔW = 0`, quindi il modello parte *identico* al GPT2
  pre-addestrato. `forward(x) = B(A(x)) * (alpha / rank)`.
- **Applicazione a Q/K/V**: in GPT2-HF Q/K/V sono fusi in `c_attn` (Conv1D →
  `3*d_model`, poi `.split`). Tre adapter separati `lora_q/lora_k/lora_v`, ciascuno
  `d_model→d_model`, sommati alle rispettive fette dopo lo split:
  ```python
  query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)  # codice lab
  query = query + self.lora_q(hidden_states)   # AGGIUNTA LoRA
  key   = key   + self.lora_k(hidden_states)
  value = value + self.lora_v(hidden_states)
  ```
- **`GPT2_LoRA`**: per ogni `block.attn`, istanzia `CustomGPT2Attention(config, rank, alpha)`,
  copia i pesi pre-addestrati con `load_state_dict(old.state_dict(), strict=False)`
  (i `lora_*` sono nuovi e non presenti nello state dict originale), poi sostituisce
  `block.attn`.
- **Freezing**: tutti i parametri `requires_grad=False`, poi `True` solo per i `lora_*`.
  `param_stats()` (dal lab) stampa trainable vs frozen.

## Dati e training

- **Dati — riuso integrale** di `lm_pipeline/data.py` (PTB, tokenizer GPT2,
  `collate_fn`, `build_dataloaders`). In B la loss la calcola HF internamente: nel
  loop si usa `input_ids` e si costruisce `labels = input_ids.clone()` con `pad → -100`
  (esattamente come il lab), ignorando le label shiftate del collate. Nessuna
  reimplementazione del data layer.
- **`train_lora.py`** (nuovo perché il forward HF restituisce `output.loss`, diverso
  da 1.A): `train_loop`/`eval_loop` dal lab (PPL = `exp(loss)`), avvolti in un `fit`
  con early stopping che rispecchia quello di 1.A. Riuso `pick_device` e
  `_build_optimizer` da `lm_pipeline/train.py`.

## Orchestrazione, CLI, config

- **`experiment_lora.py`**: come `experiment.py` ma costruisce
  `GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=..., alpha=...)`. Riusa
  `config` (load/sweep), `tracking` (seed, run dir, metrics, curves). `metrics.json`
  con lo stesso schema → riuso di `collect_runs`.
- **`run_b.py`**: `run|sweep|aggregate|finalize|export`, su `runs_b/`.
- **`report_b.py`**: riusa `collect_runs/to_markdown/to_latex` di `aggregate.py`,
  scrive `partB_summary.*` (non tocca `aggregate.py` né i suoi test).
- **Config**: `base_partB.yaml` (modello pretrained, lr adatto ad AdamW ~1e-3, epoche
  ridotte, `rank`/`alpha` di default) + `experiments_b/` con sweep
  `rank: [4, 8, 16]` e `alpha: [16, 32]`.

## Consegna standalone `LM/part_B/`

`lm_pipeline_b_export.py` genera: `model.py` (LoRA), `functions.py` (train/eval/fit),
`utils.py` (= data.py), `main.py` generato (ricostruisce `GPT2_LoRA.from_pretrained`,
carica **solo gli adapter** da `bin/` — file piccolo — e stampa la test-PPL),
`README.md`. Mirror esatto dell'export di 1.A.

## Test + smoke su CPU

- `test_model_lora.py`: (a) a init il delta LoRA è 0 → output == GPT2 base;
  (b) dopo freezing solo i `lora_*` hanno `requires_grad`; (c) rank/alpha cambiano
  l'output; (d) i pesi base restano invariati dopo un passo di ottimizzazione.
- `test_train_lora.py`: **smoke su CPU** con config minima e poche frasi, 1 step —
  verifica esecuzione senza errori e loss finita.

Per non scaricare GPT2 reale in CI, i test costruiscono un `GPT2_LoRA` da una
`GPT2Config` minima (pochi layer/dim) invece che da `from_pretrained`.

## Documentazione (`docs/2026-06-22-part1b-lora.md`)

Italiano, stile `GUIDA.md`: cosa chiede 1.B, glossario (fine-tuning, pretrained,
LoRA, rank/alpha, adapter, parametri congelati), spiegazione di ogni scelta, e una
tabella esplicita "riusato vs reimplementato vs preso dal lab".

## Riuso vs nuovo

| Componente | Origine |
|---|---|
| `data.py`/PTB/tokenizer/collate | Riuso 1.A (invariato) |
| `config.py`, `tracking.py` | Riuso 1.A (invariato) |
| `collect_runs/to_latex/to_markdown` | Riuso 1.A |
| `pick_device`, `_build_optimizer` | Riuso 1.A |
| `CustomGPT2Attention.forward`, scaffold classi | Dal lab 4 (verbatim + completamento `pass`) |
| `train_loop`/`eval_loop` HF | Dal lab 4 (adattato + avvolto in `fit`) |
| `LoRALinear`, freezing, `fit`, export, CLI, config B, test, doc | Nuovo |

## Non-obiettivi (YAGNI)

- Niente refactor di 1.A in `common/part_a/part_b`.
- Niente training reale qui (serve la VM GPU): solo codice + struttura + smoke CPU.
- Niente PEFT/librerie LoRA: implementazione manuale come richiesto.

## Dipendenze

`transformers` (già usato per il tokenizer in 1.A). Lo scaffold del lab è basato su
`transformers==4.38.0`; il `forward` di `CustomGPT2Attention` ricalca quella versione.
Da verificare la compatibilità con la versione installata in fase di implementazione.
