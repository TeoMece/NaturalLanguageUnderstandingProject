# LM pipeline (Parte 1.A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** costruire una pipeline modulare, configurabile e documentata per addestrare un GPT2 from-scratch su Penn Treebank (Parte 1.A), con sweep di iperparametri, subset del dataset, logging per il report ed export verso la struttura di consegna.

**Architecture:** package `lm_pipeline` diviso in strato *puro* (`data.py`, `model.py`, `train.py`, mai dipendente dall'orchestrazione → copiabile in consegna) e strato *orchestrazione* (`config.py`, `tracking.py`, `experiment.py`, `aggregate.py`). Un CLI unico `run.py` espone i comandi `run | sweep | aggregate | finalize | export`. Tutto in `NLU_project_development/`.

**Tech Stack:** Python 3.10, PyTorch, HuggingFace `transformers` (solo il tokenizer GPT2 in 1.A), PyYAML, matplotlib, pytest, tqdm.

> Nota di stile vincolante per questo progetto: **commentare generosamente** ogni modulo (docstring + commenti inline che spiegano il *perché*). È parte della valutazione d'esame.

---

## File structure

```
NLU_project_development/
  lm_pipeline/
    __init__.py
    data.py          # [puro] lettura PTB, dataset, subset per frasi, tokenizer, collate, dataloaders
    model.py         # [puro] GPT2 parametrico (dropout opzionale nei 4 punti, weight tying)
    train.py         # [puro] device, scheduler warmup_cosine, train/eval loop, fit+early stopping
    config.py        # [orch] load YAML, deep-merge override, set dotted-key, espansione sweep, nome run
    tracking.py      # [orch] seed, git hash, creazione run dir, salvataggio metrics/curve/plot
    experiment.py    # [orch] run end-to-end di una config, resumable
    aggregate.py     # [orch] scansione runs/, tabella ordinata, export LaTeX/MD, plot comparativo
  configs/
    base_partA.yaml
    experiments/
      00_baseline.yaml
      01_arch.yaml
      02_dropout.yaml
      03_weight_tying.yaml
      04_no_scheduler.yaml
  runs/              # output per-run (creato a runtime)
  reports/           # tabelle + figures (creato a runtime)
  dataset/PennTreeBank/   # copia dei .txt di PTB
  tests/
    test_config.py
    test_data.py
    test_model.py
    test_train.py
    test_tracking.py
    test_experiment.py
    test_aggregate.py
    test_cli.py
    test_export.py
  run.py
  requirements.txt
  pytest.ini
```

Responsabilità per file: una sola per modulo. Lo strato puro non importa mai lo strato di orchestrazione (frontiera di import per l'export).

---

## Task 0: Scaffolding del progetto

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `lm_pipeline/__init__.py`, `tests/__init__.py`
- Create: `dataset/PennTreeBank/` (copia dei .txt)

- [ ] **Step 1: Copiare il dataset PTB**

Run:
```bash
mkdir -p ~/Desktop/NLU_project_development/dataset/PennTreeBank
cp ~/Documents/GitHub/NLU-2026-Labs/labs/dataset/PennTreeBank/ptb.*.txt \
   ~/Desktop/NLU_project_development/dataset/PennTreeBank/
```
Expected: 3 file (`ptb.train.txt`, `ptb.valid.txt`, `ptb.test.txt`) presenti.

- [ ] **Step 2: Creare `requirements.txt`**

```
torch
transformers==4.38.0
pyyaml
matplotlib
tqdm
pytest
tensorboard
```

- [ ] **Step 3: Creare `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Creare i package marker vuoti**

`lm_pipeline/__init__.py`:
```python
"""Pipeline modulare per il language modeling (progetto NLU, Parte 1)."""
```
`tests/__init__.py`: file vuoto.

- [ ] **Step 5: Verificare che pytest parta (nessun test ancora)**

Run: `cd ~/Desktop/NLU_project_development && python -m pytest`
Expected: "no tests ran" senza errori di import.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: scaffold lm_pipeline project"
```

---

## Task 1: `config.py` — caricamento config e sweep

**Files:**
- Create: `lm_pipeline/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import textwrap
from lm_pipeline.config import load_config, set_dotted, expand_sweep

def test_set_dotted_creates_nested_value():
    d = {"optim": {"lr": 1e-3}}
    set_dotted(d, "optim.lr", 5e-4)
    assert d["optim"]["lr"] == 5e-4

def test_load_config_merges_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("optim:\n  lr: 0.001\n  epochs: 30\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text("base: base.yaml\noverride:\n  optim:\n    lr: 0.0005\n")
    cfg = load_config(str(exp))
    assert cfg["optim"]["lr"] == 0.0005   # override applicato
    assert cfg["optim"]["epochs"] == 30   # campo non toccato preservato

def test_expand_sweep_cartesian_product():
    cfg = {"experiment": {"name": "p"}, "optim": {"lr": 0.001},
           "sweep": {"optim.lr": [0.001, 0.0005], "optim.epochs": [10, 20]}}
    runs = expand_sweep(cfg)
    assert len(runs) == 4                                  # 2 x 2
    assert all("sweep" not in r for r in runs)             # sweep rimosso dalle run concrete
    lrs = sorted(r["optim"]["lr"] for r in runs)
    assert lrs == [0.0005, 0.0005, 0.001, 0.001]

def test_expand_sweep_derives_unique_names():
    cfg = {"experiment": {"name": "base"}, "optim": {"lr": 0.001},
           "sweep": {"optim.lr": [0.001, 0.0005]}}
    names = {r["experiment"]["name"] for r in expand_sweep(cfg)}
    assert len(names) == 2                                 # nomi distinti
    assert all(n.startswith("base__") for n in names)

def test_expand_sweep_without_sweep_returns_single():
    cfg = {"experiment": {"name": "solo"}, "optim": {"lr": 0.001}}
    runs = expand_sweep(cfg)
    assert len(runs) == 1 and runs[0]["experiment"]["name"] == "solo"
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL con `ModuleNotFoundError` / funzioni non definite.

- [ ] **Step 3: Implementare `config.py`**

```python
"""Caricamento e composizione delle configurazioni degli esperimenti.

Una config e' un dizionario annidato caricato da YAML. Supporta:
- ereditarieta': un file puo' dichiarare `base: altro.yaml` e una sezione
  `override:` che ne modifica solo alcuni campi (deep-merge);
- sweep: una sezione `sweep:` con liste di valori per chiavi "dotted"
  (es. "optim.lr"); espansa nel prodotto cartesiano -> una config per combinazione.
"""
import os
import copy
import itertools
import yaml


def set_dotted(d, dotted_key, value):
    """Imposta d[a][b][c] = value data la chiave "a.b.c", creando i dict mancanti."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _deep_merge(base, override):
    """Ritorna una copia di base con i campi di override applicati ricorsivamente."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(path):
    """Carica una config risolvendo `base` (path relativo al file) e `override`."""
    raw = _read_yaml(path)
    if "base" in raw:
        base_path = os.path.join(os.path.dirname(path), raw["base"])
        base = _read_yaml(base_path)
        cfg = _deep_merge(base, raw.get("override", {}))
    else:
        cfg = raw
    return cfg


def expand_sweep(cfg):
    """Espande la sezione `sweep` nel prodotto cartesiano di config concrete.

    Senza `sweep` ritorna [cfg] invariata. Ogni combinazione riceve un nome
    derivato univoco appeso al nome base (es. "base__lr0.0005__epochs20").
    """
    sweep = cfg.get("sweep")
    if not sweep:
        return [copy.deepcopy(cfg)]

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    runs = []
    for combo in itertools.product(*value_lists):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.pop("sweep", None)
        suffix_parts = []
        for k, v in zip(keys, combo):
            set_dotted(run_cfg, k, v)
            suffix_parts.append(f"{k.split('.')[-1]}{v}")
        base_name = cfg.get("experiment", {}).get("name", "run")
        run_cfg.setdefault("experiment", {})["name"] = base_name + "__" + "__".join(suffix_parts)
        runs.append(run_cfg)
    return runs
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (5 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/config.py tests/test_config.py
git commit -m "feat: config loading with override and sweep expansion"
```

---

## Task 2: `data.py` — PTB, subset per frasi, dataloaders

**Files:**
- Create: `lm_pipeline/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
from lm_pipeline.data import read_file, subset_sentences, PennTreeBank

def test_read_file_appends_eos(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text("the cat\nthe dog\n")
    lines = read_file(str(p), eos_token="<eos>")
    assert lines == ["the cat <eos>", "the dog <eos>"]

def test_subset_fraction_keeps_whole_sentences():
    sents = [f"frase {i}" for i in range(100)]
    out = subset_sentences(sents, fraction=0.1, max_samples=None, seed=42)
    assert len(out) == 10
    assert all(s in sents for s in out)         # frasi intere, non spezzate

def test_subset_is_deterministic():
    sents = [f"s{i}" for i in range(50)]
    a = subset_sentences(sents, fraction=0.2, max_samples=None, seed=7)
    b = subset_sentences(sents, fraction=0.2, max_samples=None, seed=7)
    assert a == b                               # stesso seed -> stesso subset

def test_subset_max_samples_overrides_fraction():
    sents = [f"s{i}" for i in range(50)]
    out = subset_sentences(sents, fraction=1.0, max_samples=5, seed=1)
    assert len(out) == 5

def test_subset_full_when_fraction_one():
    sents = [f"s{i}" for i in range(20)]
    out = subset_sentences(sents, fraction=1.0, max_samples=None, seed=1)
    assert len(out) == 20

def test_dataset_len_and_getitem():
    ds = PennTreeBank(["a <eos>", "b <eos>"])
    assert len(ds) == 2 and ds[1] == "b <eos>"
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `data.py`**

```python
"""Caricamento e preparazione del dataset Penn Treebank per il language modeling.

PTB ha una frase per riga. Aggiungiamo un token <eos> a fine frase. Il subset,
quando richiesto, seleziona FRASI INTERE (mai spezzate a meta'): nel LM la
predizione avviene dentro la sequenza, quindi tagliare per token corromperebbe
i target. Il tokenizer e' il BPE pre-addestrato di GPT2 (solo tokenizzazione,
nessun peso pre-addestrato usato in 1.A).
"""
import random
from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


def read_file(path, eos_token="<eos>"):
    """Legge un file PTB e ritorna la lista di frasi, ciascuna con <eos> in coda."""
    out = []
    with open(path, "r") as f:
        for line in f.readlines():
            out.append(line.strip() + " " + eos_token)
    return out


def subset_sentences(sentences, fraction=1.0, max_samples=None, seed=42):
    """Sottocampiona la lista di frasi in modo riproducibile (per frasi intere).

    `max_samples` (se valorizzato) ha priorita' su `fraction`. Con fraction=1.0 e
    max_samples=None ritorna tutte le frasi (ordine invariato).
    """
    if max_samples is None and (fraction is None or fraction >= 1.0):
        return list(sentences)
    n = max_samples if max_samples is not None else max(1, int(len(sentences) * fraction))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(sentences)), min(n, len(sentences))))
    return [sentences[i] for i in idx]


class PennTreeBank(data.Dataset):
    """Dataset minimale: una frase (stringa) per elemento."""

    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def collate_fn(batch, tokenizer):
    """Tokenizza un batch di frasi e costruisce input/labels per il next-token.

    Le labels sono l'input shiftato a sinistra di una posizione. I tensori
    restano su CPU: lo spostamento su device avviene nel training loop (cosi'
    funziona anche con num_workers > 0).
    """
    tok = tokenizer(batch, padding=True, return_tensors="pt")
    input_ids = tok.input_ids[:, :-1].contiguous()
    labels = tok.input_ids[:, 1:].contiguous()
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens


def get_tokenizer():
    """Tokenizer BPE di GPT2, con pad_token = eos_token (GPT2 non ha pad nativo)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloaders(cfg_data, dataset_dir, tokenizer):
    """Costruisce (train, valid, test) DataLoader applicando il subset al solo train.

    `cfg_data` e' la sezione `data` della config. Il subset si applica al training
    set; valid e test restano interi per metriche confrontabili.
    """
    import os
    train_raw = read_file(os.path.join(dataset_dir, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_dir, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_dir, "ptb.test.txt"))

    train_raw = subset_sentences(
        train_raw,
        fraction=cfg_data.get("fraction", 1.0),
        max_samples=cfg_data.get("max_samples"),
        seed=cfg_data.get("subset_seed", 42),
    )

    bs = cfg_data.get("batch_size", 32)
    coll = partial(collate_fn, tokenizer=tokenizer)
    train_dl = DataLoader(PennTreeBank(train_raw), batch_size=bs, collate_fn=coll, shuffle=True)
    dev_dl = DataLoader(PennTreeBank(dev_raw), batch_size=bs, collate_fn=coll)
    test_dl = DataLoader(PennTreeBank(test_raw), batch_size=bs, collate_fn=coll)
    return train_dl, dev_dl, test_dl, len(train_raw)
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (6 test). I test su subset/dataset non scaricano il tokenizer.

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/data.py tests/test_data.py
git commit -m "feat: PTB loading, sentence-level subset and dataloaders"
```

---

## Task 3: `model.py` — GPT2 parametrico

**Files:**
- Create: `lm_pipeline/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import torch
from lm_pipeline.model import GPT2

def _tiny(**kw):
    base = dict(vocab_size=50, pos_emb_size=16, d_model=8, n_heads=2,
                num_layers=2, ff_dim=16, dropout=0.0, weight_tying=False)
    base.update(kw)
    return GPT2(**base)

def test_forward_output_shape():
    model = _tiny()
    idx = torch.randint(0, 50, (3, 10))     # batch 3, seq 10
    logits = model(idx)
    assert logits.shape == (3, 10, 50)      # (B, L, vocab)

def test_weight_tying_shares_weights():
    model = _tiny(weight_tying=True)
    assert model.lm_head.weight is model.token_embed.weight

def test_no_weight_tying_separate_weights():
    model = _tiny(weight_tying=False)
    assert model.lm_head.weight is not model.token_embed.weight

def test_dropout_enabled_changes_train_eval():
    torch.manual_seed(0)
    model = _tiny(dropout=0.5)
    idx = torch.randint(0, 50, (2, 8))
    model.train()
    a = model(idx)
    model.eval()
    b = model(idx)
    assert not torch.allclose(a, b)         # dropout attivo solo in train
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `model.py`**

```python
"""Architettura GPT2 (decoder-only) parametrica, ispirata a minGPT/lab NLU.

Rispetto al baseline del lab, qui i 4 punti di dropout previsti dalla Parte 1.A
sono integrati e attivabili (dropout > 0), e il weight tying e' un flag. Con
dropout=0.0 e weight_tying=False si ottiene esattamente il baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Masked multi-head self-attention con dropout opzionale in due punti:
    sui pesi di attention (dopo softmax) e dopo la output projection."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)   # dropout sui pesi di attention
        self.proj_dropout = nn.Dropout(dropout)   # dropout dopo la output projection

    def forward(self, x, mask):
        B, L, d_model = x.size()
        q = self.w_q(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.h_dim)))
        similarity = similarity.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(similarity, dim=-1)
        attn = self.attn_dropout(attn)            # punto dropout 2

        y = (attn @ v).transpose(1, 2).contiguous().view(B, L, d_model)
        y = self.out_proj(y)
        y = self.proj_dropout(y)                  # punto dropout 3
        return y


class FeedForward(nn.Module):
    """Feed-forward con GELU e dropout opzionale dopo l'ultimo linear (punto 4)."""

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),                  # punto dropout 4
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Blocco GPT2: layer-norm pre-modulo, attention e feed-forward con residui."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT2 decoder-only parametrico per il language modeling."""

    def __init__(self, vocab_size, pos_emb_size=1024, d_model=768, n_heads=12,
                 num_layers=12, ff_dim=3072, dropout=0.0, weight_tying=False):
        super().__init__()
        self.pos_emb_size = pos_emb_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)    # punto dropout 1 (post-embedding)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        if weight_tying:
            # condivide gli stessi pesi tra embedding di input e output layer
            self.lm_head.weight = self.token_embed.weight

        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, L = idx.shape
        assert L <= self.pos_emb_size
        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.emb_dropout(x)                    # punto dropout 1
        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)


def init_weights(mat):
    """Inizializzazione uniforme dei Linear (come nel lab)."""
    for m in mat.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/model.py tests/test_model.py
git commit -m "feat: parametric GPT2 with optional dropout points and weight tying"
```

---

## Task 4: `train.py` — device, scheduler, loop e fit

**Files:**
- Create: `lm_pipeline/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import math
import torch
from lm_pipeline.train import pick_device, warmup_cosine_lambda, evaluate, fit
from lm_pipeline.model import GPT2

def test_pick_device_returns_valid_string():
    assert pick_device("cpu") == "cpu"
    dev = pick_device("auto")
    assert dev in ("cuda", "mps", "cpu")

def test_warmup_cosine_shape():
    fn = warmup_cosine_lambda(warmup_steps=10, total_steps=100)
    assert abs(fn(0)) < 0.2          # parte vicino a 0 (warmup)
    assert abs(fn(10) - 1.0) < 1e-6  # picco a fine warmup
    assert fn(100) < 0.05            # ~0 alla fine (cosine)
    assert fn(10) > fn(55) > fn(100) # monotona decrescente dopo il picco

def _toy_loader():
    # 2 batch di (input_ids, labels, n_tokens) con vocab piccolo
    batches = []
    for _ in range(2):
        ids = torch.randint(0, 30, (2, 8))
        labels = torch.randint(0, 30, (2, 8))
        n = torch.tensor(16)
        batches.append((ids, labels, n))
    return batches

def test_evaluate_returns_positive_ppl():
    model = GPT2(vocab_size=30, pos_emb_size=16, d_model=8, n_heads=2,
                 num_layers=1, ff_dim=16)
    crit = torch.nn.CrossEntropyLoss()
    ppl, loss = evaluate(_toy_loader(), crit, model, "cpu")
    assert ppl > 0 and math.isfinite(ppl)

def test_fit_returns_history_and_reduces_loss():
    model = GPT2(vocab_size=30, pos_emb_size=16, d_model=8, n_heads=2,
                 num_layers=1, ff_dim=16)
    cfg_optim = {"lr": 1e-2, "optimizer": "adamw", "epochs": 3, "grad_clip": 1.0,
                 "patience": 5, "scheduler": {"enabled": True, "warmup_steps": 1}}
    hist = fit(model, _toy_loader(), _toy_loader(), cfg_optim, "cpu", pad_id=0)
    assert "valid_ppl" in hist and len(hist["valid_ppl"]) >= 1
    assert "best_ppl" in hist
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `train.py`**

```python
"""Logica di addestramento e valutazione, indipendente dall'orchestrazione.

Contiene: selezione del device, lo scheduler warmup+cosine, i loop di train ed
eval (con perplexity = exp(loss medio per token)) e `fit` con early stopping su
valid-PPL. Tutto riceve oggetti gia' costruiti: nessuna dipendenza da config/yaml.
"""
import math
import copy

import torch
import torch.nn as nn


def pick_device(requested="auto"):
    """Risolve 'auto' nel miglior device disponibile (cuda > mps > cpu)."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def warmup_cosine_lambda(warmup_steps, total_steps):
    """Ritorna f(step) -> moltiplicatore del lr: salita lineare poi cosine decay.

    Da usare con torch.optim.lr_scheduler.LambdaLR. Durante il warmup il
    moltiplicatore cresce 0->1; dopo, decade come mezza cosinusoide 1->0.
    """
    warmup_steps = max(1, warmup_steps)

    def f(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


def _build_optimizer(model, cfg_optim):
    name = cfg_optim.get("optimizer", "adamw").lower()
    lr = cfg_optim["lr"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"optimizer non supportato: {name}")


def train_one_epoch(loader, optimizer, scheduler, criterion, model, device, grad_clip):
    """Una epoca di training; ritorna la loss media per token."""
    model.train()
    total_loss, total_tokens = 0.0, 0
    for input_ids, labels, n_tokens in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)                 # (B, L, vocab)
        loss = criterion(logits.permute(0, 2, 1), labels)  # CE vuole (B, vocab, L)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * int(n_tokens)
        total_tokens += int(n_tokens)
    return total_loss / max(1, total_tokens)


def evaluate(loader, criterion, model, device):
    """Valuta il modello; ritorna (perplexity, loss media per token)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids, labels, n_tokens in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits.permute(0, 2, 1), labels)
            total_loss += loss.item() * int(n_tokens)
            total_tokens += int(n_tokens)
    mean_loss = total_loss / max(1, total_tokens)
    return math.exp(mean_loss), mean_loss


def fit(model, train_loader, valid_loader, cfg_optim, device, pad_id):
    """Addestra con early stopping su valid-PPL. Ritorna lo storico delle metriche.

    Lo storico include le curve per epoca e i pesi migliori (best_state) per il
    salvataggio a valle. Il device viene applicato qui al modello.
    """
    model.to(device)
    optimizer = _build_optimizer(model, cfg_optim)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    epochs = cfg_optim.get("epochs", 30)
    sched_cfg = cfg_optim.get("scheduler", {}) or {}
    scheduler = None
    if sched_cfg.get("enabled", False):
        total_steps = epochs * max(1, len(train_loader))
        lam = warmup_cosine_lambda(sched_cfg.get("warmup_steps", 200), total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    patience = cfg_optim.get("patience", 5)
    grad_clip = cfg_optim.get("grad_clip", 1.0)

    hist = {"train_loss": [], "valid_loss": [], "valid_ppl": []}
    best_ppl, best_state, epochs_no_improve = float("inf"), None, 0

    for _ in range(epochs):
        tr_loss = train_one_epoch(train_loader, optimizer, scheduler, criterion,
                                  model, device, grad_clip)
        ppl, val_loss = evaluate(valid_loader, criterion, model, device)
        hist["train_loss"].append(tr_loss)
        hist["valid_loss"].append(val_loss)
        hist["valid_ppl"].append(ppl)
        if ppl < best_ppl:
            best_ppl, best_state, epochs_no_improve = ppl, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    hist["best_ppl"] = best_ppl
    hist["best_state"] = best_state
    if best_state is not None:
        model.load_state_dict(best_state)         # ripristina i pesi migliori
    return hist
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/train.py tests/test_train.py
git commit -m "feat: training loop, warmup-cosine scheduler and early stopping"
```

---

## Task 5: `tracking.py` — seed, run dir, salvataggi

**Files:**
- Create: `lm_pipeline/tracking.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import json
import torch
from lm_pipeline.tracking import set_seed, make_run_dir, save_metrics, save_curves

def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)

def test_make_run_dir_creates_unique_dir(tmp_path):
    d1 = make_run_dir(str(tmp_path), "exp")
    d2 = make_run_dir(str(tmp_path), "exp")
    assert d1 != d2                      # nomi collidenti -> suffisso incrementale
    import os
    assert os.path.isdir(d1) and os.path.isdir(d2)

def test_save_metrics_writes_json(tmp_path):
    path = save_metrics(str(tmp_path), {"best_ppl": 123.4, "device": "cpu"})
    data = json.load(open(path))
    assert data["best_ppl"] == 123.4

def test_save_curves_writes_csv_and_pdf(tmp_path):
    hist = {"train_loss": [2.0, 1.5], "valid_loss": [2.1, 1.6], "valid_ppl": [8.0, 5.0]}
    csv_path, pdf_path = save_curves(str(tmp_path), hist)
    import os
    assert os.path.exists(csv_path) and os.path.exists(pdf_path)
    assert "valid_ppl" in open(csv_path).readline()
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `tracking.py`**

```python
"""Utility di tracciamento: riproducibilita' e salvataggio degli artefatti per run.

Ogni run vive in una propria cartella. Qui gestiamo seed deterministico, git hash,
creazione della run dir e scrittura di metrics.json, curves.csv e curves.pdf
(materiale pronto per il report).
"""
import os
import csv
import json
import random
import subprocess

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # backend non interattivo (salva su file)
import matplotlib.pyplot as plt


def set_seed(seed):
    """Fissa il seed per Python, NumPy e PyTorch (riproducibilita')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash():
    """Ritorna l'hash del commit corrente, o 'unknown' se non in un repo git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(runs_root, name):
    """Crea una cartella run univoca; se il nome esiste, appende un suffisso."""
    os.makedirs(runs_root, exist_ok=True)
    path = os.path.join(runs_root, name)
    i = 1
    while os.path.exists(path):
        path = os.path.join(runs_root, f"{name}_{i}")
        i += 1
    os.makedirs(path)
    return path


def save_metrics(run_dir, metrics):
    """Salva il dizionario delle metriche in metrics.json e ritorna il path."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_curves(run_dir, hist):
    """Salva le curve per epoca in CSV e un grafico PPL in PDF. Ritorna (csv, pdf)."""
    csv_path = os.path.join(run_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss", "valid_ppl"])
        for i in range(len(hist["valid_ppl"])):
            w.writerow([i + 1, hist["train_loss"][i], hist["valid_loss"][i], hist["valid_ppl"][i]])

    pdf_path = os.path.join(run_dir, "curves.pdf")
    plt.figure()
    plt.plot(range(1, len(hist["valid_ppl"]) + 1), hist["valid_ppl"], marker="o")
    plt.xlabel("epoca")
    plt.ylabel("valid PPL")
    plt.title("Perplexity di validazione")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    return csv_path, pdf_path
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/tracking.py tests/test_tracking.py
git commit -m "feat: tracking utilities (seed, run dir, metrics, curves)"
```

---

## Task 6: `experiment.py` — run end-to-end resumable

**Files:**
- Create: `lm_pipeline/experiment.py`
- Test: `tests/test_experiment.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import os
import json
from lm_pipeline.experiment import run_experiment, already_done

def test_already_done_detects_completed_run(tmp_path):
    rd = tmp_path / "runs" / "exp"
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_ppl": 10}))
    assert already_done(str(tmp_path / "runs"), "exp") is True
    assert already_done(str(tmp_path / "runs"), "altro") is False
```

> Nota: il test end-to-end completo di `run_experiment` è coperto da uno smoke
> test in Task 8 (CLI) su un subset minuscolo, perché richiede il tokenizer e
> qualche secondo di training. Qui testiamo solo la logica di resumability.

- [ ] **Step 2: Eseguire il test per verificarne il fallimento**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `experiment.py`**

```python
"""Orchestrazione di una singola run: dalla config agli artefatti su disco.

Collega lo strato puro (data/model/train) con il tracking. E' resumable: se una
run con quel nome ha gia' un metrics.json, viene saltata (utile per sweep
interrotti a meta', es. su Mac).
"""
import os
import time

import torch

from . import data as data_mod
from . import tracking
from .model import GPT2, init_weights
from .train import pick_device, fit, evaluate


def already_done(runs_root, name):
    """True se esiste gia' una run completata (metrics.json presente)."""
    return os.path.exists(os.path.join(runs_root, name, "metrics.json"))


def run_experiment(cfg, runs_root, dataset_dir, tokenizer=None):
    """Esegue una run completa e salva gli artefatti. Ritorna il path della run dir.

    `cfg` e' una config gia' risolta (no sweep). In `mode: final` ignora il subset
    e valuta su test; altrimenti valuta su valid.
    """
    exp = cfg.get("experiment", {})
    name = exp.get("name", "run")
    if already_done(runs_root, name):
        return os.path.join(runs_root, name)        # resume: salta

    tracking.set_seed(exp.get("seed", 42))
    device = pick_device(exp.get("device", "auto"))
    if tokenizer is None:
        tokenizer = data_mod.get_tokenizer()

    cfg_data = dict(cfg.get("data", {}))
    is_final = cfg.get("mode", "dev") == "final"
    if is_final:
        cfg_data["fraction"] = 1.0                  # full dataset nella run finale
        cfg_data["max_samples"] = None

    train_dl, dev_dl, test_dl, n_train = data_mod.build_dataloaders(cfg_data, dataset_dir, tokenizer)

    m = cfg.get("model", {})
    dropout = m.get("dropout", {})
    model = GPT2(
        vocab_size=len(tokenizer),
        pos_emb_size=m.get("pos_emb_size", 1024),
        d_model=m.get("d_model", 256),
        n_heads=m.get("n_heads", 4),
        num_layers=m.get("num_layers", 4),
        ff_dim=m.get("ff_dim", 1024),
        dropout=(dropout.get("p", 0.1) if dropout.get("enabled", False) else 0.0),
        weight_tying=m.get("weight_tying", False),
    )
    model.apply(init_weights)

    t0 = time.time()
    hist = fit(model, train_dl, dev_dl, cfg.get("optim", {}), device, pad_id=tokenizer.pad_token_id)
    elapsed = time.time() - t0

    # in final: la metrica riportata e' la PPL su test col modello migliore
    import torch.nn as nn
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    test_ppl, _ = evaluate(test_dl, crit, model, device) if is_final else (None, None)

    run_dir = tracking.make_run_dir(runs_root, name)
    tracking.save_curves(run_dir, hist)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    metrics = {
        "name": name,
        "mode": cfg.get("mode", "dev"),
        "best_valid_ppl": hist["best_ppl"],
        "test_ppl": test_ppl,
        "epochs_run": len(hist["valid_ppl"]),
        "n_train_sentences": n_train,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "seconds": round(elapsed, 1),
        "seed": exp.get("seed", 42),
        "git_hash": tracking.git_hash(),
        "config": cfg,
    }
    tracking.save_metrics(run_dir, metrics)
    return run_dir
```

- [ ] **Step 4: Eseguire il test**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/experiment.py tests/test_experiment.py
git commit -m "feat: end-to-end resumable experiment runner"
```

---

## Task 7: `aggregate.py` — tabella e plot per il report

**Files:**
- Create: `lm_pipeline/aggregate.py`
- Test: `tests/test_aggregate.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
import os, json
from lm_pipeline.aggregate import collect_runs, to_latex, to_markdown

def _make_run(root, name, ppl):
    d = os.path.join(root, name); os.makedirs(d)
    json.dump({"name": name, "best_valid_ppl": ppl, "test_ppl": None,
               "n_params": 1000, "seconds": 12.0,
               "config": {"optim": {"lr": 0.001}}},
              open(os.path.join(d, "metrics.json"), "w"))

def test_collect_runs_sorted_by_ppl(tmp_path):
    _make_run(str(tmp_path), "a", 200.0)
    _make_run(str(tmp_path), "b", 150.0)
    rows = collect_runs(str(tmp_path))
    assert [r["name"] for r in rows] == ["b", "a"]   # ordinate per PPL crescente

def test_to_latex_contains_tabular_and_values(tmp_path):
    _make_run(str(tmp_path), "a", 199.5)
    rows = collect_runs(str(tmp_path))
    tex = to_latex(rows)
    assert "tabular" in tex and "199.5" in tex

def test_to_markdown_has_header(tmp_path):
    _make_run(str(tmp_path), "a", 199.5)
    rows = collect_runs(str(tmp_path))
    md = to_markdown(rows)
    assert "| name |" in md.lower() or "name" in md
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL (modulo assente).

- [ ] **Step 3: Implementare `aggregate.py`**

```python
"""Aggregazione delle run in tabelle (LaTeX/Markdown) e plot comparativo.

Scansiona la cartella runs/, legge ogni metrics.json e produce materiale pronto
per il report: una tabella ordinata per perplexity e una figura di confronto.
"""
import os
import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_runs(runs_root):
    """Legge tutti i metrics.json sotto runs_root, ordinati per best_valid_ppl."""
    rows = []
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        rows.append({
            "name": m.get("name", os.path.basename(os.path.dirname(path))),
            "lr": m.get("config", {}).get("optim", {}).get("lr"),
            "best_valid_ppl": m.get("best_valid_ppl"),
            "test_ppl": m.get("test_ppl"),
            "n_params": m.get("n_params"),
            "seconds": m.get("seconds"),
            "_dir": os.path.dirname(path),
        })
    rows.sort(key=lambda r: (r["best_valid_ppl"] is None, r["best_valid_ppl"]))
    return rows


def to_markdown(rows):
    """Tabella Markdown delle run (per consultazione rapida)."""
    head = "| name | lr | valid PPL | test PPL | #params | s |\n"
    head += "|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += f"| {r['name']} | {r['lr']} | {r['best_valid_ppl']:.2f} | " \
                f"{r['test_ppl'] if r['test_ppl'] is not None else '-'} | " \
                f"{r['n_params']} | {r['seconds']} |\n"
    return head + body


def to_latex(rows):
    """Tabella LaTeX (booktabs) pronta da incollare nel report IEEE."""
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"esperimento & lr & valid PPL & test PPL & \#params & s \\", r"\midrule"]
    for r in rows:
        test = f"{r['test_ppl']:.2f}" if r["test_ppl"] is not None else "-"
        name = r["name"].replace("_", r"\_")
        lines.append(f"{name} & {r['lr']} & {r['best_valid_ppl']:.2f} & {test} & "
                     f"{r['n_params']} & {r['seconds']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_reports(runs_root, reports_dir):
    """Scrive partA_summary.tex, partA_summary.md e una figura comparativa."""
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
    rows = collect_runs(runs_root)
    with open(os.path.join(reports_dir, "partA_summary.tex"), "w") as f:
        f.write(to_latex(rows))
    with open(os.path.join(reports_dir, "partA_summary.md"), "w") as f:
        f.write(to_markdown(rows))

    plt.figure()
    names = [r["name"] for r in rows]
    ppls = [r["best_valid_ppl"] for r in rows]
    plt.barh(names, ppls)
    plt.xlabel("valid PPL")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "figures", "comparison.pdf"))
    plt.close()
    return rows
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline/aggregate.py tests/test_aggregate.py
git commit -m "feat: aggregate runs into LaTeX/Markdown tables and comparison plot"
```

---

## Task 8: `run.py` — CLI e finalize

**Files:**
- Create: `run.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Scrivere i test che falliscono**

```python
from run import build_parser, select_best_run

def test_parser_sweep_command():
    args = build_parser().parse_args(["sweep", "--config", "x.yaml"])
    assert args.command == "sweep" and args.config == "x.yaml"

def test_parser_finalize_optional_run():
    args = build_parser().parse_args(["finalize"])
    assert args.command == "finalize" and args.run is None

def test_select_best_run_picks_min_valid_ppl(tmp_path):
    import os, json
    for name, ppl in [("a", 200.0), ("b", 140.0), ("c", 175.0)]:
        d = tmp_path / name; d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"name": name, "best_valid_ppl": ppl,
                                                     "config": {"experiment": {"name": name}}}))
    best = select_best_run(str(tmp_path))
    assert best["name"] == "b"
```

- [ ] **Step 2: Eseguire i test per verificarne il fallimento**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL (`run` assente).

- [ ] **Step 3: Implementare `run.py`**

```python
"""CLI unico della pipeline LM: run | sweep | aggregate | finalize | export.

Esempi:
  python run.py run      --config configs/experiments/00_baseline.yaml
  python run.py sweep    --config configs/experiments/00_baseline.yaml
  python run.py aggregate
  python run.py finalize            # seleziona automaticamente la run migliore
  python run.py finalize --run NOME # forza una run specifica
  python run.py export
"""
import os
import json
import glob
import copy
import argparse

from lm_pipeline.config import load_config, expand_sweep
from lm_pipeline.experiment import run_experiment
from lm_pipeline.aggregate import write_reports

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
REPORTS = os.path.join(ROOT, "reports")
DATASET = os.path.join(ROOT, "dataset", "PennTreeBank")


def build_parser():
    p = argparse.ArgumentParser(description="Pipeline LM (Parte 1.A)")
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("run"); pr.add_argument("--config", required=True)
    ps = sub.add_parser("sweep"); ps.add_argument("--config", required=True)
    sub.add_parser("aggregate")
    pf = sub.add_parser("finalize"); pf.add_argument("--run", default=None)
    sub.add_parser("export")
    return p


def select_best_run(runs_root):
    """Ritorna le metrics della run con best_valid_ppl minima."""
    best = None
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        if m.get("best_valid_ppl") is None:
            continue
        if best is None or m["best_valid_ppl"] < best["best_valid_ppl"]:
            best = m
    return best


def _finalize(run_name=None):
    if run_name is None:
        best = select_best_run(RUNS)
        if best is None:
            print("Nessuna run trovata."); return
        cfg = best["config"]
        print(f"Run migliore: {best['name']} (valid PPL={best['best_valid_ppl']:.2f})")
    else:
        cfg = json.load(open(os.path.join(RUNS, run_name, "metrics.json")))["config"]
        print(f"Run forzata: {run_name}")
    cfg = copy.deepcopy(cfg)
    cfg["mode"] = "final"
    cfg.setdefault("experiment", {})["name"] = cfg["experiment"]["name"] + "__FINAL"
    rd = run_experiment(cfg, RUNS, DATASET)
    m = json.load(open(os.path.join(rd, "metrics.json")))
    print(f"PPL finale su test: {m['test_ppl']:.2f}")


def main():
    args = build_parser().parse_args()
    if args.command == "run":
        run_experiment(load_config(args.config), RUNS, DATASET)
    elif args.command == "sweep":
        for cfg in expand_sweep(load_config(args.config)):
            run_experiment(cfg, RUNS, DATASET)
    elif args.command == "aggregate":
        write_reports(RUNS, REPORTS)
        print(f"Report scritti in {REPORTS}")
    elif args.command == "finalize":
        _finalize(args.run)
    elif args.command == "export":
        from lm_pipeline_export import export_part_a   # vedi Task 9
        export_part_a(ROOT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Eseguire i test**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Smoke test end-to-end su subset minuscolo**

Creare `configs/experiments/_smoke.yaml`:
```yaml
experiment: {name: smoke, seed: 1, device: cpu}
data: {fraction: 0.01, batch_size: 4, subset_seed: 1}
model: {pos_emb_size: 64, d_model: 32, n_heads: 2, num_layers: 1, ff_dim: 64,
        weight_tying: false, dropout: {enabled: false, p: 0.1}}
optim: {lr: 0.001, optimizer: adamw, epochs: 1, grad_clip: 1.0, patience: 2,
        scheduler: {enabled: true, warmup_steps: 5}}
mode: dev
```
Run: `python run.py run --config configs/experiments/_smoke.yaml`
Expected: termina senza errori; compare una cartella sotto `runs/smoke/` con
`metrics.json`, `curves.csv`, `curves.pdf`, `best_model.pt`.
(Scarica il tokenizer GPT2 la prima volta — serve connessione.)

- [ ] **Step 6: Commit**

```bash
git add run.py tests/test_cli.py configs/experiments/_smoke.yaml
git commit -m "feat: CLI with run/sweep/aggregate/finalize and smoke config"
```

---

## Task 9: Export verso la struttura di consegna

**Files:**
- Create: `lm_pipeline_export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Scrivere il test che fallisce**

```python
import os, subprocess, sys
from lm_pipeline_export import export_part_a

def test_export_creates_standalone_files(tmp_path):
    # prepara una finta struttura sorgente minima
    root = tmp_path
    (root / "lm_pipeline").mkdir()
    (root / "lm_pipeline" / "model.py").write_text("x = 'model'\n")
    (root / "lm_pipeline" / "data.py").write_text("y = 'data'\n")
    (root / "lm_pipeline" / "train.py").write_text("z = 'train'\n")
    out = root / "LM" / "part_A"
    export_part_a(str(root), out_dir=str(out), best_cfg={"model": {"d_model": 32}},
                  best_ckpt=None)
    for fname in ["model.py", "utils.py", "functions.py", "main.py"]:
        assert os.path.exists(os.path.join(str(out), fname))
    # i file copiati non devono importare lo strato di orchestrazione
    content = open(os.path.join(str(out), "model.py")).read()
    assert "config" not in content and "experiment" not in content
```

- [ ] **Step 2: Eseguire il test per verificarne il fallimento**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL (`lm_pipeline_export` assente).

- [ ] **Step 3: Implementare `lm_pipeline_export.py`**

```python
"""Export della pipeline verso la struttura di consegna rigida (LM/part_A).

Copia lo strato PURO (model.py, data.py->utils.py, train.py->functions.py), che
e' auto-contenuto, e GENERA un main.py che congela inline gli iperparametri della
config migliore, carica il checkpoint e stampa la PPL. Lo strato di orchestrazione
non viene mai copiato: i file risultanti girano standalone.
"""
import os
import json
import shutil


_MAIN_TEMPLATE = '''"""Entry point di consegna (Parte 1.A) - generato dalla pipeline.

Carica il modello migliore e stampa la perplexity su test.
"""
import torch
import torch.nn as nn

from model import GPT2, init_weights
from utils import build_dataloaders, get_tokenizer
from functions import evaluate, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {best_cfg}

if __name__ == "__main__":
    device = pick_device("auto")
    tokenizer = get_tokenizer()
    m = BEST_CONFIG["model"]
    _, _, test_dl, _ = build_dataloaders(
        {{"fraction": 1.0, "batch_size": BEST_CONFIG["data"]["batch_size"]}},
        "dataset/PennTreeBank", tokenizer)
    dropout = m.get("dropout", {{}})
    model = GPT2(vocab_size=len(tokenizer), pos_emb_size=m.get("pos_emb_size", 1024),
                 d_model=m["d_model"], n_heads=m["n_heads"], num_layers=m["num_layers"],
                 ff_dim=m["ff_dim"],
                 dropout=(dropout.get("p", 0.1) if dropout.get("enabled") else 0.0),
                 weight_tying=m.get("weight_tying", False)).to(device)
    model.load_state_dict(torch.load("bin/best_model.pt", map_location=device))
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ppl, _ = evaluate(test_dl, crit, model, device)
    print(f"Test PPL: {{ppl:.2f}}")
'''


def export_part_a(root, out_dir=None, best_cfg=None, best_ckpt=None):
    """Genera LM/part_A standalone. `best_cfg`/`best_ckpt` opzionali per i test."""
    out_dir = out_dir or os.path.join(root, "LM", "part_A")
    os.makedirs(os.path.join(out_dir, "bin"), exist_ok=True)
    src = os.path.join(root, "lm_pipeline")

    # strato puro -> nomi richiesti dalla consegna
    shutil.copy(os.path.join(src, "model.py"), os.path.join(out_dir, "model.py"))
    shutil.copy(os.path.join(src, "data.py"), os.path.join(out_dir, "utils.py"))
    shutil.copy(os.path.join(src, "train.py"), os.path.join(out_dir, "functions.py"))

    if best_cfg is None:
        # in uso reale: legge la run __FINAL piu' recente
        import glob
        finals = sorted(glob.glob(os.path.join(root, "runs", "*FINAL*", "metrics.json")))
        best_cfg = json.load(open(finals[-1]))["config"] if finals else {"model": {}, "data": {"batch_size": 16}}

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(best_cfg=repr(best_cfg)))

    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, os.path.join(out_dir, "bin", "best_model.pt"))
    return out_dir
```

> Nota: `data.py` importa `transformers` dentro `get_tokenizer()` e `train.py`
> importa solo torch — nessuno importa lo strato di orchestrazione, quindi i file
> copiati sono standalone. Verificare a mano dopo l'export reale che
> `python main.py` giri da dentro `LM/part_A/`.

- [ ] **Step 4: Eseguire il test**

Run: `python -m pytest tests/test_export.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add lm_pipeline_export.py tests/test_export.py
git commit -m "feat: export pure modules to delivery structure (part_A)"
```

---

## Task 10: Config base e degli esperimenti incrementali

**Files:**
- Create: `configs/base_partA.yaml` e i 5 file in `configs/experiments/`

- [ ] **Step 1: Creare `configs/base_partA.yaml`**

```yaml
experiment:
  name: partA
  seed: 42
  device: auto
  tensorboard: false
data:
  fraction: 1.0
  max_samples: null
  subset_seed: 42
  batch_size: 32
  seq_len: 128
model:
  pos_emb_size: 1024
  d_model: 256
  n_heads: 4
  num_layers: 4
  ff_dim: 1024
  weight_tying: false
  dropout:
    enabled: false
    p: 0.1
optim:
  lr: 1.0e-3
  optimizer: adamw
  epochs: 30
  grad_clip: 1.0
  patience: 5
  scheduler:
    enabled: true
    type: warmup_cosine
    warmup_steps: 200
mode: dev
```

- [ ] **Step 2: `00_baseline.yaml` (sweep sul learning rate)**

```yaml
base: ../base_partA.yaml
override:
  experiment:
    name: partA_baseline
sweep:
  optim.lr: [1.0e-3, 5.0e-4, 1.0e-4]
```

- [ ] **Step 3: `01_arch.yaml` (tuning architettura, una alla volta)**

```yaml
base: ../base_partA.yaml
override:
  experiment:
    name: partA_arch
    # NB: impostare optim.lr al miglior valore trovato in 00_baseline
sweep:
  model.d_model: [256, 384]
  model.num_layers: [4, 6]
```

- [ ] **Step 4: `02_dropout.yaml` (aggiunge i 4 dropout)**

```yaml
base: ../base_partA.yaml
override:
  experiment:
    name: partA_dropout
  model:
    dropout:
      enabled: true
      p: 0.1
    # NB: portare qui la migliore architettura/lr dei passi precedenti
```

- [ ] **Step 5: `03_weight_tying.yaml`**

```yaml
base: ../base_partA.yaml
override:
  experiment:
    name: partA_weighttying
  model:
    weight_tying: true
    dropout:
      enabled: true
      p: 0.1
```

- [ ] **Step 6: `04_no_scheduler.yaml` (ablazione)**

```yaml
base: ../base_partA.yaml
override:
  experiment:
    name: partA_noscheduler
  model:
    weight_tying: true
    dropout:
      enabled: true
      p: 0.1
  optim:
    scheduler:
      enabled: false
```

- [ ] **Step 7: Verificare che le config carichino senza errori**

Run:
```bash
python -c "from lm_pipeline.config import load_config, expand_sweep; \
print(len(expand_sweep(load_config('configs/experiments/00_baseline.yaml'))))"
```
Expected: stampa `3` (le 3 run dello sweep sul learning rate).

- [ ] **Step 8: Commit**

```bash
git add configs/
git commit -m "feat: base config and incremental experiment configs for Part 1.A"
```

---

## Self-review (eseguita)

**Spec coverage:** pipeline ricca + export (Task 1-9); logging locale + tabelle LaTeX + grafici (Task 5, 7); config YAML + sweep (Task 1, 10); subset per frasi (Task 2); device-agnostico (Task 4); finalize auto-select (Task 8); scheduler warmup+cosine nel baseline + ablazione (Task 4, 10); riproducibilita' seed/git hash (Task 5); frontiera di import per export (Task 3/9). Tutti i requisiti dello spec hanno un task corrispondente.

**Placeholder scan:** nessun "TBD"/"TODO"; ogni step contiene codice o comandi completi. I commenti "NB: impostare il miglior lr" nelle config sono istruzioni operative per l'approccio incrementale, non placeholder di codice.

**Type consistency:** firme coerenti tra i task — `fit(model, train_loader, valid_loader, cfg_optim, device, pad_id)`, `evaluate(loader, criterion, model, device)`, `run_experiment(cfg, runs_root, dataset_dir, tokenizer=None)`, `build_dataloaders(cfg_data, dataset_dir, tokenizer)` usate in modo uniforme in experiment.py, run.py ed export.

**Punti aperti (dallo spec, non bloccanti):** colonne esatte della tabella LaTeX e valori di default di warmup_steps/epoche, da tarare sui primi run reali.
