# What happened so far — Part 1.A lab journal

> **The quest.** Train a GPT-2 *from scratch* on Penn Treebank and find the best
> configuration by changing **one factor at a time**. The score we optimize is
> **validation perplexity (PPL)** — lower is better. We run everything in
> `mode: dev` (evaluate on validation), keeping the **test set sealed** until the
> very end (`finalize`) to avoid data leakage.

> **All numbers below are the post-fix results** (they match `reports/partA_summary.md`).
> An early weight-init bug was found and fixed mid-project — see the box right
> after Act 0; the pre-fix numbers are archived in `reports_pre_fix/`.

---

## Act 0 — Getting the machine to even run 🛠️

Before any science, a small odyssey on the Azure VM (Tesla V100): the GPU was
there but the driver wasn't, PyTorch shipped a CUDA build too new for the driver,
and the newest torch had quietly **dropped support for Volta (CC 7.0)**. The
working recipe: NVIDIA `570-server` driver via **DKMS** + **`torch==2.4.1+cu121`**.
Lesson logged in `memory/vm-gpu-setup.md` so we never re-fight this battle.

---

## ⚙️ The init fix that reshaped every result

The first pass used a flawed weight init; once fixed, the conclusions changed
(notably: weight tying went from useless to the biggest win). Two fixes landed:

1. **Embedding init.** `init_weights` ignored `nn.Embedding` (it stayed at N(0,1)),
   and weight tying clobbered the shared matrix down to the tiny Linear init →
   embeddings were ~170× off between tied/untied. Now: embeddings `normal(0, 0.02)`,
   tying-aware.
2. **Residual scaling — "Lever 1".** Residual-writing projections (`out_proj`, FFN
   2nd Linear) now init to `normal(0, 0.02/√(2·num_layers))`, keeping the
   residual-stream variance constant with depth. Full explainer:
   `docs/leva1-init-residual-scaling.md`.

Pre-fix numbers archived in `reports_pre_fix/`. Everything below is post-fix.

---

## Act 1 — Baseline: hunting the learning rate

Fixed architecture (`d_model=256`, `num_layers=4`, scheduler warmup+cosine on, no
dropout, no tying), swept the learning rate.

| lr | valid PPL |
|------|-----------|
| **1e-3** | **36.91** ← winner |
| 5e-4 | 37.15 |
| 1e-4 | 37.49 |

**Discovery.** The hottest lr we tried (1e-3) wins outright, and PPL degrades
smoothly as the lr drops — a from-scratch model needs a healthy step size to move
in the epoch budget. This is the anchor lr for everything downstream.

➡️ **Carried forward: `lr = 1e-3`.**

---

## Act 2 — Architecture: bigger is *not* better here

Swept `d_model ∈ {256, 384, 512}` × `num_layers ∈ {2, 4, 6}` (9 runs), lr fixed.

Along the way we built a small feature: **`ff_dim: auto`** (mirrors the existing
`device: auto` idiom). The feed-forward width scales as `4 × d_model` automatically,
so every model keeps canonical transformer proportions — otherwise a fixed `ff_dim`
would have **confounded** the width comparison.

| d_model \ layers | 2 | 4 | 6 |
|---|---|---|---|
| **256** | 37.00 | 36.91 | 36.85 |
| **384** | **36.31** | 36.85 | 37.29 |
| **512** | 37.96 | 39.78 | 40.97 |

**Discovery.** The sweet spot is a **moderate width, shallow depth** model:
`d384/l2` wins at **36.31**. Crucially, **scaling up hurts**: the biggest/deepest
`d512/l6` is among the *worst* (40.97). On a small corpus like PTB, with our epoch
budget, the larger models don't pay off. Reproducibility check: `d256/l4` here
scores **36.91**, identical to the baseline's lr=1e-3 run — same config, same
number → the pipeline is reproducible.

➡️ **Carried forward: `d_model = 384`, `num_layers = 2` (ff_dim = 1536).**

---

## Act 3 — Dropout: a light touch helps

Added the 4 dropout points on top of the best architecture (`d384/l2`) and swept
the rate.

| dropout p | valid PPL |
|---|---|
| none | 36.31 |
| **0.1** | **36.24** ← best |
| 0.2 | 36.85 |
| 0.3 | 37.40 |
| 0.4 | 39.40 |
| 0.5 | 47.97 |

**Discovery.** A gentle **p=0.1 helps** (36.24 vs 36.31 with none) — a small but
real regularization gain at this scale. Beyond that it degrades monotonically, hard
(p=0.5 → 47.97).

⚠️ **Methodological note.** The overfitting gap (`valid_ppl − train_ppl`) is only
trustworthy for **no-dropout** runs: training loss is logged with dropout *active*
(inflated), so the gap on dropout rows is an artifact. Use **`gap_report.py`** (see
the reminder near the end) on the no-dropout runs to read overfitting honestly.

➡️ **Carried forward: dropout p=0.1.**

---

## Act 4 — Weight tying (the real win) + scheduler ablation

On `d384/l2 + dropout p0.1`, added **weight tying** (share `token_embed ↔ lm_head`):

| config | valid PPL |
|---|---|
| **+ weight tying** | **34.18** ← best overall |
| same, scheduler OFF (ablation) | 34.23 |

**Discovery.** Weight tying is the **biggest single jump** of the whole campaign
(36.24 → **34.18**) — exactly what theory predicts, and exactly what the init fix
unlocked (pre-fix, tying looked useless). The **scheduler ablation** shows
warmup+cosine contributes only ~0.05 PPL here (34.23 vs 34.18): nice to have, not
decisive. Keep it on (harmless).

➡️ **Carried forward: weight tying ON, scheduler ON.**

---

## Act 5 — Lever 1 stress test: can depth finally win? (answered: no)

**Hypothesis.** With residual-scaled init, the biggest/deepest model should stop
underperforming. Config `05_bigarch.yaml`: full best recipe (tying, dropout p0.1,
lr 1e-3) on `d512/l6`. And `06_bigarch_dropout.yaml` swept dropout on it.

| config | valid PPL |
|---|---|
| `d512/l6` + tying + p0.1 (`bigarch`) | 38.72 |
| `d512/l6` + tying, dropout p0.2 | 44.85 |
| `d512/l6` + tying, dropout p0.3 | 45.92 |
| `d512/l6` + tying, dropout p0.4 | 47.54 |

**Verdict.** Even with Lever 1, the big/deep model (**38.72**) **loses** to the
small `d384/l2` (**34.18**), and more dropout only makes it worse (it does *not*
fall back toward 34). So depth is trainable now, but it doesn't help here: the
bottleneck is **data** (PTB is small), not trainability. The shallow `d384/l2`
stays king.

➡️ **Carried forward: nothing new — `d384/l2 + p0.1 + tying` remains the best.**

---

## Current best model 🏆

| field | value |
|---|---|
| d_model | 384 |
| num_layers | 2 |
| ff_dim | 1536 (auto, 4×) |
| weight tying | ON |
| dropout | p = 0.1 |
| lr | 1e-3 |
| scheduler | warmup + cosine (on) |
| **valid PPL** | **34.18** |

Project target is PPL < 250 → **smashed**; the chase above was for the best number.

---

## 📊 Reminder — run the gap report to analyze overfitting

Before locking decisions (and before/after any regularization), dump the
train/valid generalization gap per run:

```bash
python gap_report.py                         # reads runs/, writes reports/overfitting_gaps.csv
# or: python gap_report.py --runs runs --out reports/gaps.csv
```

It lists every run by **gap = valid_ppl − train_ppl** (descending: worst overfit on
top), using the early-stopping epoch. Use it to see *which* runs overfit before
deciding on dropout / epochs. **Caveat (see Act 3):** the gap is only meaningful for
**no-dropout** runs — dropout inflates the logged train loss, so dropout rows
understate the true gap.

---

## Tools we built along the way

- **`ff_dim: auto`** — derived feed-forward width (`ff_mult × d_model`, default 4×),
  keeps comparisons proportionate. Covered by tests in `tests/test_config.py`.
- **`gap_report.py`** — dumps the train/valid overfitting gap per run to a CSV, so
  we *look before we leap* on regularization decisions.

---

## Next chapters

1. **`finalize`** the current best (`d384/l2 + p0.1 + tying`, lr 1e-3) → the **test
   PPL**, our one official number for 1.A. *(Run on the VM — it retrains.)*
2. Move to **Part 1.B (LoRA)** — its own journal is in `reports/recap_b.md`.
