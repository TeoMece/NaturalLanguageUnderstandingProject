# What happened so far — Part 1.A lab journal

> **The quest.** Train a GPT-2 *from scratch* on Penn Treebank and find the best
> configuration by changing **one factor at a time**. The score we optimize is
> **validation perplexity (PPL)** — lower is better. We run everything in
> `mode: dev` (evaluate on validation), keeping the **test set sealed** until the
> very end (`finalize`) to avoid data leakage.

---

## Act 0 — Getting the machine to even run 🛠️

Before any science, a small odyssey on the Azure VM (Tesla V100): the GPU was
there but the driver wasn't, PyTorch shipped a CUDA build too new for the driver,
and the newest torch had quietly **dropped support for Volta (CC 7.0)**. The
working recipe: NVIDIA `570-server` driver via **DKMS** + **`torch==2.4.1+cu121`**.
Lesson logged in `memory/vm-gpu-setup.md` so we never re-fight this battle.

---

## Act 1 — Baseline: hunting the learning rate

Fixed architecture (`d_model=256`, `num_layers=4`, scheduler warmup+cosine on),
swept the learning rate.

| lr | valid PPL |
|------|-----------|
| 1e-3 | 39.65 |
| **5e-4** | **39.46** ← winner |
| 1e-4 | 40.98 |

**Discovery.** A gentle 5e-4 wins; too hot (1e-3) and too cold (1e-4) both lose.
Nothing dramatic, but it gives us the anchor lr for everything downstream.

➡️ **Carried forward: `lr = 5e-4`.**

---

## Act 2 — Architecture: bigger *and* deeper

Swept `d_model ∈ {256, 384, 512}` × `num_layers ∈ {2, 4, 6}` (9 runs), lr fixed.

Along the way we built a small feature: **`ff_dim: auto`** (mirrors the existing
`device: auto` idiom). The feed-forward width now scales as `4 × d_model`
automatically, so every model keeps the canonical transformer proportions —
otherwise a fixed `ff_dim` would have **confounded** the width comparison.

| d_model \ layers | 2 | 4 | 6 |
|---|---|---|---|
| **256** | 42.26 | 39.46 | 38.39 |
| **384** | 40.43 | 38.86 | 38.20 |
| **512** | 39.90 | 38.02 | **36.97** |

**Discovery.** A clean monotone story: **wider helps, deeper helps**, and the two
stack — the biggest/deepest model (`d512/l6`, valid **36.97**) wins.
Nice sanity check: `d256/l4` here scores **39.46**, *identical* to the baseline's
lr=5e-4 run — same config, same number → the pipeline is reproducible.

➡️ **Carried forward: `d_model = 512`, `num_layers = 6` (ff_dim = 2048).**

---

## Act 3 — Dropout: the regularizer that backfired 🎭

Added the 4 dropout points on top of the best architecture and swept the rate.

| dropout p | valid PPL |
|---|---|
| **none** | **36.97** ← still the best |
| 0.1 | 37.27 |
| 0.2 | 37.60 |
| 0.3 | 39.06 |
| 0.4 | 39.65 |
| 0.5 | 41.52 |

**Discovery.** Dropout **monotonically hurts** — even the lightest touch (p=0.1)
is worse than none, and it degrades smoothly up to p=0.5. The architecture search
had already found a model that generalizes about as well as it can at this scale;
adding regularization just removes capacity and injects noise.

We checked the obvious escape hatch — *was early stopping cutting the dropout runs
short?* No: higher-p runs actually trained **longer** (best epoch crept from 3 up
to 14) and still ended **worse**. So more epochs wouldn't rescue it.

⚠️ **Methodological note.** The overfitting gap (`valid_ppl − train_ppl`) is only
trustworthy for **no-dropout** runs: training loss is logged with dropout *active*
(inflated), so the gap on dropout rows is an artifact, not a sign of less
overfitting. The clean signal is validation PPL.

➡️ **Carried forward: dropout OFF.**

---

## Current best model 🏆

| field | value |
|---|---|
| d_model | 512 |
| num_layers | 6 |
| ff_dim | 2048 (auto, 4×) |
| lr | 5e-4 |
| scheduler | warmup + cosine (on) |
| dropout | none |
| **valid PPL** | **36.97** |

---

## Tools we built along the way

- **`ff_dim: auto`** — derived feed-forward width (`ff_mult × d_model`, default 4×),
  keeps comparisons proportionate. Covered by tests in `tests/test_config.py`.
- **`gap_report.py`** — dumps the train/valid overfitting gap per run to a CSV,
  so we can *look before we leap* on regularization decisions.

---

## Next chapters

1. **`finalize`** the current best (d512/l6, no dropout) → the **test PPL**, our
   one official number. *(Run on the VM — it retrains.)*
2. **Experiment 3 — weight tying:** does sharing embedding ↔ output weights help?
3. **Experiment 4 — scheduler ablation:** quantify the warmup+cosine contribution.
4. **Heads-up:** `03_weight_tying.yaml` and `04_no_scheduler.yaml` still have
   **dropout enabled (p=0.1)**. Given Act 3, that base is sub-optimal — decide
   whether to disable dropout there for consistency before running them.

---

## ⚙️ Mid-season rewrite — two init bugs fixed

> The numbers in Acts 1–3 above were produced with a flawed weight init and have
> been (or are being) re-run. Two fixes landed:
>
> 1. **Embedding init.** `init_weights` ignored `nn.Embedding` (stayed at N(0,1)),
>    and weight tying clobbered the shared matrix to the tiny Linear init →
>    embeddings ~170× off between tied/untied. Now: embeddings `normal(0, 0.02)`,
>    tying-aware. Pre-fix results archived in `reports_pre_fix/`.
> 2. **Residual scaling — "Lever 1".** Residual-writing projections (`out_proj`,
>    FFN 2nd Linear) now init to `normal(0, 0.02/√(2·num_layers))`, keeping the
>    residual-stream variance constant with depth. Full explainer:
>    `docs/leva1-init-residual-scaling.md`.
>
> **Post-(embedding-)fix best so far:** `d384/l2` + dropout p0.1 + weight tying,
> lr 1e-3 → **valid PPL 34.18**. Notably, weight tying now *helps* (as expected
> once the init was sane). Target of the project is PPL < 250 → already smashed;
> the chase below is for sport.

---

## Act 4 — Lever 1 stress test: can depth finally win? 🏗️ *(results pending)*

**Hypothesis.** With residual-scaled init, the biggest/deepest model should no
longer underperform a shallow one. Config `05_bigarch.yaml`: the full best recipe
(lr 1e-3, dropout p0.1, weight tying) on the **largest architecture** we have —
`d_model=512`, `num_layers=6` (ff_dim 2048 auto). Residual std at init =
`0.02/√12 ≈ 0.00577`.

| metric | shallow best (`d384/l2`, post-emb-fix) | `d512/l6` + Lever 1 |
|---|---|---|
| valid PPL | 34.18 | **_TBD_** |
| best epoch / epochs run | _TBD_ | **_TBD_** |
| train_ppl @ best (gap) | _TBD_ | **_TBD_** |

**What we're looking for:** valid PPL **below 34.18**, and a deep model that beats
the shallow `d384/l2`. If depth now helps, Lever 1 did its job.

**Result:** _to be filled after the run._

➡️ **Carried forward:** _TBD (depends on the result above)._
