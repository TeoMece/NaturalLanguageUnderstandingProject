# What happened so far — Part 2.A lab journal (NLU)

> **The quest.** Joint **intent classification** (one label per sentence) + **slot
> filling** (one IOB label per token) on **ATIS**, starting from the Part 1 GPT-2
> (decoder-only). Improve it **one change at a time**: lr → architecture → FFN ratio →
> n_heads → dropout. We select on the **dev slot F1** (the harder task; conll chunk-level
> F1) and also report **intent accuracy**; the **test set is sealed** until `finalize`.
> Metrics are in [0,1], shown here as %.

---

## Act 0 — Setup

GPT-2 backbone reused from Part 1 (duplicated, self-contained `nlu_pipeline/`) with two
heads: `slot_out` (per token) and `intent_out` (from a **CLS token appended at the end**
— with causal masking it is the only token that sees the whole sentence). Multi-task
loss = `CE(slot) + CE(intent)`. Dev set = 10% **stratified** on intent (ATIS has none).
ATIS: 4978 train / 893 test, 26 intents, 129 slot labels (IOB), ~11 tokens/sentence.

A note that will matter: ATIS is **easy and small**, so all numbers sit high (slot F1
~96%, intent acc ~98%) and differences between configs are small.

---

## Act 1 — Baseline: hunting the learning rate

Fixed architecture (`d_model=256`, `num_layers=2`, FFN 4×, `n_heads=4`), swept lr.

| lr | dev slot F1 | dev intent acc |
|------|-----------|-----------|
| **5e-4** | **96.52** | **98.19** ← winner |
| 1e-4 | 95.84 | 97.79 |
| 1e-3 | 95.15 | 97.79 |

**Discovery.** `5e-4` wins on both metrics; the gentle middle lr beats both the hotter
(1e-3) and colder (1e-4) — the same "Goldilocks" pattern seen in Part 1.

➡️ **Carried forward: `lr = 5e-4`.**

---

## Act 2 — Architecture: bigger is *not* better (again)

Swept `d_model ∈ {128, 256, 384}` × `num_layers ∈ {2, 4}` (lr 5e-4); the grid includes
the baseline `(256, 2)` as a reproducibility anchor.

| config | dev slot F1 | dev intent acc | #params |
|---|---|---|---|
| **d256 / l2** | **96.52** | 98.19 | 2.0M ← winner (slot F1) |
| d256 / l4 | 96.20 | 97.79 | 3.6M |
| d128 / l2 | 96.17 | 98.19 | 0.6M |
| d384 / l2 | 96.11 | **99.20** | 4.1M |
| d128 / l4 | 96.11 | 97.99 | 1.0M |
| d384 / l4 | 95.47 | 98.19 | 7.7M |

**Discoveries.**
- **Reproducibility ✓**: `d256/l2` here scores **96.52**, identical to the baseline's
  lr=5e-4 run — same config, same number.
- **Scaling up doesn't help**: slot F1 *decreases* as the model grows (96.52 → … →
  95.47 for the biggest `d384/l4`). On small/easy ATIS the bottleneck is **data, not
  capacity** — the same finding as Part 1.A's LM.
- ⚠️ **Intent/slot trade-off**: `d384/l2` has the **best intent accuracy (99.20)** but a
  lower slot F1 (96.11). We select on **slot F1** → `d256/l2`. Worth flagging in the
  report: the two tasks don't peak at the same config.

➡️ **Carried forward: `d_model = 256`, `num_layers = 2` (= the baseline).**

---

## Act 3 — FFN ratio: 4× is best (the convention holds)

Swept `ff_mult ∈ {2, 4, 8}` (FFN width = ff_mult × d_model) at d256/l2, in its own step
so it doesn't confound the width comparison.

| ff_mult (ff_dim) | dev slot F1 | dev intent acc |
|---|---|---|
| **4 (1024)** | **96.52** | **98.19** ← winner |
| 8 (2048) | 95.96 | 97.79 |
| 2 (512) | 95.58 | 97.19 |

**Discovery.** The canonical **4×** ratio wins; both wider (8×) and narrower (2×) hurt
— and `ff_mult=4` reproduces 96.52 again (it *is* the baseline). Another "the default is
already best" result, consistent with the data-limited picture.

➡️ **Carried forward: `ff_mult = 4` (ff_dim = 1024).**

---

## Acts pending (configs ready, lr/arch/ffn already carried in)

- **Act 4 — n_heads** (`01c_heads`): sweep `n_heads ∈ {2, 4, 8}` (256 divisible by all).
  Completes the lab's hyper-parameter list (d_model, n_heads, num_layers, ff_dim).
- **Act 5 — Dropout** (`02_dropout`): sweep `p ∈ {0.1, 0.2, 0.3, 0.5}` before the heads.
- **Finalize**: retrain the overall winner and report **test** slot F1 + intent acc.

---

## Current best config (Part 2.A) 🏆

| field | value |
|---|---|
| d_model | 256 |
| num_layers | 2 |
| ff_dim | 1024 (auto, 4×) |
| n_heads | 4 |
| dropout | none (yet) |
| lr | 5e-4 |
| **dev slot F1** | **96.52** |
| **dev intent acc** | **98.19** |
| test | _TBD (finalize)_ |

---

## Methodology notes (for the report / Q&A)

- **Selection on dev slot F1**, test sealed and scored once (`finalize`) → no leakage.
- **Dev = 10% stratified on intent** (ATIS ships no dev set).
- **Incremental, one change at a time**, carrying the best value forward (the lab's
  protocol); kept changes only if they improve dev slot F1, the rest documented anyway.
- ATIS is **data-limited/easy** → the interesting story is methodological (bigger
  doesn't help; intent vs slot peak differently), not the absolute numbers.
