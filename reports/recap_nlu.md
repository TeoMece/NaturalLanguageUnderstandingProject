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

## Act 4 — n_heads: 4 is best (completes the hyper-parameter list)

Swept `n_heads ∈ {2, 4, 8}` at d256/l2, ff 4× (256 divisible by all). Same param count
across runs (n_heads only re-partitions d_model into heads).

| n_heads | dev slot F1 | dev intent acc |
|---|---|---|
| **4** | **96.52** | 98.19 ← winner |
| 2 | 96.22 | 96.99 |
| 8 | 96.11 | **98.39** |

**Discovery.** `n_heads=4` (the default) wins on slot F1 — again the baseline.
(`n_heads=8` edges intent accuracy at 98.39, the same intent/slot trade-off seen with
`d384/l2`, but we select on slot F1.)

➡️ **Carried forward: `n_heads = 4`.**

> **Big picture so far:** the whole hyper-parameter search (lr → d_model/layers → FFN →
> n_heads) **converged back to the baseline config** (`d256/l2`, 4×, 4 heads, lr 5e-4 →
> dev slot F1 **96.52**): nothing beat it. On small/easy ATIS the default GPT-2-style
> architecture is already at its ceiling — capacity is not the bottleneck. The remaining
> lever is **regularization** (Act 5).

---

## Act 5 — Dropout: it only hurts

Added dropout before the heads on the best config and swept `p`.

| dropout p | dev slot F1 | dev intent acc |
|---|---|---|
| **none** | **96.52** | 98.19 ← best |
| 0.1 | 96.08 | 97.39 |
| 0.2 | 95.88 | 97.79 |
| 0.3 | 95.58 | 97.59 |
| 0.5 | 92.70 | 98.19 |

**Discovery.** Dropout **monotonically hurts** slot F1 — even the lightest (p=0.1) is
below no-dropout, collapsing at p=0.5. The model isn't overfitting on ATIS at this
scale, so removing capacity/adding noise only costs. Same conclusion as Part 1.A:
regularization doesn't help a data-limited, already-modest model.

➡️ **Carried forward: dropout OFF.** The search is complete.

---

## Final model (Part 2.A) 🏆 — and the whole story in one line

The entire incremental search (lr → d_model/layers → FFN → n_heads → dropout) **never
beat the baseline**: the default GPT-2-style config is already at ATIS's ceiling.

| field | value |
|---|---|
| d_model | 256 |
| num_layers | 2 |
| ff_dim | 1024 (auto, 4×) |
| n_heads | 4 |
| dropout | **none** (confirmed) |
| lr | 5e-4 |
| **dev slot F1** | **96.52** |
| **dev intent acc** | **98.19** |
| test | _TBD (run `finalize`)_ |

**Next:** `python run_nlu.py finalize` (auto-selects the 96.52 config) → test slot F1 +
intent acc, then `export`.

---

## Methodology notes (for the report / Q&A)

- **Selection on dev slot F1**, test sealed and scored once (`finalize`) → no leakage.
- **Dev = 10% stratified on intent** (ATIS ships no dev set).
- **Incremental, one change at a time**, carrying the best value forward (the lab's
  protocol); kept changes only if they improve dev slot F1, the rest documented anyway.
- ATIS is **data-limited/easy** → the interesting story is methodological (bigger
  doesn't help; intent vs slot peak differently), not the absolute numbers.
