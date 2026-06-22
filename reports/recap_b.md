# What happened so far — Part 1.B lab journal (LoRA)

> **The quest.** Take a *pre-trained* GPT-2 (HuggingFace) and fine-tune it on Penn
> Treebank by training **only manually-implemented LoRA adapters** on the Q/K/V
> projections (everything else frozen). Score = **validation perplexity (PPL)**,
> lower is better. We sweep, one factor at a time, **lr → rank → alpha**. Target:
> **PPL < 250** *and* **below Part 1.A**. Runs in `mode: dev`; test set sealed
> until `finalize`.

---

## Act 0 — Setup & sanity 🛠️

No GPU driver odyssey this time (same VM as 1.A). Two structural facts worth
pinning down up front, because they make 1.B **immune to the init bugs that bit
1.A**:

- **We never call `init_weights` in 1.B.** The base weights come *pre-trained*
  from OpenAI via `from_pretrained` (already well-scaled). The only thing we
  initialize is the LoRA adapters: `A ~ N(0, 0.02)`, **`B = 0`**. With `B = 0` the
  adapter contributes exactly **0** at init → the model starts *identical* to
  pre-trained GPT-2 (verified by `test_gpt2lora_matches_base_at_init`).
- **The "residual scaling" concern of 1.A doesn't apply.** Since `B = 0`, adapters
  add no variance to the residual stream at init; there's nothing to explode with
  depth. The whole class of init bugs from 1.A is *structurally absent here*.

Smoke check (CPU/MPS, 16 sentences, 1 epoch): pipeline runs end-to-end, only
**442,368 / 124M** params trainable (~0.35%), `adapters.pt` ≈ 1.8 MB.

---

## Act 1 — Baseline: hunting the learning rate

Fixed `rank=8`, `alpha=16`; swept the learning rate `{1e-3, 5e-4, 1e-4}`.

| lr | valid PPL |
|------|-----------|
| 1e-3 | _TBD_ |
| 5e-4 | _TBD_ |
| **1e-4** | **_TBD_** ← winner (lowest lr) |

**Discovery — and it's the *opposite* of 1.A, on purpose.** In 1.A (from scratch)
a mid lr won and too-cold (1e-4) lost: a random-init model must move *far* in few
epochs. Here the **lowest lr wins**, which is exactly the fine-tuning signature:
the pre-trained model is *already good*, the adapters start *at zero*, so big steps
**disrupt** useful pre-trained behavior — small, careful steps win. Not a bug: the
textbook intuition that *fine-tuning wants smaller lr than training-from-scratch*.

➡️ **Carried forward: lowest lr (1e-4).** Worth trying an even lower point (e.g.
**5e-5**) next time, since the trend points down.

---

## ⚠️ Methodological note — A↔B comparability (last token)

The shared `collate_fn` returns `input_ids` already shortened by one
(`ids[:, :-1]`). In 1.B we pass it to HF with `labels = input_ids.clone()` and HF
**shifts internally**, so 1.B does **not** score the last token of each sentence
(typically `<eos>`), whereas 1.A does. This is **exactly what the lab does** (it
reuses the shifted collate then clones), and it's **internally consistent across
all B runs** — so rank/alpha/lr comparisons within B are reliable. The only caveat:
B's PPL and A's PPL aren't computed on the *identical* target set (B drops ~1 token
per sentence). It doesn't change the verdict (pre-trained GPT-2 sits comfortably
below A regardless), but for a strict same-denominator A↔B number we could align B
to also score the last token. **Decision for now: keep the lab's behavior.**

---

## Next chapters

1. **Experiment B1 — rank sweep** (`01_rank_sweep.yaml`, `rank ∈ {4, 8, 16}`):
   carry the best lr in first. Bigger rank = more expressive adapters / more params
   — find the sweet spot for PPL.
2. **Experiment B2 — alpha sweep** (`02_alpha_sweep.yaml`, `alpha ∈ {16, 32}`):
   carry best lr + best rank. `alpha` sets the adaptation magnitude (`scaling =
   alpha/rank`).
3. **`finalize`** the best config → the one official **test PPL**, and **`export`**
   → fills `LM/part_B/bin/adapters.pt` + regenerates `main.py`.

---

## Current best config (Part 1.B) 🏆

| field | value |
|---|---|
| base model | openai-community/gpt2 (pre-trained, frozen) |
| LoRA on | Q, K, V (all 12 layers, independent adapters) |
| rank | 8 _(baseline; to be swept)_ |
| alpha | 16 _(baseline; to be swept)_ |
| lr | **1e-4** _(baseline winner)_ |
| trainable params | 442,368 (~0.35% of 124M) |
| **valid PPL** | **_TBD_** |
