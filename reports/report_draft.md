# NLU Course Project — Part 1: Language Modeling
*(draft for the IEEE 1-page report — prose ≤ 1 page; tables/figures/refs excluded)*

**Author:** <Nome Cognome> (mat. <id>) — <email>@studenti.unitn.it

---

## 1. Introduction

We address word-level language modeling on the Penn Treebank (PTB, ~930k training
tokens), predicting the next token and measuring perplexity (PPL). In **Part A** we
train a GPT-2-style decoder-only Transformer **from scratch** and improve it
incrementally — learning-rate and architecture search, dropout, and weight tying —
lowering the test PPL to **30.1**. In **Part B** we instead adapt the **pre-trained**
GPT-2 to PTB with a **hand-implemented LoRA**, training only low-rank adapters on the
query/key/value projections while the 124M backbone stays frozen, reaching test PPL
**19.5**. Both parts satisfy PPL < 250, and Part B clearly improves over Part A.

## 2. Implementation details

**Setup.** Sentences are encoded with the GPT-2 BPE tokenizer, terminated by an
`<eos>` marker and right-padded; padding is ignored in the loss and PPL is the
exponential of the per-token cross-entropy. Every experiment follows a greedy,
one-change-at-a-time protocol that keeps the best configuration of each step, with
AdamW, gradient clipping (max-norm 1.0), a warmup + cosine learning-rate schedule and
early stopping on the development PPL. Models are selected on dev; the test set is
kept sealed and scored once, on the final model of each part.

**Part A.** We first tune the learning rate (1e-3 best), then grow the architecture
one hyper-parameter at a time. The search favours a **moderate, shallow** model
(`d_model=384`, 2 layers, FFN 1536): larger/deeper variants are consistently worse
(e.g. `d512/l6` is among the worst), indicating that at PTB scale the bottleneck is
**data, not capacity**. We then add dropout in the four required positions — after the
token+positional embeddings, on the attention weights, after the attention output
projection, and after the feed-forward block — which helps only lightly (p=0.1).
Finally, **weight tying** (sharing the token-embedding matrix with `lm_head`) gives the
largest single gain while removing ~19M parameters — the best accuracy/size trade-off
of the search.
Two methodological points shaped these results. (i) An initial **weight-init** flaw —
embeddings left at N(0,1) and a residual-stream variance that **grew with depth** —
inverted our early conclusions; fixing it (embeddings N(0,0.02); residual-writing
projections scaled by 1/√(2·layers), keeping the residual std constant with depth)
made tying help and the architecture ranking sane. (ii) Our tight **gradient clip**
(max-norm 1.0) acts as a mild regularizer, which is why the useful dropout is small
(p=0.1); a looser clip would shift the dropout optimum upward.

**Part B.** We implement LoRA manually, without PEFT: each attention layer is replaced
by a module that injects low-rank matrices `A, B` into the Q, K, V projections,
computing `ΔW x = (α/r)·B·A·x`. `A` is drawn from N(0,0.02) and `B` is initialized to
zero, so the adapter is a no-op at initialization and the model starts exactly from the
pre-trained one; only the adapters are trainable (~0.7% of 124M parameters) while the
backbone is frozen. The scaling `α/r` decouples the adaptation strength from the
learning rate, so we search the rank `r ∈ {4,8,16}` and then `α`. Fine-tuning converges
in a few epochs, far faster than training from scratch.

## 3. Results

Table 1 traces Part A: every kept modification lowers the dev PPL, and the final model
reaches **test PPL 30.1**. Table 2 reports Part B: PPL decreases monotonically with the
rank, and the best configuration (rank 16, `α=16`) gives **test PPL 19.5** — a ~35%
relative improvement over the from-scratch model while training under 1% of the
parameters. The arc is the project's main message: from scratch, more depth does not
help on PTB (data-limited); the way to exploit a deep 124M model is **pre-training +
LoRA**, which is exactly what Part B does.

The generalization gap (Fig. 1) supports the regularization choices: the kept
no-dropout models show only a small valid−train gap, i.e. they are **not strongly
overfitting**, which is why heavy dropout does not help and our tight gradient clip
already suffices as a regularizer.

**Table 1 — Part 1.A, best model of each incremental step** (dev PPL; test for the
final). Final model: `d_model=384`, 2 layers, FFN=1536, dropout 0.1, weight tying.

| Step | Config | Param | Dev | Test |
|---|---|---|---|---|
| 0 | learning rate (1e-3) | 29.2M | 36.91 | – |
| 1 | architecture (d384/l2) | 42.6M | 36.31 | – |
| 2 | dropout 0.1 | 42.6M | 36.24 | – |
| 3 | weight tying | 23.3M | **33.56** | **30.08** |

**Table 2 — Part 1.B, LoRA fine-tuning** (lr 5e-4; “Train.” = trainable adapter params).

| Rank r | α | Train. | Dev | Test |
|---|---|---|---|---|
| 4 | 16 | 0.22M | 23.81 | – |
| 8 | 16 | 0.44M | 22.61 | – |
| 16 | 32 | 0.88M | 21.71 | – |
| 16 | 8 | 0.88M | 21.60 | – |
| 16 | 16 | 0.88M | **21.48** | **19.53** |

**Figure 1 — Overfitting gap per run** (`valid − train` PPL, from `gap_report.py`).
Dropout runs are greyed because their training loss is logged with dropout active and
their gap is therefore not reliable; the clean (no-dropout) runs show small gaps,
confirming we are not strongly overfitting. *(Generated on the VM:
`python gap_report.py` → `reports/figures/overfitting_gaps.pdf`.)*

## 4. References
- E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022.
- O. Press, L. Wolf, “Using the Output Embedding to Improve Language Models,” EACL 2017.
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.
- A. Radford et al., “Language Models are Unsupervised Multitask Learners,” 2019 (GPT-2).

*Declaration of AI use: a generative-AI assistant was used to help structure the
experiment pipeline and edit the language of this report; all architectural choices,
experiments and results are the author’s own.*

---

### Note per noi (da decidere insieme, NON vanno nel report)
- **Peso A vs B**: controlla la suddivisione punti nell'assegnazione e dai più spazio
  alla parte che vale di più (qui A e B sono bilanciate).
- **Originalità da valorizzare** (alza il voto): (a) la storia init-fix (std del
  residuo) col confronto pre/post-fix; (b) grad_clip↔dropout; (c) eventuale figura —
  es. magnitudo dell'update LoRA per layer, o il gap di overfitting (`gap_report`).
- Se serve tagliare per stare in 1 pagina: comprimere il §2 Part A (le due note
  metodologiche in una riga ciascuna) e tenere intatte le tabelle.
