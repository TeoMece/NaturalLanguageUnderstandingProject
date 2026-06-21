# Risultati PRE-FIX (snapshot congelato)

Questa cartella è uno **snapshot dei report ottenuti prima** del fix
all'inizializzazione dei pesi (vedi sotto). Conservata per confronto storico:
**non rigenerare qui** — i nuovi report post-fix vanno in `reports/`.

## Perché esiste
`init_weights` inizializzava solo i `nn.Linear` (uniform ±0.01) e **ignorava
gli `nn.Embedding`** (che restavano all'init di default N(0,1)). Con il weight
tying, la matrice condivisa `token_embed.weight == lm_head.weight` veniva
sovrascritta dall'init uniforme del Linear → embedding ~170x più piccoli nel
modello con tying rispetto a quello senza. Confronto **viziato**: la differenza
non era solo il tying, ma anche la scala di init dell'embedding.

## Cosa contengono questi risultati
Esperimenti 0–2 (baseline lr, architettura, dropout) e parte del 3 (weight
tying) girati con l'init buggato. Validi come riferimento ma **non confrontabili**
con i risultati post-fix.

## Cosa cambia il fix
Init coerente e tying-aware: `nn.Embedding` inizializzati con `normal(0, 0.02)`
(scala sensata, stile GPT-2); i `Linear` non-tied restano all'uniforme del lab;
il `Linear` con peso condiviso (tying) **non** viene più sovrascritto.
→ richiede di **ri-eseguire tutta la Parte A** da zero.
