# Parte 1.B — GPT2 pre-addestrato + LoRA

Fine-tuning di GPT2 (HuggingFace) sul Penn Treebank tramite LoRA implementato a mano
(adapter a basso rango su Q, K, V; solo gli adapter sono allenati). File:

- `model.py`     : `LoRALinear`, `CustomGPT2Attention`, `GPT2_LoRA`, freezing, param_stats
- `functions.py` : loop di train/eval (loss interna HuggingFace) + `fit` + `pick_device`
- `utils.py`     : caricamento/preprocessing del Penn Treebank e tokenizer GPT2
- `main.py`      : carica gli adapter migliori da `bin/adapters.pt` e stampa la test-PPL
- `bin/`         : `adapters.pt` (solo i pesi degli adapter; i pesi base = GPT2 pre-addestrato)

Esecuzione: `python main.py` (scarica GPT2 da HuggingFace al primo avvio).
