# Parte 1.A — Language Modeling (GPT-2 da zero su Penn Treebank)

GPT-2 decoder-only addestrato da zero su PTB. File:
- `model.py`     : `GPT2` (transformer parametrico) + `init_weights` (residual scaling)
- `functions.py` : device, scheduler warmup+cosine, train/eval, `fit` (early stopping)
- `utils.py`     : lettura PTB, tokenizer, dataloader
- `main.py`      : carica il modello migliore (`bin/best_model.pt`) e stampa la PPL su test
- `bin/`         : `best_model.pt` (pesi del modello finale)
- `dataset/PennTreeBank` : ptb.train/valid/test.txt

Esecuzione: `python main.py`.
