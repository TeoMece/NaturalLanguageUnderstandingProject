# Parte 2.A — NLU (intent classification + slot filling, GPT2 da zero)

GPT2 decoder-only (stile Parte 1) con due teste joint su ATIS: intent (dal token CLS)
e slot (per token). File:

- `model.py`     : backbone GPT2 + `GPT2JointIAS` (teste slot/intent) + `init_weights`
- `functions.py` : training/eval multi-task (loss CE_intent+CE_slot), eval con conll
                   (slot F1) + accuracy (intent), `fit`, `pick_device`
- `utils.py`     : caricamento ATIS, `Lang` (vocabolari + CLS), Dataset, collate
- `conll.py`     : scorer F1 a livello di chunk (dai lab del corso)
- `main.py`      : carica i pesi migliori (`bin/`) e stampa intent acc + slot F1 su test
- `bin/`         : `best_model.pt` (pesi) e `lang.json` (vocabolari usati in training)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py`.
