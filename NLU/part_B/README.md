# Parte 2.B — NLU fine-tuning (GPT2 / BERT pre-addestrati)

Fine-tuning multi-task (intent + slot) di un modello pre-addestrato su ATIS, con
allineamento delle label di slot ai sub-token. File:

- `model.py`     : `JointNLUTransformer` (backbone HF + teste slot/intent) + `build_joint_model`
- `functions.py` : training/eval multi-task (eval conll sugli slot ricostruiti a parola) + `fit`
- `utils.py`     : ATIS + tokenizer HF + allineamento label + collate
- `conll.py`     : scorer F1 a chunk (dai lab)
- `main.py`      : carica i pesi migliori (`bin/`) e stampa intent acc + slot F1 su test
- `bin/`         : `best_model.pt` (pesi fine-tunati) + `label_maps.json` (slot2id/intent2id)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py` (scarica il backbone pre-addestrato al primo avvio).
