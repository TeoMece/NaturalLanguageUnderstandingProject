# Parte 2.B — NLU fine-tuning (GPT2 + BERT pre-addestrati)

Fine-tuning multi-task (intent + slot) su ATIS con allineamento delle label di slot ai
sub-token. Consegniamo i **due modelli migliori**: GPT2 (decoder) e BERT (encoder). File:

- `model.py`     : `JointNLUTransformer` (backbone HF + teste slot/intent) + `build_joint_model`
- `functions.py` : training/eval multi-task (eval conll sugli slot a parola) + `fit`
- `utils.py`     : ATIS + tokenizer HF + allineamento label + collate
- `conll.py`     : scorer F1 a chunk (dai lab)
- `main.py`      : valuta ENTRAMBI i modelli su test e stampa intent acc + slot F1 per ciascuno
- `bin/`         : `gpt2.pt`, `bert-base-uncased.pt` (pesi) + `label_maps.json` (slot2id/intent2id)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py` (scarica i backbone pre-addestrati al primo avvio).
