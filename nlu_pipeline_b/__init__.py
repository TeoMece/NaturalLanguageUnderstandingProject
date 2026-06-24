"""Pacchetto della Parte 2.B (NLU): fine-tuning di GPT2 e BERT PRE-ADDESTRATI su ATIS,
multi-task (intent classification + slot filling).

Come per 1.B e 2.A, e' un package SEPARATO e **auto-contenuto**: duplica il codice
comune indipendente dal task (config, tracking, lo scorer conll) invece di importarlo,
cosi' gira da solo e si esporta pulito nella consegna `NLU/part_B`.

Differenze rispetto a 2.A:
- modelli **pre-addestrati** (GPT2 decoder, BERT encoder) invece di GPT2 da zero;
- tokenizzazione a **sub-token** (BPE/WordPiece) -> serve allineare le label di slot ai
  sub-token (label sul primo sub-token, -100 sul resto e sui token speciali);
- l'intent si legge dal token `[CLS]` (BERT, in testa) o dall'ultimo token reale (GPT2);
- **fine-tuning completo** (tutti i pesi trainabili), lr piccolo.

Contenuto:
- `data`        : ATIS + tokenizer HF + allineamento label + collate
- `model`       : JointNLUTransformer (backbone HF + teste slot/intent) + build_joint_model
- `train`       : training/eval multi-task (eval con conll ricostruendo gli slot a parola)
- `conll`/`config`/`tracking` : duplicati (scorer F1 / YAML+sweep / seed+run+metrics+curve)
- `experiment`  : una run completa; auto-tagga il modello nel nome run
- `aggregate`   : tabelle/grafici (colonne slot_f1/intent_acc)
"""
