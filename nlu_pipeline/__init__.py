"""Pacchetto della Parte 2 del progetto (NLU): intent classification + slot filling
su ATIS, in multi-task, a partire dal GPT2 della Parte 1.

Come per la Parte 1.B, questo package e' volutamente SEPARATO da `lm_pipeline` ed e'
**auto-contenuto**: invece di importare da `lm_pipeline`, **duplica** il codice comune
che e' indipendente dal task (gestione config, tracking, e il backbone GPT2). Cosi' la
pipeline NLU gira da sola e si esporta pulita nella consegna `NLU/part_A`.

Cosa contiene:
- `data`        : ATIS (Lang con token CLS, dev split stratificato, Dataset, collate)
- `model`       : backbone GPT2 (Parte 1) + due teste (slot per-token, intent dal CLS)
- `train`       : training/eval multi-task (loss CE_intent + CE_slot), eval con conll
                  (slot F1) + accuracy (intent), early stopping su slot F1 di dev
- `conll`       : scorer F1 a livello di chunk, copiato verbatim dai lab del corso
- `config`      : lettura YAML + sweep (duplicato da lm_pipeline)
- `tracking`    : seed, cartelle run, metrics.json, curve (duplicato da lm_pipeline)
- `experiment`  : una run NLU completa (colla tra data/model/train e tracking)
- `aggregate`   : tabelle/grafici delle run (colonne slot_f1 / intent_acc)

Ciclo 1 (questo): scaffold + Parte 2.A (GPT2 da zero). La Parte 2.B (fine-tuning di
GPT2/BERT pre-addestrati con sub-tokenizzazione) arrivera' in un secondo momento.
"""
