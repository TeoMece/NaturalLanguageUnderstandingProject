"""Pacchetto della Parte 1.B del progetto LM: fine-tuning di GPT2 pre-addestrato
con LoRA (Low Rank Adaptation) implementato a mano.

Questo package e' volutamente SEPARATO da `lm_pipeline` (Parte 1.A, GPT2 da zero):
- 1.A resta invariato (i suoi test non vanno toccati);
- 1.B riusa di 1.A cio' che e' indipendente dal modello: il data layer
  (`lm_pipeline.data`), la gestione config (`lm_pipeline.config`), il tracking
  (`lm_pipeline.tracking`) e l'aggregazione report (`lm_pipeline.aggregate`).

Cosa contiene di NUOVO (cio' che 1.A non ha):
- `model_lora`      : adapter LoRA + GPT2 pre-addestrato HuggingFace con LoRA su Q/K/V
- `train_lora`      : loop train/eval basati sulla loss interna di HF + early stopping
- `experiment_lora` : una run LoRA completa (colla tra data/model/train e tracking)
- `report_b`        : tabelle/grafici delle run di B (riusa le funzioni di aggregate)
"""
