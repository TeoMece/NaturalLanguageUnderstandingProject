"""Parte 1.B — orchestrazione di una singola run LoRA: dalla config agli artefatti.

E' il gemello di `lm_pipeline.experiment` (1.A), ma costruisce un GPT2 PRE-ADDESTRATO
con adapter LoRA invece di un GPT2 da zero. Riusa per intero lo strato indipendente
dal modello: i dati (`lm_pipeline.data`) e il tracking (`lm_pipeline.tracking`).

Artefatti su disco (sotto runs_root/name/):
    metrics.json   <- metriche finali (schema compatibile con aggregate.collect_runs)
    curves.csv     <- storico per epoca
    curves.pdf     <- grafico della valid-PPL
    adapters.pt    <- SOLO i pesi degli adapter LoRA (file piccolo): bastano per
                      ricostruire il modello finetunato sopra il GPT2 pre-addestrato.
"""
import os
import time

import torch

# Riuso da 1.A (indipendenti dal modello):
from lm_pipeline import data as data_mod
from lm_pipeline import tracking

from .model_lora import GPT2_LoRA, apply_lora_freezing, param_stats
from .train_lora import fit, eval_loop, pick_device


def already_done(runs_root, name):
    """True se la run e' gia' completata (metrics.json presente). Per resumabilita'."""
    return os.path.exists(os.path.join(runs_root, name, "metrics.json"))


def adapter_state_dict(model):
    """Estrae dal modello SOLO i parametri degli adapter LoRA (chiavi con 'lora_').

    E' cio' che serve salvare: i pesi base sono quelli pre-addestrati di GPT2 e si
    riottengono con `from_pretrained`. Salvare solo gli adapter rende il checkpoint
    minuscolo (pochi MB invece di ~500).
    """
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def run_experiment_lora(cfg, runs_root, dataset_dir, tokenizer=None):
    """Esegue una run LoRA completa e salva gli artefatti. Ritorna il path della run.

    Flusso: resumabilita' -> seed/device -> tokenizer -> dataloader (subset solo in
    dev) -> GPT2_LoRA.from_pretrained + freezing -> fit (early stopping) ->
    eval su test (mode 'final') -> salvataggio curve, adapters, metrics.

    `cfg` e' una config gia' risolta (no sweep). Sezione `model`:
        name  : nome del modello HuggingFace (es. 'openai-community/gpt2')
        rank  : rango LoRA
        alpha : scala LoRA
    """
    exp = cfg.get("experiment", {})
    name = exp.get("name", "partB")

    if already_done(runs_root, name):
        return os.path.join(runs_root, name)

    tracking.set_seed(exp.get("seed", 42))
    device = pick_device(exp.get("device", "auto"))

    if tokenizer is None:
        tokenizer = data_mod.get_tokenizer()

    # ---- Dati (riuso totale di 1.A) ----
    cfg_data = dict(cfg.get("data", {}))
    is_final = cfg.get("mode", "dev") == "final"
    if is_final:
        cfg_data["fraction"] = 1.0
        cfg_data["max_samples"] = None
    train_dl, dev_dl, test_dl, n_train = data_mod.build_dataloaders(
        cfg_data, dataset_dir, tokenizer
    )

    # ---- Modello: GPT2 pre-addestrato + adapter LoRA ----
    m = cfg.get("model", {})
    model_name = m.get("name", "openai-community/gpt2")
    model = GPT2_LoRA.from_pretrained(
        model_name, rank=m.get("rank", 8), alpha=m.get("alpha", 16)
    )
    # Congela tutto tranne gli adapter: solo i lora_* verranno allenati.
    apply_lora_freezing(model)
    param_stats(model)  # stampa il risparmio (trainable vs frozen)

    # ---- Training (early stopping su valid-PPL) ----
    t0 = time.time()
    hist = fit(model, train_dl, dev_dl, cfg.get("optim", {}), device,
               pad_id=tokenizer.pad_token_id, show_progress=True)
    elapsed = time.time() - t0

    # ---- Valutazione su test (solo mode 'final') ----
    if is_final:
        test_ppl, _ = eval_loop(test_dl, model, device, pad_id=tokenizer.pad_token_id)
    else:
        test_ppl = None

    # ---- Salvataggio artefatti ----
    run_dir = tracking.make_run_dir(runs_root, name)
    tracking.save_curves(run_dir, hist)
    # Salviamo SOLO gli adapter (file piccolo); il resto e' il GPT2 pre-addestrato.
    torch.save(adapter_state_dict(model), os.path.join(run_dir, "adapters.pt"))

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        "name": name,
        "mode": cfg.get("mode", "dev"),
        "best_valid_ppl": hist["best_ppl"],
        "test_ppl": test_ppl,
        "epochs_run": len(hist["valid_ppl"]),
        "n_train_sentences": n_train,
        "n_params": n_total,             # totale (per la tabella di aggregate)
        "n_trainable": n_trainable,      # adapter LoRA: la frazione davvero allenata
        "device": device,
        "seconds": round(elapsed, 1),
        "seed": exp.get("seed", 42),
        "git_hash": tracking.git_hash(),
        "config": cfg,
    }
    tracking.save_metrics(run_dir, metrics)
    return run_dir
