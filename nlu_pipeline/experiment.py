"""Orchestrazione di una singola run NLU: dalla config agli artefatti su disco.

Gemello di `lm_pipeline.experiment`, ma per il task joint intent+slot. Resumable
(salta se esiste gia' metrics.json). Artefatti sotto runs_root/name/:
    metrics.json   <- metriche finali (best dev slot F1/intent acc, e test in mode final)
    curves.csv/pdf <- storico per epoca
    best_model.pt  <- pesi del modello alla miglior slot F1 di dev
    lang.json      <- i tre vocabolari (word2id/slot2id/intent2id): servono alla consegna
"""
import os
import json
import time

import torch

from . import config
from . import data as data_mod
from . import tracking
from .model import GPT2JointIAS, init_weights
from .train import pick_device, fit, evaluate, _build_optimizer  # noqa: F401
import torch.nn as nn
from .data import PAD_TOKEN


def already_done(runs_root, name):
    """True se la run e' gia' completata (metrics.json presente)."""
    return os.path.exists(os.path.join(runs_root, name, "metrics.json"))


def run_experiment_nlu(cfg, runs_root, dataset_dir):
    """Esegue una run NLU completa e salva gli artefatti. Ritorna il path della run dir.

    Flusso: resumabilita' -> seed/device -> dataloaders+lang (dev split stratificato)
    -> modello GPT2JointIAS -> init -> fit (early stopping su slot F1 di dev) ->
    in mode 'final' valuta su test -> salva curve, pesi, lang, metrics.
    """
    exp = cfg.get("experiment", {})
    name = exp.get("name", "partA_nlu")
    if already_done(runs_root, name):
        return os.path.join(runs_root, name)

    tracking.set_seed(exp.get("seed", 42))
    device = pick_device(exp.get("device", "auto"))

    # ---- Dati ATIS (dev = split stratificato del train) ----
    cfg_data = dict(cfg.get("data", {}))
    train_dl, dev_dl, test_dl, lang = data_mod.build_dataloaders(cfg_data, dataset_dir)

    # ---- Modello ----
    m = cfg.setdefault("model", {})
    config.resolve_ff_dim(m)  # ff_dim 'auto' -> ff_mult * d_model (riuso da config)
    dropout_cfg = m.get("dropout", {})
    dropout_p = dropout_cfg.get("p", 0.1) if dropout_cfg.get("enabled", False) else 0.0

    model = GPT2JointIAS(
        vocab_size=len(lang.word2id),
        num_slots=max(lang.slot2id.values()) + 1,   # ids 0..K (pad/cls condividono 0)
        num_intents=len(lang.intent2id),
        pos_emb_size=m.get("pos_emb_size", 512),
        d_model=m.get("d_model", 256),
        n_heads=m.get("n_heads", 4),
        num_layers=m.get("num_layers", 2),
        ff_dim=m.get("ff_dim", 1024),
        dropout=dropout_p,
    )
    init_weights(model)

    # ---- Training ----
    t0 = time.time()
    hist = fit(model, train_dl, dev_dl, cfg.get("optim", {}), device, lang, show_progress=True)
    elapsed = time.time() - t0

    # ---- Valutazione su test (solo mode 'final') ----
    is_final = cfg.get("mode", "dev") == "final"
    if is_final:
        crit_slot = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        crit_intent = nn.CrossEntropyLoss()
        test_slot_f1, test_intent_acc, _ = evaluate(test_dl, crit_slot, crit_intent,
                                                    model, lang, device)
    else:
        test_slot_f1, test_intent_acc = None, None

    # ---- Salvataggio artefatti ----
    run_dir = tracking.make_run_dir(runs_root, name)
    tracking.save_curves(run_dir, hist)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    # vocabolari: indispensabili per ricostruire il modello a inference time
    with open(os.path.join(run_dir, "lang.json"), "w") as f:
        json.dump({"word2id": lang.word2id, "slot2id": lang.slot2id,
                   "intent2id": lang.intent2id}, f)

    metrics = {
        "name": name,
        "mode": cfg.get("mode", "dev"),
        "best_dev_slot_f1": hist["best_dev_slot_f1"],
        "best_dev_intent_acc": hist["best_dev_intent_acc"],
        "test_slot_f1": test_slot_f1,
        "test_intent_acc": test_intent_acc,
        "best_epoch": hist["best_epoch"],
        "epochs_run": len(hist["dev_slot_f1"]),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "seconds": round(elapsed, 1),
        "seed": exp.get("seed", 42),
        "git_hash": tracking.git_hash(),
        "config": cfg,
    }
    tracking.save_metrics(run_dir, metrics)
    return run_dir
