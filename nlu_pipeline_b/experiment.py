"""Orchestrazione di una singola run 2.B: fine-tuning di un modello HF su ATIS.

Auto-tag del modello nel nome run: la cartella si chiama `<experiment.name>__<tag>` con
`tag = model.name.split('/')[-1]` (es. partB_nlu__gpt2, partB_nlu__bert-base-uncased,
partB_nlu__bert-large-uncased). Cosi' cambiando `model.name` nel config la run e'
automaticamente distinta dalla base, senza collisioni / skip di resumability.
"""
import os
import json
import time

import torch
import torch.nn as nn

from . import data as data_mod
from . import tracking
from .model import build_joint_model
from .train import pick_device, fit, evaluate
from .data import IGNORE


def _run_name(cfg):
    base = cfg.get("experiment", {}).get("name", "partB_nlu")
    tag = cfg.get("model", {}).get("name", "model").split("/")[-1]
    return f"{base}__{tag}"


def already_done(runs_root, name):
    return os.path.exists(os.path.join(runs_root, name, "metrics.json"))


def run_experiment_nlu_b(cfg, runs_root, dataset_dir):
    """Esegue una run di fine-tuning completa e salva gli artefatti. Ritorna la run dir."""
    name = _run_name(cfg)
    if already_done(runs_root, name):
        return os.path.join(runs_root, name)

    exp = cfg.get("experiment", {})
    tracking.set_seed(exp.get("seed", 42))
    device = pick_device(exp.get("device", "auto"))

    m = cfg.get("model", {})
    model_name = m.get("name", "openai-community/gpt2")

    # ---- Dati (tokenizer del modello, sub-tokenizzazione + allineamento) ----
    tokenizer = data_mod.get_tokenizer(model_name)
    train_dl, dev_dl, test_dl, slot2id, intent2id = data_mod.build_dataloaders(
        dict(cfg.get("data", {})), dataset_dir, tokenizer)
    id2slot = {v: k for k, v in slot2id.items()}

    # ---- Modello (backbone pre-addestrato + teste) ----
    model = build_joint_model(model_name, num_slots=len(slot2id),
                              num_intents=len(intent2id),
                              intent_pool=m.get("intent_pool"))

    # ---- Fine-tuning ----
    t0 = time.time()
    hist = fit(model, train_dl, dev_dl, cfg.get("optim", {}), device, id2slot)
    elapsed = time.time() - t0

    # ---- Test (solo mode 'final') ----
    is_final = cfg.get("mode", "dev") == "final"
    if is_final:
        cs = nn.CrossEntropyLoss(ignore_index=IGNORE)
        ci = nn.CrossEntropyLoss()
        test_slot_f1, test_intent_acc, _ = evaluate(test_dl, cs, ci, model, id2slot, device)
    else:
        test_slot_f1, test_intent_acc = None, None

    # ---- Artefatti ----
    run_dir = tracking.make_run_dir(runs_root, name)
    tracking.save_curves(run_dir, hist)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    with open(os.path.join(run_dir, "label_maps.json"), "w") as f:
        json.dump({"slot2id": slot2id, "intent2id": intent2id}, f)

    metrics = {
        "name": name,
        "mode": cfg.get("mode", "dev"),
        "model_name": model_name,
        "intent_pool": model.intent_pool,
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
