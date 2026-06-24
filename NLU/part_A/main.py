"""Entry point di consegna (Parte 2.A, NLU) - generato dall'export.

Ricostruisce il modello joint e i vocabolari, carica i pesi migliori e stampa
intent accuracy + slot F1 su test (ATIS).
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import GPT2JointIAS
from utils import load_atis, IntentsAndSlots, collate_fn, Lang, PAD_TOKEN
from functions import evaluate, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {'experiment': {'name': 'partA_nlu_baseline__lr0.0005__FINAL', 'seed': 42, 'device': 'auto'}, 'data': {'dataset': 'ATIS', 'batch_size': 64, 'dev_portion': 0.1, 'split_seed': 42}, 'model': {'pos_emb_size': 512, 'd_model': 256, 'n_heads': 4, 'num_layers': 2, 'ff_dim': 1024, 'ff_mult': 4, 'dropout': {'enabled': False, 'p': 0.1}}, 'optim': {'lr': 0.0005, 'optimizer': 'adamw', 'epochs': 50, 'grad_clip': 1.0, 'patience': 5}, 'mode': 'final'}

if __name__ == "__main__":
    device = pick_device("auto")
    # vocabolari esatti usati in training
    lang = Lang.from_dicts(**json.load(open("bin/lang.json")))
    _, test_raw = load_atis("dataset/ATIS")
    test_dl = DataLoader(IntentsAndSlots(test_raw, lang),
                         batch_size=BEST_CONFIG["data"].get("batch_size", 64),
                         collate_fn=collate_fn)
    m = BEST_CONFIG["model"]
    model = GPT2JointIAS(
        vocab_size=len(lang.word2id),
        num_slots=max(lang.slot2id.values()) + 1,
        num_intents=len(lang.intent2id),
        pos_emb_size=m.get("pos_emb_size", 512),
        d_model=m["d_model"], n_heads=m["n_heads"],
        num_layers=m["num_layers"], ff_dim=m["ff_dim"],
    ).to(device)
    model.load_state_dict(torch.load("bin/best_model.pt", map_location=device))
    crit_slot = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    crit_intent = nn.CrossEntropyLoss()
    f1, acc, _ = evaluate(test_dl, crit_slot, crit_intent, model, lang, device)
    print(f"TEST  slot F1: {f1*100:.2f}%  |  intent acc: {acc*100:.2f}%")
