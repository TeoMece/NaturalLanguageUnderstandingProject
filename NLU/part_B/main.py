"""Entry point di consegna (Parte 2.B, NLU) - generato dall'export.

Ricostruisce tokenizer + modello joint dal model.name salvato, carica i pesi fine-tunati
e i vocabolari, e stampa intent accuracy + slot F1 su test (ATIS).
"""
import json
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_joint_model
from utils import get_tokenizer, load_atis, IntentsAndSlotsBERTGPT2, collate_fn, IGNORE
from functions import evaluate, pick_device

BEST_CONFIG = {'model': {'name': 'openai-community/gpt2', 'intent_pool': None}, 'data': {'batch_size': 16}}

if __name__ == "__main__":
    device = pick_device("auto")
    maps = json.load(open("bin/label_maps.json"))
    slot2id, intent2id = maps["slot2id"], maps["intent2id"]
    id2slot = {v: k for k, v in slot2id.items()}
    model_name = BEST_CONFIG["model"]["name"]

    tok = get_tokenizer(model_name)
    _, test_raw = load_atis("dataset/ATIS")
    test_dl = DataLoader(IntentsAndSlotsBERTGPT2(test_raw, tok, slot2id, intent2id),
                         batch_size=BEST_CONFIG["data"].get("batch_size", 16),
                         collate_fn=partial(collate_fn, pad_id=tok.pad_token_id))

    model = build_joint_model(model_name, num_slots=len(slot2id), num_intents=len(intent2id),
                              intent_pool=BEST_CONFIG["model"].get("intent_pool"))
    model.load_state_dict(torch.load("bin/best_model.pt", map_location=device))
    model.to(device)

    crit_slot = nn.CrossEntropyLoss(ignore_index=IGNORE)
    crit_intent = nn.CrossEntropyLoss()
    f1, acc, _ = evaluate(test_dl, crit_slot, crit_intent, model, id2slot, device)
    print(f"TEST  slot F1: {f1*100:.2f}%  |  intent acc: {acc*100:.2f}%")
