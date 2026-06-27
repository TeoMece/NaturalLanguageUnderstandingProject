"""Entry point di consegna (Parte 2.B, NLU) - generato dall'export.

Valuta su test ATIS i DUE modelli migliori (GPT2 decoder e BERT encoder): per ciascuno
ricostruisce tokenizer + modello joint, carica i pesi fine-tunati da bin/, e stampa
intent accuracy + slot F1 su test.
"""
import json
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_joint_model
from utils import get_tokenizer, load_atis, IntentsAndSlotsBERTGPT2, collate_fn, IGNORE
from functions import evaluate, pick_device

# I modelli consegnati: name (HF), intent_pool, batch_size, file dei pesi in bin/.
MODELS = [{'name': 'google-bert/bert-base-uncased', 'intent_pool': None, 'batch_size': 32, 'ckpt': 'bert-base-uncased.pt'}, {'name': 'openai-community/gpt2', 'intent_pool': None, 'batch_size': 32, 'ckpt': 'gpt2.pt'}]

if __name__ == "__main__":
    device = pick_device("auto")
    maps = json.load(open("bin/label_maps.json"))
    slot2id, intent2id = maps["slot2id"], maps["intent2id"]
    id2slot = {v: k for k, v in slot2id.items()}
    _, test_raw = load_atis("dataset/ATIS")
    crit_slot = nn.CrossEntropyLoss(ignore_index=IGNORE)
    crit_intent = nn.CrossEntropyLoss()

    for m in MODELS:
        tok = get_tokenizer(m["name"])
        test_dl = DataLoader(
            IntentsAndSlotsBERTGPT2(test_raw, tok, slot2id, intent2id),
            batch_size=m.get("batch_size", 32),
            collate_fn=partial(collate_fn, pad_id=tok.pad_token_id))
        model = build_joint_model(m["name"], num_slots=len(slot2id),
                                  num_intents=len(intent2id),
                                  intent_pool=m.get("intent_pool"))
        model.load_state_dict(torch.load("bin/" + m["ckpt"], map_location=device))
        model.to(device)
        f1, acc, _ = evaluate(test_dl, crit_slot, crit_intent, model, id2slot, device)
        print(f"[{m['name']}]  TEST slot F1: {f1*100:.2f}%  |  intent acc: {acc*100:.2f}%")
