"""Entry point di consegna (Parte 1.A) - generato dalla pipeline.

Carica il modello migliore e stampa la perplexity su test.
"""
import torch
import torch.nn as nn

from model import GPT2, init_weights
from utils import build_dataloaders, get_tokenizer
from functions import evaluate, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {'experiment': {'name': 'smoke', 'seed': 1, 'device': 'cpu'}, 'data': {'fraction': 0.01, 'batch_size': 4, 'subset_seed': 1}, 'model': {'pos_emb_size': 128, 'd_model': 32, 'n_heads': 2, 'num_layers': 1, 'ff_dim': 64, 'weight_tying': False, 'dropout': {'enabled': False, 'p': 0.1}}, 'optim': {'lr': 0.001, 'optimizer': 'adamw', 'epochs': 1, 'grad_clip': 1.0, 'patience': 2, 'scheduler': {'enabled': True, 'warmup_steps': 5}}, 'mode': 'dev'}

if __name__ == "__main__":
    device = pick_device("auto")
    tokenizer = get_tokenizer()
    m = BEST_CONFIG["model"]
    _, _, test_dl, _ = build_dataloaders(
        {"fraction": 1.0, "batch_size": BEST_CONFIG["data"]["batch_size"]},
        "dataset/PennTreeBank", tokenizer)
    dropout = m.get("dropout", {})
    model = GPT2(vocab_size=len(tokenizer), pos_emb_size=m.get("pos_emb_size", 1024),
                 d_model=m["d_model"], n_heads=m["n_heads"], num_layers=m["num_layers"],
                 ff_dim=m["ff_dim"],
                 dropout=(dropout.get("p", 0.1) if dropout.get("enabled") else 0.0),
                 weight_tying=m.get("weight_tying", False)).to(device)
    model.load_state_dict(torch.load("bin/best_model.pt", map_location=device))
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ppl, _ = evaluate(test_dl, crit, model, device)
    print(f"Test PPL: {ppl:.2f}")
