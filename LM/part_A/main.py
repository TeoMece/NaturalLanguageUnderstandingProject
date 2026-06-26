"""Entry point di consegna (Parte 1.A) - generato dalla pipeline.

Carica il modello migliore e stampa la perplexity su test.
"""
import torch
import torch.nn as nn

from model import GPT2, init_weights
from utils import build_dataloaders, get_tokenizer
from functions import evaluate, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {'experiment': {'name': 'partA_weighttying__FINAL', 'seed': 42, 'device': 'auto', 'tensorboard': False}, 'data': {'fraction': 1.0, 'max_samples': None, 'subset_seed': 42, 'batch_size': 32}, 'model': {'pos_emb_size': 1024, 'd_model': 512, 'n_heads': 8, 'num_layers': 2, 'ff_dim': 1024, 'ff_mult': 2, 'weight_tying': True, 'dropout': {'enabled': True, 'p': 0.1}}, 'optim': {'lr': 0.0005, 'optimizer': 'adamw', 'epochs': 50, 'grad_clip': 1.0, 'patience': 6, 'scheduler': {'enabled': True, 'type': 'warmup_cosine', 'warmup_steps': 200}}, 'mode': 'final'}

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
