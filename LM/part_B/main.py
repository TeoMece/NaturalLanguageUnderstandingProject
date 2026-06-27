"""Entry point di consegna (Parte 1.B, LoRA) - generato dall'export.

Ricostruisce il GPT2 pre-addestrato con adapter LoRA, carica gli adapter allenati
da bin/adapters.pt e stampa la perplexity su test.
"""
import torch

from model import GPT2_LoRA, apply_lora_freezing
from utils import build_dataloaders, get_tokenizer
from functions import eval_loop, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {'experiment': {'name': 'partB_FINAL', 'seed': 42, 'device': 'auto'}, 'data': {'fraction': 1.0, 'max_samples': None, 'subset_seed': 42, 'batch_size': 32}, 'model': {'name': 'openai-community/gpt2', 'rank': 16, 'alpha': 32}, 'optim': {'lr': 0.0005, 'optimizer': 'adamw', 'epochs': 30, 'grad_clip': 1.0, 'patience': 3}, 'mode': 'final'}

if __name__ == "__main__":
    device = pick_device("auto")
    tokenizer = get_tokenizer()
    m = BEST_CONFIG["model"]
    _, _, test_dl, _ = build_dataloaders(
        {"fraction": 1.0, "batch_size": BEST_CONFIG["data"]["batch_size"]},
        "dataset/PennTreeBank", tokenizer)
    # Ricostruisce GPT2 pre-addestrato + adapter, poi carica i pesi degli adapter.
    model = GPT2_LoRA.from_pretrained(m["name"], rank=m["rank"], alpha=m["alpha"])
    apply_lora_freezing(model)
    adapters = torch.load("bin/adapters.pt", map_location=device)
    model.load_state_dict(adapters, strict=False)  # carica solo i lora_*
    model.to(device)
    ppl, _ = eval_loop(test_dl, model, device, pad_id=tokenizer.pad_token_id)
    print(f"Test PPL: {ppl:.2f}")
