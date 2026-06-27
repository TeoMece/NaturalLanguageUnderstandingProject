"""Export della pipeline verso la struttura di consegna rigida (LM/part_A).

Copia lo strato PURO (model.py, data.py->utils.py, train.py->functions.py), che
e' auto-contenuto, e GENERA un main.py che congela inline gli iperparametri della
config migliore, carica il checkpoint e stampa la PPL. Lo strato di orchestrazione
non viene mai copiato: i file risultanti girano standalone.
"""
import os
import json
import shutil


_MAIN_TEMPLATE = '''"""Entry point di consegna (Parte 1.A) - generato dalla pipeline.

Carica il modello migliore e stampa la perplexity su test.
"""
import torch
import torch.nn as nn

from model import GPT2, init_weights
from utils import build_dataloaders, get_tokenizer
from functions import evaluate, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {best_cfg}

if __name__ == "__main__":
    device = pick_device("auto")
    tokenizer = get_tokenizer()
    m = BEST_CONFIG["model"]
    _, _, test_dl, _ = build_dataloaders(
        {{"fraction": 1.0, "batch_size": BEST_CONFIG["data"]["batch_size"]}},
        "dataset/PennTreeBank", tokenizer)
    dropout = m.get("dropout", {{}})
    model = GPT2(vocab_size=len(tokenizer), pos_emb_size=m.get("pos_emb_size", 1024),
                 d_model=m["d_model"], n_heads=m["n_heads"], num_layers=m["num_layers"],
                 ff_dim=m["ff_dim"],
                 dropout=(dropout.get("p", 0.1) if dropout.get("enabled") else 0.0),
                 weight_tying=m.get("weight_tying", False)).to(device)
    model.load_state_dict(torch.load("bin/best_model.pt", map_location=device))
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ppl, _ = evaluate(test_dl, crit, model, device)
    print(f"Test PPL: {{ppl:.2f}}")
'''


def export_part_a(root, out_dir=None, best_cfg=None, best_ckpt=None):
    """Genera LM/part_A standalone. `best_cfg`/`best_ckpt` opzionali per i test."""
    out_dir = out_dir or os.path.join(root, "LM", "part_A")
    os.makedirs(os.path.join(out_dir, "bin"), exist_ok=True)
    src = os.path.join(root, "lm_pipeline")

    # strato puro -> nomi richiesti dalla consegna
    shutil.copy(os.path.join(src, "model.py"), os.path.join(out_dir, "model.py"))
    shutil.copy(os.path.join(src, "data.py"), os.path.join(out_dir, "utils.py"))
    shutil.copy(os.path.join(src, "train.py"), os.path.join(out_dir, "functions.py"))

    if best_cfg is None:
        # in uso reale: legge la run __FINAL piu' recente
        import glob
        finals = sorted(glob.glob(os.path.join(root, "runs", "*FINAL*", "metrics.json")))
        best_cfg = json.load(open(finals[-1]))["config"] if finals else {"model": {}, "data": {"batch_size": 16}}
        # ...e prende il checkpoint dalla STESSA run FINAL (altrimenti bin/ resterebbe vuoto)
        if best_ckpt is None and finals:
            cand = os.path.join(os.path.dirname(finals[-1]), "best_model.pt")
            best_ckpt = cand if os.path.exists(cand) else None

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(best_cfg=repr(best_cfg)))

    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, os.path.join(out_dir, "bin", "best_model.pt"))
    return out_dir
