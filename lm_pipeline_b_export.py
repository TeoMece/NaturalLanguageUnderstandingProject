"""Export della Parte 1.B verso la struttura di consegna standalone (LM/part_B).

Come per 1.A, copia i moduli e GENERA un main.py con gli iperparametri congelati.
A differenza di 1.A, il modello e' un GPT2 PRE-ADDESTRATO + adapter LoRA, quindi
main.py ricostruisce il modello con `from_pretrained` e carica SOLO gli adapter.

Nota importante sull'auto-contenimento: `train_lora.py` (nel repo di sviluppo)
importa `pick_device` da `lm_pipeline.train` per non duplicare codice. Ma la
consegna deve girare da sola, senza il package `lm_pipeline`. Per questo l'export
RISCRIVE quell'import inline (sostituendolo con la definizione della funzione).
"""
import os
import json
import glob
import shutil


# Sostituzione: l'import da lm_pipeline -> definizione locale di pick_device, cosi'
# functions.py e' autosufficiente nella consegna.
_PICK_DEVICE_INLINE = '''# pick_device inline (nella consegna non importiamo lm_pipeline): sceglie il
# miglior device disponibile (cuda > mps > cpu) quando si passa "auto".
def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"'''

_IMPORT_LINE = "from lm_pipeline.train import pick_device  # noqa: F401  (riesportato per comodita')"


_MAIN_TEMPLATE = '''"""Entry point di consegna (Parte 1.B, LoRA) - generato dall'export.

Ricostruisce il GPT2 pre-addestrato con adapter LoRA, carica gli adapter allenati
da bin/adapters.pt e stampa la perplexity su test.
"""
import torch

from model import GPT2_LoRA, apply_lora_freezing
from utils import build_dataloaders, get_tokenizer
from functions import eval_loop, pick_device

# Iperparametri della configurazione migliore (congelati dall'export)
BEST_CONFIG = {best_cfg}

if __name__ == "__main__":
    device = pick_device("auto")
    tokenizer = get_tokenizer()
    m = BEST_CONFIG["model"]
    _, _, test_dl, _ = build_dataloaders(
        {{"fraction": 1.0, "batch_size": BEST_CONFIG["data"]["batch_size"]}},
        "dataset/PennTreeBank", tokenizer)
    # Ricostruisce GPT2 pre-addestrato + adapter, poi carica i pesi degli adapter.
    model = GPT2_LoRA.from_pretrained(m["name"], rank=m["rank"], alpha=m["alpha"])
    apply_lora_freezing(model)
    adapters = torch.load("bin/adapters.pt", map_location=device)
    model.load_state_dict(adapters, strict=False)  # carica solo i lora_*
    model.to(device)
    ppl, _ = eval_loop(test_dl, model, device, pad_id=tokenizer.pad_token_id)
    print(f"Test PPL: {{ppl:.2f}}")
'''


_README = '''# Parte 1.B — GPT2 pre-addestrato + LoRA

Fine-tuning di GPT2 (HuggingFace) sul Penn Treebank tramite LoRA implementato a mano
(adapter a basso rango su Q, K, V; solo gli adapter sono allenati). File:

- `model.py`     : `LoRALinear`, `CustomGPT2Attention`, `GPT2_LoRA`, freezing, param_stats
- `functions.py` : loop di train/eval (loss interna HuggingFace) + `fit` + `pick_device`
- `utils.py`     : caricamento/preprocessing del Penn Treebank e tokenizer GPT2
- `main.py`      : carica gli adapter migliori da `bin/adapters.pt` e stampa la test-PPL
- `bin/`         : `adapters.pt` (solo i pesi degli adapter; i pesi base = GPT2 pre-addestrato)

Esecuzione: `python main.py` (scarica GPT2 da HuggingFace al primo avvio).
'''


def export_part_b(root, out_dir=None, best_cfg=None, best_ckpt=None):
    """Genera LM/part_B standalone. `best_cfg`/`best_ckpt` opzionali per i test."""
    out_dir = out_dir or os.path.join(root, "LM", "part_B")
    os.makedirs(os.path.join(out_dir, "bin"), exist_ok=True)

    # model.py: copia verbatim (gia' standalone, dipende solo da torch/transformers)
    shutil.copy(os.path.join(root, "lm_pipeline_b", "model_lora.py"),
                os.path.join(out_dir, "model.py"))

    # utils.py: il data layer di 1.A, gia' standalone
    shutil.copy(os.path.join(root, "lm_pipeline", "data.py"),
                os.path.join(out_dir, "utils.py"))

    # functions.py: train_lora.py con l'import da lm_pipeline riscritto inline
    with open(os.path.join(root, "lm_pipeline_b", "train_lora.py")) as f:
        functions_src = f.read()
    functions_src = functions_src.replace(_IMPORT_LINE, _PICK_DEVICE_INLINE)
    with open(os.path.join(out_dir, "functions.py"), "w") as f:
        f.write(functions_src)

    # best_cfg: in uso reale legge la run __FINAL piu' recente sotto runs_b/
    if best_cfg is None:
        finals = sorted(glob.glob(os.path.join(root, "runs_b", "*FINAL*", "metrics.json")))
        best_cfg = (json.load(open(finals[-1]))["config"] if finals
                    else {"model": {"name": "openai-community/gpt2", "rank": 8, "alpha": 16},
                          "data": {"batch_size": 8}})
        # ...e prende gli adapter dalla STESSA run FINAL (altrimenti bin/ resterebbe vuoto)
        if best_ckpt is None and finals:
            cand = os.path.join(os.path.dirname(finals[-1]), "adapters.pt")
            best_ckpt = cand if os.path.exists(cand) else None

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(best_cfg=repr(best_cfg)))

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_README)

    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, os.path.join(out_dir, "bin", "adapters.pt"))
    return out_dir
