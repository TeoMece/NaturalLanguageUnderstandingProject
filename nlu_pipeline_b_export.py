"""Export della pipeline 2.B verso la consegna standalone (NLU/part_B).

Copia i moduli (rinominati) e GENERA un main.py che ricostruisce tokenizer + modello dal
`model.name` salvato, carica i pesi fine-tunati e i vocabolari (label_maps.json) e stampa
intent accuracy + slot F1 su test. `train.py`(->functions.py) usa import relativi: li
riscriviamo in piatti (la consegna gira senza il package).

Pesi: il fine-tuning completo salva un modello grande (~500MB). `*.pt` e' gitignored ->
il binario va nello ZIP di consegna, non in git.
"""
import os
import json
import glob
import shutil


_IMPORT_REWRITES = [
    ("from .conll import evaluate as conll_evaluate", "from conll import evaluate as conll_evaluate"),
    ("from .data import IGNORE", "from utils import IGNORE"),
]

_MAIN_TEMPLATE = '''"""Entry point di consegna (Parte 2.B, NLU) - generato dall'export.

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

BEST_CONFIG = {best_cfg}

if __name__ == "__main__":
    device = pick_device("auto")
    maps = json.load(open("bin/label_maps.json"))
    slot2id, intent2id = maps["slot2id"], maps["intent2id"]
    id2slot = {{v: k for k, v in slot2id.items()}}
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
    print(f"TEST  slot F1: {{f1*100:.2f}}%  |  intent acc: {{acc*100:.2f}}%")
'''

_README = '''# Parte 2.B — NLU fine-tuning (GPT2 / BERT pre-addestrati)

Fine-tuning multi-task (intent + slot) di un modello pre-addestrato su ATIS, con
allineamento delle label di slot ai sub-token. File:

- `model.py`     : `JointNLUTransformer` (backbone HF + teste slot/intent) + `build_joint_model`
- `functions.py` : training/eval multi-task (eval conll sugli slot ricostruiti a parola) + `fit`
- `utils.py`     : ATIS + tokenizer HF + allineamento label + collate
- `conll.py`     : scorer F1 a chunk (dai lab)
- `main.py`      : carica i pesi migliori (`bin/`) e stampa intent acc + slot F1 su test
- `bin/`         : `best_model.pt` (pesi fine-tunati) + `label_maps.json` (slot2id/intent2id)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py` (scarica il backbone pre-addestrato al primo avvio).
'''


def export_part_b_nlu(root, out_dir=None, best_cfg=None, best_ckpt=None, best_maps=None):
    """Genera NLU/part_B standalone. best_cfg/best_ckpt/best_maps opzionali per i test."""
    out_dir = out_dir or os.path.join(root, "NLU", "part_B")
    os.makedirs(os.path.join(out_dir, "bin"), exist_ok=True)
    src = os.path.join(root, "nlu_pipeline_b")

    shutil.copy(os.path.join(src, "model.py"), os.path.join(out_dir, "model.py"))
    shutil.copy(os.path.join(src, "data.py"), os.path.join(out_dir, "utils.py"))
    shutil.copy(os.path.join(src, "conll.py"), os.path.join(out_dir, "conll.py"))

    with open(os.path.join(src, "train.py")) as f:
        functions_src = f.read()
    for old, new in _IMPORT_REWRITES:
        functions_src = functions_src.replace(old, new)
    with open(os.path.join(out_dir, "functions.py"), "w") as f:
        f.write(functions_src)

    if best_cfg is None:
        finals = sorted(glob.glob(os.path.join(root, "runs_nlu_b", "*FINAL*", "metrics.json")))
        if finals:
            best_cfg = json.load(open(finals[-1]))["config"]
            run_dir = os.path.dirname(finals[-1])
            best_ckpt = best_ckpt or os.path.join(run_dir, "best_model.pt")
            best_maps = best_maps or os.path.join(run_dir, "label_maps.json")
        else:
            best_cfg = {"model": {"name": "openai-community/gpt2", "intent_pool": None},
                        "data": {"batch_size": 16}}

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(best_cfg=repr(best_cfg)))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_README)

    dst_ds = os.path.join(out_dir, "dataset", "ATIS")
    os.makedirs(dst_ds, exist_ok=True)
    for fn in ("train.json", "test.json"):
        shutil.copy(os.path.join(root, "dataset", "ATIS", fn), os.path.join(dst_ds, fn))

    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, os.path.join(out_dir, "bin", "best_model.pt"))
    if best_maps and os.path.exists(best_maps):
        shutil.copy(best_maps, os.path.join(out_dir, "bin", "label_maps.json"))
    return out_dir
