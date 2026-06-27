"""Export della pipeline 2.B verso la consegna standalone (NLU/part_B).

La consegna di 2.B richiede i DUE modelli migliori (GPT2 decoder + BERT encoder).
Copiamo i moduli (rinominati) e GENERIAMO un main.py che, per CIASCUN modello,
ricostruisce tokenizer + modello dal `model.name`, carica i pesi fine-tunati da bin/ e
stampa intent accuracy + slot F1 su test. `train.py`(->functions.py) usa import relativi:
li riscriviamo piatti (la consegna gira senza il package).

Pesi: il fine-tuning completo salva modelli grandi (~440MB l'uno). `*.pt` e' gitignored
-> i binari vanno nello ZIP di consegna, non in git.
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
MODELS = {models}

if __name__ == "__main__":
    device = pick_device("auto")
    maps = json.load(open("bin/label_maps.json"))
    slot2id, intent2id = maps["slot2id"], maps["intent2id"]
    id2slot = {{v: k for k, v in slot2id.items()}}
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
        print(f"[{{m['name']}}]  TEST slot F1: {{f1*100:.2f}}%  |  intent acc: {{acc*100:.2f}}%")
'''

_README = '''# Parte 2.B — NLU fine-tuning (GPT2 + BERT pre-addestrati)

Fine-tuning multi-task (intent + slot) su ATIS con allineamento delle label di slot ai
sub-token. Consegniamo i **due modelli migliori**: GPT2 (decoder) e BERT (encoder). File:

- `model.py`     : `JointNLUTransformer` (backbone HF + teste slot/intent) + `build_joint_model`
- `functions.py` : training/eval multi-task (eval conll sugli slot a parola) + `fit`
- `utils.py`     : ATIS + tokenizer HF + allineamento label + collate
- `conll.py`     : scorer F1 a chunk (dai lab)
- `main.py`      : valuta ENTRAMBI i modelli su test e stampa intent acc + slot F1 per ciascuno
- `bin/`         : `gpt2.pt`, `bert-base-uncased.pt` (pesi) + `label_maps.json` (slot2id/intent2id)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py` (scarica i backbone pre-addestrati al primo avvio).
'''


def export_part_b_nlu(root, out_dir=None, models=None, best_maps=None):
    """Genera NLU/part_B standalone coi DUE modelli migliori (GPT2 + BERT).

    `models` (lista di dict name/intent_pool/batch_size/ckpt) e `best_maps` sono opzionali
    (per i test); in uso reale vengono dedotti dalle run *FINAL* sotto runs_nlu_b/,
    scegliendo per ogni backbone il seed con best_dev_slot_f1 massima.
    """
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

    # Deduci i due modelli migliori dalle run *FINAL*: uno per backbone, scegliendo il
    # seed con best_dev_slot_f1 massima. Copia i pesi in bin/<tag>.pt.
    if models is None:
        models = []
        best = {}   # name -> (score, config, run_dir)
        for mp in glob.glob(os.path.join(root, "runs_nlu_b", "*FINAL*", "metrics.json")):
            m = json.load(open(mp))
            name = m["config"]["model"]["name"]
            score = m.get("best_dev_slot_f1") if m.get("best_dev_slot_f1") is not None else -1
            if name not in best or score > best[name][0]:
                best[name] = (score, m["config"], os.path.dirname(mp))
        for name, (score, cfg, run_dir) in sorted(best.items()):
            tag = name.split("/")[-1]                 # gpt2, bert-base-uncased
            ckpt_name = tag + ".pt"
            ckpt_src = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(ckpt_src):
                shutil.copy(ckpt_src, os.path.join(out_dir, "bin", ckpt_name))
            best_maps = best_maps or os.path.join(run_dir, "label_maps.json")
            models.append({"name": name,
                           "intent_pool": cfg["model"].get("intent_pool"),
                           "batch_size": cfg.get("data", {}).get("batch_size", 32),
                           "ckpt": ckpt_name})
        if not models:   # fallback: nessuna run FINAL trovata
            models = [{"name": "openai-community/gpt2", "intent_pool": None,
                       "batch_size": 32, "ckpt": "gpt2.pt"}]

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(models=repr(models)))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_README)

    dst_ds = os.path.join(out_dir, "dataset", "ATIS")
    os.makedirs(dst_ds, exist_ok=True)
    for fn in ("train.json", "test.json"):
        shutil.copy(os.path.join(root, "dataset", "ATIS", fn), os.path.join(dst_ds, fn))

    if best_maps and os.path.exists(best_maps):
        shutil.copy(best_maps, os.path.join(out_dir, "bin", "label_maps.json"))
    return out_dir
