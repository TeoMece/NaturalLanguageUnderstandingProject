"""Export della pipeline NLU verso la struttura di consegna (NLU/part_A).

Come per la Parte 1: copia i moduli (rinominati come vuole la consegna) e GENERA un
main.py che congela gli iperparametri della config migliore, ricostruisce il modello
e i vocabolari (lang.json) e stampa intent accuracy + slot F1 su test.

I moduli `nlu_pipeline/train.py` (->functions.py) usano import relativi
(`from .conll import ...`, `from .data import ...`): qui li riscriviamo in import
"piatti" (`from conll import ...`, `from utils import ...`) perche' la consegna gira
da sola, senza il package `nlu_pipeline`.
"""
import os
import json
import glob
import shutil


_MAIN_TEMPLATE = '''"""Entry point di consegna (Parte 2.A, NLU) - generato dall'export.

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
BEST_CONFIG = {best_cfg}

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
    print(f"TEST  slot F1: {{f1*100:.2f}}%  |  intent acc: {{acc*100:.2f}}%")
'''


_README = '''# Parte 2.A — NLU (intent classification + slot filling, GPT2 da zero)

GPT2 decoder-only (stile Parte 1) con due teste joint su ATIS: intent (dal token CLS)
e slot (per token). File:

- `model.py`     : backbone GPT2 + `GPT2JointIAS` (teste slot/intent) + `init_weights`
- `functions.py` : training/eval multi-task (loss CE_intent+CE_slot), eval con conll
                   (slot F1) + accuracy (intent), `fit`, `pick_device`
- `utils.py`     : caricamento ATIS, `Lang` (vocabolari + CLS), Dataset, collate
- `conll.py`     : scorer F1 a livello di chunk (dai lab del corso)
- `main.py`      : carica i pesi migliori (`bin/`) e stampa intent acc + slot F1 su test
- `bin/`         : `best_model.pt` (pesi) e `lang.json` (vocabolari usati in training)
- `dataset/ATIS` : train/test json

Esecuzione: `python main.py`.
'''

_IMPORT_REWRITES = [
    ("from .conll import evaluate as conll_evaluate", "from conll import evaluate as conll_evaluate"),
    ("from .data import PAD_TOKEN", "from utils import PAD_TOKEN"),
]


def export_part_a_nlu(root, out_dir=None, best_cfg=None, best_ckpt=None, best_lang=None):
    """Genera NLU/part_A standalone. best_cfg/best_ckpt/best_lang opzionali per i test."""
    out_dir = out_dir or os.path.join(root, "NLU", "part_A")
    os.makedirs(os.path.join(out_dir, "bin"), exist_ok=True)
    src = os.path.join(root, "nlu_pipeline")

    # moduli gia' standalone
    shutil.copy(os.path.join(src, "model.py"), os.path.join(out_dir, "model.py"))
    shutil.copy(os.path.join(src, "data.py"), os.path.join(out_dir, "utils.py"))
    shutil.copy(os.path.join(src, "conll.py"), os.path.join(out_dir, "conll.py"))

    # functions.py = train.py con import relativi riscritti in piatti
    with open(os.path.join(src, "train.py")) as f:
        functions_src = f.read()
    for old, new in _IMPORT_REWRITES:
        functions_src = functions_src.replace(old, new)
    with open(os.path.join(out_dir, "functions.py"), "w") as f:
        f.write(functions_src)

    # best_cfg: in uso reale legge la run __FINAL piu' recente sotto runs_nlu/
    if best_cfg is None:
        finals = sorted(glob.glob(os.path.join(root, "runs_nlu", "*FINAL*", "metrics.json")))
        if finals:
            best_cfg = json.load(open(finals[-1]))["config"]
            run_dir = os.path.dirname(finals[-1])
            best_ckpt = best_ckpt or os.path.join(run_dir, "best_model.pt")
            best_lang = best_lang or os.path.join(run_dir, "lang.json")
        else:
            best_cfg = {"model": {"d_model": 256, "n_heads": 4, "num_layers": 2,
                                  "ff_dim": 1024, "pos_emb_size": 512},
                        "data": {"batch_size": 64}}

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(_MAIN_TEMPLATE.format(best_cfg=repr(best_cfg)))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_README)

    # dataset ATIS
    dst_ds = os.path.join(out_dir, "dataset", "ATIS")
    os.makedirs(dst_ds, exist_ok=True)
    for fn in ("train.json", "test.json"):
        shutil.copy(os.path.join(root, "dataset", "ATIS", fn), os.path.join(dst_ds, fn))

    # pesi e vocabolari (se disponibili)
    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, os.path.join(out_dir, "bin", "best_model.pt"))
    if best_lang and os.path.exists(best_lang):
        shutil.copy(best_lang, os.path.join(out_dir, "bin", "lang.json"))
    return out_dir
