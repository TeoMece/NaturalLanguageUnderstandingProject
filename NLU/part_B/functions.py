"""Training/eval del fine-tuning 2.B (multi-task).

Fine-tuning COMPLETO (tutti i parametri trainabili) del backbone pre-addestrato + teste.
Loss = CE(slot, ignore_index=-100) + CE(intent). La valutazione ricostruisce gli slot a
livello di PAROLA prendendo la predizione al primo sub-token di ogni parola (via
word_ids in `meta`) e usa lo scorer conll (slot F1); intent = accuracy.
`pick_device`/`_build_optimizer` duplicati (auto-contenimento).
"""
import copy

import torch
import torch.nn as nn

from conll import evaluate as conll_evaluate
from utils import IGNORE


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_optimizer(model, cfg_optim):
    """AdamW su tutti i parametri (fine-tuning completo), lr piccolo dal config."""
    return torch.optim.AdamW(model.parameters(), lr=cfg_optim["lr"])


def train_one_epoch(loader, optimizer, crit_slot, crit_intent, model, device, grad_clip):
    model.train()
    total = 0.0
    for b in loader:
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        y_slots = b["slot_labels"].to(device)
        intents = b["intents"].to(device)
        optimizer.zero_grad()
        slot_logits, intent_logits = model(ids, attn)
        loss = (crit_slot(slot_logits.permute(0, 2, 1), y_slots)
                + crit_intent(intent_logits, intents))
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def evaluate(loader, crit_slot, crit_intent, model, id2slot, device):
    """Ritorna (slot_f1, intent_acc, loss). Slot ricostruiti a livello di parola."""
    model.eval()
    total, n = 0.0, 0
    ref_slots, hyp_slots, ref_int, hyp_int = [], [], [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            y_slots = b["slot_labels"].to(device)
            intents = b["intents"].to(device)
            slot_logits, intent_logits = model(ids, attn)
            total += (crit_slot(slot_logits.permute(0, 2, 1), y_slots)
                      + crit_intent(intent_logits, intents)).item()
            n += 1
            hyp_int += intent_logits.argmax(1).tolist()
            ref_int += intents.tolist()
            pred = slot_logits.argmax(-1).cpu().tolist()
            for bi, (words, gold, wids) in enumerate(b["meta"]):
                # posizione del primo sub-token di ogni parola
                first = {}
                for pos, wid in enumerate(wids):
                    if wid is not None and wid not in first:
                        first[wid] = pos
                hyp = [(words[w], id2slot[pred[bi][first[w]]] if w in first else "O")
                       for w in range(len(words))]
                ref = [(words[w], gold[w]) for w in range(len(words))]
                hyp_slots.append(hyp)
                ref_slots.append(ref)
    try:
        slot_f1 = conll_evaluate(ref_slots, hyp_slots)["total"]["f"]
    except Exception:
        slot_f1 = 0.0
    intent_acc = sum(a == b for a, b in zip(ref_int, hyp_int)) / max(1, len(ref_int))
    return slot_f1, intent_acc, total / max(1, n)


def fit(model, train_loader, dev_loader, cfg_optim, device, id2slot):
    """Fine-tuning multi-task con early stopping su dev slot F1 (massimizzazione)."""
    model.to(device)
    optimizer = _build_optimizer(model, cfg_optim)
    crit_slot = nn.CrossEntropyLoss(ignore_index=IGNORE)
    crit_intent = nn.CrossEntropyLoss()
    epochs = cfg_optim.get("epochs", 5)
    patience = cfg_optim.get("patience", 3)
    grad_clip = cfg_optim.get("grad_clip", 1.0)

    hist = {"train_loss": [], "dev_slot_f1": [], "dev_intent_acc": []}
    best_f1, best_acc, best_epoch, best_state, no_imp = -1.0, 0.0, 0, None, 0
    for ep in range(epochs):
        tr = train_one_epoch(train_loader, optimizer, crit_slot, crit_intent,
                             model, device, grad_clip)
        f1, acc, _ = evaluate(dev_loader, crit_slot, crit_intent, model, id2slot, device)
        hist["train_loss"].append(tr)
        hist["dev_slot_f1"].append(f1)
        hist["dev_intent_acc"].append(acc)
        if f1 > best_f1:
            best_f1, best_acc, best_epoch = f1, acc, ep
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    hist["best_dev_slot_f1"] = best_f1
    hist["best_dev_intent_acc"] = best_acc
    hist["best_epoch"] = best_epoch
    hist["best_state"] = best_state
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
