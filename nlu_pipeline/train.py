"""Training e valutazione multi-task per la Parte 2.A (NLU joint).

Il modello produce due output: slot (per token) e intent (per frase). La loss e' la
somma delle due cross-entropy (pesi uguali, come nel lab). La valutazione usa:
- per gli slot: lo script `conll` (F1 a livello di chunk) — la metrica standard;
- per l'intent: l'accuracy (frazione di intent corretti).

`pick_device` e `_build_optimizer` sono DUPLICATI da `lm_pipeline.train` (sono
indipendenti dal task; li duplichiamo per tenere la pipeline NLU auto-contenuta).
"""
import copy

import torch
import torch.nn as nn

from .conll import evaluate as conll_evaluate
from .data import PAD_TOKEN


# ---------------------------------------------------------------------------
# Utility duplicate da lm_pipeline (indipendenti dal task)
# ---------------------------------------------------------------------------

def pick_device(requested="auto"):
    """Risolve 'auto' nel miglior device disponibile (cuda > mps > cpu)."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_optimizer(model, cfg_optim):
    """AdamW (default) o SGD su tutti i parametri del modello."""
    lr = cfg_optim["lr"]
    name = cfg_optim.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"optimizer non supportato: {name}")


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------

def train_one_epoch(loader, optimizer, crit_slot, crit_intent, model, device, grad_clip):
    """Un'epoca di training multi-task. Ritorna la loss media per batch.

    Per ogni batch: forward -> loss = CE_slot + CE_intent -> backward -> (clip) -> step.
    CrossEntropy degli slot vuole input (B, C, L): per questo facciamo permute(0,2,1).
    """
    model.train()
    total = 0.0
    for batch in loader:
        utt = batch["utterances"].to(device)
        y_slots = batch["y_slots"].to(device)
        intents = batch["intents"].to(device)
        seq_lens = batch["slots_len"]

        optimizer.zero_grad()
        slot_logits, intent_logits = model(utt, seq_lens)
        loss = (crit_slot(slot_logits.permute(0, 2, 1), y_slots)
                + crit_intent(intent_logits, intents))
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def evaluate(loader, crit_slot, crit_intent, model, lang, device):
    """Valuta su un loader. Ritorna (slot_f1, intent_acc, loss_media).

    - slot_f1: F1 a chunk via `conll.evaluate` su liste di coppie (parola, slot);
      decodifichiamo ogni frase troncando all'ultima posizione reale MENO uno (per
      ignorare il token CLS, che non e' uno slot vero).
    - intent_acc: frazione di intent predetti correttamente (argmax).
    Nota: conll restituisce l'F1 in scala 0-100 (percentuale).
    """
    model.eval()
    total, n = 0.0, 0
    ref_slots, hyp_slots = [], []
    ref_int, hyp_int = [], []
    with torch.no_grad():
        for batch in loader:
            utt = batch["utterances"].to(device)
            y_slots = batch["y_slots"].to(device)
            intents = batch["intents"].to(device)
            seq_lens = batch["slots_len"]

            slot_logits, intent_logits = model(utt, seq_lens)
            total += (crit_slot(slot_logits.permute(0, 2, 1), y_slots)
                      + crit_intent(intent_logits, intents)).item()
            n += 1

            # intent: confronto argmax (in stringa, via id2intent)
            hyp_int += [lang.id2intent[i] for i in intent_logits.argmax(1).tolist()]
            ref_int += [lang.id2intent[i] for i in intents.tolist()]

            # slot: decodifica per frase, ignorando il CLS (ultima posizione reale)
            pred = slot_logits.argmax(-1)
            for b in range(utt.shape[0]):
                length = int(seq_lens[b]) - 1  # -1: ignora il CLS in coda
                words = [lang.id2word[w] for w in utt[b, :length].tolist()]
                gold = [lang.id2slot[s] for s in y_slots[b, :length].tolist()]
                out = [lang.id2slot[s] for s in pred[b, :length].tolist()]
                ref_slots.append([(words[i], gold[i]) for i in range(length)])
                hyp_slots.append([(words[i], out[i]) for i in range(length)])

    try:
        slot_f1 = conll_evaluate(ref_slots, hyp_slots)["total"]["f"]
    except Exception:
        # puo' capitare se il modello predice una classe assente dal ref: F1=0
        slot_f1 = 0.0
    intent_acc = sum(a == b for a, b in zip(ref_int, hyp_int)) / max(1, len(ref_int))
    return slot_f1, intent_acc, total / max(1, n)


def fit(model, train_loader, dev_loader, cfg_optim, device, lang):
    """Training multi-task con early stopping sulla SLOT F1 di dev (massimizzazione).

    Scegliamo la slot F1 come criterio perche' lo slot filling e' il task piu' difficile
    e informativo; riportiamo comunque anche l'intent accuracy. Al termine, il modello
    viene riportato ai pesi della miglior epoca.

    Ritorna lo storico con 'train_loss', 'dev_slot_f1', 'dev_intent_acc' per epoca,
    piu' 'best_dev_slot_f1', 'best_dev_intent_acc', 'best_epoch' e 'best_state'.
    """
    model.to(device)
    optimizer = _build_optimizer(model, cfg_optim)
    # slot: ignora il padding/CLS (id = PAD_TOKEN); intent: classi normali
    crit_slot = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    crit_intent = nn.CrossEntropyLoss()

    epochs = cfg_optim.get("epochs", 50)
    patience = cfg_optim.get("patience", 5)
    grad_clip = cfg_optim.get("grad_clip", 5.0)

    hist = {"train_loss": [], "dev_slot_f1": [], "dev_intent_acc": []}
    best_f1, best_acc, best_epoch = -1.0, 0.0, 0
    best_state, no_improve = None, 0

    for epoch in range(epochs):
        tr = train_one_epoch(train_loader, optimizer, crit_slot, crit_intent,
                             model, device, grad_clip)
        f1, acc, _ = evaluate(dev_loader, crit_slot, crit_intent, model, lang, device)
        hist["train_loss"].append(tr)
        hist["dev_slot_f1"].append(f1)
        hist["dev_intent_acc"].append(acc)

        if f1 > best_f1:
            best_f1, best_acc, best_epoch = f1, acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    hist["best_dev_slot_f1"] = best_f1
    hist["best_dev_intent_acc"] = best_acc
    hist["best_epoch"] = best_epoch
    hist["best_state"] = best_state
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
