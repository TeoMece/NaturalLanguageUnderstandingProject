"""Parte 1.B — addestramento e valutazione del modello LoRA.

Differenza chiave rispetto a 1.A: qui il modello e' un `GPT2LMHeadModel` di
HuggingFace, che calcola la loss INTERNAMENTE quando gli passiamo `labels=`.
Quindi non usiamo una `CrossEntropyLoss` esterna ne' lo shift manuale dei target:
HuggingFace fa lo shift dei label da solo (predice il token successivo) e ignora
i target uguali a -100. Per questo i loop sono basati su `output.loss` e seguono
lo schema del lab 4 (sezione "Models and Training for Part 1.B").

Cosa RIUSIAMO da 1.A: `pick_device` (selezione cpu/cuda/mps), perche' e'
indipendente dal modello. Tutto il resto (loop e fit) e' specifico per l'API HF.
"""
import math
import copy

import torch
from tqdm.auto import tqdm

# Riuso da 1.A: selezione del device (logica identica, nessun motivo di duplicarla).
from lm_pipeline.train import pick_device  # noqa: F401  (riesportato per comodita')


def _hf_labels(input_ids, pad_id):
    """Costruisce i label per HuggingFace a partire dagli input_ids.

    Convenzione del lab 4: NON shiftiamo i label a sinistra (lo fa HF da solo).
    Passiamo una copia di input_ids in cui i token di padding diventano -100, il
    valore che HuggingFace ignora di default nel calcolo della loss/perplexity.
    """
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    return labels


def train_loop(loader, optimizer, model, device, pad_id, grad_clip=None,
               show_progress=False, desc="train"):
    """Una epoca di training. Ritorna la loss media per token.

    Per ogni batch: costruisce i label HF, forward con `labels=` (HF calcola la
    loss), backward, gradient clipping opzionale, step. La loss viene pesata per
    il numero di token reali (n_tokens) per ottenere una media per-token corretta.

    Parametri
    ----------
    loader    : iterable di (input_ids, _, n_tokens)  — i label del collate sono
                ignorati (li ricostruiamo per l'API HF, vedi `_hf_labels`).
    optimizer : ottimizzatore (deve contenere SOLO i parametri allenabili LoRA).
    model     : GPT2_LoRA (o qualsiasi GPT2LMHeadModel) in modalita' train.
    device    : str — device su cui spostare i tensori.
    pad_id    : int — id del token di padding (mappato a -100 nei label).
    grad_clip : float|None — soglia di gradient clipping (None/0 = disattivo).
    """
    model.train()
    total_loss, total_tokens = 0.0, 0

    bar = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for input_ids, _, n_tokens in bar:
        input_ids = input_ids.to(device)
        labels = _hf_labels(input_ids, pad_id)

        optimizer.zero_grad()
        # HuggingFace: passando labels otteniamo output.loss (cross-entropy gia'
        # mediata sui token non -100), con lo shift dei target gestito internamente.
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip
            )

        optimizer.step()

        total_loss += loss.item() * int(n_tokens)
        total_tokens += int(n_tokens)
        if show_progress:
            bar.set_postfix(loss=f"{total_loss / max(1, total_tokens):.3f}")

    return total_loss / max(1, total_tokens)


def eval_loop(loader, model, device, pad_id, show_progress=False, desc="eval"):
    """Valuta senza aggiornare i pesi. Ritorna (perplexity, loss media per token).

    Perplexity = exp(loss media per token), la metrica standard del LM.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for input_ids, _, n_tokens in tqdm(loader, desc=desc, leave=False,
                                           disable=not show_progress):
            input_ids = input_ids.to(device)
            labels = _hf_labels(input_ids, pad_id)
            output = model(input_ids, labels=labels)
            total_loss += output.loss.item() * int(n_tokens)
            total_tokens += int(n_tokens)

    mean_loss = total_loss / max(1, total_tokens)
    return math.exp(mean_loss), mean_loss


def _build_optimizer(model, cfg_optim):
    """Ottimizzatore sui SOLI parametri allenabili (gli adapter LoRA).

    A differenza di 1.A (dove si ottimizzano tutti i parametri), qui filtriamo per
    `requires_grad=True`: dopo `apply_lora_freezing` restano solo i lora_*. Questo
    e' cio' che rende il fine-tuning leggero.
    """
    lr = cfg_optim["lr"]
    name = cfg_optim.get("optimizer", "adamw").lower()
    params = (p for p in model.parameters() if p.requires_grad)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"optimizer non supportato: {name}")


def fit(model, train_loader, valid_loader, cfg_optim, device, pad_id, show_progress=False):
    """Addestra con early stopping su valid-PPL. Rispecchia `lm_pipeline.train.fit`.

    Differenze rispetto a 1.A: ottimizza solo i parametri allenabili (LoRA) e usa
    i loop basati sulla loss interna di HF. Niente scheduler (il fine-tuning LoRA
    e' breve e parte da pesi gia' buoni); restano early stopping e gradient clip.

    Parametri
    ----------
    cfg_optim : dict con 'lr', 'optimizer', 'epochs', 'grad_clip', 'patience'.

    Ritorna
    -------
    dict : storico con 'train_loss', 'valid_loss', 'valid_ppl', 'best_ppl',
           'best_state' (state_dict completo alla best PPL — gli adapter sono cio'
           che conta, ma salviamo l'intero stato per ripristino fedele).
    """
    model.to(device)
    optimizer = _build_optimizer(model, cfg_optim)

    epochs = cfg_optim.get("epochs", 10)
    patience = cfg_optim.get("patience", 3)
    grad_clip = cfg_optim.get("grad_clip", 1.0)

    hist = {"train_loss": [], "valid_loss": [], "valid_ppl": []}
    best_ppl = float("inf")
    best_state = None
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="epochs", disable=not show_progress)
    for ep in epoch_bar:
        tr_loss = train_loop(train_loader, optimizer, model, device, pad_id, grad_clip,
                             show_progress=show_progress, desc=f"train e{ep + 1}/{epochs}")
        ppl, val_loss = eval_loop(valid_loader, model, device, pad_id,
                                  show_progress=show_progress, desc=f"eval e{ep + 1}")

        hist["train_loss"].append(tr_loss)
        hist["valid_loss"].append(val_loss)
        hist["valid_ppl"].append(ppl)

        if ppl < best_ppl:
            best_ppl = ppl
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                epoch_bar.set_postfix(val_ppl=f"{ppl:.2f}", best=f"{best_ppl:.2f}",
                                      no_improve=epochs_no_improve)
                break

        epoch_bar.set_postfix(val_ppl=f"{ppl:.2f}", best=f"{best_ppl:.2f}",
                              no_improve=epochs_no_improve)

    hist["best_ppl"] = best_ppl
    hist["best_state"] = best_state
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
