"""Logica di addestramento e valutazione, indipendente dall'orchestrazione.

Contiene: selezione del device, lo scheduler warmup+cosine, i loop di train ed
eval (con perplexity = exp(loss medio per token)) e `fit` con early stopping su
valid-PPL. Tutto riceve oggetti gia' costruiti: nessuna dipendenza da config/yaml.

Questo file appartiene al livello *puro* della pipeline (come model.py) e NON
importa nulla da config, experiment o tracking, per garantire massima riusabilita'.
Verra' copiato verbatim come `functions.py` nella consegna finale.
"""
import math
import copy

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def pick_device(requested="auto"):
    """Risolve 'auto' nel miglior device disponibile (cuda > mps > cpu).

    Se viene passato un device esplicito (es. 'cpu', 'cuda:0'), viene restituito
    invariato senza verifiche, cosi' il chiamante ha il controllo completo.

    Parametri
    ----------
    requested : str
        'auto' per selezione automatica, oppure stringa device esplicita.

    Ritorna
    -------
    str : nome del device selezionato
    """
    if requested != "auto":
        # Passthrough: il chiamante ha specificato un device esplicito
        return requested
    # Ordine di preferenza: GPU NVIDIA > GPU Apple Silicon > CPU
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Learning-rate schedule: warmup lineare + cosine decay
# ---------------------------------------------------------------------------

# Perche' scritta a mano e non da libreria: PyTorch ha CosineAnnealingLR (solo decay,
# SENZA warmup) ma non un warmup+cosine "tutto in uno". L'alternativa sarebbe comporre
# SequentialLR(LinearLR, CosineAnnealingLR) (piu' verboso/fragile) o importare transformers
# (get_cosine_schedule_with_warmup), dipendenza enorme per 10 righe. Qui restituiamo solo
# un MOLTIPLICATORE in [0,1] e lasciamo che LambdaLR faccia lr = lr_base * f(step): leggera,
# leggibile e testabile (vedi test sulla forma della curva in tests/test_train.py).
def warmup_cosine_lambda(warmup_steps, total_steps):
    """Ritorna f(step) -> moltiplicatore del lr: salita lineare poi cosine decay.

    Da usare con torch.optim.lr_scheduler.LambdaLR. La curva ha due fasi:
      1. Warmup lineare (0 <= step < warmup_steps): moltiplicatore 0 -> 1
         Utile per stabilizzare il training nelle prime iterazioni.
      2. Cosine decay (warmup_steps <= step <= total_steps): moltiplicatore 1 -> 0
         Il learning rate decade dolcemente seguendo una mezza cosinusoide.

    Parametri
    ----------
    warmup_steps : int
        Numero di step per la fase di warmup (garantito >= 1 per evitare div/0).
    total_steps  : int
        Numero totale di step dell'intero addestramento.

    Ritorna
    -------
    callable : f(step: int) -> float, moltiplicatore in [0, 1]
    """
    # Protezione contro warmup_steps=0 che causerebbe divisione per zero
    warmup_steps = max(1, warmup_steps)

    def f(step):
        if step < warmup_steps:
            # Fase di warmup: crescita lineare da 0 a 1
            return step / warmup_steps
        # Fase di decay coseno: progresso normalizzato in [0, 1]
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)   # clamp: non supera 1 dopo total_steps
        # Formula coseno: 1 quando progress=0, 0 quando progress=1
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


# ---------------------------------------------------------------------------
# Costruzione dell'ottimizzatore
# ---------------------------------------------------------------------------

def _build_optimizer(model, cfg_optim):
    """Costruisce l'ottimizzatore specificato in cfg_optim.

    Supporta 'adamw' (default, consigliato per transformer) e 'sgd'.

    Parametri
    ----------
    model     : nn.Module
    cfg_optim : dict con chiavi 'optimizer' (str) e 'lr' (float)

    Ritorna
    -------
    torch.optim.Optimizer
    """
    name = cfg_optim.get("optimizer", "adamw").lower()
    lr = cfg_optim["lr"]
    if name == "adamw":
        # AdamW: Adam con weight decay decoupled, standard per GPT-like models
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"optimizer non supportato: {name}")


# ---------------------------------------------------------------------------
# Loop di training per una singola epoca
# ---------------------------------------------------------------------------

def train_one_epoch(loader, optimizer, scheduler, criterion, model, device, grad_clip,
                    show_progress=False, desc="train"):
    """Esegue un'epoca completa di training sul loader dato.

    Per ogni batch: forward -> calcolo loss -> backward -> gradient clipping ->
    optimizer step -> scheduler step. La loss viene accumulata pesata per numero
    di token reali (n_tokens) cosi' da calcolare la media per-token corretta.

    Parametri
    ----------
    loader     : iterable di (input_ids, labels, n_tokens)
                 input_ids : (B, L) int  — token di input
                 labels    : (B, L) int  — token target (shiftati di 1)
                 n_tokens  : scalar int  — numero di token non-padding nel batch
    optimizer  : torch.optim.Optimizer
    scheduler  : lr_scheduler (o None se non abilitato)
    criterion  : nn.CrossEntropyLoss (con ignore_index=pad_id)
    model      : nn.Module in modalita' train
    device     : str — device su cui spostare i tensori
    grad_clip  : float — soglia per gradient clipping (0 o None = disabilitato)

    Ritorna
    -------
    float : loss media per token su tutta l'epoca
    """
    model.train()                            # attiva dropout e altri layer train-only
    total_loss, total_tokens = 0.0, 0

    # Barra di avanzamento sui batch (solo se show_progress): leave=False cosi'
    # sparisce a fine epoca e non sporca il log; disabilitata di default per non
    # interferire con i test / l'uso come libreria.
    batches = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for input_ids, labels, n_tokens in batches:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: (B, L, vocab_size)
        logits = model(input_ids)

        # CrossEntropyLoss si aspetta (B, vocab, L) come input e (B, L) come target
        loss = criterion(logits.permute(0, 2, 1), labels)

        # Backward pass: calcola i gradienti
        loss.backward()

        # Gradient clipping: evita esplosione dei gradienti, comune nei transformer
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Scheduler step a livello di batch (non di epoca) per warmup preciso
        if scheduler is not None:
            scheduler.step()

        # Accumula la loss pesata per il numero di token reali del batch
        total_loss += loss.item() * int(n_tokens)
        total_tokens += int(n_tokens)

        # Mostra la loss media corrente e il lr sulla barra (se attiva)
        if show_progress:
            lr_now = optimizer.param_groups[0]["lr"]
            batches.set_postfix(loss=f"{total_loss / max(1, total_tokens):.3f}",
                                lr=f"{lr_now:.2e}")

    # Loss media per token: metrica indipendente dalla dimensione dei batch
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Loop di valutazione
# ---------------------------------------------------------------------------

def evaluate(loader, criterion, model, device, show_progress=False, desc="eval"):
    """Valuta il modello sul loader dato e calcola perplexity e loss media.

    Eseguito in modalita' no_grad (nessun calcolo del gradiente) per efficienza.
    La perplexity e' calcolata come exp(loss_media_per_token), metrica standard
    per i language model: valori piu' bassi indicano migliore modellazione.

    Parametri
    ----------
    loader    : iterable di (input_ids, labels, n_tokens)
    criterion : nn.CrossEntropyLoss
    model     : nn.Module
    device    : str

    Ritorna
    -------
    tuple(float, float) : (perplexity, loss_media_per_token)
    """
    model.eval()                             # disattiva dropout e layer eval-incompatibili
    total_loss, total_tokens = 0.0, 0

    batches = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    with torch.no_grad():                    # nessun calcolo del gradiente -> risparmio memoria
        for input_ids, labels, n_tokens in batches:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)         # (B, L, vocab_size)
            # CrossEntropyLoss si aspetta (B, vocab, L)
            loss = criterion(logits.permute(0, 2, 1), labels)

            total_loss += loss.item() * int(n_tokens)
            total_tokens += int(n_tokens)

    mean_loss = total_loss / max(1, total_tokens)
    # Perplexity: misura di quante parole il modello considera "plausibili" in media
    return math.exp(mean_loss), mean_loss


# ---------------------------------------------------------------------------
# Fit: ciclo completo con early stopping
# ---------------------------------------------------------------------------

def fit(model, train_loader, valid_loader, cfg_optim, device, pad_id, show_progress=False):
    """Addestra il modello con early stopping su valid-PPL.

    Orchestra la costruzione dell'ottimizzatore, dello scheduler e il ciclo di
    epoche. Implementa early stopping: se la valid-PPL non migliora per
    `patience` epoche consecutive, l'addestramento si interrompe. Al termine,
    il modello viene riportato ai pesi migliori (quelli con la PPL piu' bassa).

    Parametri
    ----------
    model        : nn.Module — il modello da addestrare (modificato in-place)
    train_loader : iterable di (input_ids, labels, n_tokens)
    valid_loader : iterable di (input_ids, labels, n_tokens)
    cfg_optim    : dict con le chiavi:
                     'lr'        : float — learning rate iniziale
                     'optimizer' : str   — 'adamw' | 'sgd'
                     'epochs'    : int   — numero massimo di epoche
                     'grad_clip' : float — soglia gradient clipping
                     'patience'  : int   — epoche senza miglioramento prima dello stop
                     'scheduler' : dict  — {'enabled': bool, 'warmup_steps': int}
    device       : str — es. 'cpu', 'cuda', 'mps'
    pad_id       : int — indice del token di padding (ignorato dalla loss)

    Ritorna
    -------
    dict : storico delle metriche con le chiavi:
             'train_loss' : list[float] — loss media per token (train) per epoca
             'valid_loss' : list[float] — loss media per token (valid) per epoca
             'valid_ppl'  : list[float] — perplexity di validazione per epoca
             'best_ppl'   : float       — miglior PPL raggiunta
             'best_state' : OrderedDict — state_dict corrispondente alla best PPL
    """
    # Sposta il modello sul device scelto (cpu/cuda/mps)
    model.to(device)

    optimizer = _build_optimizer(model, cfg_optim)

    # CrossEntropyLoss con ignore_index: i token di padding non contribuiscono
    # alla loss, garantendo che non inquinino il segnale di addestramento
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    epochs = cfg_optim.get("epochs", 30)

    # Configurazione opzionale dello scheduler LR
    sched_cfg = cfg_optim.get("scheduler", {}) or {}
    scheduler = None
    if sched_cfg.get("enabled", False):
        # total_steps = epoche * batch per epoca (per step-level scheduling)
        total_steps = epochs * max(1, len(train_loader))
        lam = warmup_cosine_lambda(
            warmup_steps=sched_cfg.get("warmup_steps", 200),
            total_steps=total_steps,
        )
        # LambdaLR moltiplica il lr base per il valore restituito da lam(step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    patience = cfg_optim.get("patience", 5)
    grad_clip = cfg_optim.get("grad_clip", 1.0)

    # Inizializzazione dello storico e delle variabili per l'early stopping
    hist = {"train_loss": [], "valid_loss": [], "valid_ppl": []}
    best_ppl = float("inf")
    best_state = None
    epochs_no_improve = 0       # contatore delle epoche senza miglioramento

    # Barra esterna sulle epoche (resta a schermo); le barre dei batch sono interne.
    epoch_bar = tqdm(range(epochs), desc="epochs", disable=not show_progress)
    for epoch in epoch_bar:
        # ---- Training ----
        tr_loss = train_one_epoch(
            train_loader, optimizer, scheduler, criterion, model, device, grad_clip,
            show_progress=show_progress, desc=f"train e{epoch + 1}/{epochs}",
        )

        # ---- Validazione ----
        ppl, val_loss = evaluate(valid_loader, criterion, model, device,
                                 show_progress=show_progress, desc=f"eval e{epoch + 1}")

        # Aggiorna lo storico per la curva di apprendimento
        hist["train_loss"].append(tr_loss)
        hist["valid_loss"].append(val_loss)
        hist["valid_ppl"].append(ppl)

        # ---- Early stopping ----
        if ppl < best_ppl:
            # Miglioramento: salva snapshot dei pesi (deep copy per indipendenza)
            best_ppl = ppl
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Nessun miglioramento per `patience` epoche: interrompe
                epoch_bar.set_postfix(val_ppl=f"{ppl:.2f}", best=f"{best_ppl:.2f}",
                                      no_improve=epochs_no_improve)
                break

        # Riepilogo dell'epoca sulla barra esterna (se attiva)
        epoch_bar.set_postfix(val_ppl=f"{ppl:.2f}", best=f"{best_ppl:.2f}",
                              no_improve=epochs_no_improve)

    # Aggiunge le informazioni sul best model allo storico
    hist["best_ppl"] = best_ppl
    hist["best_state"] = best_state   # utile per salvare il checkpoint a valle

    # Ripristina i pesi migliori nel modello (model e' modificato in-place)
    if best_state is not None:
        model.load_state_dict(best_state)

    return hist
