"""Orchestrazione di una singola run: dalla config agli artefatti su disco.

Collega lo strato puro (data/model/train) con il tracking. E' resumable: se una
run con quel nome ha gia' un metrics.json, viene saltata (utile per sweep
interrotti a meta', es. su Mac).

Struttura della run su disco (sotto runs_root/name/):
    metrics.json    <- metriche finali (segna il completamento della run)
    curves.csv      <- storico per epoca (train_loss, valid_loss, valid_ppl)
    curves.pdf      <- grafico della PPL di validazione
    best_model.pt   <- state_dict del modello alla miglior PPL di validazione
"""
import os
import time

import torch
import torch.nn as nn

from . import data as data_mod
from . import tracking
from .model import GPT2, init_weights
from .train import pick_device, fit, evaluate


# ---------------------------------------------------------------------------
# Resumabilita'
# ---------------------------------------------------------------------------

def already_done(runs_root, name):
    """True se esiste gia' una run completata (metrics.json presente).

    Una run e' considerata completata quando `metrics.json` e' stato scritto
    su disco (e' l'ultimo artefatto che `run_experiment` produce). Controlla
    solo l'esistenza del file, non la sua validita', per semplicita'.

    Parametri
    ----------
    runs_root : str  — cartella radice che contiene tutte le run
    name      : str  — nome della run da controllare

    Ritorna
    -------
    bool : True se la run e' gia' completata, False altrimenti
    """
    return os.path.exists(os.path.join(runs_root, name, "metrics.json"))


# ---------------------------------------------------------------------------
# Run completa
# ---------------------------------------------------------------------------

def run_experiment(cfg, runs_root, dataset_dir, tokenizer=None):
    """Esegue una run completa e salva gli artefatti. Ritorna il path della run dir.

    `cfg` e' una config gia' risolta (no sweep). In `mode: final` ignora il subset
    e valuta su test; altrimenti valuta su valid.

    Il flusso e':
      1. Controllo di resumabilita' (salta se gia' completata)
      2. Seed e device
      3. Tokenizer (se non fornito, carica GPT2 di default)
      4. Costruzione dataloader (con subset o full dataset a seconda della mode)
      5. Costruzione e inizializzazione del modello GPT2
      6. Training con early stopping (tramite fit)
      7. Valutazione su test (solo in mode 'final')
      8. Salvataggio artefatti: curves, checkpoint, metrics (in quest'ordine)

    Parametri
    ----------
    cfg        : dict — configurazione gia' risolta (sezioni 'experiment', 'data',
                        'model', 'optim'); nessuno sweep qui.
    runs_root  : str  — cartella radice delle run (viene creata se non esiste).
    dataset_dir: str  — cartella che contiene ptb.train/valid/test.txt.
    tokenizer  : tokenizer HuggingFace opzionale; se None viene caricato GPT2.

    Ritorna
    -------
    str : path assoluto della run dir (sia se gia' completata sia se appena creata)
    """
    # Estrae la sottosezione 'experiment' dalla config (default: dict vuoto)
    exp = cfg.get("experiment", {})
    name = exp.get("name", "run")

    # ---- Resumabilita': salta se questa run e' gia' stata completata ----
    # Controlla se metrics.json esiste gia' nella cartella della run.
    # NOTA: viene controllato il path *senza* suffisso numerico (il nome esatto),
    # perche' make_run_dir crea runs_root/name, non una variante con suffisso.
    if already_done(runs_root, name):
        # La run esiste gia': ritorna direttamente il path senza rieseguire
        return os.path.join(runs_root, name)

    # ---- Riproducibilita' e device ----
    tracking.set_seed(exp.get("seed", 42))
    # pick_device gestisce 'auto' (cuda > mps > cpu) e device espliciti
    device = pick_device(exp.get("device", "auto"))

    # ---- Tokenizer ----
    # Se non fornito dall'esterno (es. in test o in run singola), carica GPT2.
    # Passare il tokenizer dall'esterno evita di riscaricare il modello ad ogni run
    # durante uno sweep.
    if tokenizer is None:
        tokenizer = data_mod.get_tokenizer()

    # ---- Preparazione della config dati ----
    # Copia per non mutare la config originale (che potrebbe essere condivisa
    # tra piu' run in uno sweep)
    cfg_data = dict(cfg.get("data", {}))
    is_final = cfg.get("mode", "dev") == "final"

    if is_final:
        # In mode 'final' si usa il dataset completo: azzera qualsiasi subset
        cfg_data["fraction"] = 1.0
        cfg_data["max_samples"] = None

    # ---- Costruzione dataloader ----
    # build_dataloaders applica il subset al solo train; valid e test restano interi
    train_dl, dev_dl, test_dl, n_train = data_mod.build_dataloaders(
        cfg_data, dataset_dir, tokenizer
    )

    # ---- Costruzione del modello ----
    m = cfg.get("model", {})
    # Il dropout e' un sottodizionario {enabled: bool, p: float}
    dropout_cfg = m.get("dropout", {})
    # Se dropout non e' abilitato, forza p=0.0 (baseline puro, senza dropout)
    dropout_p = dropout_cfg.get("p", 0.1) if dropout_cfg.get("enabled", False) else 0.0

    model = GPT2(
        vocab_size=len(tokenizer),               # vocabolario GPT2 = 50.257
        pos_emb_size=m.get("pos_emb_size", 1024),# lunghezza massima della sequenza
        d_model=m.get("d_model", 256),           # dimensione delle rappresentazioni
        n_heads=m.get("n_heads", 4),             # teste di attention
        num_layers=m.get("num_layers", 4),       # blocchi transformer
        ff_dim=m.get("ff_dim", 1024),            # dim del feed-forward interno
        dropout=dropout_p,
        weight_tying=m.get("weight_tying", False),# tying embedding/unembedding
    )
    # Inizializza i pesi con la strategia definita in model.py (Glorot/zero bias)
    model.apply(init_weights)

    # ---- Training ----
    t0 = time.time()
    # fit gestisce: spostamento su device, ottimizzatore, scheduler, early stopping
    # Ritorna lo storico delle metriche per epoca piu' best_ppl e best_state
    hist = fit(model, train_dl, dev_dl, cfg.get("optim", {}), device,
               pad_id=tokenizer.pad_token_id)
    elapsed = time.time() - t0

    # ---- Valutazione su test (solo mode 'final') ----
    # In mode 'dev' la metrica di riferimento e' la valid-PPL (gia' in hist).
    # In mode 'final' valutiamo sul test set col modello ai pesi migliori
    # (fit ha gia' ripristinato i best weights nel modello).
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    if is_final:
        test_ppl, _ = evaluate(test_dl, crit, model, device)
    else:
        test_ppl = None   # non valutato in mode 'dev' (evita data leakage)

    # ---- Salvataggio artefatti ----
    # make_run_dir crea la cartella; se il nome esiste aggiunge suffisso numerico
    run_dir = tracking.make_run_dir(runs_root, name)

    # Curve di apprendimento: curves.csv e curves.pdf
    tracking.save_curves(run_dir, hist)

    # Checkpoint del modello migliore (best weights gia' ripristinati da fit)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    # metrics.json: scritto per ULTIMO, funge da "flag" di completamento run.
    # Se il processo viene interrotto prima, la run verra' rieseguita al prossimo avvio.
    metrics = {
        "name": name,
        "mode": cfg.get("mode", "dev"),
        "best_valid_ppl": hist["best_ppl"],       # miglior PPL su validation
        "test_ppl": test_ppl,                     # PPL su test (None se mode='dev')
        "epochs_run": len(hist["valid_ppl"]),      # epoche effettivamente completate
        "n_train_sentences": n_train,             # frasi nel training set usato
        "n_params": sum(p.numel() for p in model.parameters()),  # param totali
        "device": device,                         # device effettivo usato
        "seconds": round(elapsed, 1),             # durata del training in secondi
        "seed": exp.get("seed", 42),              # seed usato (per riproducibilita')
        "git_hash": tracking.git_hash(),          # commit corrente (tracciabilita')
        "config": cfg,                            # config completa (per audit)
    }
    tracking.save_metrics(run_dir, metrics)

    return run_dir
