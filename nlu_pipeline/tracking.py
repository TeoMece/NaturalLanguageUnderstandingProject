"""Utility di tracciamento: riproducibilita' e salvataggio degli artefatti per run.

Ogni run vive in una propria cartella. Qui gestiamo seed deterministico, git hash,
creazione della run dir e scrittura di metrics.json, curves.csv e curves.pdf
(materiale pronto per il report).
"""
import os
import csv
import json
import random
import subprocess

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # backend non interattivo (salva su file)
import matplotlib.pyplot as plt


def set_seed(seed):
    """Fissa il seed per Python, NumPy e PyTorch (riproducibilita')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash():
    """Ritorna l'hash del commit corrente, o 'unknown' se non in un repo git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(runs_root, name):
    """Crea una cartella run univoca; se il nome esiste, appende un suffisso."""
    os.makedirs(runs_root, exist_ok=True)
    path = os.path.join(runs_root, name)
    i = 1
    while os.path.exists(path):
        path = os.path.join(runs_root, f"{name}_{i}")
        i += 1
    os.makedirs(path)
    return path


def save_metrics(run_dir, metrics):
    """Salva il dizionario delle metriche in metrics.json e ritorna il path."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_curves(run_dir, hist):
    """Salva le curve per epoca (NLU) in CSV e un grafico in PDF. Ritorna (csv, pdf).

    Per la Parte 2 lo storico contiene, per epoca: `train_loss`, `dev_slot_f1`,
    `dev_intent_acc`. Il PDF mostra le due metriche di dev (slot F1 e intent accuracy)
    in funzione dell'epoca.
    """
    n = len(hist["dev_slot_f1"])
    csv_path = os.path.join(run_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "dev_slot_f1", "dev_intent_acc"])
        for i in range(n):
            w.writerow([i + 1, hist["train_loss"][i], hist["dev_slot_f1"][i], hist["dev_intent_acc"][i]])

    pdf_path = os.path.join(run_dir, "curves.pdf")
    epochs = range(1, n + 1)
    plt.figure()
    plt.plot(epochs, hist["dev_slot_f1"], marker="o", label="dev slot F1")
    # intent accuracy in [0,1] -> in % per stare sulla stessa scala della F1
    plt.plot(epochs, [100 * a for a in hist["dev_intent_acc"]], marker="s", label="dev intent acc (%)")
    plt.xlabel("epoca")
    plt.ylabel("metrica")
    plt.title("Metriche di validazione (NLU)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    return csv_path, pdf_path
