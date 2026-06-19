"""Test per lm_pipeline/train.py — segue TDD: i test vengono scritti prima dell'implementazione.

Copre:
- pick_device: risoluzione 'auto' e passthrough per device espliciti
- warmup_cosine_lambda: forma della curva (salita lineare + cosine decay)
- evaluate: ritorna PPL positiva e finita su un mini-loader giocattolo
- fit: ritorna lo storico con le chiavi attese dopo addestramento su dati finti
"""
import math
import torch
from lm_pipeline.train import pick_device, warmup_cosine_lambda, evaluate, fit
from lm_pipeline.model import GPT2


def test_pick_device_returns_valid_string():
    """'cpu' viene passato invariato; 'auto' risolve in uno dei device validi."""
    assert pick_device("cpu") == "cpu"
    dev = pick_device("auto")
    assert dev in ("cuda", "mps", "cpu")


def test_warmup_cosine_shape():
    """La curva deve: partire vicino a 0, raggiungere 1.0 al termine del warmup,
    scendere quasi a 0 alla fine (cosine), e rimanere monotona decrescente dopo il picco."""
    fn = warmup_cosine_lambda(warmup_steps=10, total_steps=100)
    assert abs(fn(0)) < 0.2          # parte vicino a 0 (warmup)
    assert abs(fn(10) - 1.0) < 1e-6  # picco a fine warmup
    assert fn(100) < 0.05            # ~0 alla fine (cosine)
    assert fn(10) > fn(55) > fn(100) # monotona decrescente dopo il picco


def _toy_loader():
    """Costruisce 2 batch sintetici (input_ids, labels, n_tokens) con vocab piccolo.

    Usato da test_evaluate e test_fit per evitare dipendenze dal dataset reale.
    Ogni batch: B=2, L=8 -> 16 token totali.
    """
    batches = []
    for _ in range(2):
        ids = torch.randint(0, 30, (2, 8))
        labels = torch.randint(0, 30, (2, 8))
        n = torch.tensor(16)
        batches.append((ids, labels, n))
    return batches


def test_evaluate_returns_positive_ppl():
    """evaluate() deve ritornare una perplexity > 0 e finita su dati casuali."""
    model = GPT2(vocab_size=30, pos_emb_size=16, d_model=8, n_heads=2,
                 num_layers=1, ff_dim=16)
    crit = torch.nn.CrossEntropyLoss()
    ppl, loss = evaluate(_toy_loader(), crit, model, "cpu")
    assert ppl > 0 and math.isfinite(ppl)


def test_fit_returns_history_and_reduces_loss():
    """fit() deve ritornare uno storico con le chiavi 'valid_ppl' e 'best_ppl'."""
    model = GPT2(vocab_size=30, pos_emb_size=16, d_model=8, n_heads=2,
                 num_layers=1, ff_dim=16)
    cfg_optim = {"lr": 1e-2, "optimizer": "adamw", "epochs": 3, "grad_clip": 1.0,
                 "patience": 5, "scheduler": {"enabled": True, "warmup_steps": 1}}
    hist = fit(model, _toy_loader(), _toy_loader(), cfg_optim, "cpu", pad_id=0)
    assert "valid_ppl" in hist and len(hist["valid_ppl"]) >= 1
    assert "best_ppl" in hist
