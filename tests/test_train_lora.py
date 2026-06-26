"""Smoke test della Parte 1.B: il loop di training LoRA gira su CPU senza errori.

Usa una GPT2Config minima e un loader finto (tensori random), cosi' non serve ne'
la GPU ne' scaricare GPT2/PTB. Verifica le invarianti fondamentali del fine-tuning
LoRA: la loss e' finita, i pesi base NON cambiano, gli adapter SI'.
"""
import copy

import torch
from transformers import GPT2Config

from lm_pipeline_b.model_lora import GPT2_LoRA, apply_lora_freezing
from lm_pipeline_b.train_lora import train_loop, eval_loop, _build_optimizer, fit


def _tiny_model():
    cfg = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    return apply_lora_freezing(GPT2_LoRA(cfg, rank=4, alpha=8))


def _fake_loader(pad_id=99):
    """Un solo batch: (input_ids, labels_ignorati, n_tokens). Ids in [0,98]."""
    ids = torch.randint(0, pad_id, (2, 8))          # nessun token == pad_id
    n_tokens = int((ids != pad_id).sum())
    return [(ids, ids, n_tokens)]


def test_train_step_runs_on_cpu():
    """Un passo di train: loss finita; base congelato invariato; adapter aggiornato."""
    torch.manual_seed(0)
    model = _tiny_model()
    device, pad_id = "cpu", 99
    loader = _fake_loader(pad_id)
    optimizer = _build_optimizer(model, {"lr": 1e-2, "optimizer": "adamw"})

    # Snapshot di un peso base (deve restare invariato) e di un adapter B
    # (deve cambiare: a init B=0 ma riceve gradiente perche' A(x) != 0).
    base_w = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    lora_b = model.transformer.h[0].attn.lora_q.lora_B.weight.detach().clone()

    loss = train_loop(loader, optimizer, model, device, pad_id)
    assert torch.isfinite(torch.tensor(loss)), "la loss deve essere finita"

    base_w_after = model.transformer.h[0].attn.c_attn.weight.detach()
    lora_b_after = model.transformer.h[0].attn.lora_q.lora_B.weight.detach()

    assert torch.allclose(base_w, base_w_after), "i pesi base devono restare congelati"
    assert not torch.allclose(lora_b, lora_b_after), "l'adapter LoRA deve aggiornarsi"


def test_eval_returns_finite_ppl():
    """eval_loop ritorna una perplexity finita e positiva."""
    torch.manual_seed(0)
    model = _tiny_model()
    ppl, loss = eval_loop(_fake_loader(), model, "cpu", pad_id=99)
    assert ppl > 0 and torch.isfinite(torch.tensor(ppl))
    assert torch.isfinite(torch.tensor(loss))


def test_zeroshot_epochs0_evaluates_without_training():
    """epochs=0 (zero-shot): valuta una volta, NON addestra; best_ppl finita, adapter invariati."""
    torch.manual_seed(0)
    model = _tiny_model()
    loader = _fake_loader(99)
    # gli adapter B sono a 0 all'init: senza training devono restare invariati
    lora_b = model.transformer.h[0].attn.lora_q.lora_B.weight.detach().clone()

    hist = fit(model, loader, loader,
               {"lr": 1e-2, "optimizer": "adamw", "epochs": 0}, "cpu", pad_id=99)

    assert torch.isfinite(torch.tensor(hist["best_ppl"])), "best_ppl deve essere finita (non inf)"
    assert hist["best_state"] is not None, "lo state del modello base va salvato"
    assert hist["valid_ppl"] and torch.isfinite(torch.tensor(hist["valid_ppl"][0]))

    lora_b_after = model.transformer.h[0].attn.lora_q.lora_B.weight.detach()
    assert torch.allclose(lora_b, lora_b_after), "lo zero-shot non deve addestrare gli adapter"
