"""Test della Parte 1.B — adapter LoRA e GPT2_LoRA.

Per non scaricare i pesi reali di GPT2 (e girare anche offline/CI), i test
costruiscono un `GPT2_LoRA` da una `GPT2Config` MINIMA (pochi layer/dim), invece
che da `from_pretrained`. La logica di LoRA e del freezing e' identica.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from lm_pipeline_b.model_lora import (
    LoRALinear,
    GPT2_LoRA,
    apply_lora_freezing,
)


def _tiny_config():
    """GPT2Config minima per test veloci (no download)."""
    return GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )


def test_loralinear_zero_at_init():
    """A inizializzazione B=0 -> il delta LoRA e' identicamente nullo."""
    lora = LoRALinear(8, 8, rank=4, alpha=8)
    x = torch.randn(3, 5, 8)
    out = lora(x)
    assert torch.allclose(out, torch.zeros_like(out)), "il delta a init deve essere 0"


def test_loralinear_scaling_with_alpha():
    """Raddoppiando alpha (a parita' di A,B) l'output raddoppia: verifica scaling."""
    lora1 = LoRALinear(4, 4, rank=2, alpha=2)
    lora2 = LoRALinear(4, 4, rank=2, alpha=4)  # alpha doppio -> scaling doppio
    # Forziamo gli stessi pesi non nulli in entrambi
    with torch.no_grad():
        for m in (lora1, lora2):
            m.lora_A.weight.fill_(0.1)
            m.lora_B.weight.fill_(0.1)
    x = torch.randn(2, 4)
    out1, out2 = lora1(x), lora2(x)
    assert torch.allclose(out2, 2.0 * out1, atol=1e-6), "alpha doppio -> output doppio"


def test_gpt2lora_matches_base_at_init():
    """A init il delta LoRA e' 0 -> GPT2_LoRA da gli stessi logit del GPT2 base."""
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = GPT2_LoRA(cfg, rank=4, alpha=8).eval()

    # Modello di riferimento con gli STESSI pesi base (i lora_* vengono ignorati)
    ref = GPT2LMHeadModel(cfg).eval()
    ref.load_state_dict(model.state_dict(), strict=False)

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_model = model(ids).logits
        out_ref = ref(ids).logits
    assert torch.allclose(out_model, out_ref, atol=1e-5), (
        "con delta LoRA=0 i logit devono coincidere col GPT2 base"
    )


def test_only_lora_trainable_after_freezing():
    """Dopo il freezing solo i parametri 'lora_' sono allenabili."""
    cfg = _tiny_config()
    model = apply_lora_freezing(GPT2_LoRA(cfg, rank=4, alpha=8))
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"{name} dovrebbe essere allenabile"
        else:
            assert not p.requires_grad, f"{name} dovrebbe essere congelato"
    # almeno un adapter deve esistere (sanity check)
    assert any("lora_" in n for n, _ in model.named_parameters())
