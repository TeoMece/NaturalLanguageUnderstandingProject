import torch
from lm_pipeline.model import GPT2, init_weights

def _tiny(**kw):
    base = dict(vocab_size=50, pos_emb_size=16, d_model=8, n_heads=2,
                num_layers=2, ff_dim=16, dropout=0.0, weight_tying=False)
    base.update(kw)
    return GPT2(**base)

def test_forward_output_shape():
    model = _tiny()
    idx = torch.randint(0, 50, (3, 10))     # batch 3, seq 10
    logits = model(idx)
    assert logits.shape == (3, 10, 50)      # (B, L, vocab)

def test_weight_tying_shares_weights():
    model = _tiny(weight_tying=True)
    assert model.lm_head.weight is model.token_embed.weight

def test_no_weight_tying_separate_weights():
    model = _tiny(weight_tying=False)
    assert model.lm_head.weight is not model.token_embed.weight

def test_dropout_enabled_changes_train_eval():
    torch.manual_seed(0)
    model = _tiny(dropout=0.5)
    idx = torch.randint(0, 50, (2, 8))
    model.train()
    a = model(idx)
    model.eval()
    b = model(idx)
    assert not torch.allclose(a, b)         # dropout attivo solo in train

def test_init_embeddings_use_normal_scale():
    torch.manual_seed(0)
    model = _tiny()
    init_weights(model)
    # gli Embedding NON devono restare alla scala N(0,1) di default...
    assert model.token_embed.weight.std().item() < 0.1
    # ...ne' essere clobberati all'uniforme ±0.01 dei Linear: scala ~0.02
    assert model.token_embed.weight.abs().max().item() > 0.01

def test_init_tied_head_keeps_embedding_scale():
    torch.manual_seed(0)
    model = _tiny(weight_tying=True)
    init_weights(model)
    assert model.lm_head.weight is model.token_embed.weight   # peso condiviso
    # NON clobberato a uniform(±0.01): con normal(0.02) ci sono valori oltre 0.01
    assert model.lm_head.weight.abs().max().item() > 0.01

def test_init_untied_linear_uniform_bound():
    torch.manual_seed(0)
    model = _tiny(weight_tying=False)
    init_weights(model)
    # lm_head non-tied: init uniforme del lab -> |w| <= 0.01
    assert model.lm_head.weight.abs().max().item() <= 0.01
