import torch
from lm_pipeline.model import GPT2

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
