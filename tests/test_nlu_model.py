"""Test del modello joint GPT2JointIAS (forward shapes)."""
import torch

from nlu_pipeline.model import GPT2JointIAS, init_weights


def test_forward_shapes():
    torch.manual_seed(0)
    m = GPT2JointIAS(vocab_size=50, num_slots=10, num_intents=5,
                     pos_emb_size=32, d_model=16, n_heads=2, num_layers=2, ff_dim=32)
    init_weights(m)
    B, L = 3, 7
    idx = torch.randint(0, 50, (B, L))
    seq_lens = torch.tensor([7, 5, 4])
    slot_logits, intent_logits = m(idx, seq_lens)
    assert slot_logits.shape == (B, L, 10)
    assert intent_logits.shape == (B, 5)
