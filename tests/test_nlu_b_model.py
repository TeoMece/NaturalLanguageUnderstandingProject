"""Test del modello joint 2.B: forward shapes per BERT (CLS) e GPT2 (ultimo token).

Backbone costruiti da config MINIMA (niente download)."""
import torch
from transformers import BertModel, BertConfig, GPT2Model, GPT2Config

from nlu_pipeline_b.model import JointNLUTransformer


def test_bert_forward_cls():
    bb = BertModel(BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2,
                              num_attention_heads=2, intermediate_size=64,
                              max_position_embeddings=64))
    m = JointNLUTransformer(bb, num_slots=10, num_intents=5, intent_pool="cls_first")
    ids = torch.randint(0, 50, (3, 7))
    attn = torch.ones(3, 7, dtype=torch.long)
    s, i = m(ids, attn)
    assert s.shape == (3, 7, 10) and i.shape == (3, 5)


def test_gpt2_forward_last():
    bb = GPT2Model(GPT2Config(vocab_size=50, n_embd=32, n_layer=2, n_head=2, n_positions=64))
    m = JointNLUTransformer(bb, num_slots=10, num_intents=5, intent_pool="last")
    ids = torch.randint(0, 50, (3, 7))
    attn = torch.ones(3, 7, dtype=torch.long)
    attn[0, 5:] = 0  # padding a destra: l'intent deve leggere l'ultimo token reale (idx 4)
    s, i = m(ids, attn)
    assert s.shape == (3, 7, 10) and i.shape == (3, 5)
