"""Smoke test del training/eval 2.B su CPU (BERT minimo da config, loader finto)."""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from nlu_pipeline_b.model import JointNLUTransformer
from nlu_pipeline_b.train import train_one_epoch, evaluate
from nlu_pipeline_b.data import IGNORE


def _batch():
    ids = torch.randint(0, 50, (2, 5))
    attn = torch.ones(2, 5, dtype=torch.long)
    y = torch.full((2, 5), IGNORE)
    y[0, 1] = 1
    y[1, 1] = 0   # un sub-token etichettato per riga
    intents = torch.tensor([0, 1])
    meta = [(["a", "b"], ["O", "B-x"], [None, 0, 1, 1, None]),
            (["c"], ["O"], [None, 0, 0, 0, None])]
    return {"input_ids": ids, "attention_mask": attn, "slot_labels": y,
            "intents": intents, "meta": meta}


def test_smoke():
    torch.manual_seed(0)
    bb = BertModel(BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=64,
                              max_position_embeddings=64))
    m = JointNLUTransformer(bb, num_slots=3, num_intents=2, intent_pool="cls_first")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    cs = nn.CrossEntropyLoss(ignore_index=IGNORE)
    ci = nn.CrossEntropyLoss()
    loss = train_one_epoch([_batch()], opt, cs, ci, m, "cpu", 1.0)
    assert torch.isfinite(torch.tensor(loss))
    id2slot = {0: "O", 1: "B-x", 2: "I-x"}
    f1, acc, _ = evaluate([_batch()], cs, ci, m, id2slot, "cpu")
    assert 0.0 <= f1 <= 1.0 and 0.0 <= acc <= 1.0
