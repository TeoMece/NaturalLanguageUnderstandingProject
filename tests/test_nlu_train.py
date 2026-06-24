"""Smoke test del training/eval multi-task NLU su CPU (loader finto, vocab minimo)."""
import torch
import torch.nn as nn

from nlu_pipeline.model import GPT2JointIAS, init_weights
from nlu_pipeline.train import train_one_epoch, evaluate
from nlu_pipeline.data import PAD_TOKEN


class _Lang:
    """Lang minimale per il test (solo gli inversi servono all'eval)."""
    id2word = {0: "pad", 1: "a", 2: "b"}
    id2slot = {0: "pad", 1: "O", 2: "B-x"}
    id2intent = {0: "flight", 1: "fare"}


def _batch():
    utt = torch.tensor([[1, 2, 2], [1, 2, 0]])      # 0 = pad
    y_slots = torch.tensor([[1, 2, 0], [1, 1, 0]])  # 0 = pad/cls (ignorato)
    intents = torch.tensor([0, 1])
    seq_lens = torch.tensor([3, 2])
    return {"utterances": utt, "y_slots": y_slots, "intents": intents, "slots_len": seq_lens}


def test_train_and_eval_smoke():
    torch.manual_seed(0)
    m = GPT2JointIAS(vocab_size=3, num_slots=3, num_intents=2,
                     pos_emb_size=8, d_model=16, n_heads=2, num_layers=1, ff_dim=32)
    init_weights(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cs = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    ci = nn.CrossEntropyLoss()

    loss = train_one_epoch([_batch()], opt, cs, ci, m, "cpu", grad_clip=5.0)
    assert torch.isfinite(torch.tensor(loss))  # loss finita (non NaN)

    f1, acc, _ = evaluate([_batch()], cs, ci, m, _Lang(), "cpu")
    assert 0.0 <= f1 <= 100.0          # conll F1 in scala percentuale
    assert 0.0 <= acc <= 1.0
