import types
import torch
from lm_pipeline.data import read_file, subset_sentences, PennTreeBank, collate_fn

def test_read_file_appends_eos(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text("the cat\nthe dog\n")
    lines = read_file(str(p), eos_token="<eos>")
    assert lines == ["the cat <eos>", "the dog <eos>"]

def test_subset_fraction_keeps_whole_sentences():
    sents = [f"frase {i}" for i in range(100)]
    out = subset_sentences(sents, fraction=0.1, max_samples=None, seed=42)
    assert len(out) == 10
    assert all(s in sents for s in out)         # frasi intere, non spezzate

def test_subset_is_deterministic():
    sents = [f"s{i}" for i in range(50)]
    a = subset_sentences(sents, fraction=0.2, max_samples=None, seed=7)
    b = subset_sentences(sents, fraction=0.2, max_samples=None, seed=7)
    assert a == b                               # stesso seed -> stesso subset

def test_subset_max_samples_overrides_fraction():
    sents = [f"s{i}" for i in range(50)]
    out = subset_sentences(sents, fraction=1.0, max_samples=5, seed=1)
    assert len(out) == 5

def test_subset_full_when_fraction_one():
    sents = [f"s{i}" for i in range(20)]
    out = subset_sentences(sents, fraction=1.0, max_samples=None, seed=1)
    assert len(out) == 20

def test_dataset_len_and_getitem():
    ds = PennTreeBank(["a <eos>", "b <eos>"])
    assert len(ds) == 2 and ds[1] == "b <eos>"

def test_collate_weights_by_label_tokens():
    # n_tokens deve contare i token di LABEL non-pad (denominatore della loss),
    # non quelli di input: con padding i due conteggi differiscono.
    ids = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])   # pad_token_id = 0

    class FakeTok:
        pad_token_id = 0
        def __call__(self, batch, padding=True, return_tensors="pt"):
            return types.SimpleNamespace(input_ids=ids)

    input_ids, labels, n_tokens = collate_fn(["a", "b"], FakeTok())
    # labels = ids[:, 1:] = [[6,7,0],[9,0,0]] -> 3 token non-pad
    # input  = ids[:, :-1] = [[5,6,7],[8,9,0]] -> 5 token non-pad (valore sbagliato)
    assert int(n_tokens) == int((labels != 0).sum()) == 3
