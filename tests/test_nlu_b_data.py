"""Test del data layer 2.B: label maps, allineamento label ai sub-token, collate."""
from nlu_pipeline_b.data import (
    build_label_maps, align_labels, IntentsAndSlotsBERTGPT2, collate_fn, get_tokenizer,
)

_RAW = [
    {"utterance": "i want a flight to boston", "slots": "O O O O O B-toloc.city_name", "intent": "flight"},
    {"utterance": "what is the fare", "slots": "O O O O", "intent": "airfare"},
]


def _tok():
    # gpt2 in cache, fast -> word_ids; get_tokenizer aggiunge add_prefix_space e pad
    return get_tokenizer("openai-community/gpt2")


def test_label_maps():
    slot2id, intent2id = build_label_maps(_RAW)
    assert "O" in slot2id and "B-toloc.city_name" in slot2id
    assert set(intent2id) == {"flight", "airfare"}


def test_align_first_subtoken_gets_label_rest_ignored():
    tok = _tok()
    slot2id, _ = build_label_maps(_RAW)
    words = "i want a flight to boston".split()
    slots = "O O O O O B-toloc.city_name".split()
    enc = tok(words, is_split_into_words=True, return_tensors=None)
    labels = align_labels(enc.word_ids(), words, slots, slot2id)
    assert len(labels) == len(enc["input_ids"])
    word_ids = enc.word_ids()
    for i, wid in enumerate(word_ids):
        if wid is None:
            assert labels[i] == -100
    # ogni parola ha esattamente un sub-token etichettato (il primo)
    labeled = [labels[i] for i, wid in enumerate(word_ids) if wid is not None
               and (i == 0 or word_ids[i - 1] != wid)]
    assert len(labeled) == len(words)


def test_collate_shapes():
    tok = _tok()
    slot2id, intent2id = build_label_maps(_RAW)
    ds = IntentsAndSlotsBERTGPT2(_RAW, tok, slot2id, intent2id)
    batch = collate_fn([ds[0], ds[1]], pad_id=tok.pad_token_id)
    B = 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["slot_labels"].shape
    assert batch["input_ids"].shape[0] == B
    assert batch["intents"].shape == (B,)
    assert len(batch["meta"]) == B  # (words, gold_slots, word_ids) per esempio
