"""Test del data layer NLU (ATIS): Lang, dataset con CLS, collate, dev split."""
import torch

from nlu_pipeline.data import (
    Lang, IntentsAndSlots, collate_fn, stratified_dev_split, PAD_TOKEN,
)

_RAW = [
    {"utterance": "i want a flight to boston", "slots": "O O O O O B-toloc.city_name", "intent": "flight"},
    {"utterance": "list flights", "slots": "O O", "intent": "flight"},
    {"utterance": "what is the fare", "slots": "O O O O", "intent": "airfare"},
]


def test_lang_special_tokens():
    lang = Lang(_RAW)
    assert lang.word2id["pad"] == PAD_TOKEN
    assert "unk" in lang.word2id and "cls" in lang.word2id
    # slot 'cls' e 'pad' mappano a PAD (ignorati nella loss degli slot)
    assert lang.slot2id["pad"] == PAD_TOKEN and lang.slot2id["cls"] == PAD_TOKEN
    assert set(lang.intent2id) == {"flight", "airfare"}
    assert lang.id2intent[lang.intent2id["flight"]] == "flight"


def test_dataset_appends_cls():
    lang = Lang(_RAW)
    ds = IntentsAndSlots(_RAW, lang)
    s = ds[0]
    # ultimo token = CLS; lunghezza utt == lunghezza slot
    assert s["utterance"][-1].item() == lang.word2id["cls"]
    assert len(s["utterance"]) == len(s["slots"])
    assert s["slots"][-1].item() == PAD_TOKEN  # slot del CLS = pad


def test_collate_pads_and_aligns():
    lang = Lang(_RAW)
    ds = IntentsAndSlots(_RAW, lang)
    batch = collate_fn([ds[0], ds[1]])
    B = 2
    assert batch["utterances"].shape == batch["y_slots"].shape
    assert batch["utterances"].shape[0] == B
    assert batch["intents"].shape == (B,)
    assert batch["slots_len"].shape == (B,)


def test_stratified_split_reproducible():
    raw = _RAW * 10
    tr1, dv1 = stratified_dev_split(raw, portion=0.2, seed=42)
    tr2, dv2 = stratified_dev_split(raw, portion=0.2, seed=42)
    assert [x["utterance"] for x in dv1] == [x["utterance"] for x in dv2]
    assert len(dv1) > 0 and len(tr1) + len(dv1) == len(raw)
