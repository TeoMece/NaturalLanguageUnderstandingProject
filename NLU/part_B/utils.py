"""Data layer 2.B: ATIS sub-tokenizzato col tokenizer del modello (BPE/WordPiece).

Differenza da 2.A (tokenizzazione a parole): qui una parola puo' spezzarsi in piu'
sub-token. Allineamento (Chen et al. 2019): la label di slot della parola va al PRIMO
sub-token; gli altri sub-token e i token speciali ([CLS]/[SEP]/pad) -> -100 (ignorati
dalla CrossEntropy). In eval gli slot a livello di parola si ricostruiscono prendendo
le predizioni ai primi sub-token (via word_ids).
"""
import json
import os

import torch
import torch.utils.data as data

IGNORE = -100


def load_atis(dataset_dir):
    with open(os.path.join(dataset_dir, "train.json")) as f:
        train = json.load(f)
    with open(os.path.join(dataset_dir, "test.json")) as f:
        test = json.load(f)
    return train, test


def stratified_dev_split(train_raw, portion=0.10, seed=42):
    from collections import Counter
    from sklearn.model_selection import train_test_split
    intents = [x["intent"] for x in train_raw]
    counts = Counter(intents)
    inp, lab, singles = [], [], []
    for x in train_raw:
        if counts[x["intent"]] > 1:
            inp.append(x); lab.append(x["intent"])
        else:
            singles.append(x)
    X_tr, X_dev, _, _ = train_test_split(inp, lab, test_size=portion,
                                         random_state=seed, shuffle=True, stratify=lab)
    X_tr.extend(singles)
    return X_tr, X_dev


def build_label_maps(corpus):
    """slot2id (0..K, niente pad: gli ignorati sono -100) e intent2id, dal corpus."""
    slots, intents = set(), set()
    for x in corpus:
        slots.update(x["slots"].split())
        intents.add(x["intent"])
    slot2id = {s: i for i, s in enumerate(sorted(slots))}
    intent2id = {it: i for i, it in enumerate(sorted(intents))}
    return slot2id, intent2id


def get_tokenizer(model_name):
    """Tokenizer HF pronto per input gia' divisi in parole (is_split_into_words=True).

    - I tokenizer byte-level (GPT2/RoBERTa) richiedono `add_prefix_space=True` per
      tokenizzare input pre-divisi -> lo impostiamo quando serve.
    - GPT2 non ha un pad token nativo -> usiamo eos_token come pad.
    """
    from transformers import AutoTokenizer
    name = model_name.lower()
    kwargs = {"add_prefix_space": True} if ("gpt2" in name or "gpt" in name or "roberta" in name) else {}
    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def align_labels(word_ids, words, slots, slot2id):
    """Label per sub-token: primo sub-token di ogni parola -> slot2id[slot]; resto/special -> -100."""
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(IGNORE)
        elif wid != prev:
            labels.append(slot2id[slots[wid]])
        else:
            labels.append(IGNORE)
        prev = wid
    return labels


class IntentsAndSlotsBERTGPT2(data.Dataset):
    """Tokenizza ogni esempio col tokenizer del modello e allinea le label di slot."""

    def __init__(self, dataset, tokenizer, slot2id, intent2id, max_length=128):
        self.items = []
        for x in dataset:
            words = x["utterance"].split()
            slots = x["slots"].split()
            enc = tokenizer(words, is_split_into_words=True, truncation=True,
                            max_length=max_length)
            wids = enc.word_ids()
            labels = align_labels(wids, words, slots, slot2id)
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "slot_labels": labels,
                "intent": intent2id[x["intent"]],
                "words": words,
                "gold_slots": slots,
                "word_ids": wids,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_id):
    """Padding di input_ids/attention_mask/slot_labels; intent stacked; meta per eval.

    `meta` = lista di (words, gold_slots, word_ids) per esempio: serve in valutazione a
    ricostruire gli slot a livello di parola (predizione al primo sub-token di ogni parola).
    """
    maxlen = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    slot_labels = torch.full((B, maxlen), IGNORE, dtype=torch.long)
    intents = torch.tensor([b["intent"] for b in batch], dtype=torch.long)
    meta = []
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attn[i, :n] = torch.tensor(b["attention_mask"])
        slot_labels[i, :n] = torch.tensor(b["slot_labels"])
        meta.append((b["words"], b["gold_slots"], b["word_ids"]))
    return {"input_ids": input_ids, "attention_mask": attn,
            "slot_labels": slot_labels, "intents": intents, "meta": meta}


def build_dataloaders(cfg_data, dataset_dir, tokenizer):
    """Ritorna (train_dl, dev_dl, test_dl, slot2id, intent2id). Dev = split stratificato."""
    from functools import partial
    from torch.utils.data import DataLoader
    train_raw, test_raw = load_atis(dataset_dir)
    train_raw, dev_raw = stratified_dev_split(
        train_raw, cfg_data.get("dev_portion", 0.10), cfg_data.get("split_seed", 42))
    slot2id, intent2id = build_label_maps(train_raw + dev_raw + test_raw)
    bs = cfg_data.get("batch_size", 16)
    coll = partial(collate_fn, pad_id=tokenizer.pad_token_id)

    def mk(raw, shuffle):
        return DataLoader(IntentsAndSlotsBERTGPT2(raw, tokenizer, slot2id, intent2id),
                          batch_size=bs, collate_fn=coll, shuffle=shuffle)

    return mk(train_raw, True), mk(dev_raw, False), mk(test_raw, False), slot2id, intent2id
