"""Data layer NLU (ATIS): intent classification + slot filling.

ATIS ha esempi {utterance, slots (IOB), intent}. Costruiamo i vocabolari (Lang),
il dataset PyTorch e il collate con padding. Convenzione: PAD_TOKEN=0.

Token CLS: appendiamo un token speciale 'cls' IN CODA a ogni frase. Il modello e'
causale (decoder-only), quindi l'ultimo token e' l'unico che "vede" tutta la frase:
da li' leggiamo l'intent. Lo slot del CLS e' mappato a PAD (ignorato nella loss).
"""
import json
import os

import torch
import torch.utils.data as data

PAD_TOKEN = 0


def load_atis(dataset_dir):
    """Legge train.json e test.json di ATIS. Ritorna (train_raw, test_raw)."""
    with open(os.path.join(dataset_dir, "train.json")) as f:
        train_raw = json.load(f)
    with open(os.path.join(dataset_dir, "test.json")) as f:
        test_raw = json.load(f)
    return train_raw, test_raw


def stratified_dev_split(train_raw, portion=0.10, seed=42):
    """Crea il dev set (ATIS non lo ha) con campionamento stratificato sull'intent.

    Gli intent con una sola occorrenza restano nel train (non stratificabili), come
    nel lab. Riproducibile col seed. Ritorna (train, dev).
    """
    from collections import Counter
    from sklearn.model_selection import train_test_split

    intents = [x["intent"] for x in train_raw]
    counts = Counter(intents)
    inputs, labels, singletons = [], [], []
    for x in train_raw:
        if counts[x["intent"]] > 1:
            inputs.append(x)
            labels.append(x["intent"])
        else:
            singletons.append(x)  # intent unico -> sempre nel train
    X_tr, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=seed,
        shuffle=True, stratify=labels)
    X_tr.extend(singletons)
    return X_tr, X_dev


class Lang:
    """Vocabolari word2id / slot2id / intent2id (+ inversi).

    - word2id: pad(0), unk, cls, poi le parole del TRAIN.
    - slot2id: pad(0), le slot label, e 'cls'->PAD (lo slot del CLS si ignora).
    - intent2id: le intent label (nessun pad/cls).
    Le label (slot/intent) vengono raccolte da TUTTO il corpus passato (train+extra),
    per non avere classi sconosciute a test time; le parole solo dal train.
    """

    def __init__(self, train_raw, extra_raw=()):
        # parole: solo dal train
        words = []
        for x in train_raw:
            words += x["utterance"].split()
        # label: train + extra (dev/test) per coprire tutte le classi
        slots, intents = set(), set()
        for x in list(train_raw) + list(extra_raw):
            slots.update(x["slots"].split())
            intents.add(x["intent"])

        self.word2id = {"pad": PAD_TOKEN, "unk": 1, "cls": 2}
        for w in words:
            if w not in self.word2id:
                self.word2id[w] = len(self.word2id)

        self.slot2id = {"pad": PAD_TOKEN}
        for s in sorted(slots):
            if s not in self.slot2id:
                self.slot2id[s] = len(self.slot2id)
        self.slot2id["cls"] = PAD_TOKEN  # slot del CLS ignorato (stesso id del pad)

        self.intent2id = {it: i for i, it in enumerate(sorted(intents))}

        self.id2word = {v: k for k, v in self.word2id.items()}
        # nell'inverso degli slot escludiamo 'cls' (condivide l'id con 'pad')
        self.id2slot = {v: k for k, v in self.slot2id.items() if k != "cls"}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    @classmethod
    def from_dicts(cls, word2id, slot2id, intent2id):
        """Ricostruisce un Lang dai tre vocabolari salvati (es. da lang.json).

        Serve a inference time (consegna): si caricano i dizionari esatti usati in
        training, senza ricalcolarli dal dataset. Ricostruisce anche gli inversi.
        """
        self = cls.__new__(cls)
        self.word2id, self.slot2id, self.intent2id = word2id, slot2id, intent2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.id2slot = {v: k for k, v in slot2id.items() if k != "cls"}
        self.id2intent = {v: k for k, v in intent2id.items()}
        return self


class IntentsAndSlots(data.Dataset):
    """Dataset ATIS: mappa utterance/slots/intent in id e appende il CLS in coda.

    Parole fuori vocabolario -> 'unk'. La label di slot del token CLS e' 'cls'
    (=PAD), cosi' nella loss verra' ignorata insieme al padding.
    """

    def __init__(self, dataset, lang):
        self.lang = lang
        self.items = []
        cls_w, cls_s = lang.word2id["cls"], lang.slot2id["cls"]
        unk = lang.word2id["unk"]
        for x in dataset:
            utt = [lang.word2id.get(w, unk) for w in x["utterance"].split()] + [cls_w]
            slt = [lang.slot2id[s] for s in x["slots"].split()] + [cls_s]
            assert len(utt) == len(slt), "utterance e slots devono avere stessa lunghezza"
            self.items.append((utt, slt, lang.intent2id[x["intent"]]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt, slt, intent = self.items[idx]
        return {
            "utterance": torch.LongTensor(utt),
            "slots": torch.LongTensor(slt),
            "intent": intent,
        }


def collate_fn(batch):
    """Right-padding di utterance e slot a max_len (PAD=0). Tensori su CPU.

    Ritorna dict: utterances (B,L), y_slots (B,L), intents (B,), slots_len (B,).
    `slots_len` e' la lunghezza reale di ogni frase (CLS incluso): serve al modello
    per localizzare il token CLS (ultima posizione) da cui leggere l'intent, e all'eval
    per troncare il padding/CLS quando decodifica gli slot.
    """
    def merge(seqs):
        lengths = [len(s) for s in seqs]
        max_len = max(1, max(lengths))
        padded = torch.full((len(seqs), max_len), PAD_TOKEN, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :lengths[i]] = s
        return padded, torch.LongTensor(lengths)

    utt, lengths = merge([b["utterance"] for b in batch])
    slots, _ = merge([b["slots"] for b in batch])
    intents = torch.LongTensor([b["intent"] for b in batch])
    return {"utterances": utt, "y_slots": slots, "intents": intents, "slots_len": lengths}


def build_dataloaders(cfg_data, dataset_dir):
    """Costruisce (train_dl, dev_dl, test_dl, lang). Dev = split stratificato del train.

    Args:
        cfg_data: sezione 'data' della config (batch_size, dev_portion, split_seed).
        dataset_dir: cartella con train.json/test.json di ATIS.
    """
    from torch.utils.data import DataLoader

    train_raw, test_raw = load_atis(dataset_dir)
    train_raw, dev_raw = stratified_dev_split(
        train_raw, portion=cfg_data.get("dev_portion", 0.10),
        seed=cfg_data.get("split_seed", 42))
    # le label da train+dev+test (parole solo da train, dentro Lang)
    lang = Lang(train_raw, extra_raw=list(dev_raw) + list(test_raw))

    bs = cfg_data.get("batch_size", 64)
    train_dl = DataLoader(IntentsAndSlots(train_raw, lang), batch_size=bs,
                          collate_fn=collate_fn, shuffle=True)
    dev_dl = DataLoader(IntentsAndSlots(dev_raw, lang), batch_size=bs, collate_fn=collate_fn)
    test_dl = DataLoader(IntentsAndSlots(test_raw, lang), batch_size=bs, collate_fn=collate_fn)
    return train_dl, dev_dl, test_dl, lang
