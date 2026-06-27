"""Caricamento e preparazione del dataset Penn Treebank per il language modeling.

PTB ha una frase per riga. Aggiungiamo un token <eos> a fine frase. Il subset,
quando richiesto, seleziona FRASI INTERE (mai spezzate a meta'): nel LM la
predizione avviene dentro la sequenza, quindi tagliare per token corromperebbe
i target. Il tokenizer e' il BPE pre-addestrato di GPT2 (solo tokenizzazione,
nessun peso pre-addestrato usato in 1.A).
"""
import random
from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


def read_file(path, eos_token="<eos>"):
    """Legge un file PTB e ritorna la lista di frasi, ciascuna con <eos> in coda.

    Ogni riga del file corrisponde a una frase. Rimuoviamo gli spazi iniziali/finali
    con strip() e aggiungiamo il token di fine frase separato da uno spazio.
    Il token <eos> segnala al modello il confine di frase durante il training.
    """
    out = []
    with open(path, "r") as f:
        for line in f.readlines():
            # strip() rimuove '\n' e spazi; poi appendiamo il token di fine frase
            out.append(line.strip() + " " + eos_token)
    return out


def subset_sentences(sentences, fraction=1.0, max_samples=None, seed=42):
    """Sottocampiona la lista di frasi in modo riproducibile (per frasi intere).

    `max_samples` (se valorizzato) ha priorita' su `fraction`. Con fraction=1.0 e
    max_samples=None ritorna tutte le frasi (ordine invariato).

    Perche' frasi intere: nel language modeling il modello impara a predire il
    token successivo data la storia. Se spezzassimo le frasi in posizione casuale,
    i target sarebbero corrotti (la label del last token di un pezzo non corrisponde
    al vero token successivo del testo). Selezionare frasi intere preserva la
    coerenza input/label.

    La riproducibilita' e' garantita fissando il seed di random.Random (isolato
    dall'RNG globale di Python, cosi' non interferisce con altri moduli).
    Gli indici vengono ordinati prima della selezione per mantenere l'ordine
    originale delle frasi nel corpus.

    Args:
        sentences: lista di stringhe (una frase per elemento).
        fraction:  frazione del corpus da mantenere (0.0-1.0).
        max_samples: se intero, limita il numero di frasi a questo valore,
                     ignorando `fraction`.
        seed:      seed per la riproducibilita' del campionamento.

    Returns:
        Lista di frasi selezionate (frasi intere, ordine originale preservato).
    """
    # Caso degenere: nessun sottoinsieme richiesto -> ritorniamo tutte le frasi
    if max_samples is None and (fraction is None or fraction >= 1.0):
        return list(sentences)

    # Calcolo del numero di frasi da estrarre:
    # max_samples ha la precedenza assoluta su fraction
    n = max_samples if max_samples is not None else max(1, int(len(sentences) * fraction))

    # RNG isolato: non modifica lo stato globale di random, garantisce
    # riproducibilita' indipendente da quanto random e' stato usato altrove
    rng = random.Random(seed)

    # sorted() mantiene l'ordine originale delle frasi nel corpus
    idx = sorted(rng.sample(range(len(sentences)), min(n, len(sentences))))
    return [sentences[i] for i in idx]


class PennTreeBank(data.Dataset):
    """Dataset minimale: una frase (stringa) per elemento.

    Wrappa una lista di stringhe nell'interfaccia torch.utils.data.Dataset,
    consentendo l'uso di DataLoader con batching automatico e shuffling.
    La tokenizzazione avviene nel collate_fn (lazy, a batch time) per
    permettere padding omogeneo all'interno del batch.
    """

    def __init__(self, corpus):
        """Inizializza il dataset con la lista di frasi grezze (stringhe).

        Args:
            corpus: iterabile di stringhe (ogni stringa e' una frase con <eos>).
        """
        # Convertiamo in lista per garantire indicizzazione O(1)
        self.sents = list(corpus)

    def __len__(self):
        """Ritorna il numero di frasi nel dataset."""
        return len(self.sents)

    def __getitem__(self, idx):
        """Ritorna la frase (stringa grezza) all'indice `idx`."""
        return self.sents[idx]


def collate_fn(batch, tokenizer):
    """Tokenizza un batch di frasi e costruisce input/labels per il next-token.

    Le labels sono l'input shiftato a sinistra di una posizione. Questo implementa
    il teacher forcing standard per language modeling:
        input:  [w0, w1, ..., w_{T-1}]   (tutti i token tranne l'ultimo)
        labels: [w1, w2, ..., w_T]       (tutti i token tranne il primo)
    In questo modo il modello impara "dato il prefisso, predici il prossimo token".

    Il padding viene aggiunto dal tokenizer per allineare le sequenze del batch.
    I tensori restano su CPU: lo spostamento su device avviene nel training loop
    (cosi' funziona anche con num_workers > 0, che usa processi separati senza
    accesso alla GPU del processo principale).

    Args:
        batch:     lista di stringhe (un batch di frasi grezze).
        tokenizer: tokenizer HuggingFace con pad_token configurato.

    Returns:
        Tupla (input_ids, labels, n_tokens):
            - input_ids: LongTensor [B, T-1] — sequenza di input
            - labels:    LongTensor [B, T-1] — sequenza di target (left-shifted)
            - n_tokens:  scalare intero — numero di token non-padding nel batch
                         (usato per normalizzare la loss in modo corretto)
    """
    # Tokenizzazione con padding dinamico al massimo del batch
    tok = tokenizer(batch, padding=True, return_tensors="pt")

    # Left-shift: input_ids[:, :-1] e labels[:, 1:] sono di lunghezza T-1
    # contiguous() assicura layout di memoria contiguo (richiesto da certi op)
    input_ids = tok.input_ids[:, :-1].contiguous()
    labels = tok.input_ids[:, 1:].contiguous()

    # Contiamo i token reali (non di padding) su cui viene calcolata la loss.
    # IMPORTANTE: si contano i token di LABEL, non di input. La CrossEntropyLoss
    # con ignore_index=pad media sui soli target non-pad; per ottenere una PPL
    # token-weighted corretta il peso di ogni batch deve usare lo stesso
    # denominatore della media, cioe' il numero di label non-pad.
    n_tokens = torch.sum(labels != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens


def get_tokenizer():
    """Tokenizer BPE di GPT2, con pad_token = eos_token (GPT2 non ha pad nativo).

    GPT2 e' un tokenizer BPE (Byte-Pair Encoding) con vocabolario da 50.257 token.
    Non ha un pad_token di default: lo impostiamo uguale a eos_token (50256).
    Questo e' lo standard per GPT2 in contesti di batch padding; la loss sui
    token di padding viene mascherata nel training loop tramite n_tokens.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # GPT2 non ha un pad token nativo: usiamo eos_token come convenzione
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloaders(cfg_data, dataset_dir, tokenizer):
    """Costruisce (train, valid, test) DataLoader applicando il subset al solo train.

    `cfg_data` e' la sezione `data` della config. Il subset si applica al training
    set; valid e test restano interi per metriche confrontabili tra esperimenti.

    Il DataLoader di train usa shuffle=True per rompere le correlazioni di ordine
    tra batch (importante per la convergenza); valid e test non vengono shufflati
    per riproducibilita' della valutazione.

    Args:
        cfg_data:     dizionario con chiavi opzionali:
                        - fraction (float, default 1.0): frazione del train da usare
                        - max_samples (int|None): numero massimo di frasi di train
                        - subset_seed (int, default 42): seed per il subset
                        - batch_size (int, default 32): dimensione del batch
        dataset_dir:  cartella che contiene ptb.train.txt, ptb.valid.txt, ptb.test.txt
        tokenizer:    tokenizer HuggingFace gia' configurato (vedi get_tokenizer())

    Returns:
        Tupla (train_dl, dev_dl, test_dl, n_train_sents):
            - train_dl:      DataLoader per il training set (con shuffle)
            - dev_dl:        DataLoader per il validation set
            - test_dl:       DataLoader per il test set
            - n_train_sents: numero di frasi nel training set dopo il subset
    """
    import os

    # Lettura dei tre split del PTB
    train_raw = read_file(os.path.join(dataset_dir, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_dir, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_dir, "ptb.test.txt"))

    # Sottoinsieme del solo training set (valid e test restano interi)
    train_raw = subset_sentences(
        train_raw,
        fraction=cfg_data.get("fraction", 1.0),
        max_samples=cfg_data.get("max_samples"),
        seed=cfg_data.get("subset_seed", 42),
    )

    bs = cfg_data.get("batch_size", 32)
    # partial fissa il tokenizer come argomento keyword, rendendo collate_fn
    # compatibile con l'interfaccia DataLoader (che chiama fn(batch))
    # ora basta un solo parametro (batch) e il tokenizer e' "catturato" nella closure di partial
    coll = partial(collate_fn, tokenizer=tokenizer)

    train_dl = DataLoader(PennTreeBank(train_raw), batch_size=bs, collate_fn=coll, shuffle=True)
    dev_dl = DataLoader(PennTreeBank(dev_raw), batch_size=bs, collate_fn=coll)
    test_dl = DataLoader(PennTreeBank(test_raw), batch_size=bs, collate_fn=coll)
    return train_dl, dev_dl, test_dl, len(train_raw)
