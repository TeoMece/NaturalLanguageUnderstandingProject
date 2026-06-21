"""Architettura GPT2 (decoder-only) parametrica, ispirata a minGPT/lab NLU.

Rispetto al baseline del lab, qui i 4 punti di dropout previsti dalla Parte 1.A
sono integrati e attivabili (dropout > 0), e il weight tying e' un flag. Con
dropout=0.0 e weight_tying=False si ottiene esattamente il baseline.

Nota: questo modulo appartiene al livello *puro* della pipeline e NON importa
nulla da config, experiment o tracking, per garantire massima riusabilita'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Masked multi-head self-attention con dropout opzionale in due punti:
    sui pesi di attention (dopo softmax) e dopo la output projection.

    Parametri
    ----------
    d_model  : dimensione del modello (deve essere divisibile per n_heads), lunghezza del vettore di embedding per token
    n_heads  : numero di teste di attention parallele
    dropout  : probabilita' di dropout (0.0 = nessun dropout, baseline puro)
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) deve essere divisibile per n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads   # dimensione per singola testa

        # proiezioni lineari per query, key e value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # proiezione di output (concat delle teste -> d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Punto dropout 2: sui pesi di attention dopo softmax
        # Regolarizza il pattern di attention durante il training
        self.attn_dropout = nn.Dropout(dropout)

        # Punto dropout 3: dopo la output projection dell'attention
        # Regolarizza l'output aggregato prima della connessione residua
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Parametri
        ----------
        x    : (B, L, d_model)  — sequenza di embeddings
        mask : (1, 1, L, L)     — maschera causale (triangolare inferiore)

        Ritorna
        -------
        y    : (B, L, d_model)  — output dell'attention
        """
        B, L, d_model = x.size()

        # Calcola Q, K, V e suddivide in n_heads teste.
        # NB l'ordine conta: prima w_q(x) e' una Linear d_model->d_model che RIMESCOLA
        # l'intero embedding (ogni dim di output dipende da tutte le d_model di input);
        # solo DOPO il .view spezza il vettore gia' proiettato in n_heads fette da h_dim.
        # Quindi ogni testa non vede "un pezzo grezzo" dell'embedding, ma una proiezione
        # imparata dell'embedding intero -> e' qui che nascono i diversi punti di vista.
        # La .transpose(1,2) porta n_heads in posizione di "batch" (B, n_heads, L, h_dim):
        # cosi' il matmul successivo opera SEPARATAMENTE per ogni testa (sulle ultime 2 dim).
        # IMPORTANTE: questi .view NON sono "solo forma" -> sono l'operazione che CREA le teste
        # (spezza d_model in n_heads x h_dim) e avviene PRIMA del softmax. E' proprio questo
        # split a monte ad abilitare il softmax separato per testa (riga ~87): senza, q @ k^T
        # darebbe UNA sola matrice di attenzione e avremmo una testa sola. La matematica
        # cambia, non solo le forme. (Il .view di ricomposizione finale, riga ~97, invece e'
        # necessario solo per le forme e non cambia la distribuzione gia' calcolata.)
        q = self.w_q(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # Scores scalati: (B, n_heads, L, L) -> n_heads matrici di attenzione INDIPENDENTI,
        # una per testa (il matmul e' batchato su B e n_heads). E' questo che distingue il
        # multi-head da una testa sola: ogni parola puo' guardare cose diverse in teste diverse.
        # k.transpose(-2,-1) scambia solo le ultime 2 dim (L, h_dim)->(h_dim, L) per allineare
        # il prodotto q @ k^T: (L, h_dim) @ (h_dim, L) = (L, L) = ogni parola contro ogni parola.
        similarity = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.h_dim, dtype=torch.float)))

        # Applica la maschera causale: posizioni future diventano -inf
        # cosi' dopo softmax il loro peso e' 0 (il token non vede il futuro)
        similarity = similarity.masked_fill(mask == 0, float("-inf"))

        # Normalizza in distribuzioni di probabilita' sulle posizioni passate
        attn = F.softmax(similarity, dim=-1)

        # Punto dropout 2: applica dropout sui pesi di attention
        attn = self.attn_dropout(attn)

        # Weighted sum dei value: (B, n_heads, L, h_dim) -> riporta a (B, L, d_model).
        # La .transpose(1,2) ANNULLA quella di prima (rimette L davanti a n_heads), cosi'
        # il .view puo' ri-concatenare le fette delle teste affiancate parola per parola.
        # Il .contiguous() serve perche' la transpose cambia solo gli stride (come si legge
        # la memoria, non l'ordine fisico); .view invece pretende dati contigui in memoria.
        y = (attn @ v).transpose(1, 2).contiguous().view(B, L, d_model)

        # Proiezione di output che mescola le informazioni tra le teste (dopo la concat)
        y = self.out_proj(y)

        # Punto dropout 3: applica dropout dopo la output projection
        y = self.proj_dropout(y)
        return y


class FeedForward(nn.Module):
    """Feed-forward con GELU e dropout opzionale dopo l'ultimo linear (punto 4).

    Struttura: Linear -> GELU -> Linear -> Dropout
    La dimensione nascosta e' tipicamente 4x d_model (configurabile con ff_dim).

    Parametri
    ----------
    d_model    : dimensione di input e output, lunghezza del vettore di embedding per token
    hidden_dim : dimensione del layer nascosto (es. 4 * d_model)
    dropout    : probabilita' di dropout dopo il secondo linear
    """

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # espansione
            nn.GELU(),                        # attivazione non lineare (stile GPT2)
            nn.Linear(hidden_dim, d_model),   # proiezione di ritorno
            nn.Dropout(dropout),              # Punto dropout 4: dopo l'ultimo linear
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Blocco GPT2: layer-norm pre-modulo, attention e feed-forward con residui.

    Segue lo schema GPT2 originale con Pre-LayerNorm:
      x -> LN -> Attention -> residuo
      x -> LN -> FeedForward -> residuo

    Parametri
    ----------
    d_model  : dimensione del modello
    n_heads  : numero di teste di attention
    ff_dim   : dimensione nascosta del feed-forward
    dropout  : probabilita' di dropout (passato a attention e feed-forward)
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)   # layer norm prima dell'attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)   # layer norm prima del feed-forward
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        """
        Parametri
        ----------
        x    : (B, L, d_model)
        mask : (1, 1, L, L)

        Ritorna
        -------
        x    : (B, L, d_model) dopo attention e feed-forward con residui
        """
        # Connessione residua attorno all'attention (Pre-LN)
        x = x + self.attn(self.ln1(x), mask)
        # Connessione residua attorno al feed-forward (Pre-LN)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT2 decoder-only parametrico per il language modeling.

    Architettura:
      token_embed + pos_embed -> emb_dropout -> N * TransformerBlock -> ln_f -> lm_head

    Parametri
    ----------
    vocab_size   : dimensione del vocabolario
    pos_emb_size : lunghezza massima della sequenza (dimensione dell'embedding posizionale)
    d_model      : dimensione degli embeddings e del modello
    n_heads      : numero di teste di multi-head attention
    num_layers   : numero di TransformerBlock impilati
    ff_dim       : dimensione nascosta del feed-forward in ogni blocco
    dropout      : probabilita' di dropout applicata ai 4 punti previsti dalla Parte 1.A:
                   1) post-embedding (emb_dropout)
                   2) sui pesi di attention dopo softmax (in MultiHeadAttention)
                   3) dopo la output projection dell'attention (in MultiHeadAttention)
                   4) dopo il secondo linear del feed-forward (in FeedForward)
                   Con dropout=0.0 si ottiene il baseline puro senza regolarizzazione.
    weight_tying : se True, condivide i pesi tra token_embed e lm_head,
                   tecnica comune nei LM che riduce i parametri e migliora la generalizzazione.
                   Con weight_tying=False i pesi sono indipendenti (baseline).
    """

    # NB: questi default (d_model=768, n_heads=12, num_layers=12, ff_dim=3072) coincidono
    # con GPT2-small "ufficiale". Ma sono solo i default: negli esperimenti la config li
    # sovrascrive sempre con valori a scala ridotta (es. num_layers=4 in base_partA.yaml,
    # sweep [4,6] in 01_arch.yaml), per vincoli di costo. num_layers e d_model NON sono
    # imposti da GPT2: sono iperparametri che decidiamo/tuniamo noi.
    def __init__(self, vocab_size, pos_emb_size=1024, d_model=768, n_heads=12,
                 num_layers=12, ff_dim=3072, dropout=0.0, weight_tying=False):
        super().__init__()

        self.pos_emb_size = pos_emb_size

        # d_model = LARGHEZZA del modello: dimensione del vettore con cui ogni token e'
        # rappresentato lungo TUTTO il percorso (il "residual stream"). Non e' "il numero
        # di neuroni": il concetto piu' vicino a una hidden layer di MLP e' semmai ff_dim,
        # interno al FeedForward. d_model si divide tra le teste: h_dim = d_model / n_heads.
        # nn.Embedding = TABELLA di lookup imparata, forma (vocab_size, d_model): una riga
        # per token. Dato un id NON fa calcoli, restituisce la riga corrispondente. I numeri
        # nella tabella sono parametri: partono random e il training li rende significativi
        # (parole simili -> vettori vicini, non imposto ma appreso).
        # Token embedding -> "COSA e' questa parola": id (etichetta senza senso numerico)
        # diventa un vettore denso su cui si possono fare moltiplicazioni e gradienti.
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional embedding -> "DOVE si trova": una riga per posizione 0..pos_emb_size-1.
        # Serve perche' l'attention DA SOLA ignora l'ordine ("il gatto" == "gatto il" per
        # q @ k^T). Qui le posizioni sono IMPARATE (scelta di GPT2); il Transformer originale
        # usava posizioni fisse con seni/coseni. token_embed e pos_embed vengono poi SOMMATI
        # nel forward (stessa dim d_model): un unico vettore che codifica identita' + posizione.
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # Punto dropout 1: applicato alla somma token_embed + pos_embed
        # Regolarizza le rappresentazioni di input prima di entrare nei blocchi
        self.emb_dropout = nn.Dropout(dropout)

        # Stack di TransformerBlock (cuore del modello). Il numero di blocchi e' deciso
        # A PRIORI qui (num_layers) e resta fisso: il forward itera su quelli creati, non
        # ne aggiunge/toglie. nn.ModuleList (non una lista Python normale) e' essenziale:
        # registra i sottomoduli, cosi' i loro parametri finiscono in model.parameters(),
        # si spostano con .to(device) e si salvano nello state_dict.
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Layer norm finale prima della testa di classificazione (come in GPT2 originale)
        self.ln_f = nn.LayerNorm(d_model)

        # Testa di classificazione: proietta d_model -> vocab_size per produrre i logit
        self.lm_head = nn.Linear(d_model, vocab_size)

        if weight_tying:
            # Weight tying: l'embedding di output condivide i pesi con l'embedding di input.
            # Questo e' motivato dal fatto che token simili semanticamente dovrebbero avere
            # rappresentazioni simili sia in input che in output. Riduce anche il numero
            # di parametri da (vocab_size * d_model) a zero per lm_head.weight.
            self.lm_head.weight = self.token_embed.weight

        # Maschera causale (triangolare inferiore): il token in posizione i puo' vedere
        # solo le posizioni 0..i (autoregressive masking). Registrata come buffer in modo
        # da essere spostata automaticamente sul dispositivo corretto con .to(device).
        # torch.tril = TRIangular Lower: prende una matrice di 1 e AZZERA la parte sopra la
        # diagonale, lasciando 1 sulla diagonale e sotto. Letta come righe=chi guarda,
        # colonne=chi e' guardato: riga i ha 1 solo in 0..i -> ogni token vede se' e il
        # passato, mai il futuro. Gli 0 diventano poi -inf nel masked_fill (vedi attention)
        # e il softmax li annulla. unsqueeze(0)x2: (L,L) -> (1,1,L,L) per broadcast su B e n_heads.
        #
        # E' QUESTO a rendere il modello "decoder-only" (stile GPT, generazione):
        #  - encoder-only (BERT): SI TOGLIE questa maschera -> attenzione bidirezionale,
        #    ogni token vede tutto; obiettivo = indovinare token mascherati (comprensione).
        #  - encoder-decoder (T5): due stack + cross-attention (q dal decoder, k/v dall'encoder);
        #    l'encoder senza maschera, il decoder con. I mattoni (attention, FF, embedding,
        #    residui, layernorm) restano gli stessi: cambia soprattutto la maschera + l'obiettivo.
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        """
        Parametri
        ----------
        idx : (B, L)  — batch di sequenze di token ids (interi)

        Ritorna
        -------
        logits : (B, L, vocab_size) — distribuzioni non normalizzate per ogni posizione
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size, (
            f"La sequenza ({L}) supera la lunghezza massima supportata ({self.pos_emb_size})"
        )

        # Indici di posizione 0, 1, ..., L-1 sullo stesso device di idx
        pos = torch.arange(L, device=idx.device)

        # Combina embedding token e posizionale (somma element-wise)
        x = self.token_embed(idx) + self.pos_embed(pos)

        # Punto dropout 1: dropout post-embedding
        x = self.emb_dropout(x)

        # Ritaglia la maschera alla lunghezza effettiva della sequenza
        mask = self.mask[:, :, :L, :L]

        # Passa attraverso ogni TransformerBlock
        for block in self.blocks:
            x = block(x, mask)

        # Layer norm finale
        x = self.ln_f(x)

        # Proietta su vocab_size per ottenere i logit grezzi
        return self.lm_head(x)


def init_weights(mat):
    """Inizializza i pesi in modo coerente e compatibile col weight tying.

    Due passate, nell'ordine:
    1. ``nn.Embedding`` -> ``normal(0, 0.02)`` (scala sensata, stile GPT-2).
       Senza questo, gli Embedding restavano all'init di default N(0,1).
    2. ``nn.Linear``    -> ``uniform(-0.01, 0.01)`` (init del lab), MA i Linear
       il cui peso e' condiviso con un Embedding (weight tying:
       ``lm_head.weight is token_embed.weight``) vengono SALTATI per non
       sovrascrivere l'init dell'embedding gia' applicato. Si imposta comunque
       il bias, che nel tying NON e' condiviso.

    Cosi' la matrice condivisa mantiene la scala dell'embedding sia con tying
    attivo sia disattivo: l'unica differenza tra le due configurazioni resta il
    tying stesso (confronto pulito), e non piu' la scala di inizializzazione.

    Parametri
    ----------
    mat : nn.Module  — modello o sottomodulo da inizializzare
    """
    # Passata 1: Embedding. Registra gli id dei tensori per riconoscere i pesi
    # condivisi (tying) nella passata successiva.
    emb_weight_ids = set()
    for m in mat.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            emb_weight_ids.add(id(m.weight))

    # Passata 2: Linear. Salta il peso se condiviso con un Embedding (tying).
    for m in mat.modules():
        if isinstance(m, nn.Linear):
            if id(m.weight) not in emb_weight_ids:
                nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
