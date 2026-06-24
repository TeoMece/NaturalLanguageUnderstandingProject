"""Architettura per la Parte 2 (NLU): GPT2 decoder-only con due teste joint.

Il backbone (MultiHeadAttention, FeedForward, TransformerBlock, embedding, maschera
causale, init_weights) e' lo STESSO della Parte 1 (Language Model), qui **duplicato**
per rendere la pipeline NLU auto-contenuta. La differenza e' la testa di output:
invece della `lm_head` del LM, `GPT2JointIAS` ha due teste — una per lo slot filling
(per token) e una per l'intent classification (dal token CLS in coda) — vedi la
docstring della classe.

Nota: questo modulo NON importa nulla da config/experiment/tracking (livello "puro").
"""
import math

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
        # Scrive nel residual stream: init scalata 1/sqrt(2*num_layers) (vedi init_weights).
        self.out_proj._is_residual = True

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
        # Il 2o Linear (indice 2) scrive nel residual stream: init scalata
        # 1/sqrt(2*num_layers) (vedi init_weights).
        self.net[2]._is_residual = True

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


class GPT2JointIAS(nn.Module):
    """GPT2 decoder-only (backbone della Parte 1) con DUE teste per il task NLU joint:
    slot filling (una label per token) e intent classification (una label per frase).

    Differenza rispetto al Language Model della Parte 1: NON c'e' la `lm_head`. Al suo
    posto:
      - `slot_out`: Linear(d_model -> num_slots) applicata a OGNI posizione (sequence
        labeling). Lo slot del token CLS e del padding viene ignorato nella loss.
      - `intent_out`: Linear(d_model -> num_intents) applicata all'hidden del token
        **CLS**. Il CLS e' l'ULTIMO token di ogni frase: con la maschera causale e'
        l'unico che "vede" l'intera sequenza, quindi e' il punto giusto da cui
        classificare l'intento dell'intera frase.
      - `head_dropout`: dropout opzionale PRIMA delle due teste (e' lo step 2 di 2.A;
        con dropout=0.0 e' un no-op, cioe' il baseline).

    Parametri
    ----------
    vocab_size   : dimensione del vocabolario delle parole (da Lang.word2id)
    num_slots    : numero di classi di slot (da Lang.slot2id)
    num_intents  : numero di classi di intent (da Lang.intent2id)
    pos_emb_size : lunghezza massima di sequenza (embedding posizionale)
    d_model, n_heads, num_layers, ff_dim : iperparametri del transformer (come Parte 1)
    dropout      : probabilita' di dropout (nei blocchi e prima delle teste)
    """

    def __init__(self, vocab_size, num_slots, num_intents, pos_emb_size=512,
                 d_model=256, n_heads=4, num_layers=2, ff_dim=1024, dropout=0.0):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # embedding di token e posizione (identici alla Parte 1)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # stack di blocchi transformer (il backbone)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # dropout prima delle teste (step 2 di 2.A)
        self.head_dropout = nn.Dropout(dropout)
        # due teste: slot (per-token) e intent (dal CLS)
        self.slot_out = nn.Linear(d_model, num_slots)
        self.intent_out = nn.Linear(d_model, num_intents)

        # maschera causale (come Parte 1): il token i vede solo 0..i.
        # Restiamo decoder-only: per questo il CLS sta in CODA (vede tutto il passato).
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        """
        Parametri
        ----------
        idx      : (B, L) — batch di sequenze di token id (con CLS in coda, poi padding)
        seq_lens : (B,)   — lunghezza reale di ogni frase (CLS incluso); serve a trovare
                            la posizione del CLS da cui leggere l'intent

        Ritorna
        -------
        slot_logits   : (B, L, num_slots)   — logit di slot per ogni posizione
        intent_logits : (B, num_intents)    — logit di intent (uno per frase)
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size, (
            f"La sequenza ({L}) supera la lunghezza massima ({self.pos_emb_size})"
        )
        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.head_dropout(self.ln_f(x))

        # testa slot: una predizione per ogni posizione
        slot_logits = self.slot_out(x)  # (B, L, num_slots)

        # testa intent: hidden del CLS = ultimo token reale di ogni frase (seq_lens-1)
        cls = torch.stack([x[i, seq_lens[i] - 1] for i in range(B)])
        intent_logits = self.intent_out(cls)  # (B, num_intents)

        return slot_logits, intent_logits


def init_weights(mat):
    """Inizializza i pesi stile GPT-2, con residual scaling e tying-aware.

    Due passate, nell'ordine:
    1. ``nn.Embedding`` -> ``normal(0, 0.02)``. Senza questo, gli Embedding
       restavano all'init di default N(0,1).
    2. ``nn.Linear``    -> ``normal(0, 0.02)``, con due eccezioni:
       - proiezioni RESIDUALI (marcate ``_is_residual``: ``out_proj``
         dell'attention e il 2o ``Linear`` della FFN) -> ``normal(0, 0.02/sqrt(N))``
         con ``N = 2 * num_layers`` = numero di proiezioni residuali. E' il trucco
         di GPT-2 (Leva 1): tiene la varianza del residual stream costante con la
         profondita', cosi' i modelli profondi si allenano davvero.
       - pesi condivisi con un Embedding (weight tying:
         ``lm_head.weight is token_embed.weight``) -> SALTATI, per non
         sovrascrivere l'init dell'embedding gia' applicato.
       Il bias e' messo a 0 (standard GPT-2); nel tying NON e' condiviso.

    Cosi' la matrice condivisa mantiene la scala dell'embedding sia con tying
    attivo sia disattivo (l'unica differenza resta il tying), e la profondita'
    non gonfia piu' il residuo.

    Parametri
    ----------
    mat : nn.Module  — modello o sottomodulo da inizializzare
    """
    # Passata 1: Embedding -> normal(0, 0.02). Registra gli id dei tensori per
    # riconoscere i pesi condivisi (tying) nella passata successiva.
    emb_weight_ids = set()
    for m in mat.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            emb_weight_ids.add(id(m.weight))

    # Residual scaling (Leva 1, trucco GPT-2): le proiezioni che scrivono nel
    # residual stream (marcate _is_residual: out_proj dell'attention e il 2o
    # Linear della FFN) vanno inizializzate con std = 0.02 / sqrt(N_residual),
    # dove N_residual = numero di tali proiezioni = 2 * num_layers. Cosi' la
    # varianza del residuo resta costante con la profondita'. Contiamo le
    # proiezioni invece di passare num_layers: e' equivalente e auto-consistente.
    n_resid = sum(
        1 for m in mat.modules()
        if isinstance(m, nn.Linear) and getattr(m, "_is_residual", False)
    )
    residual_std = 0.02 / math.sqrt(n_resid) if n_resid > 0 else 0.02

    # Passata 2: Linear -> normal(0, 0.02), tranne: i residuali (std ridotto) e i
    # pesi condivisi con un Embedding (tying), che restano com'erano (gia' init).
    for m in mat.modules():
        if isinstance(m, nn.Linear):
            if id(m.weight) in emb_weight_ids:
                pass  # peso condiviso (tying): gia' inizializzato come Embedding
            elif getattr(m, "_is_residual", False):
                nn.init.normal_(m.weight, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()   # bias a 0 (standard GPT-2)
