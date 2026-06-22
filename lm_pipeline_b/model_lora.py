"""Parte 1.B — GPT2 pre-addestrato (HuggingFace) con LoRA fatto a mano su Q/K/V.

Contesto (dal lab 4): in 1.A abbiamo addestrato un GPT2 *da zero*. Qui invece
partiamo dai **pesi pre-addestrati** di GPT2 e li adattiamo al Penn Treebank
allenando SOLO un piccolo insieme di parametri aggiuntivi (gli "adapter" LoRA),
tenendo congelato tutto il resto. Cosi' otteniamo i benefici del pre-training
(il modello "sa gia' l'inglese") spendendo pochissime risorse.

Perche' LoRA (Low Rank Adaptation): fare il fine-tuning di TUTTI i ~124M parametri
di GPT2-small e' costoso. LoRA osserva che l'aggiornamento ottimale di una matrice
di pesi W e' spesso "a basso rango": puo' essere approssimato da B·A con B e A
matrici piccole (rango r << dim). Invece di toccare W (congelata), impariamo
ΔW = (alpha/r)·B·A e usiamo W_eff = W + ΔW. I parametri allenati sono solo quelli
di A e B: una frazione minima del totale.

Requisito del lab: implementare LoRA **a mano** (niente librerie tipo PEFT),
applicandolo alle matrici **query, key, value** dell'attention. Partiamo dallo
scaffold fornito dal lab (`CustomGPT2Attention`, `GPT2_LoRA`) e ne completiamo le
parti lasciate in `pass`, riusando il loro `forward` verbatim.

NB sui nomi: HuggingFace `transformers==4.38.0`. In GPT2 le tre proiezioni Q,K,V
sono FUSE in un unico layer `c_attn` (un `Conv1D` che produce 3*d_model e poi viene
splittato in q,k,v). Per applicare LoRA "a Q, K, V" aggiungiamo TRE adapter separati
(`lora_q`, `lora_k`, `lora_v`), uno per ciascuno stream dopo lo split.
"""
import torch
import torch.nn as nn

# GPT2Attention e' la classe base di HuggingFace di cui ereditiamo il forward.
# La importiamo dal modulo interno per poterne sottoclassare ed estendere il forward.
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2LMHeadModel


class LoRALinear(nn.Module):
    """Adapter LoRA per una singola proiezione lineare.

    Calcola il "delta" da sommare all'uscita della proiezione congelata:

        ΔW · x = (alpha / rank) · B · (A · x)

    dove:
      - A : (rank, in_features)  -> proietta l'input in uno spazio a basso rango
      - B : (out_features, rank) -> risale alla dimensione di uscita
      - alpha/rank               -> fattore di scala (normalizza l'effetto del rango)

    Inizializzazione (cruciale!): A ~ N(0, 0.02), **B = 0**. Con B=0 il delta e'
    NULLO a inizializzazione: il modello parte *identico* al GPT2 pre-addestrato e
    impara gradualmente lo scostamento. E' la ricetta standard di LoRA (paper
    Hu et al. 2021) e garantisce un fine-tuning stabile.

    Solo A e B sono parametri di questo modulo (pochi numeri); la matrice di pesi
    originale W vive nel layer base (es. `c_attn`) e resta congelata altrove.

    Parametri
    ----------
    in_features  : dimensione dell'input (per GPT2: d_model)
    out_features : dimensione dell'output (per Q/K/V: d_model)
    rank         : rango r dell'approssimazione (iperparametro, es. 4/8/16)
    alpha        : fattore di scala (iperparametro, es. 16/32)
    """

    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.rank = rank
        # Il fattore di scala alpha/rank disaccoppia "quanto" adatta l'adapter
        # dal rango scelto: cambiando rank a parita' di alpha l'ampiezza resta
        # confrontabile (consiglio del paper LoRA).
        self.scaling = alpha / rank

        # A e B come Linear senza bias: il bias non fa parte della formulazione LoRA.
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # A: piccoli valori casuali (rompe la simmetria, permette l'apprendimento).
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # B: zero -> ΔW = B·A = 0 a inizializzazione (vedi docstring).
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # x: (..., in_features) -> ritorna il delta (..., out_features), gia' scalato.
        return self.lora_B(self.lora_A(x)) * self.scaling


class CustomGPT2Attention(GPT2Attention):
    """GPT2Attention di HuggingFace estesa con adapter LoRA su Q, K, V.

    Eredita TUTTO il comportamento standard dell'attention di GPT2 (split delle
    teste, attenzione mascherata causale, proiezione di output, dropout) e
    aggiunge solo i tre adapter LoRA. Il metodo `forward` e' quello del lab 4
    (copiato verbatim dalla versione `transformers==4.38.0`), con UNA sola
    aggiunta: dopo aver calcolato q,k,v sommiamo il rispettivo delta LoRA.

    Parametri
    ----------
    config : GPT2Config del modello (fornisce hidden_size, num heads, ecc.)
    rank   : rango degli adapter LoRA
    alpha  : fattore di scala degli adapter LoRA
    """

    def __init__(self, config, rank, alpha):
        # Costruisce l'attention standard di GPT2 (c_attn, c_proj, dropout, mask...).
        super().__init__(config)

        # AGGIUNTA: tre adapter LoRA, uno per Q, uno per K, uno per V.
        # In GPT2 q,k,v hanno tutti dimensione d_model = config.hidden_size, sia in
        # input (hidden_states) sia in output (la fetta corrispondente di c_attn).
        d_model = config.hidden_size
        self.lora_q = LoRALinear(d_model, d_model, rank, alpha)
        self.lora_k = LoRALinear(d_model, d_model, rank, alpha)
        self.lora_v = LoRALinear(d_model, d_model, rank, alpha)

    # --- forward copiato verbatim da transformers 4.38.0 (come da scaffold del lab) ---
    # https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/gpt2/modeling_gpt2.py
    # L'UNICA modifica rispetto all'originale e' il blocco "AGGIUNTA LoRA" subito
    # dopo lo split di c_attn: tutto il resto e' identico, cosi' ereditiamo
    # esattamente la semantica dell'attention di GPT2.
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # c_attn produce [q | k | v] concatenati (3*d_model) e li separa in 3.
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            # ============================ AGGIUNTA LoRA ============================
            # Qui sta tutta la modifica richiesta dalla Parte 1.B: sommiamo a Q, K, V
            # il delta a basso rango calcolato dagli adapter a partire dallo STESSO
            # input dell'attention (hidden_states). Equivale a usare le matrici
            # "efficaci" W_q + ΔW_q, W_k + ΔW_k, W_v + ΔW_v senza toccare i pesi
            # congelati di c_attn. A init i delta sono 0 (B=0), quindi questo blocco
            # non altera il comportamento del GPT2 pre-addestrato al primo passo;
            # durante il training solo gli adapter imparano lo scostamento.
            # NB: lo facciamo PRIMA di _split_heads, sugli stream "interi" (d_model),
            # perche' gli adapter mappano d_model -> d_model esattamente come c_attn.
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(hidden_states)
            value = value + self.lora_v(hidden_states)
            # =======================================================================

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2_LoRA(GPT2LMHeadModel):
    """GPT2LMHeadModel di HuggingFace in cui ogni blocco usa `CustomGPT2Attention`.

    Si usa con `from_pretrained` per partire dai pesi pre-addestrati:

        model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=8, alpha=16)

    Meccanica (importante): `from_pretrained` prima costruisce il modello chiamando
    questo `__init__`, poi carica i pesi pre-addestrati nel modello PER NOME di
    parametro. Nel nostro `__init__` sostituiamo ogni `block.attn` con una
    `CustomGPT2Attention`: i nomi dei parametri base (`...attn.c_attn.weight`, ecc.)
    restano IDENTICI, quindi `from_pretrained` li popola comunque correttamente.
    I parametri nuovi (`...attn.lora_*`) non esistono nel checkpoint: HuggingFace
    li segnala come "newly initialized" e li lascia all'inizializzazione di LoRA
    (delta=0). E' esattamente il comportamento voluto.

    Parametri (oltre a quelli standard di GPT2LMHeadModel)
    ----------
    rank  : rango degli adapter LoRA
    alpha : fattore di scala degli adapter LoRA
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        # Costruisce il GPT2LMHeadModel standard (transformer + lm_head).
        super().__init__(*model_args, **model_kwargs)

        # Sostituisce l'attention di ogni blocco con la versione LoRA.
        for block in self.transformer.h:
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # Copia i pesi base attuali nel nuovo modulo (strict=False perche' il
            # nuovo modulo ha in piu' i parametri lora_*, assenti nel vecchio
            # state_dict). NB: a questo punto i pesi sono ancora quelli "freschi" di
            # __init__; quelli pre-addestrati veri verranno caricati DOPO da
            # from_pretrained, grazie ai nomi identici. Manteniamo comunque questa
            # copia perche' rende il modulo coerente anche se istanziato senza
            # from_pretrained (es. nei test con GPT2Config minima).
            new_attn.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        # Nessuna modifica al forward del LM: deleghiamo al padre. Il calcolo della
        # loss (se passiamo `labels=`) e lo shift interno dei target sono gestiti da
        # HuggingFace.
        return super().forward(*args, **kwargs)


def apply_lora_freezing(model):
    """Congela tutti i parametri tranne gli adapter LoRA (in-place).

    E' il passo che rende il fine-tuning "leggero": dopo questa chiamata
    l'ottimizzatore aggiornera' SOLO i pesi degli adapter (quelli con "lora_" nel
    nome), mentre i ~124M parametri pre-addestrati di GPT2 restano fissi.

    Ritorna il modello stesso per comodita' di concatenazione.
    """
    # 1) congela tutto
    for p in model.parameters():
        p.requires_grad = False
    # 2) riattiva solo gli adapter
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    return model


def param_stats(model):
    """Stampa e ritorna (totale, allenabili): utile per verificare il risparmio LoRA.

    Mostra a colpo d'occhio quanti pochi parametri stiamo davvero allenando
    rispetto al totale del modello (la "frazione di adapter").
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")
    return total, trainable
