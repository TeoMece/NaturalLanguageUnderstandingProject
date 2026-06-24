"""Modello joint 2.B: backbone HuggingFace (GPT2/BERT) + teste slot e intent.

A differenza di 2.A (GPT2 da zero), qui il backbone e' un modello PRE-ADDESTRATO
caricato da HuggingFace. Aggiungiamo due teste lineari sopra le hidden states:
- slot (per sub-token), intent (dal token di frase).
L'intent si legge da posizioni diverse a seconda dell'architettura (vedi sotto).
"""
import torch
import torch.nn as nn


class JointNLUTransformer(nn.Module):
    """Avvolge un backbone HF (che espone last_hidden_state) con due teste.

    - slot_out: Linear(H, num_slots) su ogni sub-token.
    - intent_out: Linear(H, num_intents) sul token di frase:
        intent_pool='cls_first' (BERT) -> hidden della posizione 0 ([CLS], bidirezionale);
        intent_pool='last'      (GPT2) -> hidden dell'ultimo token reale (causale; lo
        localizziamo con attention_mask, perche' col padding a destra l'ultimo token
        valido e' in posizione sum(mask)-1).
    """

    def __init__(self, backbone, num_slots, num_intents, intent_pool):
        super().__init__()
        self.backbone = backbone
        H = backbone.config.hidden_size
        self.slot_out = nn.Linear(H, num_slots)
        self.intent_out = nn.Linear(H, num_intents)
        assert intent_pool in ("cls_first", "last"), intent_pool
        self.intent_pool = intent_pool

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        slot_logits = self.slot_out(out)                      # (B, L, num_slots)
        if self.intent_pool == "cls_first":
            sent = out[:, 0]                                  # [CLS] in testa (BERT)
        else:
            last_idx = attention_mask.sum(1) - 1              # ultimo token reale (GPT2)
            sent = out[torch.arange(out.size(0), device=out.device), last_idx]
        intent_logits = self.intent_out(sent)                 # (B, num_intents)
        return slot_logits, intent_logits


def build_joint_model(model_name, num_slots, num_intents, intent_pool=None):
    """Factory: AutoModel.from_pretrained + intent_pool dedotto dal nome se non passato.

    BERT (encoder) -> intent dal [CLS] in testa; GPT2 (decoder) -> dall'ultimo token.
    """
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(model_name)
    if intent_pool is None:
        intent_pool = "cls_first" if "bert" in model_name.lower() else "last"
    return JointNLUTransformer(backbone, num_slots, num_intents, intent_pool)
