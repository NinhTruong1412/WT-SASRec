"""
models/wt_sasrec.py
-------------------
Watch-Time Self-Attentive Sequential Recommendation (WTSASRec)

Core thesis contribution (Direction 1):
  "Modify SASRec to weight attention scores by normalized watch time.
   Items watched longer get higher influence on the next prediction."

Mechanism
---------
Standard SASRec attention:
  attn_scores = QK^T / sqrt(d)
  output      = softmax(attn_scores + mask) · V

WTSASRec adds a watch-time bias per key position:
  wt_norm   = log(1 + wt) / log(1 + max_wt + ε)       # (B, L), in [0,1]
  wt_bias   = Linear(1 → num_heads)(wt_norm)            # (B, L, H_heads)
  wt_bias   = alpha * wt_bias                           # alpha: learnable scalar
  attn_scores += wt_bias.permute(0,2,1).unsqueeze(2)    # broadcast over queries
                                                        # (B, H, 1, L)

This lets every query attend more to keys (past items) that were watched longer,
without changing the model's parameters for non-watch-time data.
"""

import math
import torch
import torch.nn as nn
from recbole.model.layers import MultiHeadAttention, FeedForward
from recbole.model.sequential_recommender.sasrec import SASRec


# ── Watch-Time Augmented Attention ────────────────────────────────────────────

class WTMultiHeadAttention(MultiHeadAttention):
    """Extends MultiHeadAttention with an additive watch-time bias."""

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob,
                 attn_dropout_prob, layer_norm_eps):
        super().__init__(n_heads, hidden_size, hidden_dropout_prob,
                         attn_dropout_prob, layer_norm_eps)
        # wt_proj lives in WTSASRec and is computed once per forward pass;
        # each layer only needs its own learnable scale alpha.
        self.wt_alpha = nn.Parameter(torch.ones(1))   # learnable scale

    def forward(self, input_tensor, attention_mask, wt_bias=None):
        """
        Args:
            input_tensor:    (B, L, H)
            attention_mask:  (B, 1, 1, L)  — causal / padding mask
            wt_bias:         (B, L, H_heads) — pre-computed per-position bias
        """
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer   = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer   = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # ── Inject watch-time bias ────────────────────────────────────────
        if wt_bias is not None:
            # wt_bias: (B, L, H_heads) → (B, H_heads, L) → (B, H_heads, 1, L)
            wt_bias_t = wt_bias.permute(0, 2, 1).unsqueeze(2)
            attention_scores = attention_scores + self.wt_alpha * wt_bias_t

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape     = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ── Watch-Time Transformer Layer ──────────────────────────────────────────────

class WTTransformerLayer(nn.Module):
    """One transformer block with WTMultiHeadAttention."""

    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob,
                 attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.multi_head_attention = WTMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask, wt_bias=None):
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask, wt_bias=wt_bias
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


# ── Watch-Time Transformer Encoder ────────────────────────────────────────────

class WTTransformerEncoder(nn.Module):
    """Stack of WTTransformerLayer blocks."""

    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.layer = nn.ModuleList([
            WTTransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob,
                               attn_dropout_prob, hidden_act, layer_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask, wt_bias=None,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, wt_bias)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# ── WTSASRec Model ────────────────────────────────────────────────────────────

class WTSASRec(SASRec):
    """Watch-Time Self-Attentive Sequential Recommendation."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        n_heads = config["n_heads"]

        # Replace the standard transformer encoder with our WT-augmented one
        self.trm_encoder = WTTransformerEncoder(
            n_layers         = config["n_layers"],
            n_heads          = n_heads,
            hidden_size      = config["hidden_size"],
            inner_size       = config["inner_size"],
            hidden_dropout_prob = config["hidden_dropout_prob"],
            attn_dropout_prob   = config["attn_dropout_prob"],
            hidden_act       = config["hidden_act"],
            layer_norm_eps   = config["layer_norm_eps"],
        )

        # Project scalar watch time to one value per attention head
        self.wt_proj = nn.Linear(1, n_heads)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_watch_time(self, wt_seq: torch.Tensor) -> torch.Tensor:
        """Log-normalize watch time per sequence to [0, 1]."""
        log_wt  = torch.log1p(wt_seq)
        max_wt  = log_wt.max(dim=1, keepdim=True).values
        return log_wt / (max_wt + 1e-8)

    def _compute_wt_bias(self, wt_seq: torch.Tensor) -> torch.Tensor:
        """
        Returns watch-time bias: (B, L, num_heads).
        Each position's bias tells each attention head how strongly to up-weight that key.
        """
        wt_norm = self._normalize_watch_time(wt_seq)   # (B, L)
        return self.wt_proj(wt_norm.unsqueeze(-1))      # (B, L, H_heads)

    # ── Override forward ──────────────────────────────────────────────────────

    def forward(self, item_seq, item_seq_len, wt_seq=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        position_embedding = self.position_embedding(position_ids)
        item_emb           = self.item_embedding(item_seq)
        input_emb          = item_emb + position_embedding
        input_emb          = self.LayerNorm(input_emb)
        input_emb          = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        wt_bias = self._compute_wt_bias(wt_seq) if wt_seq is not None else None

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, wt_bias=wt_bias,
            output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return self.gather_indexes(output, item_seq_len - 1)

    # ── Override loss / predict to pass wt_seq ────────────────────────────────

    def calculate_loss(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        wt_seq       = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        seq_output   = self.forward(item_seq, item_seq_len, wt_seq)

        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_emb   = self.item_embedding(pos_items)
        neg_emb   = self.item_embedding(neg_items)
        pos_score = torch.sum(seq_output * pos_emb, dim=-1)
        neg_score = torch.sum(seq_output * neg_emb, dim=-1)
        return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        wt_seq       = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        seq_output   = self.forward(item_seq, item_seq_len, wt_seq)
        test_item    = interaction[self.ITEM_ID]
        test_emb     = self.item_embedding(test_item)
        return torch.mul(seq_output, test_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        wt_seq       = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        seq_output   = self.forward(item_seq, item_seq_len, wt_seq)
        test_items   = self.item_embedding.weight
        return torch.matmul(seq_output, test_items.T)
