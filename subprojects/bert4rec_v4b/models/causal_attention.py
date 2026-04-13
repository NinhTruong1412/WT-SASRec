from __future__ import annotations

import torch
import torch.nn as nn

from recbole.model.layers import FeedForward, MultiHeadAttention


class CausalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with an additive watch-time bias."""

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super().__init__(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.causal_beta = nn.Parameter(torch.zeros(1))

    def forward(self, input_tensor, attention_mask, causal_log_weight=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        if causal_log_weight is not None:
            causal_bias = causal_log_weight.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + self.causal_beta * causal_bias

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CausalTransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.multi_head_attention = CausalMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask, causal_log_weight=None):
        attn_output = self.multi_head_attention(
            hidden_states, attention_mask, causal_log_weight=causal_log_weight
        )
        return self.feed_forward(attn_output)


class CausalTransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                CausalTransformerLayer(
                    n_heads,
                    hidden_size,
                    inner_size,
                    hidden_dropout_prob,
                    attn_dropout_prob,
                    hidden_act,
                    layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask, causal_log_weight=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, causal_log_weight)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
