"""
models/wt_gru4rec.py
--------------------
Watch-Time Gated GRU4Rec (WTGru4Rec)

Extends GRU4Rec by gating each item embedding with a learned function of its
normalized watch time before the sequence enters the GRU.

Mechanism
---------
  wt_norm  = log(1 + wt) / log(1 + max_wt + ε)   # per-sequence normalization
  gate     = sigmoid( W_g · wt_norm )              # W_g: Linear(1 → embedding_size)
  input    = item_emb * gate                       # element-wise gating
  ...rest identical to GRU4Rec...

Rationale: items with higher watch time (stronger preference signal) modulate
their embedding magnitude, giving the GRU a richer input signal.
"""

import torch
import torch.nn as nn
from recbole.model.sequential_recommender.gru4rec import GRU4Rec


class WTGru4Rec(GRU4Rec):
    """Watch-Time Gated GRU4Rec."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.wt_gate = nn.Linear(1, self.embedding_size)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_watch_time(self, wt_seq: torch.Tensor) -> torch.Tensor:
        """Log-normalize watch time per sequence to [0, 1]."""
        log_wt = torch.log1p(wt_seq)                        # (B, L)
        max_wt = log_wt.max(dim=1, keepdim=True).values     # (B, 1)
        return log_wt / (max_wt + 1e-8)                     # (B, L)

    # ── Override forward ──────────────────────────────────────────────────────

    def forward(self, item_seq, item_seq_len, wt_seq=None):
        item_seq_emb = self.item_embedding(item_seq)          # (B, L, E)

        if wt_seq is not None:
            wt_norm = self._normalize_watch_time(wt_seq)      # (B, L)
            gate = torch.sigmoid(
                self.wt_gate(wt_norm.unsqueeze(-1))           # (B, L, H)
            )
            item_seq_emb = item_seq_emb * gate                # gated embeddings

        item_seq_emb = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense(gru_output)
        return self.gather_indexes(gru_output, item_seq_len - 1)

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
