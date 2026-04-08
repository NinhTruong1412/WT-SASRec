"""
models/wt_bert4rec.py
---------------------
Watch-Time Enhanced BERT4Rec (WTBert4Rec)

Extends BERT4Rec by injecting a watch-time embedding into the input layer.

Mechanism:
  wt_norm   = log(1 + wt) / log(1 + max_wt + ε)    # (B, L), in [0,1]
  wt_emb    = Linear(1 → hidden_size)(wt_norm)       # (B, L, H)
  input_emb = item_emb + position_emb + wt_emb
"""

import torch
import torch.nn as nn
from recbole.model.sequential_recommender.bert4rec import BERT4Rec


class WTBert4Rec(BERT4Rec):
    """Watch-Time Enhanced BERT4Rec."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.wt_proj = nn.Linear(1, config["hidden_size"])

    def _normalize_watch_time(self, wt_seq: torch.Tensor) -> torch.Tensor:
        log_wt = torch.log1p(wt_seq)
        max_wt = log_wt.max(dim=1, keepdim=True).values
        return log_wt / (max_wt + 1e-8)

    def forward(self, item_seq, wt_seq=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        position_embedding = self.position_embedding(position_ids)
        item_emb  = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        if wt_seq is not None:
            wt_norm = self._normalize_watch_time(wt_seq)
            wt_emb  = self.wt_proj(wt_norm.unsqueeze(-1))           # (B, L, H)
            is_real = (item_seq != 0) & (item_seq != self.mask_token)
            wt_emb  = wt_emb * is_real.unsqueeze(-1).float()
            input_emb = input_emb + wt_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output     = self.output_ln(ffn_output)
        return output  # [B, L, H]

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items       = interaction[self.POS_ITEMS]
        neg_items       = interaction[self.NEG_ITEMS]
        masked_index    = interaction[self.MASK_INDEX]
        wt_seq = interaction["watch_time_list"] if "watch_time_list" in interaction else None

        seq_output = self.forward(masked_item_seq, wt_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B, mask_len, H]

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) + self.output_bias[pos_items]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) + self.output_bias[neg_items]
            targets = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets
            ) / torch.sum(targets)
            return loss
        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self.item_embedding.weight[: self.n_items]
            logits  = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.output_bias
            targets = (masked_index > 0).float().view(-1)
            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets
            ) / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item    = interaction[self.ITEM_ID]
        wt_seq = interaction["watch_time_list"] if "watch_time_list" in interaction else None

        item_seq   = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq, wt_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_emb   = self.item_embedding(test_item)
        return torch.mul(seq_output, test_emb).sum(dim=1) + self.output_bias[test_item]

    def full_sort_predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        wt_seq = interaction["watch_time_list"] if "watch_time_list" in interaction else None

        item_seq   = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq, wt_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items = self.item_embedding.weight[: self.n_items]
        return torch.matmul(seq_output, test_items.transpose(0, 1)) + self.output_bias
