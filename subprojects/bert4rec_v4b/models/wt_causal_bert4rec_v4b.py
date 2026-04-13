from __future__ import annotations

import torch
import torch.nn as nn

from recbole.model.sequential_recommender.bert4rec import BERT4Rec

from .causal_attention import CausalTransformerEncoder


def _completion_ratio(
    watch_time: torch.Tensor,
    duration: torch.Tensor,
    item_seq: torch.Tensor,
    mask_token: int,
) -> torch.Tensor:
    eps = 1e-6
    ratio = (watch_time.float() / duration.float().clamp(min=eps)).clamp(0.0, 1.0)
    real = (item_seq != 0) & (item_seq != mask_token)
    return ratio * real.float()


class WTCausalBERT4RecV2(BERT4Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.wt_beta_init = float(config["wt_beta_init"] if "wt_beta_init" in config else 1.0)
        self.trm_encoder = CausalTransformerEncoder(
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            hidden_size=config["hidden_size"],
            inner_size=config["inner_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attn_dropout_prob=config["attn_dropout_prob"],
            hidden_act=config["hidden_act"],
            layer_norm_eps=config["layer_norm_eps"],
        )

        for layer in self.trm_encoder.layer:
            nn.init.constant_(layer.multi_head_attention.causal_beta, self.wt_beta_init)

        self.apply(self._init_weights)
        for layer in self.trm_encoder.layer:
            nn.init.constant_(layer.multi_head_attention.causal_beta, self.wt_beta_init)

    def _get_causal_log_weight(self, watch_time, duration, item_seq):
        ratio = _completion_ratio(watch_time, duration, item_seq, self.mask_token)
        return torch.log1p(ratio)

    def forward(self, item_seq, watch_time=None, duration=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)

        causal_log_weight = None
        if watch_time is not None and duration is not None:
            causal_log_weight = self._get_causal_log_weight(watch_time, duration, item_seq)

        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            causal_log_weight=causal_log_weight,
            output_all_encoded_layers=True,
        )

        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        return self.output_ln(ffn_output)

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        neg_items = interaction[self.NEG_ITEMS]
        masked_index = interaction[self.MASK_INDEX]
        watch_time = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        duration = interaction["duration_list"] if "duration_list" in interaction else None

        seq_output = self.forward(masked_item_seq, watch_time, duration)

        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)
        seq_output = torch.bmm(pred_index_map, seq_output)

        if self.loss_type == "CE":
            test_item_emb = self.item_embedding.weight[: self.n_items]
            logits = torch.matmul(seq_output, test_item_emb.T) + self.output_bias
            targets = (masked_index > 0).float().view(-1)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            return torch.sum(
                loss_fct(logits.view(-1, self.n_items), pos_items.view(-1)) * targets
            ) / torch.sum(targets)

        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_score = torch.sum(seq_output * pos_emb, -1) + self.output_bias[pos_items]
        neg_score = torch.sum(seq_output * neg_emb, -1) + self.output_bias[neg_items]
        targets = (masked_index > 0).float()
        return -torch.sum(
            torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets
        ) / torch.sum(targets)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        watch_time = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        duration = interaction["duration_list"] if "duration_list" in interaction else None

        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq, watch_time, duration)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_emb).sum(dim=1) + self.output_bias[test_item]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        watch_time = interaction["watch_time_list"] if "watch_time_list" in interaction else None
        duration = interaction["duration_list"] if "duration_list" in interaction else None

        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq, watch_time, duration)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items = self.item_embedding.weight[: self.n_items]
        return torch.matmul(seq_output, test_items.T) + self.output_bias


class WTCausalBERT4RecV3(WTCausalBERT4RecV2):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        hidden_size = config["hidden_size"]

        self.wt_input_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
        )
        self.film_gamma = nn.Linear(1, hidden_size)
        self.film_beta = nn.Linear(1, hidden_size)

        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        nn.init.xavier_uniform_(self.wt_input_proj[0].weight)
        nn.init.zeros_(self.wt_input_proj[0].bias)

    def forward(self, item_seq, watch_time=None, duration=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)

        causal_log_weight = None
        if watch_time is not None and duration is not None:
            ratio = _completion_ratio(watch_time, duration, item_seq, self.mask_token)
            log_ratio = torch.log1p(ratio)
            causal_log_weight = log_ratio
            item_emb = item_emb + self.wt_input_proj(log_ratio.unsqueeze(-1))

        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            causal_log_weight=causal_log_weight,
            output_all_encoded_layers=True,
        )

        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)

        if watch_time is not None and duration is not None:
            ratio_norm = ratio.unsqueeze(-1)
            output = self.film_gamma(ratio_norm) * output + self.film_beta(ratio_norm)

        return output


class WTCausalBERT4RecV4b(WTCausalBERT4RecV3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        hidden_size = config["hidden_size"]
        self.n_wt_quantiles = int(config["n_wt_quantiles"] if "n_wt_quantiles" in config else 8)

        del self.wt_input_proj
        self.wt_quantile_emb = nn.Embedding(self.n_wt_quantiles + 1, hidden_size)
        nn.init.normal_(self.wt_quantile_emb.weight, std=0.02)
        self.wt_quantile_emb.weight.data[0].zero_()

    def _quantile_bucket(self, ratio: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        bucket = (ratio * self.n_wt_quantiles).long().clamp(1, self.n_wt_quantiles)
        real = (item_seq != 0) & (item_seq != self.mask_token)
        return bucket * real.long()

    def forward(self, item_seq, watch_time=None, duration=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)

        causal_log_weight = None
        if watch_time is not None and duration is not None:
            ratio = _completion_ratio(watch_time, duration, item_seq, self.mask_token)
            causal_log_weight = torch.log1p(ratio)
            item_emb = item_emb + self.wt_quantile_emb(self._quantile_bucket(ratio, item_seq))

        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            causal_log_weight=causal_log_weight,
            output_all_encoded_layers=True,
        )

        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)

        if watch_time is not None and duration is not None:
            ratio_norm = ratio.unsqueeze(-1)
            output = self.film_gamma(ratio_norm) * output + self.film_beta(ratio_norm)

        return output
