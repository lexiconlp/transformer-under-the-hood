"""
Define different attention models.
"""
from typing import Tuple, Type, Union

import torch
from torch import nn
from torch.nn.functional import softmax

from transformer_uth.vectorizer import SequenceData


def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Equation 1 in Vaswani et al. (2017)
    # Scaled dot product between Query and Keys
    scaling_factor = torch.sqrt(
        torch.tensor(query.shape[2], dtype=torch.float64)
    )
    output = torch.bmm(query, key.transpose(1, 2)) / scaling_factor

    # Softmax to get attention weights
    attention_weights = softmax(output, dim=2)

    # Multiply weights by Values
    weighted_sum = torch.bmm(attention_weights, value)

    # Following Figure 1 and Section 3.1 in Vaswani et al. (2017)
    # Residual connection ie. add weighted sum to original query
    output = weighted_sum + query

    return output, attention_weights


class AttentionModel(nn.Module):
    def __init__(self, seq_data: SequenceData, hidden=64):
        super().__init__()

        self.vocab_in = seq_data.vocab_in
        self.vocab_out = seq_data.vocab_out
        self.len_out = seq_data.len_out
        self.hidden = hidden

        self.query = nn.Parameter(
            torch.zeros((1, self.len_out, self.hidden), requires_grad=True)
        )
        self.key_val_dense = nn.Linear(self.vocab_in, self.hidden)
        self.layer_norm = nn.LayerNorm(self.hidden)
        self.final_dense = nn.Linear(self.hidden, self.vocab_out)

    def forward(self, x_batch):
        key_val = self.key_val_dense(x_batch)
        decoding, attention_weights = _attention(
            self.query.repeat(key_val.shape[0], 1, 1), key_val, key_val
        )
        decoding = self.layer_norm(decoding)

        logits = self.final_dense(decoding)
        return logits, attention_weights


class SelfAttentionModel(nn.Module):
    def __init__(self, seq_data: SequenceData, hidden=64):
        super().__init__()

        self.vocab_in = seq_data.vocab_in
        self.vocab_out = seq_data.vocab_out
        self.len_out = seq_data.len_out
        self.hidden = hidden

        self.query = nn.Parameter(
            torch.zeros((1, self.len_out, self.hidden), requires_grad=True)
        )
        self.key_val_dense = nn.Linear(self.vocab_in, self.hidden)
        self.layer_norm = nn.LayerNorm(self.hidden)
        self.layer_norm_self = nn.LayerNorm(self.hidden)
        self.final_dense = nn.Linear(self.hidden, self.vocab_out)

    def forward(self, x_batch):
        key_val = self.key_val_dense(x_batch)

        decoding, attention_weights = _attention(
            self.query.repeat(key_val.shape[0], 1, 1), key_val, key_val
        )
        decoding = self.layer_norm(decoding)

        decoding, self_attention_weights = _attention(
            decoding, decoding, decoding
        )
        decoding = self.layer_norm_self(decoding)

        logits = self.final_dense(decoding)
        return logits, attention_weights, self_attention_weights


class PositionalAttentionModel(nn.Module):
    def __init__(self, seq_data: SequenceData, hidden=64):
        super().__init__()

        self.vocab_in = seq_data.vocab_in
        self.vocab_out = seq_data.vocab_out
        self.len_out = seq_data.len_out
        self.len_in = seq_data.len_in
        self.hidden = hidden

        self.query = nn.Parameter(
            torch.zeros((1, self.len_out, self.hidden), requires_grad=True)
        )
        self.positional_emb = nn.Parameter(
            torch.zeros((1, self.len_in, self.hidden), requires_grad=True)
        )
        self.key_val_dense = nn.Linear(self.vocab_in, self.hidden)
        self.layer_norm = nn.LayerNorm(self.hidden)
        self.layer_norm_self = nn.LayerNorm(self.hidden)
        self.final_dense = nn.Linear(self.hidden, self.vocab_out)

    def forward(self, x_batch):
        key_val = self.key_val_dense(x_batch)
        key_val += self.positional_emb

        decoding, attention_weights = _attention(
            self.query.repeat(key_val.shape[0], 1, 1), key_val, key_val
        )
        decoding = self.layer_norm(decoding)

        decoding, self_attention_weights = _attention(
            decoding, decoding, decoding
        )
        decoding = self.layer_norm_self(decoding)

        logits = self.final_dense(decoding)
        return logits, attention_weights, self_attention_weights


TransformerModel = Type[
    Union[AttentionModel, SelfAttentionModel, PositionalAttentionModel]
]
