import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Optional

def attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    # K: .., m, d_k
    # Q: .., n, d_k
    # V: .., m, d_v
    # mask: n, m
    # output: .., n, d_v
    scores = (Q @ K.mT) / math.sqrt(Q.shape[-1])  # n, m
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    scores = F.softmax(scores, dim=-1)

    if pdrop is not None:
        scores = F.dropout(scores, p=pdrop)
    return scores @ V

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.w2(F.gelu(self.w1(x)))

class MultiheadSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, pdrop: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pdrop = pdrop

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=True)

        self._apply_xavier_initialization()

    def _apply_xavier_initialization(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split_heads(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: .., m, d_model
        # return: .., n_heads, m, d_k

        # .., m, n_heads, d_k
        x = x.view(*x.shape[:-1], self.n_heads, self.d_k)
        # .., n_heads, m, d_k
        return x.transpose(-2, -3)

    def _merge_heads(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: .., n_heads, m, d_k
        # return: .., m, d_model
        # .., m, n_heads, d_k
        x = x.transpose(-2, -3)
        # .., m, d_model
        return x.contiguous().view(*x.shape[:-2], self.n_heads * self.d_k)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: .., m, d_model
        Q = self.q_proj(x)  # .., n_heads, m, d_model
        K = self.k_proj(x)
        V = self.v_proj(x)

        # separate heads
        # .., n_heads, m, d_k
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # attention
        # .., n_heads, m, d_k
        seq_len = Q.shape[-2]
        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = attention(K=K, Q=Q, V=V, mask=None, pdrop=self.pdrop)
        # .., m, d_model
        scores = self._merge_heads(scores)

        return self.output_proj(scores)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
    ):
        super().__init__()

        self.ln1 = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = MultiheadSelfAttention(
            d_model=d_model, n_heads=num_heads, pdrop=attn_pdrop
        )

        self.ln2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.residual_dropout = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.residual_dropout(self.attn(self.ln1(x)))
        x = x + self.residual_dropout(self.ffn(self.ln2(x)))
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.attention_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (n, sequence_length, d_model)
        attention_weights = torch.softmax(self.attention_layer(x), dim=1)  # (n, sequence_length, 1)
        pooled_output = torch.sum(attention_weights * x, dim=1)  # (n, d_model)
        return pooled_output