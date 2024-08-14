import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

from replay_trainer.models.transformer_utils import TransformerBlock, AttentionPooling

class Transformer(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        sequence_length: int,
        objective: Literal["classification", "regression"],
        config: dict = dict(),
        
    ):
        super().__init__()
        self.objective = objective
        self._load_config(config)

        self.action_embedding = nn.Embedding(action_size, self.d_model)
        self.obs_proj = nn.Linear(obs_size, self.d_model)
        self.position_embeddings = nn.Embedding(sequence_length, self.d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    attn_pdrop=self.attn_pdrop,
                    residual_pdrop=self.residual_pdrop,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = nn.RMSNorm(self.d_model, eps=1e-5)
        self.attention_pooling = AttentionPooling(self.d_model)
        output_size = action_size if objective == "classification" else 1
        self.lm_head = nn.Linear(self.d_model, output_size, bias=False)

    def _load_config(self, config: dict):
        self.d_model = config.get("d_model", 128)
        self.num_heads = config.get("num_heads", 4)
        self.d_ff = config.get("d_ff", 512)
        self.attn_pdrop = config.get("attn_pdrop", 0.1)
        self.residual_pdrop = config.get("residual_pdrop", 0.1)
        self.num_layers = config.get("num_layers", 8)

    def forward(self, actions: torch.LongTensor, obs: torch.FloatTensor) -> torch.FloatTensor:
        # actions: (n, sequence_length, 1)
        # obs: (n, sequence_length, obs_size)
        # x = torch.cat([self.action_embedding(actions.squeeze(-1)), self.obs_proj(obs)], dim=2)
        # x = self.action_embedding(actions.squeeze(-1)) + self.obs_proj(obs)
        x = self.obs_proj(obs)
        # x: (n, sequence_length, d_model)
        x = x + self.position_embeddings(torch.arange(x.shape[1], device=x.device))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.attention_pooling(x)
        x = self.lm_head(x)
        if self.objective == "regression":
            x = torch.sigmoid(x).squeeze(-1)
        return x
    
# class RegressionTransformer(nn.Module):
#     def __init__(
#         self,
#         obs_size: int,
#         action_size: int,
#         sequence_length: int,
#         config: dict = dict(),
        
#     ):
#         super().__init__()

#         self._load_config(config)

#         self.action_embedding = nn.Embedding(action_size, self.d_model)
#         self.obs_proj = nn.Linear(obs_size, self.d_model)
#         self.position_embeddings = nn.Embedding(sequence_length, self.d_model)

#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     d_model=self.d_model,
#                     num_heads=self.num_heads,
#                     d_ff=self.d_ff,
#                     attn_pdrop=self.attn_pdrop,
#                     residual_pdrop=self.residual_pdrop,
#                 )
#                 for _ in range(self.num_layers)
#             ]
#         )
#         self.ln_final = nn.RMSNorm(self.d_model, eps=1e-5)
#         self.attention_pooling = AttentionPooling(self.d_model)
#         self.lm_head = nn.Linear(self.d_model, 1, bias=True)

#     def _load_config(self, config: dict):
#         self.d_model = config.get("d_model", 128)
#         self.num_heads = config.get("num_heads", 4)
#         self.d_ff = config.get("d_ff", 512)
#         self.attn_pdrop = config.get("attn_pdrop", 0.1)
#         self.residual_pdrop = config.get("residual_pdrop", 0.1)
#         self.num_layers = config.get("num_layers", 8)

#     def forward(self, actions: torch.LongTensor, obs: torch.FloatTensor) -> torch.FloatTensor:
#         # actions: (n, sequence_length, 1)
#         # obs: (n, sequence_length, obs_size)
#         # x = torch.cat([self.action_embedding(actions.squeeze(-1)), self.obs_proj(obs)], dim=2)
#         # x = self.action_embedding(actions.squeeze(-1)) + self.obs_proj(obs)
#         x = self.obs_proj(obs)
#         # x: (n, sequence_length, d_model)
#         x = x + self.position_embeddings(torch.arange(x.shape[1], device=x.device))
#         for layer in self.layers:
#             x = layer(x)
#         x = self.ln_final(x)
#         x = self.attention_pooling(x)
#         x = self.lm_head(x)
#         x = F.sigmoid(x).squeeze(-1)
#         return x