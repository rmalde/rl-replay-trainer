import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

from replay_trainer.models.transformer_utils import TransformerBlock
from replay_trainer.models.physics_transformer import PhysicsProjection

class TransformerCLS(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        objective: Literal["classification", "regression"],
        config: dict = dict(),
    ):
        super().__init__()
        self.objective = objective
        self._load_config(config)
        
        # Physics projection layer
        self.physics_proj = PhysicsProjection(self.d_model)
        
        # CLS token is a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Position embeddings for physics features + CLS token
        self.position_embeddings = nn.Embedding(self.physics_proj.seq_len + 1, self.d_model)

        # Transformer layers
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
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, action_size if objective == "classification" else 1)
        )

    def _load_config(self, config: dict):
        self.d_model = config.get("d_model", 128)
        self.num_heads = config.get("num_heads", 4)
        self.d_ff = config.get("d_ff", 512)
        self.attn_pdrop = config.get("attn_pdrop", 0.1)
        self.residual_pdrop = config.get("residual_pdrop", 0.1)
        self.num_layers = config.get("num_layers", 8)

    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        # obs: (batch_size, obs_size)
        batch_size = obs.shape[0]
        
        # Project observations through physics layer
        x = self.physics_proj(obs)  # (batch_size, seq_len, d_model)
        
        # Expand CLS token for batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        
        # Prepend CLS token to sequence
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, d_model)
        
        # Add position embeddings
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embeddings(positions)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)
        
        # Take only the CLS token output
        x = x[:, 0]  # (batch_size, d_model)
        
        # Pass through MLP head
        x = self.mlp_head(x)
        
        if self.objective == "regression":
            x = torch.sigmoid(x).squeeze(-1)
            
        return x
