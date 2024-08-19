import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

from replay_trainer.models.transformer_utils import TransformerBlock, AttentionPooling

class PhysicsProjection(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = 22

        self.pos_proj = self._make_proj(3)
        self.vel_proj = self._make_proj(3)
        self.ang_vel_proj = self._make_proj(3)
        self.pad_proj = self._make_proj(34)
        self.rel_pos_proj = self._make_proj(3)
        self.rel_vel_proj = self._make_proj(3)
        self.rotmat_proj = self._make_proj(9)
        self.pyr_proj = self._make_proj(3)
        self.stats_proj = self._make_proj(4)
    
    def _make_proj(self, d_in: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_in, self.d_model, bias=False),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )


    def forward(self, x: torch.FloatTensor):
        # x: (n, 113)

        # x: (n, 22, d_model)
        return torch.stack([
            self.pos_proj(x[:, :3]), # ball pos
            self.vel_proj(x[:, 3:6]), # ball vel
            self.ang_vel_proj(x[:, 6:9]), # ball ang vel
            self.pad_proj(x[:, 9:43]), # pad pos
            self.rel_pos_proj(x[:, 43:46]), # player rel pos
            self.rel_vel_proj(x[:, 46:49]), # player rel vel
            self.pos_proj(x[:, 49:52]), # player pos
            self.rotmat_proj(x[:, 52:61]), # player rotmat
            self.vel_proj(x[:, 61:64]), # player vel
            self.ang_vel_proj(x[:, 64:67]), # player ang vel
            self.pyr_proj(x[:, 67:70]), # player pyr
            self.stats_proj(x[:, 70:74]), # player stats
            self.rel_pos_proj(x[:, 74:77]), # opp rel pos
            self.rel_vel_proj(x[:, 77:80]), # opp rel vel
            self.pos_proj(x[:, 80:83]), # opp pos
            self.rotmat_proj(x[:, 83:92]), # opp rotmat
            self.vel_proj(x[:, 92:95]), # opp vel
            self.ang_vel_proj(x[:, 95:98]), # opp ang vel
            self.pyr_proj(x[:, 98:101]), # opp pyr
            self.stats_proj(x[:, 101:105]), # opp stats
            self.rel_pos_proj(x[:, 105:108]), # player_opp rel pos
            self.rel_vel_proj(x[:, 108:111]), # player_opp rel vel
        ], dim=1)



class PhysicsTransformer(nn.Module):
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

        self.physics_proj = PhysicsProjection(self.d_model)
        seq_len = self.physics_proj.seq_len
            
        self.position_embeddings = nn.Embedding(seq_len, self.d_model)

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

    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        # obs: (n, sequence_length, obs_size)

        # assert that sequence length is 1
        if len(obs.shape) == 3:
            assert obs.shape[1] == 1
            obs = obs.squeeze(1)

        # (n, 22, d_model)
        x = self.physics_proj(obs)
        # (n, 23, d_model)
        x = x + self.position_embeddings(torch.arange(x.shape[1], device=x.device))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.attention_pooling(x)
        x = self.lm_head(x)
        if self.objective == "regression":
            x = F.sigmoid(x).squeeze(-1)
        return x