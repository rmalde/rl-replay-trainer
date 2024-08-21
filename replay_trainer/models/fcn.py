import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from replay_trainer.models.physics_transformer import PhysicsProjection


class FCN(nn.Module):
    def __init__(
        self,
        obs_size,
        action_size,
        layer_sizes,
        objective: Literal["classification", "regression", "value"],
        config: dict = dict(),
    ):
        super().__init__()
        self.objective = objective
        self._load_config(config)

        self.physics_proj = PhysicsProjection(32)

        output_size = action_size if objective == "classification" else 1
        layer_sizes = [self.physics_proj.proj_out] + layer_sizes + [output_size]
        # [X, 2048, 2048, 2048, 2048, Y]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if self.use_batch_norm:
                    # layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                    layers.append(nn.LayerNorm(layer_sizes[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def _load_config(self, config: dict):
        self.dropout = config.get("dropout", 0.2)
        self.use_batch_norm = config.get("use_batch_norm", False)

    def forward(self, obs):
        """
        obs: (n, obs_size)
        """
        x = self.physics_proj(obs)
        x = self.layers(x)
        if self.objective == "regression":
            x = F.sigmoid(x).squeeze(-1)
        return x
