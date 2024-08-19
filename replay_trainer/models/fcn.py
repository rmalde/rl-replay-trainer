import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal


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

        output_size = action_size if objective == "classification" else 1
        layer_sizes = [obs_size] + layer_sizes + [output_size]
        # [X, 2048, 2048, 2048, 2048, Y]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def _load_config(self, config: dict):
        self.dropout = config.get("dropout", 0.2)
        self.use_batch_norm = config.get("use_batch_norm", False)

    def forward(self, obs):
        """
        obs: (n, obs_size)
        """
        x = self.layers(obs)
        if self.objective == "regression":
            x = F.sigmoid(x).squeeze(-1)
        return x
