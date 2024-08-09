import torch

import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, obs_size, action_size, sequence_length, config: dict = dict()):
        super(FCN, self).__init__()
        self._load_config(config)

        input_size = obs_size * sequence_length
        layer_sizes = [input_size] + [self.hidden_size] * self.num_layers + [action_size]
        # [X, 2048, 2048, 2048, 2048, Y]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def _load_config(self, config: dict):
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_layers = config.get("num_layers", 6)
        self.dropout = config.get("dropout", 0.2)

    def forward(self, actions, obs):
        """
        actions: (n, sequence_length)
        obs: (n, sequence_length, obs_size)
        """
        obs_flat = obs.view(obs.size(0), -1) # (n, sequence_length * obs_size)
        x = self.layers(obs_flat)
        return x # (n, action_size)