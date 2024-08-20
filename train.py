import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split

from replay_trainer.data import get_obsact_dataloaders
from replay_trainer.models import FCN, Transformer, PhysicsTransformer
from replay_trainer import Trainer


def train(dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4096
    train_loader, test_loader, obs_size, action_size = get_obsact_dataloaders(
        dataset_dir, batch_size=batch_size
    )

    trainer_config = {
        "learning_rate": 5e-4,
        "num_epochs": 100_000,
        "wandb_project": "rl-replay-trainer",
        # "wandb_project": None,
    }
    model_config = {
        "dropout": 0.2,
        "use_batch_norm": False
    }

    print("Initializing model...")
    model = FCN(obs_size=obs_size,
                action_size=action_size,
                layer_sizes=[2048, 2048, 2048, 1024, 1024],
                objective="classification",
                config=model_config)
    # model = Transformer(
    #     obs_size=obs_size,
    #     action_size=action_size,
    #     config=model_config,
    # )
    trainer = Trainer(model, train_loader, test_loader, trainer_config, device)

    print("Training model...")
    trainer.train()


if __name__ == "__main__":
    dataset_dir = "dataset/ssl-1v1-8k"
    train(dataset_dir)
