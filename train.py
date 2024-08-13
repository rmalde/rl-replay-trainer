import torch
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split

from replay_trainer.data import ObsActDataset, get_obsact_dataloaders
from replay_trainer.models import FCN, Transformer
from replay_trainer import Trainer



def train(dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequence_length = 10
    train_loader, test_loader, obs_size, action_size = get_obsact_dataloaders(dataset_dir, sequence_length)

    trainer_config = {
        "learning_rate": 5e-4,
        "num_epochs": 100_000,
        # "wandb_project": "rl-replay-trainer",
        "wandb_project": None,
    }
    model_config = {
        "d_model": 128,
        "num_heads": 8,
        "d_ff": 512,
        "attn_pdrop": 0.2,
        "residual_pdrop": 0.2,
        "num_layers": 16,
    }

    print("Initializing model...")
    # model = FCN(obs_size=train_dataset[0][0]['obs'].shape[1],
    #             action_size=90,
    #             sequence_length=train_dataset.sequence_length,
    #             config=model_config)
    model = Transformer(
        obs_size=obs_size,
        action_size=action_size,
        sequence_length=sequence_length,
        config=model_config,
    )

    trainer = Trainer(model, train_loader, test_loader, trainer_config, device)

    print("Training model...")
    trainer.train()


if __name__ == "__main__":
    dataset_dir = "dataset/ssl-1v1-40-trig-v2"
    train(dataset_dir)
