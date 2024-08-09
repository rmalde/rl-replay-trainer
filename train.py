import torch
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split

from replay_trainer.data import ObsActDataset
from replay_trainer.models import FCN
from replay_trainer import Trainer


def train(dataset_dir):
    print("Loading train and test datasets...")
    # initialize data
    filenames = []
    for filename in os.listdir(os.path.join(dataset_dir, 'actions')):
        filenames.append(filename.split('.')[0])
    
    #TEMP
    filenames = filenames[:60]
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2)

    sequence_length = 1
    train_dataset = ObsActDataset(dataset_dir, train_filenames, sequence_length)
    test_dataset = ObsActDataset(dataset_dir, test_filenames, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Train and test datasets loaded.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_config = {
        "learning_rate": 5e-4,
        "num_epochs": 100_000,
        "wandb_project": "rl-replay-trainer"
    }
    model_config = {
        "dropout": 0.5,
    }

    print("Initializing model...")
    model = FCN(obs_size=train_dataset[0][0]['obs'].shape[1],
                action_size=90,
                sequence_length=train_dataset.sequence_length,
                config=model_config)
    
    trainer = Trainer(model, train_loader, test_loader, trainer_config, device)

    print("Training model...")
    trainer.train()

if __name__ == "__main__":
    train('dataset/ssl-1v1-400')

