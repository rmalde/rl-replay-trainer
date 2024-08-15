import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

rank_to_skill = {
    "bronze-1": 0.0,
    "silver-1": 0.1,
    "gold-1": 0.15,
    "platinum-1": 0.20,
    "diamond-1": 0.25,
    "champion-1": 0.3,
    "grand-champion-1": 0.4,
    "grand-champion-2": 0.5,
    "grand-champion-3": 0.6,
    "supersonic-legend": 0.7,
    "pro": 1.0
}
# rank_to_skill = {
#     "diamond-1": 0.0,
#     "champion-1": 0.167,
#     "grand-champion-1": 0.333,
#     "grand-champion-2": 0.5,
#     "grand-champion-3": 0.677,
#     "supersonic-legend": 0.833,
#     "pro": 1.0
# }

class SkillDataset(Dataset):
    def __init__(self, dataset_dir, filenames, filename_to_rank, sequence_length=30):
        """
        Args:
            dataset_dir (str): Path to the dataset, ie 'dataset/ssl-1v1-100'
            filenames (list of str): List of names for the action/obs numpy files
            filename_to_rank (dict): Dictionary mapping filename to respective ranks
            sequence_length (int): Number of timesteps to use as input.
        """
        
        self.filenames = filenames
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.filename_to_rank = filename_to_rank
        self.data = self._load_data()

        # set obs and act space for external reference
        self.obs_size = self.data[0][0]['obs'].shape[1]
        self.action_size = 90


    def _load_data(self):
        data = []
        skipped_ranks = set()
        for filename in tqdm(self.filenames):
            if self.filename_to_rank[filename] not in rank_to_skill:
                skipped_ranks.add(self.filename_to_rank[filename])
                continue
            actions_path = os.path.join(self.dataset_dir, 'actions', f'{filename}.npz')
            obs_path = os.path.join(self.dataset_dir, 'obs', f'{filename}.npz')

            actions = np.load(actions_path)['array']
            obs = np.load(obs_path)['array']
            num_actions = actions.shape[1]
            num_obs = obs.shape[1]

            actions = torch.from_numpy(actions).long()
            obs = torch.from_numpy(obs).float()
            skill = torch.tensor(rank_to_skill[self.filename_to_rank[filename]])

            # Pad the start with 18 (boost and throttle) for actions and zeros for observations
            padded_actions = torch.ones((self.sequence_length + len(actions), num_actions), dtype=torch.long) * 18
            padded_obs = torch.zeros((self.sequence_length + len(obs), num_obs), dtype=torch.float)
            padded_actions[self.sequence_length:] = actions
            padded_obs[self.sequence_length:] = obs

            # Create sequences of the previous 10 timesteps of actions and obs
            for i in range(len(actions)):
                input_seq = {
                    'actions': padded_actions[i:i+self.sequence_length],
                    'obs': padded_obs[i:i+self.sequence_length]
                }
                target = skill
                data.append((input_seq, target))
        print(f"Skipped ranks: {list(skipped_ranks)}")
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return input_seq, target

