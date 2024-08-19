import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# rank_to_skill = {
#     "bronze-1": 0.0,
#     "silver-1": 0.1,
#     "gold-1": 0.15,
#     "platinum-1": 0.20,
#     "diamond-1": 0.25,
#     "champion-1": 0.3,
#     "grand-champion-1": 0.4,
#     "grand-champion-2": 0.5,
#     "grand-champion-3": 0.6,
#     "supersonic-legend": 0.7,
#     "pro": 1.0
# }

rank_to_skill = {
    "bronze-1": 0.0,
    "silver-1": 0.2,
    "gold-1": 0.4,
    "platinum-1": 0.6,
    "diamond-1": 0.8,
    "champion-1": 1.0
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
    def __init__(self, dataset_dir, filenames, filename_to_rank):
        """
        Args:
            dataset_dir (str): Path to the dataset, ie 'dataset/ssl-1v1-100'
            filenames (list of str): List of names for the action/obs numpy files
            filename_to_rank (dict): Dictionary mapping filename to respective ranks
        """
        
        self.filenames = filenames
        self.dataset_dir = dataset_dir
        self.filename_to_rank = filename_to_rank
        self.data = self._load_data()

        # set obs and act space for external reference
        self.obs_size = len(self.data[0][0])
        self.action_size = 90


    def _load_data(self):
        data = []
        skipped_ranks = set()
        for filename in tqdm(self.filenames):
            if self.filename_to_rank[filename] not in rank_to_skill:
                skipped_ranks.add(self.filename_to_rank[filename])
                continue
            # actions_path = os.path.join(self.dataset_dir, 'actions', f'{filename}.npz')
            obs_path = os.path.join(self.dataset_dir, 'obs', f'{filename}.npz')
            
            # (seq_len, 111)
            obs = np.load(obs_path)['array']

            obs = torch.from_numpy(obs).float()
            skill = torch.tensor(rank_to_skill[self.filename_to_rank[filename]])

            for i in range(len(obs)):
                data.append((obs[i], skill))
        print(f"Skipped ranks: {list(skipped_ranks)}")
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # obs, skill
        return self.data[idx]

