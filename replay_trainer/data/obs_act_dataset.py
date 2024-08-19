import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class ObsActDataset(Dataset):
    def __init__(self, dataset_dir, filenames):
        """
        Args:
            dataset_dir (str): Path to the dataset, ie 'dataset/ssl-1v1-100'
            filenames (list of str): List of names for the action/obs numpy files
        """
        
        self.filenames = filenames
        self.dataset_dir = dataset_dir
        self.data = self._load_data()

        # set obs and act space for external reference
        self.obs_size = len(self.data[0][0])
        self.action_size = 90


    def _load_data(self):
        data = []
        for filename in tqdm(self.filenames):
            actions_path = os.path.join(self.dataset_dir, 'actions', f'{filename}.npz')
            obs_path = os.path.join(self.dataset_dir, 'obs', f'{filename}.npz')

            actions = np.load(actions_path)['array']
            obs = np.load(obs_path)['array']

            actions = torch.from_numpy(actions).long()
            obs = torch.from_numpy(obs).float()

            # pair of obs and action from next timestep
            for i in range(len(obs) - 1):
                data.append((obs[i], actions[i+1]))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # obs, act
        return self.data[idx]

