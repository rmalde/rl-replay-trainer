import torch
from torch.utils.data import DataLoader
from replay_trainer.models import FCN
import os
import numpy as np

OBS_SIZE = 99
ACT_SIZE = 90

def test_inference(model_path: str, act_path: str, obs_path: str):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 1
    model = FCN(OBS_SIZE, ACT_SIZE, sequence_length).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load data
    actions_full = np.load(act_path) # (frames, 1)
    obs_full = np.load(obs_path) # (frames, obs_size)

    idx = 0
    act = actions_full[idx:idx+sequence_length]
    obs = obs_full[idx:idx+sequence_length]
    act = np.array([16])
    obs = np.array([-0.0000e+00, -0.0000e+00,  4.0326e-02, -0.0000e+00, -0.0000e+00,
           0.0000e+00, -0.0000e+00, -0.0000e+00,  0.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
           1.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,  2.0035e+00,
           3.2930e-02,  0.0000e+00,  0.0000e+00, -1.1783e-04, -0.0000e+00,
          -2.0035e+00,  7.3957e-03,  4.3709e-08,  9.9995e-01, -9.5872e-03,
           4.1907e-10,  9.5872e-03,  9.9995e-01, -0.0000e+00, -0.0000e+00,
           1.1783e-04, -1.9417e-04, -0.0000e+00,  0.0000e+00,  3.4000e-01,
           1.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00, -2.0035e+00,
           3.2930e-02,  0.0000e+00,  0.0000e+00, -1.1783e-04, -0.0000e+00,
           2.0035e+00,  7.3957e-03,  4.3709e-08, -9.9995e-01, -9.5872e-03,
           4.1907e-10, -9.5872e-03,  9.9995e-01, -0.0000e+00, -0.0000e+00,
           1.1783e-04,  1.9417e-04, -0.0000e+00,  0.0000e+00,  3.4000e-01,
           1.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,  4.0070e+00,
           0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
    

    # Run inference
    act = torch.from_numpy(act).long().unsqueeze(0).to(device)
    obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    pred = model(act, obs)

    print(f"==>> obs: {obs}")
    print(f"==>> act: {act}")
    print(f"==>> pred: {pred.argmax(dim=1)}")
    print(f"==>> gt: {actions_full[idx + sequence_length]}")
    print(actions_full[:30].flatten())


if __name__ == "__main__":
    model_path = "checkpoints/fcn_t1_e32.pt"
    data_idx = "00000"
    dataset_path = "dataset/ssl-1v1-400"
    act_path = os.path.join(dataset_path, "actions", f"{data_idx}.npy")
    obs_path = os.path.join(dataset_path, "obs", f"{data_idx}.npy")
    test_inference(model_path, act_path, obs_path)