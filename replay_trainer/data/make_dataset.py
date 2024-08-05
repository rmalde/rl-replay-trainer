from tqdm.rich import tqdm
import argparse
import os
import sys
from typing import List
import time
import warnings

from replay_trainer.data.utils import get_replay_ids, get_replay_data
from replay_trainer.data.convert import ParsedReplay, replay_to_rlgym, ReplayFrame

# Suppress TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=UserWarning, message="rich is experimental/alpha")

def download_replays(replay_dir: str, n_download: int):
    ballchasing_params = {
        "playlist": "ranked-duels",
        "season": "f13", 
        # "match-result": "win",
        "min-rank": "grand-champion",
        "max-rank": "grand-champion",
        "count": n_download,
    }
    replay_ids = get_replay_ids(ballchasing_params, verbose=True)
    print(f"Found {len(replay_ids)} replays")
    if len(replay_ids) == 0:
        print("No replays found, adjust ballchasing_params")
        sys.exit(1)
    print(f"Downloading replays to {replay_dir}...")

    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)
    for replay_id in tqdm(replay_ids):
        data = get_replay_data(replay_id)

        with open(f"{replay_dir}/{replay_id}.replay", "wb") as f:
            f.write(data)
        time.sleep(1)
    

def main(args):
    # download fresh replays
    if args.n_download is not None:
        download_replays(args.replay_dir, args.n_download)

    if not os.path.exists(args.replay_dir) or len(os.listdir(args.replay_dir)) == 0:
        print("No replays found in replay_dir, make sure to specify --n_download")
        sys.exit(1)
    
    # convert to rlgym replays
    print("Converting replays to action and obs...")
    for replay_id in tqdm(os.listdir(args.replay_dir)):
        replay_path = os.path.abspath(os.path.join(args.replay_dir, replay_id))  

        parsed_replay = ParsedReplay.load(replay_path, from_wsl=True)

        replay_frames = replay_to_rlgym(parsed_replay)

        for frame in replay_frames:
            print("frame: ", frame)
            quit()





        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make dataset of replays')
    parser.add_argument('--n_download', type=int, default=None, help='Number of replays to download. If not specified, will use existing replays in replay-dir')
    parser.add_argument('--replay-dir', type=str, default='dataset/replays', help='Directory to store replays')
    args = parser.parse_args()

    main(args)
