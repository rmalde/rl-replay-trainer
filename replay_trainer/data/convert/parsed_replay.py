import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# CARBALL_COMMAND = 'wine {} -i "{}" -o "{}" parquet'
CARBALL_COMMAND = '{} -i {} -o {} parquet'

ENV = os.environ.copy()
ENV["NO_COLOR"] = "1"
ENV["RUST_BACKTRACE"] = "1"

def wsl_to_windows_path(path: str, distro_name: str = "Ubuntu") -> str:
    # Check if the path starts with '/mnt/'
    path = str(path)
    if path.startswith("/mnt/"):
        # Extract the drive letter and replace it with Windows drive format
        drive_letter = path[5]
        windows_path = path.replace(f"/mnt/{drive_letter}/", f"{drive_letter.upper()}:\\")
    else:
        # Convert the WSL path to a UNC path using '\\wsl$'
        # Remove the leading slash to format it correctly
        windows_path = f"\\\\wsl$\\{distro_name}{path}"

    # Replace forward slashes with backslashes for Windows format
    windows_path = windows_path.replace("/", "\\")

    return windows_path


def process_replay(replay_path, output_folder, carball_path=None, skip_existing=True, from_wsl=False):
    if carball_path is None:
        # Use carball.exe in the same directory as this script
        carball_path = os.path.join(os.path.dirname(__file__), "carball.exe")
    folder, fn = os.path.split(replay_path)
    replay_name = fn.replace(".replay", "")
    
    processed_folder = os.path.join(output_folder, replay_name)
    if os.path.isdir(processed_folder) and len(os.listdir(processed_folder)) > 0:
        if skip_existing:
            return
        else:
            os.rmdir(processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    stdout_log_path = os.path.join(processed_folder, "carball.o.log")
    stderr_log_path = os.path.join(processed_folder, "carball.e.log")


    with open(os.path.join(processed_folder, "carball.o.log"), "w", encoding="utf8") as stdout_f:
        with open(os.path.join(processed_folder, "carball.e.log"), "w", encoding="utf8") as stderr_f:
            if from_wsl:
                replay_path = wsl_to_windows_path(replay_path)
                processed_folder = wsl_to_windows_path(processed_folder)
            command = CARBALL_COMMAND.format(carball_path, replay_path, processed_folder)
            print(f"==>> command: {command}")
            result = subprocess.run(
                command.split(),
                stdout=stdout_f,
                stderr=stderr_f,
                env=ENV
            )
            print("carball finished with code: ", result.returncode)
            print(result)

            print("\n=== carball.o.log ===")
            with open(stdout_log_path, "r", encoding="utf8") as stdout_f:
                print(stdout_f.read())

            print("\n=== carball.e.log ===")
            with open(stderr_log_path, "r", encoding="utf8") as stderr_f:
                print(stderr_f.read())

            return result


def load_parquet(*args, **kwargs):
    return pd.read_parquet(*args, engine="pyarrow", **kwargs)


@dataclass
class ParsedReplay:
    metadata: dict
    analyzer: dict
    game_df: pd.DataFrame
    ball_df: pd.DataFrame
    player_dfs: Dict[str, pd.DataFrame]

    @staticmethod
    def load(replay_dir, carball_path=None, from_wsl=False) -> "ParsedReplay":
        if isinstance(replay_dir, str):
            replay_dir = Path(replay_dir)

        def load_files(rdir):
            with (rdir / "metadata.json").open("r", encoding="utf8") as f:
                metadata = json.load(f)
            with (rdir / "analyzer.json").open("r", encoding="utf8") as f:
                analyzer = json.load(f)
            ball_df = load_parquet(rdir / "__ball.parquet")
            game_df = load_parquet(rdir / "__game.parquet")

            player_dfs = {}
            for player_file in rdir.glob("player_*.parquet"):
                player_id = player_file.name.split("_")[1].split(".")[0]
                player_dfs[player_id] = load_parquet(player_file)

            return ParsedReplay(metadata, analyzer, game_df, ball_df, player_dfs)

        if not replay_dir.is_dir():
            # Assume it's a replay file
            with tempfile.TemporaryDirectory() as temp_dir:
                process_replay(replay_dir, temp_dir, carball_path=carball_path, skip_existing=False, from_wsl=from_wsl)
                replay_dir = Path(temp_dir) / replay_dir.stem
                return load_files(replay_dir)
        else:
            return load_files(replay_dir)