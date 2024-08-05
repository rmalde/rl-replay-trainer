import requests
import dotenv
import os
from typing import List

dotenv.load_dotenv()
BALLCHASING_API_KEY = os.getenv("BALLCHASING_API_KEY")

def get_replay_ids(ballchasing_params: dict = {}, verbose: bool = False) -> List[str]:
    if ballchasing_params == {}:
        ballchasing_params = {
        # "playlist": "ranked-duels",
        # # "season": "f15",
        # "pro": "true", 
        # "match-result": "win",
        # "min-rank": "grand-champion",
        "count": "10",  # Number of replays to return
        }
    url = "https://ballchasing.com/api/replays"
    headers = {"Authorization": BALLCHASING_API_KEY}
    
    response = requests.get(url, headers=headers, params=ballchasing_params)

    if response.status_code == 200:
        replay_info = response.json()
        if verbose:
            for replay in replay_info["list"]:
                print(replay["replay_title"])
        return [replay["id"] for replay in replay_info["list"]]
    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_replay_data(replay_id: str):
    url = f"https://ballchasing.com/api/replays/{replay_id}/file"
    headers = {"Authorization": BALLCHASING_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}, {response.text}")

def main():

    replay_ids = get_replay_ids()
    print(f"==>> replay_ids: {replay_ids}")

    data = get_replay_data(replay_ids[0])
    print(f"==>> data: {data}")

if __name__ == "__main__":
    main()