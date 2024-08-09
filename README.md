# rl-replay-trainer
Train a model from rocket league replays

# Setup
```
pip install -e .
```

Use [replay-to-action-obs](https://github.com/rmalde/replay-to-action-) repo to generate dataset from ballchasing.com
Note you will HAVE to use windows to generate the dataset in the repo above, but after that, for this repo you're free to use any OS. 
Place the dataset in `dataset` directory.
The structure should be 
```
rl-replay-trainer/
|-- dataset/
    |--[dataset_name]
        |--actions
        |--obs
        |--replays
        idx_to_replay_id.csv
        [dataset_name].zip
```


## TODO:
- for batchnorm I should have a flag for eval mode so it used the train statistics during eval rather than computing new norm
- put th emetrics in the trainer in a separate class
