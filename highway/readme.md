# Highway Environments Pipeline

This folder contains a pipeline to train policies for highway-env environments, collect dataset episodes from these policies, and then train sequence models on those datasets.

## Pipeline Steps

### 1. Train Environments (`train_env.py`)

First, train a PPO reinforcement learning agent on the target environments (e.g., `highway-v0`, `roundabout-v0`). 

```bash
python train_env.py
```

This script will run PPO training using `stable-baselines3`, save `ppo_model.zip` for the environments defined in `ENV_IDS`, and optionally save videos of the testing episodes. 

### 2. Collect Episodes (`collect_episodes.py`)

Once you have trained the PPO models, collect a dataset of states/observations of the ego and other vehicles. This dataset contains both safe trajectories and trajectories with collisions.

```bash
python collect_episodes.py --num_train 10000 --num_test 2000 --output_dir collision_data --multiprocess
```

- `--num_train`: The number of safe episodes to collect for the training dataset.
- `--num_test`: The number of testing episodes to collect (half secure, half collisions).
- `--output_dir`: The directory to save the HDF5 datasets.
- `--env_ids`: The environments to run data collection for.
- `--multiprocess`: Provide to run collections in parallel for different environment IDs.

The outputs are `{env_id}_train.h5` and `{env_id}_test.h5` saved in the output directory.

### 3. Train the Model (`train_paper.py`)

Finally, train a trajectory prediction model using the collected HDF5 datasets.

```bash
python train_paper.py --h5_path_train collision_data/highway-v0_train.h5 --h5_path_test collision_data/highway-v0_test.h5 --model_type encoder_mlp
```

Useful arguments:
- `--h5_path_train`: Path to the collected train HDF5 dataset.
- `--h5_path_test`: Path to the collected test HDF5 dataset.
- `--obs_len` & `--pred_len`: The lengths for observation and prediction horizons.
- `--model_type`: The architecture type (`transformer` or `encoder_mlp`).
- `--target_relative`: Instead of absolute coordinates for target values, predict relative differences.

The outputs (KDE plots, configuration files, and metrics) are saved under `outputs/`.
