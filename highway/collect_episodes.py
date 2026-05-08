"""
HDF5 Dataset Structure:
-----------------------
Root Attributes:
    - env_id (str): The ID of the environment (e.g., 'intersection-v0').
    - num_episodes_simulated (int): Total number of episodes simulated.

Groups:
    - /episode_{i}: A group for each episode 'i'.
    
Group Attributes:
    - collision (bool): True if a collision occurred during the episode, False otherwise.
    
Datasets within each episode group:
    - ego_positions: A numpy array of shape (num_steps, 2) containing the (x, y) 
                     coordinates of the ego vehicle at each time step.
    - other_positions: A numpy array of shape (num_steps, max_others, 2) containing  
                       the (x, y) coordinates of all other vehicles. Since the number  
                       of vehicles can vary per step, the array is padded with NaNs.
"""

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
import h5py
import numpy as np
import os
import argparse
import multiprocessing as mp
import torch
# from train_env import ENV_IDS

ENV_IDS = ["roundabout-v0", "merge-v0", "intersection-v0", "highway-v0"]

def store_lane_info(env, env_id, output_dir, lock=None):
    if lock: lock.acquire()
    try:
        lane_pth_path = os.path.join(output_dir, f"{env_id}_lanes.pth")
        if not os.path.exists(lane_pth_path):
            all_lanes = env.unwrapped.road.network.lanes_list()
            lane_geometries = []
            for lane in all_lanes:
                length = lane.length
                width = int(lane.width_at(0)) if hasattr(lane, 'width_at') else lane.width
                longitudinals = np.arange(0, length, step=5)
                center_line = []
                for s in longitudinals:
                    x, y = lane.position(longitudinal=s, lateral=0)
                    heading = lane.heading_at(longitudinal=s)
                    center_line.append({"s": float(s), "x": float(x), "y": float(y), "heading": float(heading)})
                lane_geometries.append({
                    "type": type(lane).__name__,
                    "length": float(length),
                    "width": float(width),
                    "center_line": center_line
                })
            torch.save(lane_geometries, lane_pth_path)
            print(f"Saved lane geometries to {lane_pth_path}")
    finally:
        if lock: lock.release()

def collect_for_env(env_id, args, lock=None):
    model_path = f"{env_id}_ppo_model.zip"
    train_h5_path = os.path.join(args.output_dir, f"{env_id}_train.h5")
    test_h5_path = os.path.join(args.output_dir, f"{env_id}_test.h5")
    
    print(f"\n--- Collecting data for {env_id} ---")
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Skipping.")
        return

    model = PPO.load(model_path, device="cpu")
    env = gym.make(env_id, config={"policy_frequency": 5, "absolute": True})
    
    # Extract lane geometries
    store_lane_info(env, env_id, args.output_dir, lock)

    # Initialize HDF5 files (overwrite if exists) safely
    if lock: lock.acquire()
    with h5py.File(train_h5_path, 'w') as f_train, h5py.File(test_h5_path, 'w') as f_test:
        f_train.attrs['env_id'] = env_id
        f_test.attrs['env_id'] = env_id
    if lock: lock.release()
        
    target_test_safe = args.num_test // 2
    target_test_collision = args.num_test - target_test_safe
    
    train_safe_count = 0
    test_safe_count = 0
    test_collision_count = 0
    total_simulated = 0

    print(f"Target: {args.num_train} train (safe), {target_test_safe} test (safe), {target_test_collision} test (collision)")
    
    while train_safe_count < args.num_train or \
          test_safe_count < target_test_safe or \
          test_collision_count < target_test_collision:
          
        obs, info = env.reset()
        done = truncated = False
        
        episode_features = []
        collision_occurred = False

        while not (done or truncated):
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            new_obs, reward, done, truncated, info = env.step(action)
            
            # Get the ego vehicle
            ego_vehicle = env.unwrapped.vehicle
            
            # Check for collision
            if ego_vehicle.crashed:
                collision_occurred = True

            # Record all features from the observation directly
            # obs[0] is ego, obs[1:] are others
            episode_features.append({
                'ego': obs[0].copy(),
                'others': obs[1:].copy(),
                'collision': collision_occurred,
            })
            obs = new_obs

        # Determine destination file based on quotas
        dest_filename = None
        ep_name = ""
        
        if collision_occurred:
            if test_collision_count < target_test_collision:
                dest_filename = test_h5_path
                ep_name = f"episode_{test_safe_count + test_collision_count}"
                test_collision_count += 1
                print(f"Sim {total_simulated}: Collision! -> Test ({test_collision_count}/{target_test_collision})")
            else:
                print(f"Sim {total_simulated}: Collision! -> Discarded (Quota met)")
        else:
            if train_safe_count < args.num_train:
                dest_filename = train_h5_path
                ep_name = f"episode_{train_safe_count}"
                train_safe_count += 1
                print(f"Sim {total_simulated}: Safe -> Train ({train_safe_count}/{args.num_train})")
            elif test_safe_count < target_test_safe:
                dest_filename = test_h5_path
                ep_name = f"episode_{test_safe_count + test_collision_count}"
                test_safe_count += 1
                print(f"Sim {total_simulated}: Safe -> Test ({test_safe_count}/{target_test_safe})")
            else:
                print(f"Sim {total_simulated}: Safe -> Discarded (Quota met)")

        # Save the episode data to the chosen H5 file with lock
        if dest_filename is not None:
            num_steps = len(episode_features)
            feature_dim = episode_features[0]['ego'].shape[0]
            max_others = max(len(step['others']) for step in episode_features) if episode_features else 0
            
            # Prepare numpy arrays
            ego_arr = np.zeros((num_steps, feature_dim))
            others_arr = np.full((num_steps, max_others, feature_dim), np.nan)
            
            for t, step_data in enumerate(episode_features):
                ego_arr[t] = step_data['ego']
                for i, feat in enumerate(step_data['others']):
                    others_arr[t, i] = feat

            if lock: lock.acquire()
            with h5py.File(dest_filename, 'a') as f_dest:
                ep_group = f_dest.create_group(ep_name)
                ep_group.attrs['collision'] = collision_occurred
                ep_group.create_dataset('ego_positions', data=ego_arr, compression="gzip")
                ep_group.create_dataset('other_positions', data=others_arr, compression="gzip")
                
                # Update root attributes on the fly
                is_train = "train" in dest_filename
                f_dest.attrs['num_episodes_simulated'] = train_safe_count if is_train else (test_safe_count + test_collision_count)
            if lock: lock.release()
        
        total_simulated += 1

    env.close()
    print(f"\nSimulation complete for {env_id}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="collision_data2")
    parser.add_argument("--env_ids", nargs="+", default=ENV_IDS)
    parser.add_argument("--multiprocess", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.multiprocess:
        manager = mp.Manager()
        lock = manager.Lock()
        processes = []
        for env_id in args.env_ids:
            p = mp.Process(target=collect_for_env, args=(env_id, args, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for env_id in args.env_ids:
            collect_for_env(env_id, args, None)

if __name__ == "__main__":
    main()
