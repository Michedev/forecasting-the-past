import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import RecordVideo

import highway_env  # noqa: F401 (required to register the environments)

# =======================================================
# Configuration
# =======================================================
# ENV_IDS = ["roundabout-v0", "merge-v0", "intersection-v0", "highway-v0"]
ENV_IDS = ["highway-v0"]



TRAIN = True
TEST = True
TOTAL_TIMESTEPS = int(2e5)  # Increase to ~1M for better SOTA performance
NUM_CPU = 6                 # Adjust based on your machine's CPU cores
BATCH_SIZE = 64

def run_experiment(env_id):
    model_save_path = f"{env_id}_ppo_model"
    tensorboard_log = f"./logs/{env_id}_ppo/"

    if TRAIN:
        print(f"Starting training on {env_id} using {NUM_CPU} CPU cores...")
        
        # 1. Create a vectorized environment for faster data collection
        # We use SubprocVecEnv to run environments on separate processes
        env = make_vec_env(
            env_id, 
            n_envs=NUM_CPU, 
            vec_env_cls=SubprocVecEnv
        )

        # 2. Define the PPO model architecture
        # We use a slightly wider and deeper network (256, 256) since these
        # environments require slightly more complex reasoning than basic highway driving.
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=BATCH_SIZE * 12 // NUM_CPU,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,       # Discount factor. Lower gamma makes agent value short-term rewards (good for tight merges)
            verbose=2,
            tensorboard_log=tensorboard_log,
        )

        # 3. Train and save
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(model_save_path)
        print(f"Training completed and model saved to {model_save_path}.zip")
        
        # Clean up the vector env
        env.close()

    if TEST:
        print(f"Running evaluation on {env_id}...")
        
        # 1. Load the model
        model = PPO.load(model_save_path)

        # 2. Create a single environment for testing, with rgb_array for video recording
        eval_env = gym.make(env_id, render_mode="rgb_array")
        
        # Optionally, increase rendering framerate to make videos look smoother
        eval_env.unwrapped.config["simulation_frequency"] = 15
        
        # 3. Wrap with a video recorder to save the agent's performance
        eval_env = RecordVideo(
            eval_env, 
            video_folder=f"{env_id}_videos", 
            episode_trigger=lambda e: True # Record every episode
        )
        eval_env.unwrapped.set_record_video_wrapper(eval_env)

        # 4. Run the simulation
        for episode in range(5):  # Run 5 episodes
            obs, info = eval_env.reset()
            done = truncated = False
            while not (done or truncated):
                # Predict the next action
                action, _states = model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, done, truncated, info = eval_env.step(action)
                
                # Render (Required to generate the video frames)
                eval_env.render()
                
        eval_env.close()
        print(f"Evaluation complete. Videos saved in '{env_id}_videos/'")

def main():
    for env_id in ENV_IDS:
        print(f"\n{'='*50}\nStarting experiment for: {env_id}\n{'='*50}")
        run_experiment(env_id)

if __name__ == "__main__":
    main()