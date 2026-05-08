#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME="highway_pipeline"

# Initialize conda for the script
# This generic approach tries to source conditionally or uses eval
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda is not installed or not in PATH."
    exit 1
fi

echo "=================================="
echo "Creating and Activating Conda Env"
echo "=================================="
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

echo "=================================="
echo "Installing Requirements"
echo "=================================="
pip install -r requirements.txt

echo "=================================="
echo "Step 1: Training RL Agent"
echo "=================================="
# Trains the agent on highway-v0 (as set in train_env.py)
python train_env.py

echo "=================================="
echo "Step 2: Collecting Episodes Data"
echo "=================================="
# Collects episodes into the collision_data directory specifically for the environment we trained
python collect_episodes.py --output_dir collision_data --env_ids highway-v0

echo "=================================="
echo "Step 3: Training Trajectory Model"
echo "=================================="
# Trains the generative model using the collected highway-v0 data, targeting relative differences
python train_paper.py \
    --h5_path_train collision_data/highway-v0_train.h5 \
    --h5_path_test collision_data/highway-v0_test.h5 \
    --target_relative

echo "=================================="
echo "Pipeline finished successfully!"
echo "=================================="
