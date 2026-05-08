# Forecasting the Past
<img width="1909" height="560" alt="qualitative sample forecasting the past" src="https://github.com/user-attachments/assets/6d924bb4-83ad-40d8-b39b-a3669b9c263e" />
This repository is the official reference implementation for the paper: [Forecasting the Past](https://arxiv.org/abs/2604.12425).

## Repository Structure

The project is split into two main components:

### [`highway/`](highway/)
This directory contains the pipeline for generating driving simulations and training base models using `highway-env` and `stable-baselines3`.
- **Environment Training**: Scripts to train PPO reinforcement learning agents on highway and roundabout scenarios (`train_env.py`).
- **Dataset Generation**: Tooling to collect trajectories from the agent, explicitly capturing both safe episodes and collisions (`collect_episodes.py`).
- **Trajectory Modeling**: Base modeling approaches for sequence prediction tasks (Transformers/Encoder-MLP) directly on the simulated data (`train_paper.py`).
- **Automation**: An end-to-end automated setup and execution script (`run_pipeline.sh`).

For detailed usage, please see the [`highway/readme.md`](highway/readme.md).

### [`shifts/`](shifts/)
This directory focuses on shifted trajectory dynamics, advanced sequence modeling, and state-space (SS) decoders. It builds upon the generated data and introduces robust modeling architectures.
- **Decoding Strategies**: Custom trajectory decoding implementations and their corresponding loss formulations (`traj_decoder_ss.py`, `traj_decoder_ss_loss.py`).
- **Pre-trained Integration**: Scripts for running shifted trajectory predictions using pre-trained network encoders (`train_with_pretrain_encoder.py`).
- **Prediction Core**: Main trajectory prediction algorithms adapting to the distribution shifts (`traj_pred.py`).

## Getting Started

To get started with generating datasets or training the foundational models, head into the `highway/` directory which has its own `requirements.txt` and an automated startup bash script:

```bash
cd highway
bash run_pipeline.sh
```
