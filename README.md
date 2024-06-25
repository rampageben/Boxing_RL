# Boxing_RL
Custom DQN Training for Gym Environment
This project implements a Custom Deep Q-Network (DQN) for training on the "ALE/Boxing-v5" environment from OpenAI Gym.

Prerequisites
Python 3.x
PyTorch
Gym
Stable Baselines3
TorchSummary
Installation
Clone the repository:

bash
複製程式碼
git clone https://github.com/your_username/your_repo.git
cd your_repo
Install the required packages:

bash
複製程式碼
pip install torch gym stable-baselines3 torchsummary
Model Architecture
The Custom DQN model consists of convolutional layers followed by fully connected layers:

Convolutional Layers:
3x3 kernel, stride 2, padding 1
Channels: 3 -> 16 -> 32 -> 32 -> 16
Fully Connected Layers:
Input: Flattened features from convolutional layers
Output: Number of actions
Replay Buffer
A class to store and sample experiences:

Stores state, action, reward, next state, and done flag.
Samples experiences for training the DQN.
Training
The training loop includes:

Action selection using epsilon-greedy strategy.
Storing experiences in the replay buffer.
Sampling from the replay buffer and updating the model.
Saving the best model based on the minimum loss.
Evaluation
The model is evaluated using the evaluate_policy function from Stable Baselines3, providing mean and standard deviation of rewards over 10 episodes.

Usage
Run the training script with optional parameters:

python
複製程式碼
train_agent(load_model=True, human_mode=False)
load_model (bool): Load a pre-trained model if available.
human_mode (bool): Run in human mode without training.
Example
python
複製程式碼
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque
from stable_baselines3.common.evaluation import evaluate_policy
from torchsummary import summary

# CustomDQN and ReplayBuffer classes...

def train_agent(total_timesteps=100000, load_model=False, human_mode=False):
    # Environment setup and model initialization...
    # Training loop...

# Start training
train_agent(load_model=True, human_mode=False)
License
This project is licensed under the MIT License.

