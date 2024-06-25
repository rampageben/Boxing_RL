# Boxing_RL Readme
## Custom DQN Training for Gym Environment
This project implements a Custom Deep Q-Network (DQN) for training on the "ALE/Boxing-v5" environment from OpenAI Gym.

### Prerequisites
Python 3.11.9
PyTorch
Gym
Stable Baselines3
TorchSummary
### Installation
Clone the repository:
```bash
git clone https://github.com/your_username/your_repo.git
cd your_repo
```
### Install the required packages:
+ torch
+ gym
+ stable-baselines3
+ torchsummary
+ numpy

### Model Architecture<br>

The Custom DQN model consists of convolutional layers followed by fully connected layers:  
+ **Convolutional Layers:**  
    + 3x3 kernel, stride 2, padding 1  
    - Channels: 3 -> 16 -> 32 -> 32 -> 16  
+ **Fully Connected Layers:**  
    + Input: Flattened features from convolutional layers  
    + Output: Number of actions
  
### Replay Buffer
A class to store and sample experiences:
+ Stores state, action, reward, next state, and done flag.
+ Samples experiences for training the DQN.
  
### Training
The training loop includes:
+ Action selection using epsilon-greedy strategy.
+ Storing experiences in the replay buffer.
+ Sampling from the replay buffer and updating the model.
+ Saving the best model based on the minimum loss.
  
### Evaluation
The model is evaluated using the `evaluate_policy` function from Stable Baselines3, providing mean and standard deviation of rewards over 10 episodes.

### Usage
Run the training script with optional parameters:

```python
train_agent(load_model=True, human_mode=False)
```

+ load_model (bool): Load a pre-trained model if available.
+ human_mode (bool): Run in human mode without training.

### Author
Rampagebing - adam23ben1012@gmail.com

