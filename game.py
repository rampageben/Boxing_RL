import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque
from stable_baselines3.common.evaluation import evaluate_policy
from torchsummary import summary


class CustomDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def _feature_size(self, input_shape):
        with torch.no_grad():
            input_tensor = torch.zeros(1, *input_shape)
            output_tensor = self.conv(input_tensor)
            return output_tensor.shape[1]

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            self._format_state(state),  # 格式化 state
            action,
            reward,
            self._format_state(next_state),  # 格式化 next_state
            done
        ))

    def _format_state(self, state):
        # 確保 state 是浮點型並有適當的形狀
        return np.array(state, dtype=np.float32)
    
    def sample(self, batch_size):
        sample_indices = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in sample_indices])
        
        return (
            np.stack(states),  # 使用 stack 確保維度一致
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=bool)
        )
    
    def __len__(self):
        return len(self.buffer)  # 返回緩衝區中的元素數量

def train_agent(total_timesteps=100000, load_model=False, human_mode=False):
    if human_mode:
        env = gym.make("ALE/Boxing-v5", render_mode='human')
        input_shape = (3, env.observation_space.shape[1], env.observation_space.shape[0])  # 調整維度順序
        num_actions = env.action_space.n
        device = torch.device("mps")
        model = CustomDQN(input_shape, num_actions).to(device)
        if load_model:
            try:
                model.load_state_dict(torch.load("./best_model.pth"))
                print("Model loaded successfully")
            except FileNotFoundError:
                print("No saved model found, training from scratch")
        print("Running in human mode, no training will be performed.")
        while True:
            state = env.reset()
            state = state[0]
            done = False
            while not done:
                env.render()
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0).permute(0, 3, 1, 2)
                    q_values = model(state_tensor)
                    action = q_values.argmax(dim=1).item()  # 選擇最優動作
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                done = terminated or truncated
        env.close()
        return
    env = gym.make("ALE/Boxing-v5", render_mode='rgb_array',difficulty = 1)
    input_shape = (3, env.observation_space.shape[1], env.observation_space.shape[0])  # 調整維度順序
    num_actions = env.action_space.n
    device = torch.device("mps")
    model = CustomDQN(input_shape, num_actions).to(device)

    if load_model:
        try:
            model.load_state_dict(torch.load("./best_model.pth"))
            print("Model loaded successfully")
        except FileNotFoundError:
            print("No saved model found, training from scratch")

    

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    replay_buffer = ReplayBuffer(buffer_size=100000)
    epsilon = 1.0  # 初始探索率
    epsilon_decay = 0.9995  # 探索率衰減
    epsilon_min = 0.01  # 最小探索率
    min_loss = float('inf')  # 初始化最小損失
    state = env.reset()
    state = state[0]
    
    for timestep in range(total_timesteps):
        env.render()
        if random.random() < epsilon:
            action = env.action_space.sample()  # 隨機選擇動作（探索）
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0).permute(0, 3, 1, 2)
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()  # 選擇最優動作（利用）
        next_state, reward, terminated, truncated, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, terminated or truncated)

        state = next_state
        
        if terminated or truncated:
            state = env.reset()
            state = state[0]

        if len(replay_buffer) > 10000:
            states, actions, rewards, next_states, dones = replay_buffer.sample(1024)
            states = torch.tensor(states, dtype=torch.float, device=device).permute(0, 3, 1, 2)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float, device=device).permute(0, 3, 1, 2)
            dones = torch.tensor(dones, dtype=torch.float, device=device)
            q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_q_values = model(next_states).max(1)[0]
            target_q_values = rewards+ (1 - dones) * 0.99 * next_q_values

            loss = nn.MSELoss()(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"timestep:{timestep}  Loss: {loss.item()}")
            
            # 保存具有最小損失的模型
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved with loss: {min_loss}")
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)  # 衰減探索率

    mean_reward, std_reward = evaluate_policy(
    lambda obs: model(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0).permute(0, 3, 1, 2)).argmax(dim=1).item(),
    env, n_eval_episodes=10
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    print("Finished training")

# 訓練模型時可以選擇是否加載先前的模型以及是否啟用人類模式
train_agent(load_model=True, human_mode=False)
