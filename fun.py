import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from tetris_gymnasium.envs.tetris import Tetris


# Define the experience tuple
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Preprocess observation function
def preprocess_observation(observation):
    board = observation['board'].flatten()
    active_tetromino = observation['active_tetromino_mask'].flatten()
    holder = observation['holder'].flatten()
    queue = observation['queue'].flatten()
    processed_observation = np.concatenate((board, active_tetromino, holder, queue))
    processed_observation = processed_observation / 9.0
    return processed_observation

# DQN network
class DQN(nn.Module):
    def __init__(self, input_size, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(256, action_space)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

# Main training function
def train_dqn():
    env = gym.make("tetris_gymnasium/Tetris")
    observation, _ = env.reset()
    state = preprocess_observation(observation)
    input_size = len(state)
    action_space = env.action_space.n
    
    policy_net = DQN(input_size, action_space)
    target_net = DQN(input_size, action_space)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    num_episodes = 1000
    target_update = 10
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = preprocess_observation(observation)
        done = False
        total_reward = 0
        
        while not done:
            # Select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()
            
            # Perform action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_observation)
            done = terminated or truncated
            total_reward += reward
            
            # Store experience
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # Learn
            if len(memory) > BATCH_SIZE:
                experiences = memory.sample(BATCH_SIZE)
                batch = Experience(*zip(*experiences))
                
                state_batch = torch.FloatTensor(batch.state)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1)
                reward_batch = torch.FloatTensor(batch.reward)
                next_state_batch = torch.FloatTensor(batch.next_state)
                done_batch = torch.FloatTensor(batch.done)
                
                q_values = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q_values = target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + (GAMMA * max_next_q_values * (1 - done_batch))
                
                loss = nn.functional.mse_loss(q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Decay epsilon
            if epsilon > EPSILON_END:
                epsilon -= EPSILON_DECAY
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode}, Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    # Hyperparameters
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE = 1e-4
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 1e-4
    train_dqn()
