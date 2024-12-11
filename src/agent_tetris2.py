from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

import cv2
from src.tetris import Tetris
from collections import deque



WIDTH = 10  # Width of board
HEIGHT = 20  # Height of board
BLOCK_SIZE = 30  # Block size when rendering
BATCH_SIZE = 512  # High batch size
LEARNING_RATE = 1e-3
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 1e-3
NUM_DECAY_EPOCHS = 1350
NUM_EPOCHS = 2000
SAVE_INTERVAL = 500
REPLAY_MEMORY_SIZE = 22000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    
def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE)
    model = DQN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    state = env.reset().to(DEVICE)
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    epoch = 0

    while epoch < NUM_EPOCHS:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = FINAL_EPSILON + (max(NUM_DECAY_EPOCHS - epoch, 0) * 
                                   (INITIAL_EPSILON - FINAL_EPSILON) / NUM_DECAY_EPOCHS)
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(DEVICE)

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random() <= epsilon:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=True)
        next_state = next_state.to(DEVICE)
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset().to(DEVICE)
        else:
            state = next_state
            continue

        if len(replay_memory) < REPLAY_MEMORY_SIZE / 10:
            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), BATCH_SIZE))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch)).to(DEVICE)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(DEVICE)
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(DEVICE)

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + GAMMA * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            NUM_EPOCHS,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))

        if epoch > 0 and epoch % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), "policy_net.pth")

    torch.save(model.state_dict(), "policy_net.pth")
    return model


if __name__ == '__main__':
    agent_tetris = train()