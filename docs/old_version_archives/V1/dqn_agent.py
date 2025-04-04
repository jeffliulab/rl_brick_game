import torch
import torch.nn as nn  # Import torch.nn to fix the NameError
import torch.optim as optim
import numpy as np
from dqn_model import DQN
from replay_memory import ReplayMemory

class DQNAgent:
    def __init__(self, action_space):
        self.model = DQN(action_space)
        self.target_model = DQN(action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        # Ensure the state has the correct shape for the model
        if len(state.shape) == 3:  # If state shape is (channels, height, width)
            state = np.expand_dims(state, axis=0)  # Add batch dimension, making it (1, channels, height, width)
        
        with torch.no_grad():
            return self.model(torch.FloatTensor(state)).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample and convert to NumPy arrays for faster tensor creation
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
