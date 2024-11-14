# train_improvedDQN.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import json
import os
from datetime import datetime
import time
from skimage.color import rgb2gray
from skimage.transform import resize

def setup_device():
    """Setup and return the best available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

class DQN(nn.Module):
    def __init__(self, n_actions, frame_stack=4):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate size of flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_width * conv_width * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size):
        self.device = setup_device()
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_target_frequency = 1000
        self.batch_size = 32
        
        # Networks
        self.policy_net = DQN(action_size).to(self.device)
        self.target_net = DQN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Memory
        self.memory = deque(maxlen=100000)
        self.steps = 0
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        self.steps += 1
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def preprocess_frame(frame):
    """Convert frame to grayscale and resize"""
    gray_frame = rgb2gray(frame)
    resized_frame = resize(gray_frame, (84, 84), anti_aliasing=True)
    return np.array(resized_frame, dtype=np.float32)

def stack_frames(frames, frame, is_new_episode):
    """Stack frames for temporal information"""
    if is_new_episode:
        frames = deque([frame] * 4, maxlen=4)
    else:
        frames.append(frame)
    return np.array(frames), frames

def train(env, agent, num_episodes=10000, max_steps=50000):
    scores = []
    epsilon_history = []
    start_time = time.time()
    best_score = float('-inf')
    
    print("\n=== Training Configuration ===")
    print(f"Device: {agent.device}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Batch size: {agent.batch_size}")
    print("============================\n")
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state, _ = env.reset()
        state = preprocess_frame(state)
        frame_stack = deque(maxlen=4)
        state, frame_stack = stack_frames(frame_stack, state, True)
        score = 0
        steps = 0
        episode_losses = []
        
        for step in range(max_steps):
            steps += 1
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            next_state = preprocess_frame(next_state)
            next_state, frame_stack = stack_frames(frame_stack, next_state, False)
            
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            score += reward
            state = next_state
            
            if done:
                break
        
        # Record episode statistics
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        
        # Update best score and save model
        if score > best_score:
            best_score = score
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'score': score
            }, 'improved_dqn_breakout_model_best.pth')
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-100:] if len(scores) >= 100 else scores)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print("\n=== Training Progress ===")
            print(f"Episode: {episode + 1}/{num_episodes} ({(episode + 1)/num_episodes*100:.1f}% complete)")
            print(f"Time: {total_time/3600:.1f} hours elapsed")
            print(f"Memory: {len(agent.memory)}/{agent.memory.maxlen} transitions stored")
            print(f"Statistics:")
            print(f"  Current Score: {score:.1f}")
            print(f"  Best Score: {best_score:.1f}")
            print(f"  Average Score (last 100): {avg_score:.1f}")
            print(f"  Steps this episode: {steps}")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Episode time: {episode_time:.1f} seconds")
            
            if torch.cuda.is_available():
                print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            
            # Estimate remaining time
            avg_time_per_episode = total_time / (episode + 1)
            remaining_episodes = num_episodes - (episode + 1)
            estimated_time_remaining = avg_time_per_episode * remaining_episodes
            print(f"Estimated time remaining: {estimated_time_remaining/3600:.1f} hours")
            print("========================\n")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            checkpoint_path = f'improved_dqn_breakout_model_episode_{episode+1}.pth'
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'score': score,
                'scores': scores,
                'epsilon_history': epsilon_history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return scores, epsilon_history

if __name__ == "__main__":
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print CUDA information
    print("\nCUDA Configuration:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Device Count: {torch.cuda.device_count()}")
    
    # Create environment and agent
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    agent = ImprovedDQNAgent(state_size=(4, 84, 84), action_size=4)
    
    try:
        # Train the agent
        scores, epsilon_history = train(env, agent)
        
        # Save final results
        np.save('training_scores.npy', scores)
        np.save('epsilon_history.npy', epsilon_history)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Saving checkpoint...")
        torch.save({
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }, 'interrupted_training_checkpoint.pth')
        print("Checkpoint saved!")
        
    finally:
        env.close()