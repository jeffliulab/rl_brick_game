"""
Custom Breakout Environment and DQN Training Script - Fixed version with explicit shape debugging
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import pygame
import pygame.gfxdraw

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Ensure output directory exists
os.makedirs('models', exist_ok=True)

# Custom Breakout Environment
class BreakoutEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super(BreakoutEnv, self).__init__()
        
        # Pygame initialization
        pygame.init()
        
        # Screen parameters
        self.screen_width = 160
        self.screen_height = 210
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        # Paddle parameters
        self.paddle_width = 30
        self.paddle_height = 5
        self.paddle_speed = 5
        self.paddle_y = self.screen_height - 20
        self.paddle_x = self.screen_width // 2 - self.paddle_width // 2
        
        # Ball parameters
        self.ball_radius = 3
        self.ball_speed_x = 2
        self.ball_speed_y = -2
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        
        # Brick parameters
        self.brick_rows = 6
        self.brick_cols = 8
        self.brick_width = 18
        self.brick_height = 10
        self.brick_padding = 2
        self.brick_offset_top = 40
        self.brick_offset_left = 6
        self.bricks = []
        
        # Game state
        self.score = 0
        self.lives = 3
        self.terminated = False
        self.truncated = False
        
        # Render setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Colors
        self.colors = [
            (255, 0, 0),     # Red
            (255, 165, 0),   # Orange
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (75, 0, 130),    # Indigo
        ]
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Left, Stay, Right
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.screen_height, self.screen_width, 3), 
            dtype=np.uint8
        )
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset()
        
        # Reset paddle
        self.paddle_x = self.screen_width // 2 - self.paddle_width // 2
        
        # Reset ball
        self.ball_x = self.screen_width // 2
        self.ball_y = self.paddle_y - self.ball_radius - 1
        
        # Random ball direction
        self.ball_speed_x = random.choice([-2, 2])
        self.ball_speed_y = -2
        
        # Reset bricks
        self.bricks = []
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                brick_x = col * (self.brick_width + self.brick_padding) + self.brick_offset_left
                brick_y = row * (self.brick_height + self.brick_padding) + self.brick_offset_top
                self.bricks.append([brick_x, brick_y, True, self.colors[row % len(self.colors)]])
        
        # Reset game state
        self.score = 0
        self.lives = 3
        self.terminated = False
        self.truncated = False
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Move paddle based on action
        if action == 0:  # Left
            self.paddle_x = max(0, self.paddle_x - self.paddle_speed)
        elif action == 2:  # Right
            self.paddle_x = min(self.screen_width - self.paddle_width, self.paddle_x + self.paddle_speed)
        
        # Move ball
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        
        # Ball-wall collision
        if self.ball_x <= self.ball_radius or self.ball_x >= self.screen_width - self.ball_radius:
            self.ball_speed_x *= -1
        if self.ball_y <= self.ball_radius:
            self.ball_speed_y *= -1
        
        # Ball-paddle collision
        if (self.ball_y >= self.paddle_y - self.ball_radius and
            self.ball_y <= self.paddle_y + self.paddle_height and
            self.ball_x >= self.paddle_x and
            self.ball_x <= self.paddle_x + self.paddle_width):
            
            # Adjust ball angle based on where it hit the paddle
            relative_intersect_x = (self.paddle_x + self.paddle_width / 2) - self.ball_x
            normalized_intersect_x = relative_intersect_x / (self.paddle_width / 2)
            bounce_angle = normalized_intersect_x * (np.pi / 3)  # Max angle: 60 degrees
            
            # Calculate new velocity
            speed = np.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)
            self.ball_speed_x = speed * -np.sin(bounce_angle)
            self.ball_speed_y = speed * -np.cos(bounce_angle)
            
            # Ensure minimum vertical velocity
            if abs(self.ball_speed_y) < 1:
                self.ball_speed_y = -1 if self.ball_speed_y < 0 else 1
        
        # Ball-brick collision
        reward = 0
        for i, brick in enumerate(self.bricks):
            if brick[2]:  # If brick is active
                brick_x, brick_y, _, brick_color = brick
                if (self.ball_x >= brick_x and
                    self.ball_x <= brick_x + self.brick_width and
                    self.ball_y >= brick_y and
                    self.ball_y <= brick_y + self.brick_height):
                    
                    # Disable brick
                    self.bricks[i][2] = False
                    
                    # Reverse ball direction
                    self.ball_speed_y *= -1
                    
                    # Award points based on brick row (color)
                    row = (brick_y - self.brick_offset_top) // (self.brick_height + self.brick_padding)
                    points = (self.brick_rows - row) * 10
                    self.score += points
                    reward += points / 10  # Scale down reward
        
        # Check if ball is lost
        if self.ball_y >= self.screen_height:
            self.lives -= 1
            reward -= 1  # Penalty for losing a ball
            
            if self.lives <= 0:
                self.terminated = True
                reward -= 3  # Additional penalty for game over
            else:
                # Reset ball position
                self.ball_x = self.screen_width // 2
                self.ball_y = self.paddle_y - self.ball_radius - 1
                self.ball_speed_x = random.choice([-2, 2])
                self.ball_speed_y = -2
        
        # Check if all bricks are destroyed
        if all(not brick[2] for brick in self.bricks):
            self.terminated = True
            reward += 10  # Bonus for clearing all bricks
        
        # Cap maximum steps
        if self.score > 5000:
            self.truncated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, self.terminated, self.truncated, info
    
    def _get_obs(self):
        # Render the game state to the surface
        self._render_game()
        
        # Convert the pygame surface to a numpy array
        observation = pygame.surfarray.array3d(self.screen)
        
        # Transpose to match the expected shape (H, W, C)
        observation = np.transpose(observation, (1, 0, 2))
        
        return observation
    
    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "bricks_remaining": sum(1 for brick in self.bricks if brick[2]),
            "paddle_position": self.paddle_x,
            "ball_position": (self.ball_x, self.ball_y)
        }
    
    def _render_game(self):
        # Clear screen with black
        self.screen.fill((0, 0, 0))
        
        # Draw paddle (white)
        pygame.draw.rect(
            self.screen, (200, 200, 200),
            pygame.Rect(self.paddle_x, self.paddle_y, self.paddle_width, self.paddle_height)
        )
        
        # Draw ball (white)
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_x), int(self.ball_y), self.ball_radius, (200, 200, 200)
        )
        
        # Draw bricks
        for brick in self.bricks:
            if brick[2]:  # If brick is active
                pygame.draw.rect(
                    self.screen, brick[3],
                    pygame.Rect(brick[0], brick[1], self.brick_width, self.brick_height)
                )
        
        # Draw score and lives
        font = pygame.font.SysFont(None, 20)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (self.screen_width - 80, 10))
    
    def render(self):
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Custom Breakout")
            self.clock = pygame.time.Clock()
        
        if self.render_mode == "human":
            self.window.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

# Preprocess game frames
def preprocess_frame(frame):
    # Convert to grayscale
    gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    
    # Resize to 84x84
    # Using a consistent resize method
    from PIL import Image
    import numpy as np
    
    h, w = gray.shape
    # Convert to PIL Image for proper resizing
    img = Image.fromarray((gray * 255).astype(np.uint8)).resize((84, 84), Image.BILINEAR)
    
    # Convert back to numpy array and normalize
    resized = np.array(img).astype(np.float32) / 255.0
    
    return resized

# Stack frames for state representation
def stack_frames(frames, frame, is_new_episode):
    if is_new_episode:
        # For new episodes, create a deque with the same frame 4 times
        frames = deque([frame] * 4, maxlen=4)
    else:
        # Add the frame to the deque
        frames.append(frame)
    
    # Stack frames into a numpy array
    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames, frames

# DQN Network with proper shape handling
class DQNetwork(nn.Module):
    def __init__(self, input_shape=(4, 84, 84), action_size=3):
        super(DQNetwork, self).__init__()
        
        # Print the input shape for debugging
        print(f"Initializing DQNetwork with input shape {input_shape} and action size {action_size}")
        
        # Activation function
        self.relu = nn.ReLU()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self._forward_conv(dummy_input)
        self.fc_input_dim = int(np.prod(dummy_output.shape[1:]))
        
        print(f"Calculated fc_input_dim: {self.fc_input_dim}")
        
        # FC layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_size)
    
    def _forward_conv(self, x):
        print(f"Conv input shape: {x.shape}")
        x = self.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")
        x = self.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")
        x = self.relu(self.conv3(x))
        print(f"After conv3: {x.shape}")
        return x
    
    def forward(self, x):
        # Forward convolutional layers
        x = self._forward_conv(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        
        # Forward fully connected layers
        x = self.relu(self.fc1(x))
        print(f"After fc1: {x.shape}")
        x = self.fc2(x)
        print(f"After fc2: {x.shape}")
        return x

# DQN Agent
class DQNAgent:
    def __init__(self, state_size=(4, 84, 84), action_size=3, memory_size=10000, gamma=0.99, lr=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Main and target networks
        self.model = DQNetwork(state_size, action_size).to(device)
        # Disable print statements for target model
        with torch.no_grad():
            self.target_model = DQNetwork(state_size, action_size).to(device)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(q_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sample minibatch
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Print shapes for debugging
        print(f"States batch shape: {states.shape}")
        print(f"Actions batch shape: {actions.shape}")
        
        # Current Q values
        self.model.train()
        curr_q_values = self.model(states).gather(1, actions)
        
        # Print Q-values shape
        print(f"Current Q-values shape: {curr_q_values.shape}")
        
        # Next Q values from target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        
        # Target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(curr_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

def train_dqn():
    # Create environment
    env = BreakoutEnv(render_mode="rgb_array")
    print(f"Environment created. Action space: {env.action_space}, Observation space: {env.observation_space}")
    
    # Reset environment to get observation shape
    obs, _ = env.reset()
    print(f"Raw observation shape: {obs.shape}")
    
    # Process frame to get actual input shape
    processed_frame = preprocess_frame(obs)
    print(f"Processed frame shape: {processed_frame.shape}")
    
    # Create agent with correct shapes
    state_size = (4, processed_frame.shape[0], processed_frame.shape[1])
    action_size = env.action_space.n
    
    print(f"Using state size: {state_size}, action size: {action_size}")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training parameters
    num_episodes = 10  # Reduced for debugging
    batch_size = 32
    target_update_freq = 5
    save_freq = 5
    
    # Initial frame stack
    state, _ = env.reset()
    state = preprocess_frame(state)
    state, frames_deque = stack_frames([], state, True)
    
    print(f"Initial stacked state shape: {state.shape}")
    
    # Track performance
    scores = []
    
    print("Starting training...")
    
    # Starting with collection phase
    print("Collecting initial experiences...")
    done = False
    for _ in range(batch_size + 5):  # Collect a few more than batch size
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = preprocess_frame(next_state)
        next_state, frames_deque = stack_frames(frames_deque, next_state, False)
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            state, _ = env.reset()
            state = preprocess_frame(state)
            state, frames_deque = stack_frames([], state, True)
    
    # Main training loop
    for episode in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        state = preprocess_frame(state)
        state, frames_deque = stack_frames([], state, True)
        
        # Episode tracking
        done = False
        score = 0
        step = 0
        
        # Episode loop
        while not done:
            step += 1
            
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = preprocess_frame(next_state)
            next_state, frames_deque = stack_frames(frames_deque, next_state, False)
            
            # Store in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Train model
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if step % 10 == 0 and loss is not None:
                    print(f"Step {step}, Loss: {loss:.4f}")
            
            # Limit steps for debugging
            if step >= 100:
                break
        
        # Update target model
        if episode % target_update_freq == 0:
            agent.update_target_model()
            print(f"Episode {episode}: Target model updated")
        
        # Save model
        if episode % save_freq == 0:
            model_path = f"models/breakout_dqn_episode_{episode}.pth"
            torch.save(agent.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        # Add score to history
        scores.append(score)
        
        # Display episode stats
        print(f"Episode {episode}: Score = {score:.1f}, Steps: {step}, Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    torch.save(agent.model.state_dict(), "models/breakout_dqn_final.pth")
    print("Training complete! Final model saved.")
    env.close()

if __name__ == "__main__":
    try:
        train_dqn()
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        print(traceback.format_exc())