import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import time

def inspect_model(path):
    """Inspect the contents of the saved model file"""
    print("\n=== Inspecting Model File ===")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print("\nCheckpoint type:", type(checkpoint))
        
        if isinstance(checkpoint, dict):
            print("\nCheckpoint keys:")
            for key in checkpoint.keys():
                print(f"- {key}")
            
            if 'model_state_dict' in checkpoint:
                print("\nModel state dict keys:")
                for key in checkpoint['model_state_dict'].keys():
                    print(f"- {key}")
                    if 'conv' in key or 'fc' in key:
                        param = checkpoint['model_state_dict'][key]
                        print(f"  Shape: {param.shape}")
        else:
            print("\nDirect state dict keys:")
            for key in checkpoint.keys():
                print(f"- {key}")
                if 'conv' in key or 'fc' in key:
                    param = checkpoint[key]
                    print(f"  Shape: {param.shape}")
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        # First convolutional layer + batch norm
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer + batch norm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer + batch norm
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Calculate the size of flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
            
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # Apply convolutions with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class BreakoutDQNAgent:
    def __init__(self, action_space=4, frame_stack_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        self.model = DQN(action_space).to(self.device)
        self.frame_stack_size = frame_stack_size
        self.frames = deque(maxlen=frame_stack_size)
        
    def preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        gray_frame = rgb2gray(frame)
        resized_frame = resize(gray_frame, (84, 84), anti_aliasing=True)
        return np.array(resized_frame, dtype=np.float32)
    
    def stack_frames(self, frame, is_new_episode=False):
        """Stack frames for temporal information"""
        if is_new_episode:
            self.frames.clear()
            for _ in range(self.frame_stack_size):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
            
        stacked_frames = np.stack(self.frames, axis=0)
        return stacked_frames
    
    def get_action(self, state):
        """Get action from model using current state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def load_model(self, path):
        """Load pretrained model weights with detailed error reporting"""
        print("\n=== Loading Model ===")
        try:
            # First inspect the model file
            inspect_model(path)
            
            # Load the checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Print model structure before loading
            print("\nCurrent model structure:")
            print(self.model)
            
            # Extract state dict based on checkpoint structure
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try loading the state dict
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            print("\nModel loaded successfully!")
            
            # Print training info if available
            if isinstance(checkpoint, dict):
                if 'episode' in checkpoint:
                    print(f"Loaded from episode: {checkpoint['episode']}")
                if 'score' in checkpoint:
                    print(f"Best score: {checkpoint['score']}")
                    
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            raise

def main():
    try:
        # Initialize environment and agent
        print("\n=== Initializing Environment and Agent ===")
        env = gym.make('ALE/Breakout-v5', render_mode='human')
        agent = BreakoutDQNAgent()
        
        # Load the model
        model_path = "improved_dqn_breakout_model_best.pth"
        agent.load_model(model_path)
        
        print("\n=== Starting Game ===")
        # Game loop
        episode_reward = 0
        lives = env.ale.lives()
        
        # Initial reset
        state, _ = env.reset()
        state = agent.preprocess_frame(state)
        stacked_state = agent.stack_frames(state, is_new_episode=True)
        
        # Fire to start the game
        env.step(1)  # Fire action
        time.sleep(0.5)  # Wait for ball to appear
        
        while True:
            # Get action from agent
            action = agent.get_action(stacked_state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check for life lost
            current_lives = env.ale.lives()
            life_lost = current_lives < lives
            lives = current_lives
            
            if life_lost:
                # Fire to restart after losing a life
                time.sleep(0.5)
                env.step(1)
                
            # Preprocess next state
            next_state = agent.preprocess_frame(next_state)
            stacked_state = agent.stack_frames(next_state, is_new_episode=life_lost)
            
            episode_reward += reward
            
            # Check for game over
            if terminated or truncated:
                print(f"\nGame Over! Total Score: {episode_reward}")
                break
            
            env.render()
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    main()